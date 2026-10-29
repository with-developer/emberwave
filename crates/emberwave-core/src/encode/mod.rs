//! A PyroWave encoder on the CPU: 8-bit pictures in, frames of bitstream v1
//! out, each within a byte budget.
//!
//! It is the reference for the GPU encoder, as [`crate::cpu`] is for the GPU
//! decoder (decisions D8 and D28), and plain sequential Rust. The steps:
//!
//! 1. Each plane is padded to the aligned size and taken through five levels
//!    of the forward CDF 9/7 transform (`fdwt.rs`).
//! 2. Each 8x8 block of a transmitted band is quantised: a scale, dead-zone
//!    magnitudes, bit-planes per 4x2 subblock, and for dropping 0 to 15 of
//!    those planes what the block then costs and how far it is from the
//!    coefficients (`quant.rs`).
//! 3. The rate control chooses how many planes each 32x32 block drops so
//!    that the frame fits the budget, spending the bits where they reduce a
//!    contrast-weighted error most (`rate.rs`).
//! 4. The blocks are packed in block index order behind a start of frame
//!    header (`pack.rs`).
//!
//! A frame decodes with any decoder of bitstream v1, upstream's included.

pub(crate) mod fdwt;
pub(crate) mod pack;
pub(crate) mod quant;
pub(crate) mod rate;
mod weights;

use thiserror::Error;

use crate::bitstream::consts::{DECOMPOSITION_LEVELS, HEADER_BYTES, NUM_COMPONENTS};
use crate::bitstream::header::Header;
use crate::bitstream::packet::DEFAULT_PRISTINE_BANDS;
use crate::bitstream::{
    Band, ChromaSiting, ChromaSubsampling, ColorPrimaries, Component, Error, FrameLayout,
    FrameView, SequenceHeader, TransferFunction, YcbcrRange, YcbcrTransform,
};
use quant::{BandQuant, Blocks8};
use rate::BlockCost;

/// What a stream says about its colours. It changes nothing in the coding,
/// only the start of frame headers and a PWV1 file's header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Signalling {
    /// Colour primaries.
    pub color_primaries: ColorPrimaries,
    /// Transfer function.
    pub transfer_function: TransferFunction,
    /// Y'CbCr matrix.
    pub ycbcr_transform: YcbcrTransform,
    /// Sample range.
    pub ycbcr_range: YcbcrRange,
    /// Position of 4:2:0 chroma.
    pub chroma_siting: ChromaSiting,
}

impl Default for Signalling {
    /// BT.709 in video range with centred chroma: what most video is.
    fn default() -> Self {
        Self {
            color_primaries: ColorPrimaries::Srgb,
            transfer_function: TransferFunction::Srgb,
            ycbcr_transform: YcbcrTransform::Bt709,
            ycbcr_range: YcbcrRange::Limited,
            chroma_siting: ChromaSiting::Center,
        }
    }
}

/// What can go wrong while encoding a frame.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EncodeError {
    /// A plane does not have the size the stream's pictures have.
    #[error("the {component:?} plane has {actual} samples, {expected} expected")]
    PlaneSize {
        /// Which plane.
        component: Component,
        /// `width * height` of the plane.
        expected: usize,
        /// Samples given.
        actual: usize,
    },
    /// Not even the smallest frame the encoder can make fits the budget.
    #[error("a budget of {budget} bytes is less than the {minimum} bytes the smallest frame takes")]
    BudgetTooSmall {
        /// Bytes allowed.
        budget: usize,
        /// Bytes of the smallest frame.
        minimum: usize,
    },
    /// A plane's rows are closer together than the plane is wide.
    #[error(
        "the rows of the {component:?} plane are {stride} bytes apart, less than its {width} samples"
    )]
    PlaneStride {
        /// Which plane.
        component: Component,
        /// Bytes from the start of one row to the start of the next.
        stride: usize,
        /// Samples in a row.
        width: usize,
    },
    /// A frame to measure with [`coefficient_error`] does not decode.
    #[error("the frame does not decode: {0}")]
    Frame(Error),
}

/// The blocks that a partial decode needs whole, which every frame carries
/// even when they are all zero: those of the bands that
/// [`FrameAssembler::is_ready`](crate::bitstream::FrameAssembler::is_ready)
/// checks, the coarsest LL band and the detail bands of the coarsest
/// levels. They come first in block index order, every level's bands
/// before those of the next finer level, so they are the blocks below the
/// index returned (decision D58).
pub(crate) fn pristine_blocks(layout: &FrameLayout) -> u32 {
    const {
        // One band would be the coarsest LL band alone, whose blocks the
        // detail bands of each component interleave: not a prefix.
        // More than DECOMPOSITION_LEVELS - 1 bands, `is_ready` counts as that
        // many, which the prefix below would overshoot.
        assert!(DEFAULT_PRISTINE_BANDS >= 2 && DEFAULT_PRISTINE_BANDS < DECOMPOSITION_LEVELS);
    }
    let finest = DECOMPOSITION_LEVELS + 1 - DEFAULT_PRISTINE_BANDS;
    layout
        .bands()
        .find(|&(_, level, _, _)| level < finest)
        .map_or(layout.block_count(), |(_, _, _, band)| band.first_block)
}

/// The wavelet coefficients of a picture: `[component][level][band]`,
/// row-major at the size of the level's bands; empty where a component has
/// no such band.
pub(crate) type Bands = [[[Vec<f32>; 4]; DECOMPOSITION_LEVELS]; NUM_COMPONENTS];

/// One quantised 32x32 block and what the rate control needs to know of it.
#[derive(Debug, Clone)]
pub(crate) struct Block32 {
    pub index: u32,
    pub quant: BandQuant,
    pub blocks8: Blocks8,
    pub cost: BlockCost,
}

/// Encodes the pictures of one stream: one size and chroma format.
#[derive(Debug, Clone)]
pub struct Encoder {
    layout: FrameLayout,
    signalling: Signalling,
    /// Sequence number of the last frame, modulo 8.
    sequence: u8,
}

impl Encoder {
    /// An encoder for pictures of `width` x `height` pixels. Width and height
    /// are from 1 to 16384, and even for 4:2:0.
    pub fn new(
        width: u32,
        height: u32,
        chroma: ChromaSubsampling,
        signalling: Signalling,
    ) -> Result<Self, Error> {
        Ok(Self {
            layout: FrameLayout::new(width, height, chroma)?,
            signalling,
            sequence: 0,
        })
    }

    /// The block layout of the frames.
    pub fn layout(&self) -> &FrameLayout {
        &self.layout
    }

    /// The stream as a PWV1 file header describes it: size, chroma format
    /// and signalling, with the per-frame fields zero.
    pub fn stream(&self) -> SequenceHeader {
        stream_header(&self.layout, &self.signalling)
    }

    /// Encodes one picture into a frame of at most `budget` bytes: a start of
    /// frame header and the coded blocks in block index order, as a PWV1 file
    /// stores a frame. [`packets`] splits it for a network.
    ///
    /// `planes` are the Y, Cb and Cr samples, row-major without padding, at
    /// the sizes of [`FrameLayout::plane_size`].
    pub fn encode(&mut self, planes: [&[u8]; 3], budget: usize) -> Result<Vec<u8>, EncodeError> {
        check_planes(&self.layout, planes)?;
        let blocks = self.quantise(&self.transform(planes));
        let costs: Vec<BlockCost> = blocks.iter().map(|block| block.cost.clone()).collect();
        let budget_words = (budget.saturating_sub(HEADER_BYTES) / 4) as u64;
        let drops = (budget >= HEADER_BYTES)
            .then(|| rate::choose(&costs, budget_words))
            .flatten()
            .ok_or_else(|| {
                let smallest: usize = costs
                    .iter()
                    .map(|cost| usize::from(cost.words[usize::from(cost.max_drop)]) * 4)
                    .sum();
                EncodeError::BudgetTooSmall {
                    budget,
                    minimum: HEADER_BYTES + smallest,
                }
            })?;

        self.sequence = next_sequence(self.sequence);
        let mut frame = vec![0u8; HEADER_BYTES];
        let mut total_blocks = 0u32;
        let pristine = pristine_blocks(&self.layout);
        for (block, &drop) in blocks.iter().zip(&drops) {
            if block.cost.words[usize::from(drop)] == 0 {
                continue;
            }
            pack::pack_block(
                &block.blocks8,
                &block.quant,
                drop,
                self.sequence,
                block.index,
                block.index < pristine,
                &mut frame,
            );
            total_blocks += 1;
        }

        frame[..HEADER_BYTES].copy_from_slice(&start_of_frame(
            &self.stream(),
            self.sequence,
            total_blocks,
        ));
        debug_assert!(frame.len() <= budget);
        Ok(frame)
    }

    /// The forward transform of a picture whose planes have the right sizes.
    pub(crate) fn transform(&self, planes: [&[u8]; 3]) -> Bands {
        let layout = &self.layout;
        let mut bands: Bands = Default::default();
        for (component, plane) in Component::ALL.into_iter().zip(planes) {
            let (width, height) = layout.plane_size(component);
            let first = layout.finest_level(component);
            let (mut level_width, mut level_height) = (
                (layout.aligned_width() >> first) as usize,
                (layout.aligned_height() >> first) as usize,
            );
            let mut image = fdwt::pad_plane(
                plane,
                width as usize,
                height as usize,
                level_width,
                level_height,
            );
            let levels = &mut bands[component.index()];
            for slot in levels.iter_mut().skip(first) {
                let fdwt::Level { ll, hl, lh, hh } =
                    fdwt::forward_2d(image, level_width, level_height);
                slot[Band::Hl.index()] = hl;
                slot[Band::Lh.index()] = lh;
                slot[Band::Hh.index()] = hh;
                image = ll;
                level_width /= 2;
                level_height /= 2;
            }
            levels[DECOMPOSITION_LEVELS - 1][Band::Ll.index()] = image;
        }
        bands
    }

    /// Quantises the coefficients: every block of the frame, in block index
    /// order, with its costs.
    pub(crate) fn quantise(&self, bands: &Bands) -> Vec<Block32> {
        let layout = &self.layout;
        let pristine = pristine_blocks(layout);
        let mut blocks = Vec::with_capacity(layout.block_count() as usize);
        for (component, level, band, band_layout) in layout.bands() {
            let coefficients = &bands[component.index()][level][band.index()];
            let quant = BandQuant::new(layout.chroma(), component, level, band);
            for block_y in 0..band_layout.blocks_y {
                for block_x in 0..band_layout.blocks_x {
                    let blocks8: Blocks8 = std::array::from_fn(|raster| {
                        let values = quant::gather_block8(
                            coefficients,
                            band_layout,
                            block_x,
                            block_y,
                            raster,
                        );
                        quant::analyse_block8(&values, &quant)
                    });
                    let index = band_layout.first_block + block_y * band_layout.blocks_x + block_x;
                    let mut cost = BlockCost {
                        words: [0; quant::DROP_STEPS],
                        distortion: [0.0; quant::DROP_STEPS],
                        max_drop: quant.max_drop(),
                    };
                    for drop in 0..quant::DROP_STEPS {
                        cost.words[drop] = pack::block_words(&blocks8, drop as u8);
                        if index < pristine {
                            cost.words[drop] = cost.words[drop].max(pack::EMPTY_BLOCK_WORDS);
                        }
                        cost.distortion[drop] =
                            blocks8.iter().flatten().map(|b| b.distortion[drop]).sum();
                    }
                    blocks.push(Block32 {
                        index,
                        quant,
                        blocks8,
                        cost,
                    });
                }
            }
        }
        blocks
    }
}

/// The header of a PWV1 file of a stream of `layout` with `signalling`.
pub(crate) fn stream_header(layout: &FrameLayout, signalling: &Signalling) -> SequenceHeader {
    SequenceHeader {
        width: layout.width(),
        height: layout.height(),
        sequence: 0,
        total_blocks: 0,
        code: 0,
        chroma: layout.chroma(),
        color_primaries: signalling.color_primaries,
        transfer_function: signalling.transfer_function,
        ycbcr_transform: signalling.ycbcr_transform,
        ycbcr_range: signalling.ycbcr_range,
        chroma_siting: signalling.chroma_siting,
    }
}

/// The sequence number of the frame after one numbered `sequence`. The first
/// frame is number 1, as with upstream's encoder.
pub(crate) fn next_sequence(sequence: u8) -> u8 {
    (sequence + 1) & 0x7
}

/// The start of frame header of frame `sequence` of `stream`, which carries
/// `total_blocks` blocks.
pub(crate) fn start_of_frame(
    stream: &SequenceHeader,
    sequence: u8,
    total_blocks: u32,
) -> [u8; HEADER_BYTES] {
    let header = SequenceHeader {
        sequence,
        total_blocks,
        ..*stream
    };
    let (word0, word1) = header.start_of_frame_words();
    let mut bytes = [0u8; HEADER_BYTES];
    bytes[..4].copy_from_slice(&word0.to_le_bytes());
    bytes[4..].copy_from_slice(&word1.to_le_bytes());
    bytes
}

/// An error unless `planes` have the sizes of [`FrameLayout::plane_size`].
pub(crate) fn check_planes(layout: &FrameLayout, planes: [&[u8]; 3]) -> Result<(), EncodeError> {
    for (component, plane) in Component::ALL.into_iter().zip(planes) {
        let (width, height) = layout.plane_size(component);
        let expected = width as usize * height as usize;
        if plane.len() != expected {
            return Err(EncodeError::PlaneSize {
                component,
                expected,
                actual: plane.len(),
            });
        }
    }
    Ok(())
}

/// An error unless `planes` hold the planes of [`FrameLayout::plane_size`]
/// with their rows `strides` bytes apart: each at least its width, and at
/// least `(height - 1) * stride + width` bytes in the plane. For the GPU
/// encoder's strided input.
#[cfg(feature = "gpu")]
pub(crate) fn check_strided_planes(
    layout: &FrameLayout,
    planes: [&[u8]; 3],
    strides: [usize; 3],
) -> Result<(), EncodeError> {
    for ((component, plane), stride) in Component::ALL.into_iter().zip(planes).zip(strides) {
        let (width, height) = layout.plane_size(component);
        let (width, height) = (width as usize, height as usize);
        if stride < width {
            return Err(EncodeError::PlaneStride {
                component,
                stride,
                width,
            });
        }
        let expected = (height - 1).saturating_mul(stride).saturating_add(width);
        if plane.len() < expected {
            return Err(EncodeError::PlaneSize {
                component,
                expected,
                actual: plane.len(),
            });
        }
    }
    Ok(())
}

/// How far a frame is from the picture it was encoded from, in the terms of
/// the rate control: the squared error of the wavelet coefficients, per
/// component, Y, Cb and Cr.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CoefficientError {
    /// The squared error, weighted per band by the eye's contrast sensitivity
    /// as the rate control weighs it (decision D28).
    pub weighted: [f64; NUM_COMPONENTS],
    /// The squared error, unweighted.
    pub squared: [f64; NUM_COMPONENTS],
}

impl CoefficientError {
    /// The weighted error of the three components: what the rate control
    /// makes as small as the budget allows.
    pub fn weighted_total(&self) -> f64 {
        self.weighted.iter().sum()
    }
}

/// The coefficient error of `frame` against the picture `planes`, the
/// samples of [`Encoder::encode`] at the sizes of the frame's layout: the
/// error the rate control makes as small as it can, so that frames of any
/// encoder, upstream's too, can be compared in its terms. Blocks the frame
/// does not carry count as zero coefficients, as a decoder decodes them.
pub fn coefficient_error(
    planes: [&[u8]; 3],
    frame: &FrameView<'_>,
) -> Result<CoefficientError, EncodeError> {
    let layout = frame.layout();
    check_planes(layout, planes)?;
    let encoder = Encoder {
        layout: layout.clone(),
        signalling: Signalling::default(),
        sequence: 0,
    };
    let picture = encoder.transform(planes);
    let decoded = crate::cpu::dequant::dequantize(frame).map_err(EncodeError::Frame)?;
    let mut error = CoefficientError::default();
    for (component, level, band, _) in layout.bands() {
        let c = component.index();
        let coded = decoded.plane(component, level, band).unwrap_or_default();
        // The decoder's bands are the layout's, as the encoder's are.
        debug_assert_eq!(coded.len(), picture[c][level][band.index()].len());
        let squared: f64 = picture[c][level][band.index()]
            .iter()
            .zip(coded)
            .map(|(&a, &b)| f64::from(a - b).powi(2))
            .sum();
        let weight = BandQuant::new(layout.chroma(), component, level, band).weight;
        error.weighted[c] += f64::from(weight) * squared;
        error.squared[c] += squared;
    }
    Ok(error)
}

/// Splits an encoded frame into packets of at most `max_bytes` each, at
/// block boundaries. The first packet starts with the start of frame header.
/// A block larger than `max_bytes` gets a packet of its own, and so does a
/// frame that is not well formed from the first malformed block on.
pub fn packets(frame: &[u8], max_bytes: usize) -> Vec<&[u8]> {
    let mut packets = Vec::new();
    let mut start = 0;
    let mut end = HEADER_BYTES.min(frame.len());
    while end < frame.len() {
        let size = match Header::parse(&frame[end..]) {
            Ok(Header::Block(header)) if header.payload_bytes() >= HEADER_BYTES => {
                header.payload_bytes().min(frame.len() - end)
            }
            _ => frame.len() - end,
        };
        if end - start + size > max_bytes && end > start {
            packets.push(&frame[start..end]);
            start = end;
        }
        end += size;
    }
    if end > start {
        packets.push(&frame[start..end]);
    }
    packets
}

/// The header of the block that starts `bytes`, for the tests.
#[cfg(test)]
fn block_header(bytes: &[u8]) -> Option<crate::bitstream::header::BlockHeader> {
    match Header::parse(bytes) {
        Ok(Header::Block(header)) => Some(header),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
