//! Slow, single-threaded CPU decoder that follows `bitstream.md` literally.
//!
//! The real decoder runs these stages on the GPU. This module exists so that
//! the bitstream parser can be checked against real encoder output without a
//! GPU, and so that shader stages can be compared against something simple.
//! It is not meant to be fast.

pub mod dequant;
pub mod idwt;

use crate::bitstream::consts::{BLOCK_SIZE, DC_SHIFT, DECOMPOSITION_LEVELS, NUM_COMPONENTS};
use crate::bitstream::{Band, Component, Error, FrameLayout, FrameView};
use dequant::Bands;

/// One decoded colour plane with samples in `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Plane {
    /// Width in samples.
    pub width: u32,
    /// Height in samples.
    pub height: u32,
    /// Row-major samples.
    pub samples: Vec<f32>,
}

impl Plane {
    /// Converts to 8-bit UNORM the way a GPU storage write does: scale to
    /// 255 and round to nearest, ties to even.
    pub fn to_unorm8(&self) -> Vec<u8> {
        self.samples
            .iter()
            .map(|v| (v * 255.0).round_ties_even() as u8)
            .collect()
    }

    /// Converts to 16-bit UNORM, rounding to nearest, ties to even.
    pub fn to_unorm16(&self) -> Vec<u16> {
        self.samples
            .iter()
            .map(|v| (v * 65535.0).round_ties_even() as u16)
            .collect()
    }
}

/// A decoded frame: Y, Cb and Cr planes.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedFrame {
    /// The planes in bitstream order (Y, Cb, Cr). Chroma planes are half
    /// size for 4:2:0.
    pub planes: [Plane; 3],
}

/// The LL band that enters the inverse transform of each level, per
/// component: level 4 comes from the bitstream and the others are produced by
/// the level above. Row-major with the width of the level as stride.
#[derive(Debug, Clone, Default)]
pub struct LlBands {
    levels: [[Option<Vec<f32>>; DECOMPOSITION_LEVELS]; NUM_COMPONENTS],
}

impl LlBands {
    /// The LL band of `level`, or `None` if the component has no such level.
    pub fn get(&self, component: Component, level: usize) -> Option<&[f32]> {
        self.levels.get(component.index())?.get(level)?.as_deref()
    }
}

/// Decodes a frame: dequantisation, inverse DWT and the DC shift.
///
/// Blocks that were not received decode as zero coefficients.
pub fn decode_frame(frame: &FrameView<'_>) -> Result<DecodedFrame, Error> {
    let layout = frame.layout();
    let mut bands = dequant::dequantize(frame)?;
    let planes = Component::ALL.map(|component| reconstruct(layout, &mut bands, component, None));
    Ok(DecodedFrame { planes })
}

/// Like [`decode_frame`], and also returns the intermediate LL bands.
pub fn decode_frame_with_ll(frame: &FrameView<'_>) -> Result<(DecodedFrame, LlBands), Error> {
    let layout = frame.layout();
    let mut bands = dequant::dequantize(frame)?;
    let mut ll_bands = LlBands::default();
    let planes = Component::ALL.map(|component| {
        reconstruct(
            layout,
            &mut bands,
            component,
            Some(&mut ll_bands.levels[component.index()]),
        )
    });
    Ok((DecodedFrame { planes }, ll_bands))
}

/// The largest change of the coarsest level around a block of detail, in the
/// coefficients' units, at which a frame drawn in part keeps the last
/// decode's coefficients for the block when it lacks it (decision D66).
/// Measured to keep a still picture's detail whole, and to stay within
/// 0.03 dB of the zeros of before on pictures that moved by one to eight
/// pixels a frame.
pub const CONCEALMENT_THRESHOLD: f32 = 0.03;

/// Decodes the frames of a stream one after another, as a client draws
/// them, and keeps what the last decode left of the frame's detail, so that
/// a frame drawn in part conceals the blocks it lacks where the picture is
/// still (decision D66). The GPU decoder does the same with
/// [`set_concealment`](crate::gpu::Decoder::set_concealment).
#[derive(Debug, Clone, Default)]
pub struct Concealer {
    /// The coefficients the last decode left, and the layout of its frames.
    last: Option<(FrameLayout, Bands)>,
}

impl Concealer {
    /// A concealer that has decoded nothing yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// The coefficients the last decode left, concealment included, as the
    /// GPU decoder's pyramid holds them after the same decode
    /// ([`gpu::Decoder::read_band`](crate::gpu::Decoder::read_band) reads
    /// the bands that are not LL, and the coarsest LL).
    pub fn bands(&self) -> Option<&Bands> {
        self.last.as_ref().map(|(_, bands)| bands)
    }

    /// Decodes `frame`. A whole frame decodes as [`decode_frame`] does. A
    /// frame drawn in part, `partial`, lacks blocks that were sent: where
    /// it lacks one of detail, below the coarsest level, it takes the last
    /// decode's coefficients for the block if the coarsest level around the
    /// block changed by at most [`CONCEALMENT_THRESHOLD`] since then, and
    /// zeros otherwise, as for every block it lacks when it is the first.
    pub fn decode(&mut self, frame: &FrameView<'_>, partial: bool) -> Result<DecodedFrame, Error> {
        let layout = frame.layout();
        let mut bands = dequant::dequantize(frame)?;
        // Not a let chain: those need Rust 1.88, and emberwave builds on 1.87.
        if let (true, Some((last_layout, last))) = (partial, &self.last) {
            if last_layout == layout {
                conceal(frame, &mut bands, last);
            }
        }
        let kept = bands.clone();
        let planes =
            Component::ALL.map(|component| reconstruct(layout, &mut bands, component, None));
        self.last = Some((layout.clone(), kept));
        Ok(DecodedFrame { planes })
    }
}

/// Puts the `last` decode's coefficients into the blocks of detail that
/// `frame` lacks where the picture is still around them (see
/// [`Concealer::decode`]).
fn conceal(frame: &FrameView<'_>, bands: &mut Bands, last: &Bands) {
    let layout = frame.layout();
    let coarsest = DECOMPOSITION_LEVELS - 1;
    for block_index in 0..layout.block_count() {
        if frame.block(block_index).is_some() {
            continue;
        }
        let Some(location) = layout.locate(block_index) else {
            continue;
        };
        if location.level == coarsest
            || coarse_change(
                layout,
                bands,
                last,
                location.component,
                location.level,
                location.block_x,
                location.block_y,
            ) > CONCEALMENT_THRESHOLD
        {
            continue;
        }
        let (Some(band), Some(old), Some(plane)) = (
            layout.band(location.component, location.level, location.band),
            last.plane(location.component, location.level, location.band),
            bands.plane_mut(location.component, location.level, location.band),
        ) else {
            continue;
        };
        let (width, height) = (band.width as usize, band.height as usize);
        let x0 = (location.block_x * BLOCK_SIZE) as usize;
        let y0 = (location.block_y * BLOCK_SIZE) as usize;
        let x1 = (x0 + BLOCK_SIZE as usize).min(width);
        for y in y0..(y0 + BLOCK_SIZE as usize).min(height) {
            plane[y * width + x0..y * width + x1]
                .copy_from_slice(&old[y * width + x0..y * width + x1]);
        }
    }
}

/// The largest change, between `last` and `current`, of the coarsest
/// level's four bands of `component` where the block (`block_x`,
/// `block_y`) of `level` lies, one coefficient around included.
fn coarse_change(
    layout: &FrameLayout,
    current: &Bands,
    last: &Bands,
    component: Component,
    level: usize,
    block_x: u32,
    block_y: u32,
) -> f32 {
    let coarsest = DECOMPOSITION_LEVELS - 1;
    let shift = coarsest - level;
    let (x0, y0) = (
        ((block_x * BLOCK_SIZE) >> shift).saturating_sub(1),
        ((block_y * BLOCK_SIZE) >> shift).saturating_sub(1),
    );
    // The block's end falls on a coefficient of the coarsest level: 32 is a
    // multiple of every 2^shift.
    let reach = |block: u32| (((block + 1) * BLOCK_SIZE) >> shift) + 1;
    let mut largest = 0.0f32;
    for band in Band::ALL {
        let (Some(info), Some(now), Some(then)) = (
            layout.band(component, coarsest, band),
            current.plane(component, coarsest, band),
            last.plane(component, coarsest, band),
        ) else {
            continue;
        };
        let width = info.width as usize;
        for y in y0..reach(block_y).min(info.height) {
            for x in x0..reach(block_x).min(info.width) {
                let at = y as usize * width + x as usize;
                largest = largest.max((now[at] - then[at]).abs());
            }
        }
    }
    largest
}

/// Runs the inverse transform from the coarsest level up for one component,
/// optionally keeping a copy of the LL band that enters each level.
fn reconstruct(
    layout: &FrameLayout,
    bands: &mut Bands,
    component: Component,
    mut trace: Option<&mut [Option<Vec<f32>>; DECOMPOSITION_LEVELS]>,
) -> Plane {
    let coarsest = DECOMPOSITION_LEVELS - 1;
    let (coarsest_width, coarsest_height) = layout.level_size(coarsest);
    let (mut width, mut height) = (coarsest_width as usize, coarsest_height as usize);

    let mut take = |level: usize, band: Band, width: usize, height: usize| {
        bands
            .take(component, level, band)
            .unwrap_or_else(|| vec![0.0; width * height])
    };

    let mut ll = take(coarsest, Band::Ll, width, height);
    for level in (layout.finest_level(component)..=coarsest).rev() {
        if let Some(trace) = trace.as_deref_mut() {
            trace[level] = Some(ll.clone());
        }
        let hl = take(level, Band::Hl, width, height);
        let lh = take(level, Band::Lh, width, height);
        let hh = take(level, Band::Hh, width, height);
        ll = idwt::inverse_2d(&ll, &hl, &lh, &hh, width, height);
        width *= 2;
        height *= 2;
    }

    // Crop the padding away and undo the DC shift.
    let (plane_width, plane_height) = layout.plane_size(component);
    let mut samples = Vec::with_capacity(plane_width as usize * plane_height as usize);
    for y in 0..plane_height as usize {
        let row = &ll[y * width..y * width + plane_width as usize];
        samples.extend(row.iter().map(|v| (v + DC_SHIFT).clamp(0.0, 1.0)));
    }

    Plane {
        width: plane_width,
        height: plane_height,
        samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unorm_conversion_rounds_ties_to_even() {
        let plane = Plane {
            width: 4,
            height: 1,
            samples: vec![0.0, 1.0, 0.5, 2.5 / 255.0],
        };
        // 0.5 * 255 = 127.5 -> 128 (even); 2.5 -> 2 (even).
        assert_eq!(plane.to_unorm8(), [0, 255, 128, 2]);
        assert_eq!(plane.to_unorm16()[..2], [0, 65535]);
    }

    use crate::bitstream::header::Header;
    use crate::bitstream::{ChromaSubsampling, FrameAssembler};
    use crate::encode::{Encoder, Signalling, packets, pristine_blocks};

    /// A 192x128 4:2:0 picture with texture everywhere, its luma raised by
    /// `lift`.
    fn picture(lift: u8) -> [Vec<u8>; 3] {
        let (width, height) = (192, 128);
        let texture = |x: usize, y: usize| ((x * 37 + y * 101 + (x * y) % 23) % 171) as u8 + 20;
        [
            (0..width * height)
                .map(|i| texture(i % width, i / width) + lift)
                .collect(),
            vec![110; width * height / 4],
            vec![140; width * height / 4],
        ]
    }

    /// The frame of `planes`, and the assembler with its packets but every
    /// third that holds blocks of detail only, as a stream that lost them.
    fn coded(planes: &[Vec<u8>; 3]) -> (Vec<u8>, FrameAssembler) {
        let mut encoder =
            Encoder::new(192, 128, ChromaSubsampling::Yuv420, Signalling::default()).unwrap();
        let frame = encoder
            .encode([&planes[0], &planes[1], &planes[2]], 20_000)
            .unwrap();
        let layout = encoder.layout().clone();
        let pristine = pristine_blocks(&layout);
        let mut assembler = FrameAssembler::new(layout);
        for (n, packet) in packets(&frame, 64).into_iter().enumerate() {
            let detail = match Header::parse(packet).unwrap() {
                Header::Block(block) => block.block_index >= pristine,
                Header::StartOfFrame(_) => false,
            };
            if !(detail && n % 3 == 0) {
                assembler.push_packet(packet).unwrap();
            }
        }
        assert!(assembler.received_blocks() < assembler.total_blocks());
        (frame, assembler)
    }

    fn whole(frame: &[u8], layout: &FrameLayout) -> DecodedFrame {
        let mut assembler = FrameAssembler::new(layout.clone());
        assembler.push_packet(frame).unwrap();
        decode_frame(&assembler.frame()).unwrap()
    }

    #[test]
    fn a_still_picture_drawn_in_part_conceals_what_it_lacks_with_the_last_frame() {
        // The same picture twice: the frame drawn in part takes the first
        // frame's coefficients for every block it lacks, and so decodes as
        // the whole frame does.
        let planes = picture(0);
        let (first, _) = coded(&planes);
        let (second, partial) = coded(&planes);
        let layout = partial.layout().clone();
        let mut concealer = Concealer::new();
        let mut assembler = FrameAssembler::new(layout.clone());
        assembler.push_packet(&first).unwrap();
        assert_eq!(
            concealer.decode(&assembler.frame(), false).unwrap(),
            whole(&first, &layout)
        );
        let concealed = concealer.decode(&partial.frame(), true).unwrap();
        assert_eq!(concealed, whole(&second, &layout));
        assert_ne!(concealed, decode_frame(&partial.frame()).unwrap());
    }

    #[test]
    fn a_changed_picture_drawn_in_part_keeps_zeros_where_it_lacks_blocks() {
        // The picture brighter everywhere: the coarsest level changed around
        // every block, so none takes the last frame's coefficients.
        let (first, _) = coded(&picture(0));
        let (_, partial) = coded(&picture(40));
        let layout = partial.layout().clone();
        let mut concealer = Concealer::new();
        let mut assembler = FrameAssembler::new(layout.clone());
        assembler.push_packet(&first).unwrap();
        concealer.decode(&assembler.frame(), false).unwrap();
        assert_eq!(
            concealer.decode(&partial.frame(), true).unwrap(),
            decode_frame(&partial.frame()).unwrap()
        );
    }

    #[test]
    fn without_a_last_frame_or_of_another_size_a_frame_in_part_has_zeros() {
        let (_, partial) = coded(&picture(0));
        let zeros = decode_frame(&partial.frame()).unwrap();
        let mut concealer = Concealer::new();
        assert_eq!(concealer.decode(&partial.frame(), true).unwrap(), zeros);
        // A frame of another size before: nothing to take from it.
        let mut concealer = Concealer::new();
        let other = FrameLayout::new(128, 128, ChromaSubsampling::Yuv420).unwrap();
        let mut encoder =
            Encoder::new(128, 128, ChromaSubsampling::Yuv420, Signalling::default()).unwrap();
        let frame = encoder
            .encode([&[90; 128 * 128], &[128; 64 * 64], &[128; 64 * 64]], 10_000)
            .unwrap();
        let mut assembler = FrameAssembler::new(other);
        assembler.push_packet(&frame).unwrap();
        concealer.decode(&assembler.frame(), false).unwrap();
        assert_eq!(concealer.decode(&partial.frame(), true).unwrap(), zeros);
    }
}
