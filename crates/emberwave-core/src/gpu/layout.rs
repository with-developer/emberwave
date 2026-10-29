//! Buffer layouts and shader parameter blocks of the GPU pipeline.
//!
//! Everything here is plain arithmetic over a [`FrameLayout`], so it is
//! tested without a GPU. The parameter structs mirror the uniform blocks in
//! the WGSL files field for field.

use bytemuck::{Pod, Zeroable};

use crate::bitstream::consts::{BANDS_PER_LEVEL, DECOMPOSITION_LEVELS, NUM_COMPONENTS};
use crate::bitstream::{Band, ChromaSiting, ChromaSubsampling, Component, FrameLayout, YcbcrRange};
use crate::color::{CHROMA_CENTRE, CHROMA_SCALE, ColorParams, INV_255, LUMA_BLACK, LUMA_SCALE};
use crate::encode::quant::{BandQuant, DROP_STEPS};

use super::GpuError;

/// Largest number of transmitted bands: four at level 4, three at each of the
/// other levels, for three components. Sizes the band tables of
/// `dequant.wgsl` and `quantise.wgsl`.
pub(crate) const MAX_BANDS: usize =
    NUM_COMPONENTS * (BANDS_PER_LEVEL + (DECOMPOSITION_LEVELS - 1) * (BANDS_PER_LEVEL - 1));

/// Invocations per workgroup of `yuv2rgb.wgsl`.
const YUV_GROUP_SIZE: u64 = 256;

/// Largest workgroup grid dimension that every backend supports.
pub(crate) const MAX_GRID_DIMENSION: u32 = 65_535;

/// Where everything lives in the GPU buffers, in elements (floats or words).
#[derive(Debug, Clone)]
pub(crate) struct BufferLayout {
    /// Float offset of band `[component][level][band]` in the pyramid.
    band_offsets: [[[u32; BANDS_PER_LEVEL]; DECOMPOSITION_LEVELS]; NUM_COMPONENTS],
    /// Size of the pyramid in floats.
    pub pyramid_floats: u64,
    /// Word offset of each packed 8-bit plane.
    pub packed_offsets: [u32; NUM_COMPONENTS],
    /// Words per row of each packed plane.
    pub packed_strides: [u32; NUM_COMPONENTS],
    /// Size of the packed planes in words.
    pub packed_words: u64,
}

/// Marks a band that does not exist in `band_offsets`.
const ABSENT: u32 = u32::MAX;

/// Words per row of a decoder's packed planes are a multiple of this: rows
/// of a multiple of 256 bytes, which a copy from a buffer into a texture
/// takes, so that a plane goes into a texture with a copy (decision D41).
pub(crate) const DECODER_ROW_WORDS: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT / 4;

/// A decoder's packed planes start at a multiple of this many words: 512
/// bytes, `D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT`, where Direct3D 12 copies
/// into a texture from. From other offsets wgpu copies through a buffer it
/// allocates on the spot, and panics when that fails (decision D41).
pub(crate) const DECODER_PLANE_WORDS: u32 = 512 / 4;

impl BufferLayout {
    /// The layout of an encoder: rows of the packed planes padded to whole
    /// words only, and the planes one after the other.
    pub(crate) fn new(layout: &FrameLayout) -> Result<Self, GpuError> {
        Self::with_alignment(layout, 1, 1)
    }

    /// The layout of a decoder: rows of the packed planes padded to a
    /// multiple of [`DECODER_ROW_WORDS`], planes at multiples of
    /// [`DECODER_PLANE_WORDS`].
    pub(crate) fn for_decoder(layout: &FrameLayout) -> Result<Self, GpuError> {
        Self::with_alignment(layout, DECODER_ROW_WORDS, DECODER_PLANE_WORDS)
    }

    /// The layout with packed rows of a multiple of `row_words` words, and
    /// planes that start at multiples of `plane_words`.
    fn with_alignment(
        layout: &FrameLayout,
        row_words: u32,
        plane_words: u32,
    ) -> Result<Self, GpuError> {
        let mut band_offsets = [[[ABSENT; BANDS_PER_LEVEL]; DECOMPOSITION_LEVELS]; NUM_COMPONENTS];
        let mut pyramid_floats = 0u64;

        for component in Component::ALL {
            // 4:2:0 chroma has no finest level.
            let finest = layout.finest_level(component);
            for (level, offsets) in band_offsets[component.index()]
                .iter_mut()
                .enumerate()
                .skip(finest)
            {
                let (width, height) = layout.level_size(level);
                for band in Band::ALL {
                    offsets[band.index()] = to_offset(pyramid_floats)?;
                    pyramid_floats += u64::from(width) * u64::from(height);
                }
            }
        }

        let mut packed_offsets = [0; NUM_COMPONENTS];
        let mut packed_strides = [0; NUM_COMPONENTS];
        let mut packed_words = 0u64;
        for component in Component::ALL {
            let (width, height) = layout.plane_size(component);
            let stride = width.div_ceil(4).next_multiple_of(row_words);
            packed_words = packed_words.next_multiple_of(u64::from(plane_words));
            packed_offsets[component.index()] = to_offset(packed_words)?;
            packed_strides[component.index()] = stride;
            packed_words += u64::from(stride) * u64::from(height);
        }

        Ok(Self {
            band_offsets,
            pyramid_floats,
            packed_offsets,
            packed_strides,
            packed_words,
        })
    }

    /// Float offset of a band in the pyramid, if the component has that level.
    pub(crate) fn band_offset(
        &self,
        component: Component,
        level: usize,
        band: Band,
    ) -> Option<u32> {
        let offset = *self
            .band_offsets
            .get(component.index())?
            .get(level)?
            .get(band.index())?;
        (offset != ABSENT).then_some(offset)
    }
}

/// Buffer offsets must fit the `u32` the shaders index with.
fn to_offset(value: u64) -> Result<u32, GpuError> {
    u32::try_from(value).map_err(|_| GpuError::TooLarge)
}

/// A workgroup grid for `count` workgroups that stays within the per-dimension
/// limit: `(x, y)` with `x * y >= count`. Shaders rebuild the linear index as
/// `y * x_dimension + x`.
pub(crate) fn grid_for(count: u64) -> (u32, u32) {
    let max = u64::from(MAX_GRID_DIMENSION);
    let x = count.clamp(1, max);
    let y = count.div_ceil(x).max(1);
    (x as u32, y as u32)
}

/// One transmitted band in the dequantisation band table.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct BandInfo {
    pub first_block: u32,
    pub blocks_x: u32,
    pub width: u32,
    pub height: u32,
    pub dst_offset: u32,
    pub pad: [u32; 3],
}

/// Uniform block of `dequant.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct DequantParams {
    pub band_count: u32,
    pub block_count: u32,
    pub grid_width: u32,
    pub pad: u32,
    pub bands: [BandInfo; MAX_BANDS],
}

impl DequantParams {
    pub(crate) fn new(layout: &FrameLayout, buffers: &BufferLayout) -> Self {
        let mut params = Self::zeroed();
        let mut count = 0;
        for (component, level, band, info) in layout.bands() {
            params.bands[count] = BandInfo {
                first_block: info.first_block,
                blocks_x: info.blocks_x,
                width: info.width,
                height: info.height,
                dst_offset: buffers
                    .band_offset(component, level, band)
                    .expect("a transmitted band has pyramid space"),
                pad: [0; 3],
            };
            count += 1;
        }
        params.band_count = count as u32;
        params.block_count = layout.block_count();
        params.grid_width = grid_for(u64::from(layout.block_count())).0;
        params
    }

    /// Workgroup grid of the dispatch: one workgroup per block.
    pub(crate) fn grid(&self) -> (u32, u32) {
        grid_for(u64::from(self.block_count))
    }
}

/// One transmitted band in the quantisation table of the encoder.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct QuantBand {
    pub first_block: u32,
    pub blocks_x: u32,
    pub width: u32,
    pub height: u32,
    /// Float offset of the band in the pyramid.
    pub offset: u32,
    /// [`BandQuant::exponent`].
    pub exponent: u32,
    /// [`BandQuant::weight`].
    pub weight: f32,
    /// [`BandQuant::max_drop`].
    pub max_drop: u32,
}

/// Uniform block of `equant.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct QuantParams {
    pub band_count: u32,
    pub block_count: u32,
    pub grid_width: u32,
    /// The blocks every frame carries, even empty: those below this index
    /// (`encode::pristine_blocks`).
    pub pristine_blocks: u32,
    pub bands: [QuantBand; MAX_BANDS],
}

impl QuantParams {
    /// The band table, with the CPU encoder's quantisation of every band.
    pub(crate) fn new(layout: &FrameLayout, buffers: &BufferLayout) -> Self {
        let mut params = Self::zeroed();
        let mut count = 0;
        for (component, level, band, info) in layout.bands() {
            let quant = BandQuant::new(layout.chroma(), component, level, band);
            params.bands[count] = QuantBand {
                first_block: info.first_block,
                blocks_x: info.blocks_x,
                width: info.width,
                height: info.height,
                offset: buffers
                    .band_offset(component, level, band)
                    .expect("a transmitted band has pyramid space"),
                // 7 to 12, positive.
                exponent: quant.exponent as u32,
                weight: quant.weight,
                max_drop: u32::from(quant.max_drop()),
            };
            count += 1;
        }
        params.band_count = count as u32;
        params.block_count = layout.block_count();
        params.grid_width = grid_for(u64::from(layout.block_count())).0;
        params.pristine_blocks = crate::encode::pristine_blocks(layout);
        params
    }

    /// Workgroup grid of the dispatch: one workgroup per block.
    pub(crate) fn grid(&self) -> (u32, u32) {
        grid_for(u64::from(self.block_count))
    }
}

/// What `equant.wgsl` writes for every block: the CPU encoder's
/// [`BlockCost`](crate::encode::rate::BlockCost) in 32-bit words.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub(crate) struct CostRecord {
    pub words: [u32; DROP_STEPS],
    pub distortion: [f32; DROP_STEPS],
    pub max_drop: u32,
}

#[cfg(test)]
impl CostRecord {
    /// The record as the CPU encoder's rate control takes it, or `None` if a
    /// number does not fit its type, which no shader writes.
    pub(crate) fn to_cost(self) -> Option<crate::encode::rate::BlockCost> {
        let mut words = [0u16; DROP_STEPS];
        for (word, &value) in words.iter_mut().zip(&self.words) {
            *word = u16::try_from(value).ok()?;
        }
        Some(crate::encode::rate::BlockCost {
            words,
            distortion: self.distortion,
            max_drop: u8::try_from(self.max_drop).ok()?,
        })
    }

    /// The CPU encoder's record in 32-bit words.
    pub(crate) fn from_cost(cost: &crate::encode::rate::BlockCost) -> Self {
        Self {
            words: cost.words.map(u32::from),
            distortion: cost.distortion,
            max_drop: u32::from(cost.max_drop),
        }
    }
}

/// Blocks per workgroup of the dispatches of `rate.wgsl` over the blocks.
pub(crate) const RATE_GROUP_SIZE: u64 = 64;

/// The most words a block takes: a header and 16 8x8 blocks of 3 bytes of
/// tables, 8 subblocks of at most 18 bit-planes (15 base planes and 3 more)
/// and 64 sign bits. `pack.wgsl` assembles blocks of up to this size.
pub(crate) const MAX_BLOCK_WORDS: u64 = (8 + 16 * (3 + 8 * 18 + 8)) / 4;

/// Values a workgroup of `scan.wgsl` sums, and the most chunks of that many
/// it sums: 2^20 values at most.
pub(crate) const SCAN_CHUNK: u64 = 1024;

/// Uniform block of `rate.wgsl` and `pack.wgsl`, written for every frame.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct FrameParams {
    pub block_count: u32,
    /// Workgroups in x of the dispatches over the blocks, 64 blocks each.
    pub grid_width: u32,
    /// Words the blocks may take: the budget less the start of frame header.
    pub budget_words: u32,
    pub sequence: u32,
    /// Words the payload buffer holds.
    pub capacity_words: u32,
    pub pad: [u32; 3],
}

/// Workgroup grid of the dispatches of `rate.wgsl` over the blocks.
pub(crate) fn rate_grid(layout: &FrameLayout) -> (u32, u32) {
    grid_for(u64::from(layout.block_count()).div_ceil(RATE_GROUP_SIZE))
}

/// Number of keys of the rate control, the size of its histogram.
pub(crate) const RATE_KEYS: u64 = 512;

/// What the CPU reads of `rate.wgsl`'s state, which follows the histogram.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub(crate) struct RateSummary {
    /// Words of all blocks with nothing dropped.
    pub total: u32,
    /// Words of all blocks with all dropped that they may drop.
    pub minimum: u32,
    /// Blocks that are sent.
    pub sent: u32,
    pub cutoff_key: u32,
    pub remaining: u32,
    /// 1 if the frame fits its budget.
    pub fits: u32,
}

/// Bytes of `rate.wgsl`'s state: the histogram and the summary.
pub(crate) const RATE_STATE_BYTES: u64 = RATE_KEYS * 4 + std::mem::size_of::<RateSummary>() as u64;

/// Uniform block of `scan.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct ScanParams {
    pub count: u32,
    pub chunk_count: u32,
    pub grid_width: u32,
    pub pad: u32,
}

impl ScanParams {
    /// The parameters of a scan of `count` values, or an error if they are
    /// more than two passes of the scan can sum.
    pub(crate) fn new(count: u32) -> Result<Self, GpuError> {
        let chunk_count = u64::from(count).div_ceil(SCAN_CHUNK);
        if chunk_count > SCAN_CHUNK {
            return Err(GpuError::TooLarge);
        }
        Ok(Self {
            count,
            chunk_count: chunk_count as u32,
            grid_width: grid_for(chunk_count).0,
            pad: 0,
        })
    }

    /// Workgroup grid of the dispatches over the chunks.
    pub(crate) fn grid(&self) -> (u32, u32) {
        grid_for(u64::from(self.chunk_count))
    }
}

/// Uniform block of `gather.wgsl`: where each component's plane goes in the
/// planes buffer, and which channel of its texture holds it.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct GatherParams {
    /// Per component, Y, Cb and Cr (the fourth is unused): the plane's width
    /// and height in samples.
    pub width: [u32; 4],
    pub height: [u32; 4],
    /// Where the plane's rows start in the planes buffer, and words per row.
    pub offset: [u32; 4],
    pub stride: [u32; 4],
    /// The channel of the component in its texture: 0 red, 1 green.
    pub channel: [u32; 4],
}

impl GatherParams {
    /// The parameters for pictures of `layout`, with Cr in the green channel
    /// of its texture if `cr_in_green`.
    pub(crate) fn new(layout: &FrameLayout, buffers: &BufferLayout, cr_in_green: bool) -> Self {
        let mut params = Self::zeroed();
        for component in Component::ALL {
            let i = component.index();
            (params.width[i], params.height[i]) = layout.plane_size(component);
            params.offset[i] = buffers.packed_offsets[i];
            params.stride[i] = buffers.packed_strides[i];
        }
        params.channel[Component::Cr.index()] = u32::from(cr_in_green);
        params
    }

    /// Workgroups of the dispatch: the words of the longest row in x and the
    /// pairs of rows of the tallest plane in y, 16 of each to a workgroup,
    /// and the components in z.
    pub(crate) fn grid(&self) -> (u32, u32, u32) {
        let words = self.stride.iter().copied().max().unwrap_or(0);
        let rows = self.height.iter().copied().max().unwrap_or(0);
        (words.div_ceil(16), rows.div_ceil(2).div_ceil(16), 3)
    }
}

/// Per component part of a transform level's uniform block.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct CompParams {
    /// Float offsets of the LL, HL, LH and HH bands in the pyramid.
    pub band_offset: [u32; 4],
    /// Where the level's image is: at the component's finest level the packed
    /// 8-bit plane, in words, otherwise the LL band of the next finer level,
    /// in floats.
    pub image_offset: u32,
    /// Row length of the image, in the same unit.
    pub image_stride: u32,
    /// Size of the plane at the finest level, the size of the image otherwise.
    pub plane_width: u32,
    pub plane_height: u32,
    /// 1 at the component's finest level, whose image is the packed plane.
    pub is_plane: u32,
    pub pad: [u32; 3],
}

/// Uniform block of `idwt.wgsl` and `fdwt.wgsl` for one decomposition level.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct LevelParams {
    pub band_width: u32,
    pub band_height: u32,
    /// Size of the level's image, twice that of its bands.
    pub image_width: u32,
    pub image_height: u32,
    pub comps: [CompParams; NUM_COMPONENTS],
}

/// One level of the wavelet transform: parameters and dispatch size.
///
/// A level works between its four bands and its image, twice their size: the
/// inverse transform (`idwt.wgsl`) combines the bands into the image and the
/// forward transform (`fdwt.wgsl`) splits the image into the bands, so both
/// take the same parameters.
#[derive(Debug, Clone)]
pub(crate) struct TransformLevel {
    pub params: LevelParams,
    /// Number of components that have this level: the z size of the dispatch.
    pub components: u32,
}

impl TransformLevel {
    /// Parameters for the transform between the bands of `level` and its image.
    pub(crate) fn new(layout: &FrameLayout, buffers: &BufferLayout, level: usize) -> Self {
        let (band_width, band_height) = layout.level_size(level);
        let mut params = LevelParams {
            band_width,
            band_height,
            image_width: 2 * band_width,
            image_height: 2 * band_height,
            ..LevelParams::zeroed()
        };

        let mut components = 0;
        for component in Component::ALL {
            // Components without this level are skipped; they are always the
            // last ones, so the z index of a dispatch is the component index.
            if level < layout.finest_level(component) {
                continue;
            }

            let offsets = Band::ALL.map(|band| {
                buffers
                    .band_offset(component, level, band)
                    .expect("the level exists for this component")
            });

            let comp = &mut params.comps[component.index()];
            comp.band_offset = offsets;
            if level == layout.finest_level(component) {
                let (width, height) = layout.plane_size(component);
                comp.image_offset = buffers.packed_offsets[component.index()];
                comp.image_stride = buffers.packed_strides[component.index()];
                comp.plane_width = width;
                comp.plane_height = height;
                comp.is_plane = 1;
            } else {
                // The LL band of the next finer level is twice as wide.
                comp.image_offset = buffers
                    .band_offset(component, level - 1, Band::Ll)
                    .expect("the next finer level exists");
                comp.image_stride = 2 * band_width;
                comp.plane_width = 2 * band_width;
                comp.plane_height = 2 * band_height;
                comp.is_plane = 0;
            }
            components += 1;
        }

        Self { params, components }
    }

    /// Workgroup grid: 32x32 tiles of the image in x and y, components in z.
    pub(crate) fn grid(&self) -> (u32, u32, u32) {
        (
            self.params.image_width.div_ceil(32),
            self.params.image_height.div_ceil(32),
            self.components,
        )
    }
}

/// Uniform block of `yuv2rgb.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct YuvParams {
    pub width: u32,
    pub height: u32,
    pub chroma_420: u32,
    pub siting_left: u32,
    pub y_offset: u32,
    pub y_stride: u32,
    pub cb_offset: u32,
    pub cb_stride: u32,
    pub cr_offset: u32,
    pub cr_stride: u32,
    pub chroma_width: u32,
    pub chroma_height: u32,
    pub limited: u32,
    pub grid_width: u32,
    /// 1: blue in the low byte of the words, for a copy into a BGRA texture.
    pub bgra: u32,
    pub pad: u32,
    pub coefficients: [f32; 4],
    pub levels: [f32; 4],
    pub scale: [f32; 4],
}

impl YuvParams {
    pub(crate) fn new(layout: &FrameLayout, buffers: &BufferLayout, color: &ColorParams) -> Self {
        let (chroma_width, chroma_height) = layout.plane_size(Component::Cb);
        let mut params = Self {
            width: layout.width(),
            height: layout.height(),
            chroma_420: u32::from(layout.chroma() == ChromaSubsampling::Yuv420),
            siting_left: u32::from(color.chroma_siting == ChromaSiting::Left),
            y_offset: buffers.packed_offsets[Component::Y.index()],
            y_stride: buffers.packed_strides[Component::Y.index()],
            cb_offset: buffers.packed_offsets[Component::Cb.index()],
            cb_stride: buffers.packed_strides[Component::Cb.index()],
            cr_offset: buffers.packed_offsets[Component::Cr.index()],
            cr_stride: buffers.packed_strides[Component::Cr.index()],
            chroma_width,
            chroma_height,
            limited: u32::from(color.range == YcbcrRange::Limited),
            grid_width: 0,
            bgra: 0,
            pad: 0,
            coefficients: color.matrix.coefficients(),
            levels: [LUMA_BLACK, LUMA_SCALE, CHROMA_SCALE, CHROMA_CENTRE],
            scale: [INV_255, 0.0, 0.0, 0.0],
        };
        params.grid_width = params.grid().0;
        params
    }

    /// Workgroup grid: one invocation per pixel.
    pub(crate) fn grid(&self) -> (u32, u32) {
        let pixels = u64::from(self.width) * u64::from(self.height);
        grid_for(pixels.div_ceil(YUV_GROUP_SIZE))
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::*;
    use crate::bitstream::ChromaSubsampling;

    fn layout_420() -> FrameLayout {
        FrameLayout::new(200, 120, ChromaSubsampling::Yuv420).unwrap()
    }

    #[test]
    fn parameter_blocks_have_uniform_friendly_sizes() {
        // Uniform buffers need struct sizes that are multiples of 16 bytes.
        assert_eq!(size_of::<BandInfo>(), 32);
        assert_eq!(size_of::<DequantParams>(), 16 + 32 * MAX_BANDS);
        assert_eq!(size_of::<CompParams>(), 48);
        assert_eq!(size_of::<LevelParams>(), 16 + 3 * 48);
        assert_eq!(size_of::<YuvParams>(), 64 + 3 * 16);
        assert_eq!(size_of::<QuantBand>(), 32);
        assert_eq!(size_of::<QuantParams>(), 16 + 32 * MAX_BANDS);
        assert_eq!(MAX_BANDS, 48);
        // A storage array element in `equant.wgsl`: 33 words, no padding.
        assert_eq!(size_of::<CostRecord>(), 4 * 33);
        assert_eq!(size_of::<FrameParams>(), 32);
        assert_eq!(size_of::<ScanParams>(), 16);
        // RateState in `rate.wgsl`.
        assert_eq!(RATE_STATE_BYTES, 4 * (512 + 6));
        assert_eq!(MAX_BLOCK_WORDS, 622);
    }

    #[test]
    fn the_largest_frame_fits_the_scans_and_32_bit_word_counts() {
        // 16384x16384 at 4:4:4: three components of 262144 blocks.
        let layout = FrameLayout::new(16384, 16384, ChromaSubsampling::Yuv444).unwrap();
        assert_eq!(layout.block_count(), 786_432);
        let scan = ScanParams::new(layout.block_count()).unwrap();
        assert_eq!(scan.chunk_count, 768);
        // The rate control adds up words of every block in 32 bits.
        assert!(u64::from(layout.block_count()) * MAX_BLOCK_WORDS < u64::from(u32::MAX));
        // One value more than two passes of the scan sum is refused.
        assert!(ScanParams::new(1 << 20).is_ok());
        assert!(ScanParams::new((1 << 20) + 1).is_err());
    }

    #[test]
    fn pyramid_regions_tile_the_buffer_without_gaps() {
        for layout in [
            layout_420(),
            FrameLayout::new(131, 67, ChromaSubsampling::Yuv444).unwrap(),
        ] {
            let buffers = BufferLayout::new(&layout).unwrap();
            let mut regions = Vec::new();
            for component in Component::ALL {
                for level in 0..DECOMPOSITION_LEVELS {
                    let (w, h) = layout.level_size(level);
                    for band in Band::ALL {
                        if let Some(offset) = buffers.band_offset(component, level, band) {
                            regions.push((offset as u64, u64::from(w) * u64::from(h)));
                        }
                    }
                }
            }
            regions.sort();
            let mut next = 0;
            for (offset, size) in regions {
                assert_eq!(offset, next, "regions are contiguous and disjoint");
                next = offset + size;
            }
            assert_eq!(next, buffers.pyramid_floats);
        }
    }

    #[test]
    fn four_two_zero_chroma_has_no_finest_level() {
        let layout = layout_420();
        let buffers = BufferLayout::new(&layout).unwrap();
        assert!(buffers.band_offset(Component::Y, 0, Band::Hh).is_some());
        assert!(buffers.band_offset(Component::Cb, 0, Band::Ll).is_none());
        assert!(buffers.band_offset(Component::Cb, 1, Band::Ll).is_some());
    }

    #[test]
    fn packed_plane_sizes() {
        let layout = layout_420(); // 200x120, chroma 100x60
        let buffers = BufferLayout::new(&layout).unwrap();
        assert_eq!(buffers.packed_strides, [50, 25, 25]);
        assert_eq!(buffers.packed_words, 50 * 120 + 2 * 25 * 60);

        // Odd widths round up to whole words.
        let layout = FrameLayout::new(131, 67, ChromaSubsampling::Yuv444).unwrap();
        assert_eq!(
            BufferLayout::new(&layout).unwrap().packed_strides,
            [33, 33, 33]
        );

        // A decoder's rows take 256 bytes or a multiple of it, and its planes
        // start at multiples of 512 bytes.
        let layout = layout_420();
        let buffers = BufferLayout::for_decoder(&layout).unwrap();
        assert_eq!(buffers.packed_strides, [64, 64, 64]);
        assert_eq!(buffers.packed_offsets, [0, 64 * 120, 64 * 120 + 64 * 60]);
        assert_eq!(buffers.packed_words, 64 * 120 + 2 * 64 * 60);
        let layout = FrameLayout::new(3840, 2160, ChromaSubsampling::Yuv420).unwrap();
        assert_eq!(
            BufferLayout::for_decoder(&layout).unwrap().packed_strides,
            [960, 512, 512]
        );
        // Odd heights leave gaps between the planes.
        let layout = FrameLayout::new(131, 67, ChromaSubsampling::Yuv444).unwrap();
        let buffers = BufferLayout::for_decoder(&layout).unwrap();
        assert_eq!(buffers.packed_strides, [64, 64, 64]);
        assert_eq!(buffers.packed_offsets, [0, 64 * 68, 2 * 64 * 68]);
        // Cb ends 256 bytes short of a multiple of 512: Cr starts after a gap.
        let layout = FrameLayout::new(1280, 722, ChromaSubsampling::Yuv420).unwrap();
        let buffers = BufferLayout::for_decoder(&layout).unwrap();
        assert_eq!(buffers.packed_strides, [320, 192, 192]);
        assert_eq!(
            buffers.packed_offsets,
            [0, 320 * 722, 320 * 722 + 192 * 361 + 64]
        );
        for offset in buffers.packed_offsets {
            assert_eq!(offset % DECODER_PLANE_WORDS, 0);
        }
    }

    #[test]
    fn dequant_band_table_follows_block_order() {
        let layout = layout_420();
        let buffers = BufferLayout::new(&layout).unwrap();
        let params = DequantParams::new(&layout, &buffers);

        assert_eq!(params.block_count, layout.block_count());
        let bands: Vec<_> = layout.bands().collect();
        assert_eq!(params.band_count as usize, bands.len());
        for (slot, (component, level, band, info)) in bands.iter().enumerate() {
            let entry = params.bands[slot];
            assert_eq!(entry.first_block, info.first_block);
            assert_eq!((entry.width, entry.height), (info.width, info.height));
            assert_eq!(
                Some(entry.dst_offset),
                buffers.band_offset(*component, *level, *band)
            );
        }
        // Sorted by first block, which the shader's search relies on.
        assert!(
            params.bands[..bands.len()]
                .windows(2)
                .all(|pair| pair[0].first_block < pair[1].first_block)
        );
    }

    #[test]
    fn transform_levels_chain_into_each_other() {
        let layout = layout_420();
        let buffers = BufferLayout::new(&layout).unwrap();

        for level in 1..DECOMPOSITION_LEVELS {
            let this = TransformLevel::new(&layout, &buffers, level);
            assert_eq!(this.components, 3);
            let (bw, bh) = layout.level_size(level);
            assert_eq!(
                (this.params.image_width, this.params.image_height),
                (2 * bw, 2 * bh)
            );

            for component in Component::ALL {
                let comp = this.params.comps[component.index()];
                if level == layout.finest_level(component) {
                    // 4:2:0 chroma finishes at level 1.
                    // The image of the finest level is the packed plane, in words.
                    assert_eq!(comp.is_plane, 1);
                    assert_eq!(comp.image_offset, buffers.packed_offsets[component.index()]);
                    assert_eq!(comp.image_stride, buffers.packed_strides[component.index()]);
                } else {
                    // The image is the LL band of the next finer level.
                    let next = buffers.band_offset(component, level - 1, Band::Ll).unwrap();
                    assert_eq!(comp.is_plane, 0);
                    assert_eq!(comp.image_offset, next);
                    assert_eq!(comp.image_stride, 2 * bw);
                }
            }
        }

        // The finest level has luma only in 4:2:0.
        let finest = TransformLevel::new(&layout, &buffers, 0);
        assert_eq!(finest.components, 1);
        assert_eq!(finest.params.comps[0].is_plane, 1);
        assert_eq!(
            (
                finest.params.comps[0].plane_width,
                finest.params.comps[0].plane_height
            ),
            (200, 120)
        );
        assert_eq!(finest.grid(), (7, 4, 1)); // 224 x 128 output: 7 x 4 tiles

        let layout = FrameLayout::new(131, 67, ChromaSubsampling::Yuv444).unwrap();
        let buffers = BufferLayout::new(&layout).unwrap();
        assert_eq!(TransformLevel::new(&layout, &buffers, 0).components, 3);
    }

    #[test]
    fn grids_stay_within_the_dimension_limit() {
        assert_eq!(grid_for(0), (1, 1));
        assert_eq!(grid_for(1), (1, 1));
        assert_eq!(grid_for(12_429), (12_429, 1));
        assert_eq!(grid_for(65_535), (65_535, 1));
        assert_eq!(grid_for(65_536), (65_535, 2));
        for count in [1u64, 1000, 65_535, 65_536, 1_000_000, 100_000_000] {
            let (x, y) = grid_for(count);
            assert!(x <= MAX_GRID_DIMENSION && y <= MAX_GRID_DIMENSION);
            assert!(u64::from(x) * u64::from(y) >= count);
        }
    }

    #[test]
    fn yuv_params_describe_the_packed_planes_and_the_conversion() {
        use crate::bitstream::{ChromaSiting, YcbcrRange};
        use crate::color::ColorMatrix;

        let layout = layout_420(); // 200x120, chroma 100x60
        let buffers = BufferLayout::new(&layout).unwrap();
        let color = ColorParams {
            matrix: ColorMatrix::Bt2020,
            range: YcbcrRange::Full,
            chroma_siting: ChromaSiting::Left,
        };
        let p = YuvParams::new(&layout, &buffers, &color);

        assert_eq!((p.width, p.height, p.chroma_420), (200, 120, 1));
        assert_eq!((p.siting_left, p.limited), (1, 0));
        assert_eq!((p.chroma_width, p.chroma_height), (100, 60));
        // The packed planes follow each other, rows padded to words.
        assert_eq!((p.y_offset, p.y_stride), (0, 50));
        assert_eq!((p.cb_offset, p.cb_stride), (50 * 120, 25));
        assert_eq!((p.cr_offset, p.cr_stride), (50 * 120 + 25 * 60, 25));
        assert_eq!(p.coefficients, ColorMatrix::Bt2020.coefficients());
        assert_eq!(p.scale[0], 1.0 / 255.0);

        // 24000 pixels is 94 workgroups of 256.
        assert_eq!(p.grid(), (94, 1));
        assert_eq!(p.grid_width, 94);

        // A picture of more than 65535 workgroups wraps into the second dimension.
        let big = FrameLayout::new(8192, 4320, ChromaSubsampling::Yuv420).unwrap();
        let big_buffers = BufferLayout::new(&big).unwrap();
        let p = YuvParams::new(&big, &big_buffers, &ColorParams::default());
        assert_eq!(p.grid(), (65_535, 3));
        assert_eq!(p.grid_width, 65_535);
        assert!(u64::from(p.grid().0) * u64::from(p.grid().1) * 256 >= 8192 * 4320);

        let planar = FrameLayout::new(131, 67, ChromaSubsampling::Yuv444).unwrap();
        let planar_buffers = BufferLayout::new(&planar).unwrap();
        let p = YuvParams::new(&planar, &planar_buffers, &ColorParams::default());
        assert_eq!(
            (p.chroma_420, p.chroma_width, p.chroma_height),
            (0, 131, 67)
        );
    }

    proptest::proptest! {
        // DequantParams, QuantParams and TransformLevel look bands up in the BufferLayout of the
        // same FrameLayout and `expect` to find them. Every picture size the
        // bitstream allows must keep that promise, so building the parameter
        // blocks never panics.
        #[test]
        fn every_valid_picture_size_builds_its_parameter_blocks(
            width in 1u32..=16384,
            height in 1u32..=16384,
            planar in proptest::bool::ANY,
        ) {
            let (width, height, chroma) = if planar {
                (width, height, ChromaSubsampling::Yuv444)
            } else {
                // 4:2:0 needs even sizes; 16384 is even.
                (width.next_multiple_of(2), height.next_multiple_of(2), ChromaSubsampling::Yuv420)
            };
            let layout = FrameLayout::new(width, height, chroma).unwrap();
            // A picture too large for 32-bit offsets is an error, not a panic.
            let Ok(buffers) = BufferLayout::new(&layout) else {
                return Ok(());
            };
            DequantParams::new(&layout, &buffers);
            let quant = QuantParams::new(&layout, &buffers);
            for band in &quant.bands[..quant.band_count as usize] {
                proptest::prop_assert!((7..=12).contains(&band.exponent));
                proptest::prop_assert!(band.max_drop <= 15);
                proptest::prop_assert!(band.weight > 0.0 && band.weight.is_finite());
            }
            for level in 0..DECOMPOSITION_LEVELS {
                TransformLevel::new(&layout, &buffers, level);
            }
            YuvParams::new(&layout, &buffers, &ColorParams::default());
        }
    }
}
