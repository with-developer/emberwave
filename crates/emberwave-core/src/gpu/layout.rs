//! Buffer layouts and shader parameter blocks of the GPU pipeline.
//!
//! Everything here is plain arithmetic over a [`FrameLayout`], so it is
//! tested without a GPU. The parameter structs mirror the uniform blocks in
//! the WGSL files field for field.

use bytemuck::{Pod, Zeroable};

use crate::bitstream::consts::{BANDS_PER_LEVEL, DECOMPOSITION_LEVELS, NUM_COMPONENTS};
use crate::bitstream::{Band, Component, FrameLayout};

use super::GpuError;

/// Largest number of transmitted bands: four at level 4, three at each of the
/// other levels, for three components. Sizes the band table in `dequant.wgsl`.
pub(crate) const MAX_BANDS: usize =
    NUM_COMPONENTS * (BANDS_PER_LEVEL + (DECOMPOSITION_LEVELS - 1) * (BANDS_PER_LEVEL - 1));

/// Invocations per workgroup of `pack.wgsl`.
const PACK_GROUP_SIZE: u64 = 256;

/// Largest workgroup grid dimension that every backend supports.
pub(crate) const MAX_GRID_DIMENSION: u32 = 65_535;

/// Where everything lives in the GPU buffers, in elements (floats or words).
#[derive(Debug, Clone)]
pub(crate) struct BufferLayout {
    /// Float offset of band `[component][level][band]` in the pyramid.
    band_offsets: [[[u32; BANDS_PER_LEVEL]; DECOMPOSITION_LEVELS]; NUM_COMPONENTS],
    /// Size of the pyramid in floats.
    pub pyramid_floats: u64,
    /// Float offset of each final plane in the plane buffer.
    pub plane_offsets: [u32; NUM_COMPONENTS],
    /// Size of the plane buffer in floats.
    pub plane_floats: u64,
    /// Word offset of each packed 8-bit plane.
    pub packed_offsets: [u32; NUM_COMPONENTS],
    /// Words per row of each packed plane.
    pub packed_strides: [u32; NUM_COMPONENTS],
    /// Size of the packed planes in words.
    pub packed_words: u64,
}

/// Marks a band that does not exist in `band_offsets`.
const ABSENT: u32 = u32::MAX;

impl BufferLayout {
    pub(crate) fn new(layout: &FrameLayout) -> Result<Self, GpuError> {
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

        let mut plane_offsets = [0; NUM_COMPONENTS];
        let mut packed_offsets = [0; NUM_COMPONENTS];
        let mut packed_strides = [0; NUM_COMPONENTS];
        let mut plane_floats = 0u64;
        let mut packed_words = 0u64;
        for component in Component::ALL {
            let (width, height) = layout.plane_size(component);
            plane_offsets[component.index()] = to_offset(plane_floats)?;
            plane_floats += u64::from(width) * u64::from(height);

            let stride = width.div_ceil(4);
            packed_offsets[component.index()] = to_offset(packed_words)?;
            packed_strides[component.index()] = stride;
            packed_words += u64::from(stride) * u64::from(height);
        }

        Ok(Self {
            band_offsets,
            pyramid_floats,
            plane_offsets,
            plane_floats,
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

/// Per component part of the inverse transform's uniform block.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct CompParams {
    pub band_offset: [u32; 4],
    pub dst_offset: u32,
    pub dst_stride: u32,
    pub crop_width: u32,
    pub crop_height: u32,
    pub is_final: u32,
    pub pad: [u32; 3],
}

/// Uniform block of `idwt.wgsl` for one decomposition level.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct IdwtParams {
    pub band_width: u32,
    pub band_height: u32,
    pub out_width: u32,
    pub out_height: u32,
    pub comps: [CompParams; NUM_COMPONENTS],
}

/// One level of the inverse transform: parameters and dispatch size.
#[derive(Debug, Clone)]
pub(crate) struct IdwtLevel {
    pub params: IdwtParams,
    /// Number of components that have this level: the z size of the dispatch.
    pub components: u32,
}

impl IdwtLevel {
    /// Parameters for the inverse transform that consumes the bands of `level`.
    pub(crate) fn new(layout: &FrameLayout, buffers: &BufferLayout, level: usize) -> Self {
        let (band_width, band_height) = layout.level_size(level);
        let mut params = IdwtParams {
            band_width,
            band_height,
            out_width: 2 * band_width,
            out_height: 2 * band_height,
            ..IdwtParams::zeroed()
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
                comp.dst_offset = buffers.plane_offsets[component.index()];
                comp.dst_stride = width;
                comp.crop_width = width;
                comp.crop_height = height;
                comp.is_final = 1;
            } else {
                // The LL band of the next finer level is twice as wide.
                comp.dst_offset = buffers
                    .band_offset(component, level - 1, Band::Ll)
                    .expect("the next finer level exists");
                comp.dst_stride = 2 * band_width;
                comp.crop_width = 2 * band_width;
                comp.crop_height = 2 * band_height;
                comp.is_final = 0;
            }
            components += 1;
        }

        Self { params, components }
    }

    /// Workgroup grid: 32x32 output tiles in x and y, components in z.
    pub(crate) fn grid(&self) -> (u32, u32, u32) {
        (
            self.params.out_width.div_ceil(32),
            self.params.out_height.div_ceil(32),
            self.components,
        )
    }
}

/// Per component part of the packing pass's uniform block.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct PackComp {
    pub src_offset: u32,
    pub width: u32,
    pub height: u32,
    pub dst_offset: u32,
    pub dst_stride: u32,
    pub pad: [u32; 3],
}

/// Uniform block of `pack.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct PackParams {
    pub grid_width: u32,
    pub pad: [u32; 3],
    pub comps: [PackComp; NUM_COMPONENTS],
}

impl PackParams {
    pub(crate) fn new(layout: &FrameLayout, buffers: &BufferLayout) -> Self {
        let mut params = Self::zeroed();
        let mut largest = 0u64;
        for component in Component::ALL {
            let (width, height) = layout.plane_size(component);
            let i = component.index();
            params.comps[i] = PackComp {
                src_offset: buffers.plane_offsets[i],
                width,
                height,
                dst_offset: buffers.packed_offsets[i],
                dst_stride: buffers.packed_strides[i],
                pad: [0; 3],
            };
            largest = largest.max(u64::from(buffers.packed_strides[i]) * u64::from(height));
        }
        params.grid_width = grid_for(largest.div_ceil(PACK_GROUP_SIZE)).0;
        params
    }

    /// Workgroup grid: enough groups for the largest plane, three planes in z.
    pub(crate) fn grid(&self) -> (u32, u32, u32) {
        let largest = self
            .comps
            .iter()
            .map(|c| u64::from(c.dst_stride) * u64::from(c.height))
            .max()
            .unwrap_or(0);
        let (x, y) = grid_for(largest.div_ceil(PACK_GROUP_SIZE));
        (x, y, NUM_COMPONENTS as u32)
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
        assert_eq!(size_of::<IdwtParams>(), 16 + 3 * 48);
        assert_eq!(size_of::<PackComp>(), 32);
        assert_eq!(size_of::<PackParams>(), 16 + 3 * 32);
        assert_eq!(MAX_BANDS, 48);
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
    fn plane_and_packed_sizes() {
        let layout = layout_420(); // 200x120, chroma 100x60
        let buffers = BufferLayout::new(&layout).unwrap();
        assert_eq!(buffers.plane_offsets, [0, 200 * 120, 200 * 120 + 100 * 60]);
        assert_eq!(buffers.plane_floats, 200 * 120 + 2 * 100 * 60);
        assert_eq!(buffers.packed_strides, [50, 25, 25]);
        assert_eq!(buffers.packed_words, 50 * 120 + 2 * 25 * 60);

        // Odd widths round up to whole words.
        let layout = FrameLayout::new(131, 67, ChromaSubsampling::Yuv444).unwrap();
        assert_eq!(
            BufferLayout::new(&layout).unwrap().packed_strides,
            [33, 33, 33]
        );
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
    fn idwt_levels_chain_into_each_other() {
        let layout = layout_420();
        let buffers = BufferLayout::new(&layout).unwrap();

        for level in 1..DECOMPOSITION_LEVELS {
            let this = IdwtLevel::new(&layout, &buffers, level);
            assert_eq!(this.components, 3);
            let (bw, bh) = layout.level_size(level);
            assert_eq!(
                (this.params.out_width, this.params.out_height),
                (2 * bw, 2 * bh)
            );

            for component in Component::ALL {
                let comp = this.params.comps[component.index()];
                if level == layout.finest_level(component) {
                    // 4:2:0 chroma finishes at level 1.
                    assert_eq!(comp.is_final, 1);
                    assert_eq!(comp.dst_offset, buffers.plane_offsets[component.index()]);
                } else {
                    // The destination is the LL band the next level reads.
                    let next = buffers.band_offset(component, level - 1, Band::Ll).unwrap();
                    assert_eq!(comp.is_final, 0);
                    assert_eq!(comp.dst_offset, next);
                    assert_eq!(comp.dst_stride, 2 * bw);
                }
            }
        }

        // The finest level has luma only in 4:2:0.
        let finest = IdwtLevel::new(&layout, &buffers, 0);
        assert_eq!(finest.components, 1);
        assert_eq!(finest.params.comps[0].is_final, 1);
        assert_eq!(
            (
                finest.params.comps[0].crop_width,
                finest.params.comps[0].crop_height
            ),
            (200, 120)
        );
        assert_eq!(finest.grid(), (7, 4, 1)); // 224 x 128 output: 7 x 4 tiles

        let layout = FrameLayout::new(131, 67, ChromaSubsampling::Yuv444).unwrap();
        let buffers = BufferLayout::new(&layout).unwrap();
        assert_eq!(IdwtLevel::new(&layout, &buffers, 0).components, 3);
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
    fn pack_grid_covers_the_largest_plane() {
        let layout = layout_420();
        let buffers = BufferLayout::new(&layout).unwrap();
        let params = PackParams::new(&layout, &buffers);
        let (x, y, z) = params.grid();
        assert_eq!(z, 3);
        // 50 words per row x 120 rows = 6000 words = 24 groups of 256.
        assert_eq!((x, y), (24, 1));
        assert_eq!(params.grid_width, 24);
    }
}
