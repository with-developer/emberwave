//! Where every 32x32 block of a frame lives in the wavelet pyramid.
//!
//! The layout depends only on the picture size and the chroma format. It is
//! derived the way `bitstream.md` ("Block index ordering") describes it.

use super::consts::{
    ALIGNMENT, BANDS_PER_LEVEL, BLOCK_SIZE, DECOMPOSITION_LEVELS, MAX_DIMENSION,
    MINIMUM_IMAGE_SIZE, NUM_COMPONENTS,
};
use super::error::Error;
use super::header::ChromaSubsampling;

/// A colour component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Component {
    /// Luma.
    Y,
    /// Blue difference chroma.
    Cb,
    /// Red difference chroma.
    Cr,
}

impl Component {
    /// All components in bitstream order.
    pub const ALL: [Component; NUM_COMPONENTS] = [Component::Y, Component::Cb, Component::Cr];

    /// Index of the component in bitstream order.
    pub fn index(self) -> usize {
        match self {
            Component::Y => 0,
            Component::Cb => 1,
            Component::Cr => 2,
        }
    }
}

/// A frequency band of one decomposition level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Band {
    /// Low-pass in both directions.
    Ll,
    /// Horizontal high-pass, vertical low-pass.
    Hl,
    /// Horizontal low-pass, vertical high-pass.
    Lh,
    /// High-pass in both directions.
    Hh,
}

impl Band {
    /// All bands in bitstream order.
    pub const ALL: [Band; BANDS_PER_LEVEL] = [Band::Ll, Band::Hl, Band::Lh, Band::Hh];

    /// Index of the band in bitstream order.
    pub fn index(self) -> usize {
        match self {
            Band::Ll => 0,
            Band::Hl => 1,
            Band::Lh => 2,
            Band::Hh => 3,
        }
    }
}

/// Size and block range of one transmitted band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BandLayout {
    /// Width of the band in coefficients.
    pub width: u32,
    /// Height of the band in coefficients.
    pub height: u32,
    /// 32x32 blocks per row of the band.
    pub blocks_x: u32,
    /// 32x32 blocks per column of the band.
    pub blocks_y: u32,
    /// Block index of the band's first block. Blocks are row-major.
    pub first_block: u32,
}

impl BandLayout {
    /// Number of 32x32 blocks in the band.
    pub fn block_count(&self) -> u32 {
        self.blocks_x * self.blocks_y
    }
}

/// Position of a block in the pyramid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLocation {
    /// Colour component.
    pub component: Component,
    /// Decomposition level, 0 (finest) to 4 (coarsest).
    pub level: usize,
    /// Frequency band.
    pub band: Band,
    /// Block column inside the band.
    pub block_x: u32,
    /// Block row inside the band.
    pub block_y: u32,
}

/// Block layout of a frame: padded size, band sizes and block numbering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameLayout {
    width: u32,
    height: u32,
    aligned_width: u32,
    aligned_height: u32,
    chroma: ChromaSubsampling,
    bands: [[[Option<BandLayout>; BANDS_PER_LEVEL]; DECOMPOSITION_LEVELS]; NUM_COMPONENTS],
    /// Transmitted bands in block index order.
    order: Vec<(Component, usize, Band)>,
    block_count: u32,
}

impl FrameLayout {
    /// Builds the layout for a picture of `width` x `height` luma pixels.
    ///
    /// Dimensions must be in `1..=16384`, and 4:2:0 needs both to be even
    /// (`bitstream.md`, "Start of frame header").
    pub fn new(width: u32, height: u32, chroma: ChromaSubsampling) -> Result<Self, Error> {
        let in_range =
            (1..=MAX_DIMENSION).contains(&width) && (1..=MAX_DIMENSION).contains(&height);
        let even_if_420 = chroma == ChromaSubsampling::Yuv444 || ((width | height) & 1) == 0;
        if !in_range || !even_if_420 {
            return Err(Error::InvalidDimensions {
                width,
                height,
                chroma,
            });
        }

        let aligned_width = width.next_multiple_of(ALIGNMENT).max(MINIMUM_IMAGE_SIZE);
        let aligned_height = height.next_multiple_of(ALIGNMENT).max(MINIMUM_IMAGE_SIZE);

        let mut layout = Self {
            width,
            height,
            aligned_width,
            aligned_height,
            chroma,
            bands: [[[None; BANDS_PER_LEVEL]; DECOMPOSITION_LEVELS]; NUM_COMPONENTS],
            order: Vec::new(),
            block_count: 0,
        };

        // Coarsest level first; within a level by component, then by band.
        // The LL band is only transmitted at the coarsest level. At the finest
        // level 4:2:0 has no chroma.
        for level in (0..DECOMPOSITION_LEVELS).rev() {
            let (level_width, level_height) = layout.level_size(level);
            let blocks_x = level_width.div_ceil(BLOCK_SIZE);
            let blocks_y = level_height.div_ceil(BLOCK_SIZE);

            for component in Component::ALL {
                if level == 0 && component != Component::Y && chroma == ChromaSubsampling::Yuv420 {
                    continue;
                }

                let first_band = if level == DECOMPOSITION_LEVELS - 1 {
                    0
                } else {
                    1
                };
                for band in &Band::ALL[first_band..] {
                    layout.bands[component.index()][level][band.index()] = Some(BandLayout {
                        width: level_width,
                        height: level_height,
                        blocks_x,
                        blocks_y,
                        first_block: layout.block_count,
                    });
                    layout.order.push((component, level, *band));
                    layout.block_count += blocks_x * blocks_y;
                }
            }
        }

        Ok(layout)
    }

    /// Luma width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Luma height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Luma width after padding to the transform alignment.
    pub fn aligned_width(&self) -> u32 {
        self.aligned_width
    }

    /// Luma height after padding to the transform alignment.
    pub fn aligned_height(&self) -> u32 {
        self.aligned_height
    }

    /// Chroma subsampling.
    pub fn chroma(&self) -> ChromaSubsampling {
        self.chroma
    }

    /// Number of blocks the frame can have, transmitted or not.
    pub fn block_count(&self) -> u32 {
        self.block_count
    }

    /// Size of the bands of decomposition `level` (0 finest, 4 coarsest).
    pub fn level_size(&self, level: usize) -> (u32, u32) {
        let shift = u32::try_from(level).unwrap_or(u32::MAX);
        let half = |aligned: u32| (aligned / 2).checked_shr(shift).unwrap_or(0).max(1);
        (half(self.aligned_width), half(self.aligned_height))
    }

    /// Size of the decoded plane of `component` in samples.
    pub fn plane_size(&self, component: Component) -> (u32, u32) {
        if component == Component::Y || self.chroma == ChromaSubsampling::Yuv444 {
            (self.width, self.height)
        } else {
            (self.width / 2, self.height / 2)
        }
    }

    /// The finest decomposition level that is transmitted for `component`.
    /// Decoding it yields the plane.
    pub fn finest_level(&self, component: Component) -> usize {
        if component != Component::Y && self.chroma == ChromaSubsampling::Yuv420 {
            1
        } else {
            0
        }
    }

    /// Layout of one band, or `None` if it is not transmitted.
    pub fn band(&self, component: Component, level: usize, band: Band) -> Option<&BandLayout> {
        self.bands
            .get(component.index())?
            .get(level)?
            .get(band.index())?
            .as_ref()
    }

    /// Transmitted bands in block index order.
    pub fn bands(&self) -> impl Iterator<Item = (Component, usize, Band, &BandLayout)> + '_ {
        self.order.iter().filter_map(|&(component, level, band)| {
            self.band(component, level, band)
                .map(|layout| (component, level, band, layout))
        })
    }

    /// Finds the band and the position inside it of a block index.
    pub fn locate(&self, block_index: u32) -> Option<BlockLocation> {
        if block_index >= self.block_count {
            return None;
        }

        let after = self.order.partition_point(|&(component, level, band)| {
            self.band(component, level, band)
                .is_some_and(|layout| layout.first_block <= block_index)
        });
        let &(component, level, band) = self.order.get(after.checked_sub(1)?)?;
        let layout = self.band(component, level, band)?;
        let relative = block_index.checked_sub(layout.first_block)?;
        if relative >= layout.block_count() {
            return None;
        }

        Some(BlockLocation {
            component,
            level,
            band,
            block_x: relative % layout.blocks_x,
            block_y: relative / layout.blocks_x,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_dimensions() {
        for (w, h, chroma) in [
            (0, 64, ChromaSubsampling::Yuv444),
            (64, 0, ChromaSubsampling::Yuv444),
            (16385, 64, ChromaSubsampling::Yuv444),
            (65, 64, ChromaSubsampling::Yuv420),
            (64, 65, ChromaSubsampling::Yuv420),
        ] {
            assert_eq!(
                FrameLayout::new(w, h, chroma),
                Err(Error::InvalidDimensions {
                    width: w,
                    height: h,
                    chroma
                })
            );
        }
        // Odd sizes are fine for 4:4:4.
        assert!(FrameLayout::new(131, 67, ChromaSubsampling::Yuv444).is_ok());
    }

    #[test]
    fn alignment_and_minimum_size() {
        let l = FrameLayout::new(160, 96, ChromaSubsampling::Yuv420).unwrap();
        assert_eq!((l.aligned_width(), l.aligned_height()), (160, 128));

        let l = FrameLayout::new(2, 2, ChromaSubsampling::Yuv420).unwrap();
        assert_eq!((l.aligned_width(), l.aligned_height()), (128, 128));

        let l = FrameLayout::new(1920, 1080, ChromaSubsampling::Yuv420).unwrap();
        assert_eq!((l.aligned_width(), l.aligned_height()), (1920, 1088));
    }

    #[test]
    fn level_size_is_total() {
        let l = FrameLayout::new(160, 96, ChromaSubsampling::Yuv420).unwrap();
        // Beyond the last level the bands degenerate to one sample; no panic.
        assert_eq!(l.level_size(32), (1, 1));
        assert_eq!(l.level_size(usize::MAX), (1, 1));
        assert_eq!(l.level_size(10), (1, 1));
    }

    #[test]
    fn level_sizes() {
        // 160x128 aligned: bands are 80x64 at level 0 and halve per level.
        let l = FrameLayout::new(160, 96, ChromaSubsampling::Yuv420).unwrap();
        let sizes: Vec<_> = (0..DECOMPOSITION_LEVELS).map(|v| l.level_size(v)).collect();
        assert_eq!(sizes, [(80, 64), (40, 32), (20, 16), (10, 8), (5, 4)]);
    }

    #[test]
    fn block_numbering_420() {
        let l = FrameLayout::new(160, 96, ChromaSubsampling::Yuv420).unwrap();

        // Level 4: all four bands of all three components, one block each.
        // Level 3 and 2: three bands per component, one block each.
        // Level 1: three bands per component, 2x1 blocks each.
        // Level 0: three bands of luma only, 3x2 blocks each.
        assert_eq!(l.block_count(), 12 + 9 + 9 + 18 + 18);

        let ll4_y = l.band(Component::Y, 4, Band::Ll).unwrap();
        assert_eq!((ll4_y.first_block, ll4_y.block_count()), (0, 1));
        let hl4_y = l.band(Component::Y, 4, Band::Hl).unwrap();
        assert_eq!(hl4_y.first_block, 1);
        let ll4_cb = l.band(Component::Cb, 4, Band::Ll).unwrap();
        assert_eq!(ll4_cb.first_block, 4);

        // LL is transmitted at level 4 only.
        assert!(l.band(Component::Y, 3, Band::Ll).is_none());
        // 4:2:0 has no level 0 chroma.
        assert!(l.band(Component::Cb, 0, Band::Hl).is_none());
        let hl0_y = l.band(Component::Y, 0, Band::Hl).unwrap();
        assert_eq!((hl0_y.blocks_x, hl0_y.blocks_y), (3, 2));
        assert_eq!(hl0_y.first_block, 48);
    }

    #[test]
    fn block_counts_of_common_resolutions() {
        // Worked out by hand per level: bands per level (4 at level 4, else 3)
        // times components (luma only at level 0) times 32x32 blocks per band.
        // 1080p: 48 + 108 + 360 + 1215 + 1530.
        for (w, h, expected) in [(1920, 1080, 3261), (2560, 1440, 5667), (3840, 2160, 12429)] {
            let l = FrameLayout::new(w, h, ChromaSubsampling::Yuv420).unwrap();
            assert_eq!(l.block_count(), expected, "{w}x{h}");
        }
        let l = FrameLayout::new(1920, 1080, ChromaSubsampling::Yuv420).unwrap();
        let per_level: Vec<u32> = (0..DECOMPOSITION_LEVELS)
            .map(|level| {
                l.bands()
                    .filter(|(_, v, _, _)| *v == level)
                    .map(|(_, _, _, b)| b.block_count())
                    .sum()
            })
            .collect();
        assert_eq!(per_level, [1530, 1215, 360, 108, 48]);
    }

    #[test]
    fn block_numbering_444_has_level_zero_chroma() {
        let l = FrameLayout::new(160, 96, ChromaSubsampling::Yuv444).unwrap();
        assert_eq!(l.block_count(), 12 + 9 + 9 + 18 + 3 * 3 * 6);
        assert!(l.band(Component::Cr, 0, Band::Hh).is_some());
    }

    #[test]
    fn locate_inverts_numbering() {
        for (w, h, chroma) in [
            (160, 96, ChromaSubsampling::Yuv420),
            (200, 120, ChromaSubsampling::Yuv420),
            (131, 67, ChromaSubsampling::Yuv444),
            (1920, 1080, ChromaSubsampling::Yuv420),
        ] {
            let l = FrameLayout::new(w, h, chroma).unwrap();
            let mut expected = 0;
            for (component, level, band, bl) in l.bands() {
                for y in 0..bl.blocks_y {
                    for x in 0..bl.blocks_x {
                        assert_eq!(
                            l.locate(expected),
                            Some(BlockLocation {
                                component,
                                level,
                                band,
                                block_x: x,
                                block_y: y
                            })
                        );
                        expected += 1;
                    }
                }
            }
            assert_eq!(expected, l.block_count());
            assert_eq!(l.locate(l.block_count()), None);
            assert_eq!(l.locate(u32::MAX), None);
        }
    }

    #[test]
    fn plane_sizes_and_finest_level() {
        let l = FrameLayout::new(200, 120, ChromaSubsampling::Yuv420).unwrap();
        assert_eq!(l.plane_size(Component::Y), (200, 120));
        assert_eq!(l.plane_size(Component::Cb), (100, 60));
        assert_eq!(l.finest_level(Component::Y), 0);
        assert_eq!(l.finest_level(Component::Cr), 1);

        let l = FrameLayout::new(131, 67, ChromaSubsampling::Yuv444).unwrap();
        assert_eq!(l.plane_size(Component::Cb), (131, 67));
        assert_eq!(l.finest_level(Component::Cb), 0);
    }

    #[test]
    fn largest_frame_fits() {
        let l = FrameLayout::new(16384, 16384, ChromaSubsampling::Yuv444).unwrap();
        assert!(l.block_count() > 0);
        assert!(l.locate(l.block_count() - 1).is_some());
    }
}
