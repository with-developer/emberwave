//! Reference coefficient decoding: bit-planes and sign bits to scaled floats.
//!
//! This is the stage that the GPU runs in the real decoder. Here it is plain
//! sequential code that follows `bitstream.md` ("Decoding 8x8 blocks" and
//! "Dequantization"), so it can serve as an oracle for the shaders.

use crate::bitstream::block::{BlockBytes, BlockStructure};
use crate::bitstream::consts::{
    BANDS_PER_LEVEL, BLOCK_SIZE, BLOCK8_PER_BLOCK, BLOCK8_PER_SIDE, BLOCK8_SCALE_BIAS,
    BLOCK8_SCALE_STEP, BLOCK8_SIZE, DEADZONE_OFFSET, DECOMPOSITION_LEVELS, MAX_SCALE_EXP,
    NUM_COMPONENTS, QUANT_MANTISSA_BASE, QUANT_MANTISSA_BITS, SUBBLOCK_HEIGHT, SUBBLOCK_WIDTH,
    SUBBLOCKS_PER_BLOCK8, VALUES_PER_SUBBLOCK,
};
use crate::bitstream::{Band, BandLayout, Component, Error, FrameLayout, FrameView};

/// 2^`exponent` as an exact `f32`, for exponents in the normal range.
fn exact_pow2(exponent: i32) -> f32 {
    f32::from_bits(((127 + exponent) as u32) << 23)
}

/// Scale shared by all coefficients of a 32x32 block (`Block32x32Scale`).
///
/// The reference formula is `(8 + m) * 2^(e - 3)` with
/// `e = MAX_SCALE_EXP - (quant_code >> 3)` and `m = quant_code & 7`. Both the
/// mantissa and the power of two are exact in `f32`, so the result is exact.
/// Upstream shifts an integer by `20 + e`, which is undefined for
/// `quant_code >= 200`; this version extends the formula to smaller scales.
pub fn block_scale(quant_code: u8) -> f32 {
    let e = MAX_SCALE_EXP - i32::from(quant_code >> QUANT_MANTISSA_BITS);
    let mantissa = QUANT_MANTISSA_BASE + u32::from(quant_code & ((1 << QUANT_MANTISSA_BITS) - 1));
    mantissa as f32 * exact_pow2(e - 3)
}

/// Scale of one 8x8 block from its 4-bit code (`Block8x8Scale`).
pub fn block8_scale(code: u8) -> f32 {
    f32::from(code) * BLOCK8_SCALE_STEP + BLOCK8_SCALE_BIAS
}

/// Decodes one 32x32 block into `plane`, a row-major band image that is
/// `band.width` samples wide.
///
/// Coefficients that fall outside the band (blocks at the right and bottom
/// edge) are dropped. Coefficients of 8x8 blocks that are not in the block's
/// ballot are left untouched, so `plane` should start zeroed.
///
/// Returns the number of sign bits the block carries, which is the number of
/// non-zero coefficients including those outside the band.
pub fn decode_block(
    words: &[u32],
    band: &BandLayout,
    block_x: u32,
    block_y: u32,
    plane: &mut [f32],
) -> Result<usize, Error> {
    let structure = BlockStructure::parse(words)?;
    let bytes = BlockBytes::new(words);
    let block_scale = block_scale(structure.header().quant_code);

    // Saturating, so that absurd block coordinates fall outside the band
    // instead of overflowing.
    let origin_x = block_x.saturating_mul(BLOCK_SIZE);
    let origin_y = block_y.saturating_mul(BLOCK_SIZE);
    let mut sign_bit = structure.sign_offset() * 8;

    // Signs are consumed in coefficient order: 8x8 block, subblock, value.
    for raster in 0..BLOCK8_PER_BLOCK {
        let Some(block8) = structure.block8(raster) else {
            continue;
        };

        // An 8x8 block whose control word is zero decodes to nothing, whatever
        // its base bit-planes say, and the reference decoder skips it too.
        // Upstream's encoder never produces one (its ballot bit would be
        // clear); emberwave's does, as the only 8x8 block of a block of the
        // coarsest level with nothing left, which it sends anyway (D58).
        if block8.control_word == 0 {
            continue;
        }

        let scale = block_scale * block8_scale(block8.scale_code);
        let block8_x = origin_x.saturating_add((raster % BLOCK8_PER_SIDE) as u32 * BLOCK8_SIZE);
        let block8_y = origin_y.saturating_add((raster / BLOCK8_PER_SIDE) as u32 * BLOCK8_SIZE);
        let mut plane_byte = block8.plane_offset;

        for subblock in 0..SUBBLOCKS_PER_BLOCK8 {
            // Subblocks run down two columns of four: 0..3 on the left, 4..7 on the right.
            let sub_x = block8_x.saturating_add((subblock / 4 * SUBBLOCK_WIDTH) as u32);
            let sub_y = block8_y.saturating_add((subblock % 4 * SUBBLOCK_HEIGHT) as u32);

            // Bit-planes come most significant first, one byte per plane,
            // with bit `v` of the byte belonging to value `v`.
            let mut magnitudes = [0u32; VALUES_PER_SUBBLOCK];
            for plane_index in (0..block8.subblock_planes(subblock)).rev() {
                let byte = bytes.u8(plane_byte);
                plane_byte += 1;
                for (value, magnitude) in magnitudes.iter_mut().enumerate() {
                    *magnitude |= u32::from((byte >> value) & 1) << plane_index;
                }
            }

            for (value, &magnitude) in magnitudes.iter().enumerate() {
                if magnitude == 0 {
                    continue;
                }

                let sign_byte = sign_bit / 8;
                if sign_byte >= bytes.len() {
                    return Err(Error::BlockContentOverrun {
                        needed: sign_byte + 1,
                        available: bytes.len(),
                    });
                }
                let negative = (bytes.u8(sign_byte) >> (sign_bit % 8)) & 1 != 0;
                sign_bit += 1;

                // Values run down the columns of the 4x2 subblock.
                let x = sub_x.saturating_add((value / SUBBLOCK_HEIGHT) as u32);
                let y = sub_y.saturating_add((value % SUBBLOCK_HEIGHT) as u32);
                if x >= band.width || y >= band.height {
                    continue;
                }

                let coefficient = (magnitude as f32 + DEADZONE_OFFSET) * scale;
                if let Some(slot) = plane.get_mut(y as usize * band.width as usize + x as usize) {
                    *slot = if negative { -coefficient } else { coefficient };
                }
            }
        }
    }

    Ok(sign_bit - structure.sign_offset() * 8)
}

/// The dequantised coefficients of every transmitted band of a frame.
#[derive(Debug, Clone)]
pub struct Bands {
    planes: [[[Option<Vec<f32>>; BANDS_PER_LEVEL]; DECOMPOSITION_LEVELS]; NUM_COMPONENTS],
}

impl Bands {
    fn zeroed(layout: &FrameLayout) -> Self {
        let mut bands = Self {
            planes: Default::default(),
        };
        for (component, level, band, band_layout) in layout.bands() {
            bands.planes[component.index()][level][band.index()] =
                Some(vec![
                    0.0;
                    band_layout.width as usize * band_layout.height as usize
                ]);
        }
        bands
    }

    /// The coefficients of a band, row-major with the band's width as stride,
    /// or `None` if the band is not transmitted.
    pub fn plane(&self, component: Component, level: usize, band: Band) -> Option<&[f32]> {
        self.planes
            .get(component.index())?
            .get(level)?
            .get(band.index())?
            .as_deref()
    }

    /// Moves a band out, leaving it empty.
    pub(crate) fn take(
        &mut self,
        component: Component,
        level: usize,
        band: Band,
    ) -> Option<Vec<f32>> {
        self.planes
            .get_mut(component.index())?
            .get_mut(level)?
            .get_mut(band.index())?
            .take()
    }

    pub(crate) fn plane_mut(
        &mut self,
        component: Component,
        level: usize,
        band: Band,
    ) -> Option<&mut [f32]> {
        self.planes
            .get_mut(component.index())?
            .get_mut(level)?
            .get_mut(band.index())?
            .as_deref_mut()
    }
}

/// Decodes every received block of a frame into per-band coefficient images.
///
/// Blocks that were not received stay zero.
pub fn dequantize(frame: &FrameView<'_>) -> Result<Bands, Error> {
    let layout = frame.layout();
    let mut bands = Bands::zeroed(layout);

    for block_index in 0..layout.block_count() {
        let (Some(words), Some(location)) = (frame.block(block_index), layout.locate(block_index))
        else {
            continue;
        };
        let Some(band_layout) = layout.band(location.component, location.level, location.band)
        else {
            continue;
        };
        let Some(plane) = bands.plane_mut(location.component, location.level, location.band) else {
            continue;
        };
        decode_block(
            words,
            band_layout,
            location.block_x,
            location.block_y,
            plane,
        )?;
    }

    Ok(bands)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_scale_matches_the_reference_formula() {
        // Reference: (1 / 2^23) * ((8 + m) << (20 + e)) with e = 4 - (code >> 3).
        for code in 0..=199u8 {
            let e = MAX_SCALE_EXP - i32::from(code >> 3);
            let m = i32::from(code & 7);
            let reference =
                (1.0f32 / (8.0 * 1024.0 * 1024.0)) * (((8 + m) * (1 << (20 + e))) as f32);
            assert_eq!(block_scale(code), reference, "quant_code {code}");
        }
    }

    #[test]
    fn block_scale_endpoints() {
        assert_eq!(block_scale(0), 16.0); // m = 0, e = 4: 8 * 2^1
        assert_eq!(block_scale(7), 30.0); // m = 7, e = 4: 15 * 2^1
        assert_eq!(block_scale(8), 8.0); // e = 3: 8 * 2^0
        assert_eq!(block_scale(255), 15.0 * 2f32.powi(-30));
        assert!(block_scale(255) > 0.0);
    }

    #[test]
    fn block8_scale_range() {
        assert_eq!(block8_scale(0), 0.25);
        assert_eq!(block8_scale(8), 1.25);
        assert_eq!(block8_scale(15), 2.125);
    }

    /// Hand-assembles a block with a single active 8x8 block (raster 0) and
    /// checks the decoded coefficients against values worked out on paper.
    #[test]
    fn decodes_a_hand_built_block() {
        // Control word: subblock 0 has 1 extra plane, subblock 1 has 2, the
        // rest 0. Base planes 0. Scale code 4 -> 4/8 + 0.25 = 0.75.
        let control_word: u16 = 0b0000_0000_0000_1001;
        let qscale: u8 = 0x40;
        let quant_code: u32 = 8; // block scale 8.0

        // Subblock 0: one plane. Bit v of the byte is value v. Values 0, 3 and 5 set.
        let sb0_plane = 0b0010_1001u8;
        // Subblock 1: two planes, MSB first. Value 1 = 0b10 (2), value 2 = 0b01 (1),
        // value 6 = 0b11 (3).
        let sb1_msb = 0b0100_0010u8;
        let sb1_lsb = 0b0100_0100u8;
        // Signs, one bit per non-zero value in order: sb0 values 0, 3, 5 then
        // sb1 values 1, 2, 6. Make values 3 and 6 negative.
        let signs = 0b0010_0010u8; // bit 1 -> second value (sb0 value 3), bit 5 -> sixth (sb1 value 6)

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&[0, 0, 0, 0]); // word 0, patched below
        bytes.extend_from_slice(&(quant_code).to_le_bytes()); // word 1: quant_code 8, block_index 0
        bytes.extend_from_slice(&control_word.to_le_bytes());
        bytes.push(qscale);
        bytes.extend_from_slice(&[sb0_plane, sb1_msb, sb1_lsb, signs]);
        while !bytes.len().is_multiple_of(4) {
            bytes.push(0);
        }
        let words: Vec<u32> = bytes
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut words = words;
        words[0] = 0b1 | ((words.len() as u32) << 16); // ballot: 8x8 block 0 only

        let band = BandLayout {
            width: 32,
            height: 32,
            blocks_x: 1,
            blocks_y: 1,
            first_block: 0,
        };
        let mut plane = vec![0.0f32; 32 * 32];
        decode_block(&words, &band, 0, 0, &mut plane).unwrap();

        let scale = 8.0 * 0.75;
        let at = |x: usize, y: usize| plane[y * 32 + x];

        // Subblock 0 covers x 0..4, y 0..2. Value v is at (v / 2, v % 2).
        assert_eq!(at(0, 0), 1.5 * scale); // value 0
        assert_eq!(at(1, 1), -1.5 * scale); // value 3, negative
        assert_eq!(at(2, 1), 1.5 * scale); // value 5
        assert_eq!(at(0, 1), 0.0); // value 1
        // Subblock 1 covers x 0..4, y 2..4.
        assert_eq!(at(0, 3), 2.5 * scale); // value 1, magnitude 2
        assert_eq!(at(1, 2), 1.5 * scale); // value 2, magnitude 1
        assert_eq!(at(3, 2), -3.5 * scale); // value 6, magnitude 3, negative
        // Everything else is zero.
        let nonzero = plane.iter().filter(|v| **v != 0.0).count();
        assert_eq!(nonzero, 6);
    }

    #[test]
    fn missing_sign_bits_are_an_error_not_a_panic() {
        // One plane in subblock 0 with a non-zero value but no room for its sign.
        let control_word: u16 = 0b01;
        let mut bytes = vec![0u8; 8];
        bytes.extend_from_slice(&control_word.to_le_bytes());
        bytes.push(0x00);
        bytes.push(0x01); // the plane: value 0 set; this is the last byte of the block
        while !bytes.len().is_multiple_of(4) {
            bytes.push(0);
        }
        let mut words: Vec<u32> = bytes
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        words[0] = 0b1 | ((words.len() as u32) << 16);

        let band = BandLayout {
            width: 32,
            height: 32,
            blocks_x: 1,
            blocks_y: 1,
            first_block: 0,
        };
        let mut plane = vec![0.0f32; 32 * 32];
        // 8 + 3 = 11 bytes of tables, plane at 11, signs from byte 12 = the block size (12).
        assert!(matches!(
            decode_block(&words, &band, 0, 0, &mut plane),
            Err(Error::BlockContentOverrun { .. })
        ));
    }

    #[test]
    fn coefficients_outside_the_band_are_dropped() {
        // A band of 3x2 coefficients; the block sets values across a 4x2 subblock.
        let control_word: u16 = 0b01;
        let mut bytes = vec![0u8; 8];
        bytes.extend_from_slice(&control_word.to_le_bytes());
        bytes.push(0x40);
        bytes.push(0xFF); // all 8 values have magnitude 1
        bytes.push(0x00); // 8 sign bits, all positive
        while !bytes.len().is_multiple_of(4) {
            bytes.push(0);
        }
        let mut words: Vec<u32> = bytes
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        words[0] = 0b1 | ((words.len() as u32) << 16);

        let band = BandLayout {
            width: 3,
            height: 2,
            blocks_x: 1,
            blocks_y: 1,
            first_block: 0,
        };
        let mut plane = vec![0.0f32; 3 * 2];
        decode_block(&words, &band, 0, 0, &mut plane).unwrap();
        assert!(
            plane.iter().all(|v| *v > 0.0),
            "all 6 in-band values are set: {plane:?}"
        );
    }
}
