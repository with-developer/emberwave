//! Quantisation: wavelet coefficients to the integer magnitudes that the
//! bit-planes of a block carry, and for every number of dropped bit-planes
//! what the block then costs and how far it is from the coefficients.
//!
//! The decoder reconstructs `(m + 0.5) * block_scale * block8_scale`
//! (`bitstream.md`, "Dequantization"), so the encoder is free in how it
//! chooses the scales and the magnitudes. Its choices, in this file, follow
//! the same plan as upstream's encoder (decision D28).

use super::weights::WEIGHTS;
use crate::bitstream::consts::{
    BLOCK_SIZE, BLOCK8_PER_BLOCK, BLOCK8_PER_SIDE, BLOCK8_SIZE, DEADZONE_OFFSET, MAX_SCALE_EXP,
    QUANT_MANTISSA_BITS, SUBBLOCK_HEIGHT, SUBBLOCK_WIDTH, SUBBLOCKS_PER_BLOCK8,
    VALUES_PER_SUBBLOCK,
};
use crate::bitstream::{Band, BandLayout, ChromaSubsampling, Component};

/// Numbers of dropped bit-planes the rate control chooses from: 0 to 15.
pub(crate) const DROP_STEPS: usize = 16;

/// Coefficients of an 8x8 block.
pub(crate) const BLOCK8_VALUES: usize = SUBBLOCKS_PER_BLOCK8 * VALUES_PER_SUBBLOCK;

/// The largest bit exponent of a band: coefficients are scaled by at most
/// `2^12` before quantisation, which keeps every block within 15 base
/// bit-planes and its quant code within a byte.
const MAX_EXPONENT: i32 = 12;

/// The largest 8x8 scale code the encoder uses: scale 1. Finer scales than
/// the coefficients need are pointless, see [`analyse_block8`].
const MAX_SCALE_CODE: u8 = 6;

/// 2^`exponent` as an exact `f32`, for exponents in the normal range.
fn exact_pow2(exponent: i32) -> f32 {
    f32::from_bits(((127 + exponent) as u32) << 23)
}

/// The 8x8 scale of a scale code, `code / 8 + 0.25` (`Block8x8Scale`).
fn block8_scale(code: u8) -> f32 {
    f32::from(code + 2) / 8.0
}

/// `1 / block8_scale(code)`, rounded once, the same on every system: the
/// encoder multiplies by it rather than divide, as GPUs divide less exactly.
const INVERSE_SCALES: [f32; MAX_SCALE_CODE as usize + 1] = {
    let mut inverse = [0.0; MAX_SCALE_CODE as usize + 1];
    let mut code = 0;
    while code < inverse.len() {
        inverse[code] = 8.0 / (code as f32 + 2.0);
        code += 1;
    }
    inverse
};

/// The bit exponent `b` of a band before the cap. A coefficient error of one
/// in `c * 2^b` costs about the same in the picture in every band, as `4^b`
/// follows the energy the inverse transform gives a coefficient of the band.
/// Low-pass bands and coarser levels get more bits, chroma one bit less.
pub(crate) fn band_exponent(component: Component, level: usize, band: Band) -> i32 {
    let band_bits = match band {
        Band::Ll => 2,
        Band::Hl | Band::Lh => 1,
        Band::Hh => 0,
    };
    8 + band_bits + level as i32 - i32::from(component != Component::Y)
}

/// How a band is quantised: its capped exponent and its distortion weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BandQuant {
    /// The capped bit exponent: coefficients are quantised as `c * 2^exponent`.
    pub exponent: i32,
    /// What a squared coefficient error of the band counts for in the rate
    /// control (`weights.rs`).
    pub weight: f32,
}

impl BandQuant {
    pub(crate) fn new(
        chroma: ChromaSubsampling,
        component: Component,
        level: usize,
        band: Band,
    ) -> Self {
        let class = match (component, chroma) {
            (Component::Y, _) => 0,
            (_, ChromaSubsampling::Yuv420) => 1,
            (_, ChromaSubsampling::Yuv444) => 2,
        };
        Self {
            exponent: band_exponent(component, level, band).min(MAX_EXPONENT),
            weight: WEIGHTS[class][level][band.index()],
        }
    }

    /// The most bit-planes a block of the band may drop: the decoder's block
    /// scale is `2^(4 - (quant_code >> 3))`, and the exponent `4 + b - q` of
    /// the quant code cannot go below zero.
    pub(crate) fn max_drop(&self) -> u8 {
        (self.exponent + MAX_SCALE_EXP).clamp(0, DROP_STEPS as i32 - 1) as u8
    }

    /// The quant code of a block of the band that drops `drop` bit-planes:
    /// block scale `2^(drop - b)`, mantissa zero.
    pub(crate) fn quant_code(&self, drop: u8) -> u8 {
        ((self.exponent + MAX_SCALE_EXP - i32::from(drop)) as u8) << QUANT_MANTISSA_BITS
    }
}

/// The position of coefficient `index` (subblock, then value) in its 8x8
/// block: subblocks run down two columns of four, values down the two rows
/// of a subblock's four columns (`bitstream.md`, "Decoding 8x8 blocks").
pub(crate) fn value_position(index: usize) -> (usize, usize) {
    let (subblock, value) = (index / VALUES_PER_SUBBLOCK, index % VALUES_PER_SUBBLOCK);
    let x = subblock / 4 * SUBBLOCK_WIDTH + value / SUBBLOCK_HEIGHT;
    let y = subblock % 4 * SUBBLOCK_HEIGHT + value % SUBBLOCK_HEIGHT;
    (x, y)
}

/// The coefficients of the 8x8 block `raster` (0 to 15, row-major) of the
/// 32x32 block at (`block_x`, `block_y`) of a band, in coefficient order.
/// Positions outside the band are zero.
pub(crate) fn gather_block8(
    band: &[f32],
    layout: &BandLayout,
    block_x: u32,
    block_y: u32,
    raster: usize,
) -> [f32; BLOCK8_VALUES] {
    let x0 = (block_x * BLOCK_SIZE) as usize + raster % BLOCK8_PER_SIDE * BLOCK8_SIZE as usize;
    let y0 = (block_y * BLOCK_SIZE) as usize + raster / BLOCK8_PER_SIDE * BLOCK8_SIZE as usize;
    let (width, height) = (layout.width as usize, layout.height as usize);
    std::array::from_fn(|index| {
        let (dx, dy) = value_position(index);
        let (x, y) = (x0 + dx, y0 + dy);
        if x < width && y < height {
            band[y * width + x]
        } else {
            0.0
        }
    })
}

/// One quantised 8x8 block and what dropping bit-planes does to it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Block8 {
    /// The 8x8 scale code, so that the scale is `code / 8 + 0.25`.
    pub scale_code: u8,
    /// Magnitudes in coefficient order, with no plane dropped.
    pub magnitudes: [u32; BLOCK8_VALUES],
    /// Bit `i` set: coefficient `i` is negative.
    pub negative: u64,
    /// Bit-planes of each subblock with no plane dropped.
    pub planes: [u8; SUBBLOCKS_PER_BLOCK8],
    /// Bit-planes every subblock has, with no plane dropped.
    pub base: u8,
    /// Per number of dropped planes: bytes of bit-planes.
    pub plane_bytes: [u16; DROP_STEPS],
    /// Per number of dropped planes: non-zero coefficients, one sign bit each.
    pub signs: [u8; DROP_STEPS],
    /// Per number of dropped planes: the weighted squared error of the
    /// reconstruction against the coefficients.
    pub distortion: [f32; DROP_STEPS],
}

impl Block8 {
    /// Whether the block is coded with `drop` planes dropped: some subblock
    /// keeps a plane above the base. Otherwise every magnitude is zero.
    pub(crate) fn active(&self, drop: u8) -> bool {
        self.plane_bytes[usize::from(drop)] != 0
    }

    /// Bit-planes of subblock `subblock` with `drop` planes dropped.
    pub(crate) fn subblock_planes(&self, subblock: usize, drop: u8) -> u8 {
        self.planes[subblock].saturating_sub(drop)
    }

    /// Base bit-planes with `drop` planes dropped.
    pub(crate) fn base_planes(&self, drop: u8) -> u8 {
        self.base.saturating_sub(drop)
    }
}

/// Quantises an 8x8 block of a band with the given quantisation, or `None`
/// if every magnitude would be zero.
///
/// The coefficients are scaled to `t = c * 2^b`. The scale code is the
/// smallest whose scale still needs no more bit-planes for the largest `|t|`
/// than scale 1 does: within those planes the finest steps. Magnitudes are
/// `trunc(|t| / scale)`, a dead zone around zero, and the decoder's `+ 0.5`
/// puts the reconstruction in the middle of each step. The base bit-planes
/// are all but the top three of the largest magnitude, and each 4x2
/// subblock has as many more as its own largest magnitude needs, 0 to 3.
pub(crate) fn analyse_block8(
    coefficients: &[f32; BLOCK8_VALUES],
    quant: &BandQuant,
) -> Option<Block8> {
    let resolution = exact_pow2(quant.exponent);
    let scaled = coefficients.map(|c| (c * resolution).abs());
    let largest = scaled.iter().copied().fold(0.0f32, f32::max);
    if largest < 1.0 {
        return None;
    }

    // Planes at scale 1, then the smallest scale with no more: all exact.
    let unit_planes = 32 - (largest as u32).leading_zeros();
    let limit = exact_pow2(unit_planes as i32);
    let scale_code = (0..MAX_SCALE_CODE)
        .find(|&code| largest < limit * block8_scale(code))
        .unwrap_or(MAX_SCALE_CODE);
    let inverse = INVERSE_SCALES[usize::from(scale_code)];

    let magnitudes = scaled.map(|t| (t * inverse) as u32);
    let largest_magnitude = magnitudes.iter().copied().max().unwrap_or(0);
    if largest_magnitude == 0 {
        return None;
    }
    let negative = coefficients.iter().enumerate().fold(0u64, |bits, (i, c)| {
        bits | u64::from(c.is_sign_negative() && *c != 0.0) << i
    });

    let top_bit = 31 - largest_magnitude.leading_zeros();
    let base = top_bit.saturating_sub(2) as u8;
    let planes: [u8; SUBBLOCKS_PER_BLOCK8] = std::array::from_fn(|subblock| {
        let values = &magnitudes[subblock * VALUES_PER_SUBBLOCK..][..VALUES_PER_SUBBLOCK];
        let largest = values.iter().copied().max().unwrap_or(0) >> base;
        base + (32 - largest.leading_zeros()) as u8
    });

    // Multiplications by exact powers of two only, so that a GPU gets the
    // same: GPUs need not divide exactly.
    let to_coefficient = exact_pow2(-quant.exponent);
    let step = block8_scale(scale_code) * to_coefficient;
    let mut plane_bytes = [0u16; DROP_STEPS];
    let mut signs = [0u8; DROP_STEPS];
    let mut distortion = [0.0f32; DROP_STEPS];
    for drop in 0..DROP_STEPS as u8 {
        let kept = |m: u32| m.checked_shr(u32::from(drop)).unwrap_or(0);
        let extra: u8 = planes
            .iter()
            .map(|&p| p.saturating_sub(drop) - base.saturating_sub(drop))
            .sum();
        if extra > 0 {
            plane_bytes[usize::from(drop)] = planes
                .iter()
                .map(|&p| u16::from(p.saturating_sub(drop)))
                .sum();
            signs[usize::from(drop)] = magnitudes.iter().filter(|&&m| kept(m) != 0).count() as u8;
        }

        let drop_scale = exact_pow2(i32::from(drop)) * step;
        let error: f32 = magnitudes
            .iter()
            .zip(&scaled)
            .map(|(&m, &t)| {
                let reconstructed = match kept(m) {
                    0 => 0.0,
                    k => (k as f32 + DEADZONE_OFFSET) * drop_scale,
                };
                let difference = t * to_coefficient - reconstructed;
                difference * difference
            })
            .sum();
        distortion[usize::from(drop)] = quant.weight * error;
    }

    Some(Block8 {
        scale_code,
        magnitudes,
        negative,
        planes,
        base,
        plane_bytes,
        signs,
        distortion,
    })
}

/// The 8x8 blocks of one 32x32 block, those that carry anything.
pub(crate) type Blocks8 = [Option<Block8>; BLOCK8_PER_BLOCK];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::dequant;

    /// The weighted squared error of an 8x8 block that is not coded at all.
    fn uncoded_distortion(coefficients: &[f32; BLOCK8_VALUES], weight: f32) -> f32 {
        weight * coefficients.iter().map(|c| c * c).sum::<f32>()
    }

    fn quant() -> BandQuant {
        BandQuant::new(ChromaSubsampling::Yuv420, Component::Y, 1, Band::Hl)
    }

    #[test]
    fn band_exponents_and_quant_codes() {
        // Luma LL of the coarsest level: 8 + 2 + 4 = 14, capped to 12; with
        // nothing dropped its quant code is (12 + 4) << 3 = 128.
        let q = BandQuant::new(ChromaSubsampling::Yuv420, Component::Y, 4, Band::Ll);
        assert_eq!(q.exponent, 12);
        assert_eq!(q.quant_code(0), 128);
        assert_eq!(dequant::block_scale(q.quant_code(0)), exact_pow2(-12));
        assert_eq!(dequant::block_scale(q.quant_code(3)), exact_pow2(3 - 12));
        // The finest luma HH band and a chroma band.
        assert_eq!(band_exponent(Component::Y, 0, Band::Hh), 8);
        assert_eq!(band_exponent(Component::Cb, 1, Band::Lh), 9);
        // The quant code's exponent cannot go negative: 8 + 4 - 12 = 0, and
        // the block scale is then 2^(12 - 8).
        let hh = BandQuant::new(ChromaSubsampling::Yuv420, Component::Y, 0, Band::Hh);
        assert_eq!(hh.max_drop(), 12);
        assert_eq!(hh.quant_code(12), 0);
        assert_eq!(dequant::block_scale(hh.quant_code(12)), 16.0);
    }

    #[test]
    fn positions_follow_the_decoders_order() {
        assert_eq!(value_position(0), (0, 0));
        assert_eq!(value_position(1), (0, 1));
        assert_eq!(value_position(2), (1, 0));
        assert_eq!(value_position(7), (3, 1));
        assert_eq!(value_position(8), (0, 2)); // subblock 1
        assert_eq!(value_position(32), (4, 0)); // subblock 4
        assert_eq!(value_position(63), (7, 7));
    }

    #[test]
    fn small_blocks_carry_nothing() {
        // |t| below 1 at every position: nothing to code.
        let resolution = exact_pow2(quant().exponent);
        let coefficients = [0.9 / resolution; BLOCK8_VALUES];
        assert_eq!(analyse_block8(&coefficients, &quant()), None);
    }

    #[test]
    fn the_reconstruction_is_within_half_a_step() {
        // Coefficients spread over a range, both signs.
        let coefficients: [f32; BLOCK8_VALUES] = std::array::from_fn(|i| {
            ((i as f32 * 0.37).sin() * 0.05) * if i % 3 == 0 { -1.0 } else { 1.0 }
        });
        let q = quant();
        let block = analyse_block8(&coefficients, &q).unwrap();
        let step = block8_scale(block.scale_code) / exact_pow2(q.exponent);
        for (i, (&c, &m)) in coefficients.iter().zip(&block.magnitudes).enumerate() {
            if m == 0 {
                // The dead zone: below one step, give or take the rounding of
                // the inverse scale.
                assert!(c.abs() <= step * 1.0001, "{i}: {c} with step {step}");
            } else {
                let reconstructed = (m as f32 + 0.5) * step;
                assert!(
                    (c.abs() - reconstructed).abs() <= 0.5 * step * 1.0001,
                    "{i}"
                );
                assert_eq!(block.negative >> i & 1 == 1, c < 0.0, "{i}");
            }
        }
        // The largest magnitude uses all of its planes: base plus three.
        let largest = *block.magnitudes.iter().max().unwrap();
        assert_eq!(u32::from(block.base) + 3, 32 - largest.leading_zeros());
        // No subblock needs more than three planes above the base.
        assert!(
            block
                .planes
                .iter()
                .all(|&p| p >= block.base && p <= block.base + 3)
        );
    }

    #[test]
    fn dropping_planes_costs_less_and_loses_more() {
        let coefficients: [f32; BLOCK8_VALUES] =
            std::array::from_fn(|i| (i as f32 * 0.91).cos() * 0.08);
        let block = analyse_block8(&coefficients, &quant()).unwrap();
        for drop in 1..DROP_STEPS {
            assert!(block.plane_bytes[drop] <= block.plane_bytes[drop - 1]);
            assert!(block.signs[drop] <= block.signs[drop - 1]);
        }
        // With everything dropped, the error is that of not coding the block.
        let last = DROP_STEPS - 1;
        assert!(!block.active(last as u8));
        assert_eq!(
            block.distortion[last],
            uncoded_distortion(&coefficients, quant().weight)
        );
        assert!(block.distortion[0] < block.distortion[last]);
    }
}
