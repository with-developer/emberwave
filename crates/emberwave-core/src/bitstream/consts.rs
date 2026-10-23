//! Named constants of the PyroWave v1 bitstream.
//!
//! Each constant says where it comes from. "Upstream" is the PyroWave
//! repository at the revision pinned in `testdata/README.md`, and
//! `bitstream.md` is its `bitstream/bitstream.md`.

/// Wavelet decomposition levels.
///
/// Source: `bitstream.md`, "Wavelet transform"; upstream `pyrowave_common.hpp`
/// `DecompositionLevels`.
pub const DECOMPOSITION_LEVELS: usize = 5;

/// Image dimensions are padded up to a multiple of this.
///
/// Source: `bitstream.md`, "Image dimension alignment and padding", `Alignment`.
pub const ALIGNMENT: u32 = 1 << DECOMPOSITION_LEVELS;

/// Smallest padded image dimension. Below this the mirroring of the coarsest
/// band would start to mirror twice.
///
/// Source: `bitstream.md`, `MinimumImageSize`.
pub const MINIMUM_IMAGE_SIZE: u32 = 4 << DECOMPOSITION_LEVELS;

/// Colour components: Y, Cb, Cr.
///
/// Source: `bitstream.md`, `NumComponents`.
pub const NUM_COMPONENTS: usize = 3;

/// Frequency bands per decomposition level: LL, HL, LH, HH.
///
/// Source: `bitstream.md`, `NumFrequencyBandsPerLevel`.
pub const BANDS_PER_LEVEL: usize = 4;

/// Largest image width or height. `width_minus_1` and `height_minus_1` are
/// 14-bit fields.
///
/// Source: `bitstream.md`, "Image dimension alignment and padding".
pub const MAX_DIMENSION: u32 = 1 << 14;

/// Side of a block of coefficients that is coded in isolation.
///
/// Source: `bitstream.md`, "Decoding wavelet coefficients".
pub const BLOCK_SIZE: u32 = 32;

/// Side of the 8x8 blocks a 32x32 block is made of.
///
/// Source: `bitstream.md`, "Decoding 8x8 blocks".
pub const BLOCK8_SIZE: u32 = 8;

/// 8x8 blocks per side of a 32x32 block.
pub const BLOCK8_PER_SIDE: usize = (BLOCK_SIZE / BLOCK8_SIZE) as usize;

/// 8x8 blocks in a 32x32 block, which is also the width of the `ballot` field.
pub const BLOCK8_PER_BLOCK: usize = BLOCK8_PER_SIDE * BLOCK8_PER_SIDE;

/// 4x2 subblocks in an 8x8 block.
///
/// Source: `bitstream.md`, "Decoding 8x8 blocks" (subblock order).
pub const SUBBLOCKS_PER_BLOCK8: usize = 8;

/// Coefficients in a 4x2 subblock.
pub const VALUES_PER_SUBBLOCK: usize = 8;

/// Width of a 4x2 subblock in coefficients.
pub const SUBBLOCK_WIDTH: usize = 4;

/// Height of a 4x2 subblock in coefficients.
pub const SUBBLOCK_HEIGHT: usize = 2;

/// Size in bytes of the header that starts every block and of the start of
/// frame header.
///
/// Source: `bitstream.md`, `BitstreamHeader` and `BitstreamSequenceHeader`.
pub const HEADER_BYTES: usize = 8;

/// Size in bytes of one control word (`CodeWords[i]`) in a block.
pub const CONTROL_WORD_BYTES: usize = 2;

/// Size in bytes of one `QScale[i]` entry in a block.
pub const QSCALE_BYTES: usize = 1;

/// The sequence counter is 3 bits wide and wraps.
///
/// Source: upstream `pyrowave_common.hpp`, `SequenceCountMask`.
pub const SEQUENCE_COUNT_MASK: u8 = 0x7;

/// Largest forward distance (modulo 8) of a sequence number that still counts
/// as newer. A larger distance means the packet belongs to an older frame.
///
/// Source: upstream `pyrowave_decoder.cpp`, `Impl::push_packet`
/// (`SequenceCountMask / 2`).
pub const SEQUENCE_MAX_FORWARD_DISTANCE: u8 = SEQUENCE_COUNT_MASK / 2;

/// `code` value of the start of frame extended header.
///
/// Source: `bitstream.md`, `BITSTREAM_EXTENDED_CODE_START_OF_FRAME`.
pub const EXTENDED_CODE_START_OF_FRAME: u8 = 0;

/// Exponent bias of the block quantiser scale: `e = MAX_SCALE_EXP - (quant_code >> 3)`.
///
/// Source: `bitstream.md`, "Dequantization", `MaxScaleExp`.
pub const MAX_SCALE_EXP: i32 = 4;

/// Mantissa bits of `quant_code` (the low three bits).
pub const QUANT_MANTISSA_BITS: u32 = 3;

/// Implicit leading one of the `quant_code` mantissa: the mantissa is `8 + m`.
pub const QUANT_MANTISSA_BASE: u32 = 8;

/// Bias of the 8x8 block scale: `code / 8 + 0.25`.
///
/// Source: `bitstream.md`, "Dequantization", `Block8x8Scale`; upstream
/// `shaders/dwt_quant_scale.h`.
pub const BLOCK8_SCALE_BIAS: f32 = 0.25;

/// Step of the 8x8 block scale per unit of its 4-bit code.
pub const BLOCK8_SCALE_STEP: f32 = 1.0 / 8.0;

/// Added to every non-zero decoded magnitude before scaling (deadzone
/// quantiser reconstruction point).
///
/// Source: `bitstream.md`, "Dequantization".
pub const DEADZONE_OFFSET: f32 = 0.5;

/// Offset added to every decoded sample before clamping to `[0, 1]`.
///
/// Source: `bitstream.md`, "Inverse DC shift".
pub const DC_SHIFT: f32 = 0.5;

/// Packed in the second byte of a block's `QScale` entry: the low four bits
/// carry the number of base bit-planes, the high four bits the 8x8 scale code.
///
/// Source: upstream `shaders/constants.h` (`Q_PLANES_BITS`, `QUANT_SCALE_BITS`).
pub const QSCALE_PLANES_MASK: u8 = 0x0f;

/// Shift that moves the 8x8 scale code out of a `QScale` byte.
pub const QSCALE_SCALE_SHIFT: u32 = 4;
