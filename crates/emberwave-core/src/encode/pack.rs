//! Packing: the bytes of a coded 32x32 block (layout in
//! `crate::bitstream::block` and `bitstream.md`, "Decoding 8x8 blocks").

use super::quant::{BandQuant, Block8, Blocks8};
use crate::bitstream::consts::{
    CONTROL_WORD_BYTES, HEADER_BYTES, QSCALE_BYTES, QSCALE_SCALE_SHIFT, SUBBLOCKS_PER_BLOCK8,
    VALUES_PER_SUBBLOCK,
};
use crate::bitstream::header::BlockHeader;

/// The size in words of a block sent although none of its 8x8 blocks is left
/// to send: its header, then the code word and the scale of one 8x8 block
/// with no bit-planes, which decodes as zeros (decision D58).
pub(crate) const EMPTY_BLOCK_WORDS: u16 =
    ((HEADER_BYTES + CONTROL_WORD_BYTES + QSCALE_BYTES) as u16).div_ceil(4);

/// The size in words of a block with `drop` planes dropped, or 0 if none of
/// its 8x8 blocks is left to send: what [`pack_block`] writes, unless it
/// keeps the block.
pub(crate) fn block_words(blocks8: &Blocks8, drop: u8) -> u16 {
    let (mut active, mut plane_bytes, mut signs) = (0usize, 0usize, 0usize);
    for block8 in blocks8.iter().flatten().filter(|b| b.active(drop)) {
        active += 1;
        plane_bytes += usize::from(block8.plane_bytes[usize::from(drop)]);
        signs += usize::from(block8.signs[usize::from(drop)]);
    }
    if active == 0 {
        return 0;
    }
    let bytes = HEADER_BYTES
        + active * (CONTROL_WORD_BYTES + QSCALE_BYTES)
        + plane_bytes
        + signs.div_ceil(8);
    bytes.div_ceil(4) as u16
}

/// Appends a block with `drop` planes dropped to `out`, which must end at a
/// multiple of four bytes. If none of its 8x8 blocks is left, it writes
/// nothing, or with `keep` a block of [`EMPTY_BLOCK_WORDS`]: its first 8x8
/// block, which is always inside the band, with no bit-planes. A 32x32 block
/// with no 8x8 block at all should not be sent (`bitstream.md`).
pub(crate) fn pack_block(
    blocks8: &Blocks8,
    quant: &BandQuant,
    drop: u8,
    sequence: u8,
    block_index: u32,
    keep: bool,
    out: &mut Vec<u8>,
) {
    let active: Vec<(usize, &Block8)> = blocks8
        .iter()
        .enumerate()
        .filter_map(|(raster, block8)| block8.as_ref().map(|b| (raster, b)))
        .filter(|(_, block8)| block8.active(drop))
        .collect();
    if active.is_empty() {
        if keep {
            let header = BlockHeader {
                ballot: 1,
                payload_words: EMPTY_BLOCK_WORDS,
                sequence,
                quant_code: quant.quant_code(drop),
                block_index,
            };
            let (word0, word1) = header.to_words();
            out.extend_from_slice(&word0.to_le_bytes());
            out.extend_from_slice(&word1.to_le_bytes());
            // A code word and a scale of zero: no planes above a base of none.
            out.resize(
                out.len() + usize::from(EMPTY_BLOCK_WORDS) * 4 - HEADER_BYTES,
                0,
            );
        }
        return;
    }

    let start = out.len();
    out.extend_from_slice(&[0; HEADER_BYTES]);

    // Two bits per subblock: its planes above the base.
    for (_, block8) in &active {
        let base = block8.base_planes(drop);
        let control = (0..SUBBLOCKS_PER_BLOCK8).fold(0u16, |control, subblock| {
            let extra = block8.subblock_planes(subblock, drop) - base;
            control | u16::from(extra) << (2 * subblock)
        });
        out.extend_from_slice(&control.to_le_bytes());
    }
    for (_, block8) in &active {
        out.push(block8.scale_code << QSCALE_SCALE_SHIFT | block8.base_planes(drop));
    }

    // Bit-planes, per subblock, most significant first; bit v of a byte is
    // value v.
    for (_, block8) in &active {
        for subblock in 0..SUBBLOCKS_PER_BLOCK8 {
            let values =
                &block8.magnitudes[subblock * VALUES_PER_SUBBLOCK..][..VALUES_PER_SUBBLOCK];
            for plane in (0..block8.subblock_planes(subblock, drop)).rev() {
                let bit = u32::from(drop + plane);
                out.push(
                    values
                        .iter()
                        .enumerate()
                        .fold(0u8, |byte, (v, &m)| byte | (((m >> bit) & 1) as u8) << v),
                );
            }
        }
    }

    // One sign bit per non-zero magnitude in coefficient order, the first in
    // the lowest bit; unused bits of the last byte are zero.
    let mut signs = 0usize;
    for (_, block8) in &active {
        for (i, &m) in block8.magnitudes.iter().enumerate() {
            if m >> drop == 0 {
                continue;
            }
            if signs.is_multiple_of(8) {
                out.push(0);
            }
            if block8.negative >> i & 1 == 1 {
                *out.last_mut().expect("a byte was pushed") |= 1 << (signs % 8);
            }
            signs += 1;
        }
    }
    while !(out.len() - start).is_multiple_of(4) {
        out.push(0);
    }

    let header = BlockHeader {
        ballot: active
            .iter()
            .fold(0u16, |ballot, (raster, _)| ballot | 1 << raster),
        payload_words: ((out.len() - start) / 4) as u16,
        sequence,
        quant_code: quant.quant_code(drop),
        block_index,
    };
    debug_assert_eq!(header.payload_words, block_words(blocks8, drop));
    let (word0, word1) = header.to_words();
    out[start..start + 4].copy_from_slice(&word0.to_le_bytes());
    out[start + 4..start + 8].copy_from_slice(&word1.to_le_bytes());
}
