//! Internal structure of one coded 32x32 block.
//!
//! A block is a sequence of little-endian bytes, stored here as `u32` words:
//!
//! ```text
//! header        8 bytes     BlockHeader
//! control[N]    2 bytes     per active 8x8 block: 2 bits per subblock = extra bit-planes
//! qscale[N]     1 byte      per active 8x8 block: low nibble = base bit-planes, high = scale code
//! planes        P bytes     bit-planes, most significant first, one byte per 4x2 subblock plane
//! signs         ceil(S/8)   one bit per non-zero coefficient
//! padding                   up to the next 32-bit boundary, ignored
//! ```
//!
//! `N` is the number of set bits in `ballot`. Active 8x8 blocks follow each
//! other in row-major order. See `ARCHITECTURE.md` for the derivation.

use super::consts::{
    BLOCK8_PER_BLOCK, CONTROL_WORD_BYTES, HEADER_BYTES, QSCALE_BYTES, QSCALE_PLANES_MASK,
    QSCALE_SCALE_SHIFT, SUBBLOCKS_PER_BLOCK8,
};
use super::error::Error;
use super::header::BlockHeader;

/// Words in a block header.
const HEADER_WORDS: usize = HEADER_BYTES / 4;

/// Little-endian byte view over the words of a block.
///
/// Reads outside the block return 0. Callers check sizes first, and this
/// keeps a malformed block from ever causing a panic.
#[derive(Debug, Clone, Copy)]
pub struct BlockBytes<'a> {
    words: &'a [u32],
}

impl<'a> BlockBytes<'a> {
    /// Wraps the words of a block, header included.
    pub fn new(words: &'a [u32]) -> Self {
        Self { words }
    }

    /// Size of the block in bytes.
    pub fn len(&self) -> usize {
        self.words.len() * 4
    }

    /// Whether the block has no words.
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// The byte at `offset`, or 0 past the end.
    pub fn u8(&self, offset: usize) -> u8 {
        self.words
            .get(offset / 4)
            .map_or(0, |word| (word >> (8 * (offset % 4))) as u8)
    }

    /// The little-endian 16-bit value at `offset`, or 0 past the end.
    pub fn u16(&self, offset: usize) -> u16 {
        u16::from_le_bytes([self.u8(offset), self.u8(offset.saturating_add(1))])
    }
}

/// An 8x8 block that carries data (its `ballot` bit is set).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveBlock8 {
    /// Two bits per subblock: bit-planes on top of `base_planes`.
    pub control_word: u16,
    /// Bit-planes every subblock has.
    pub base_planes: u8,
    /// Four bit code of the 8x8 block scale.
    pub scale_code: u8,
    /// Byte offset from the start of the block of this 8x8 block's first bit-plane.
    pub plane_offset: usize,
    /// Bytes of bit-planes this 8x8 block occupies.
    pub plane_bytes: usize,
}

impl ActiveBlock8 {
    /// Bit-planes of one 4x2 subblock (`0..8`).
    pub fn subblock_planes(&self, subblock: usize) -> usize {
        usize::from((self.control_word >> (2 * subblock)) & 0x3) + usize::from(self.base_planes)
    }
}

/// Where the parts of a block are, worked out from its header and tables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockStructure {
    header: BlockHeader,
    blocks8: [Option<ActiveBlock8>; BLOCK8_PER_BLOCK],
    sign_offset: usize,
}

impl BlockStructure {
    /// Parses the structure of the block made of `words` (header included).
    ///
    /// Checks that the tables and bit-planes fit in the block. The sign bits
    /// follow the bit-planes; how many there are depends on the decoded
    /// magnitudes, so that is checked while decoding.
    pub fn parse(words: &[u32]) -> Result<Self, Error> {
        if words.len() < HEADER_WORDS {
            return Err(Error::BlockTooSmall { words: words.len() });
        }

        let header = BlockHeader::from_words(words[0], words[1]);
        let bytes = BlockBytes::new(words);
        let size = bytes.len();

        let active = header.ballot.count_ones() as usize;
        let tables_end = HEADER_BYTES + active * (CONTROL_WORD_BYTES + QSCALE_BYTES);
        if tables_end > size {
            return Err(Error::BlockContentOverrun {
                needed: tables_end,
                available: size,
            });
        }

        let qscale_base = HEADER_BYTES + active * CONTROL_WORD_BYTES;
        let mut blocks8 = [None; BLOCK8_PER_BLOCK];
        let mut plane_cursor = tables_end;
        let mut index = 0;
        for (raster, slot) in blocks8.iter_mut().enumerate() {
            if header.ballot & (1 << raster) == 0 {
                continue;
            }

            let control_word = bytes.u16(HEADER_BYTES + index * CONTROL_WORD_BYTES);
            let qscale = bytes.u8(qscale_base + index * QSCALE_BYTES);
            let base_planes = qscale & QSCALE_PLANES_MASK;

            let extra_planes: usize = (0..SUBBLOCKS_PER_BLOCK8)
                .map(|subblock| usize::from((control_word >> (2 * subblock)) & 0x3))
                .sum();
            let plane_bytes = extra_planes + SUBBLOCKS_PER_BLOCK8 * usize::from(base_planes);

            *slot = Some(ActiveBlock8 {
                control_word,
                base_planes,
                scale_code: qscale >> QSCALE_SCALE_SHIFT,
                plane_offset: plane_cursor,
                plane_bytes,
            });
            plane_cursor += plane_bytes;
            index += 1;
        }

        if plane_cursor > size {
            return Err(Error::BlockContentOverrun {
                needed: plane_cursor,
                available: size,
            });
        }

        Ok(Self {
            header,
            blocks8,
            sign_offset: plane_cursor,
        })
    }

    /// The block header.
    pub fn header(&self) -> &BlockHeader {
        &self.header
    }

    /// The 8x8 block at row-major position `raster` (`0..16`), if it is active.
    pub fn block8(&self, raster: usize) -> Option<&ActiveBlock8> {
        self.blocks8.get(raster)?.as_ref()
    }

    /// Number of active 8x8 blocks.
    pub fn active_count(&self) -> usize {
        self.header.ballot.count_ones() as usize
    }

    /// Byte offset from the start of the block where the sign bits begin.
    pub fn sign_offset(&self) -> usize {
        self.sign_offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Packs little-endian bytes into words, zero padding the last one.
    fn pack(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks(4)
            .map(|chunk| {
                let mut word = [0u8; 4];
                word[..chunk.len()].copy_from_slice(chunk);
                u32::from_le_bytes(word)
            })
            .collect()
    }

    /// A block of `words` words: a header with `ballot`, then `tables`, then
    /// zero padding.
    fn block(ballot: u16, tables: &[u8], words: usize) -> Vec<u32> {
        let mut bytes = Vec::new();
        let word0 = u32::from(ballot) | ((words as u32) << 16);
        bytes.extend_from_slice(&word0.to_le_bytes());
        bytes.extend_from_slice(&(7u32 << 8).to_le_bytes()); // quant_code 0, block_index 7
        bytes.extend_from_slice(tables);
        bytes.resize(words * 4, 0);
        pack(&bytes)
    }

    #[test]
    fn plane_offsets_follow_the_tables() {
        // Active 8x8 blocks at raster 0 and 2.
        // Block A: subblocks 0 and 1 have 2 and 1 extra planes, 2 base planes:
        //          3 + 8 * 2 = 19 bytes of planes.
        // Block B: all subblocks have 3 extra planes, no base planes: 24 bytes.
        let control_a: u16 = 0b0110;
        let control_b: u16 = 0xFFFF;
        let mut tables = Vec::new();
        tables.extend_from_slice(&control_a.to_le_bytes());
        tables.extend_from_slice(&control_b.to_le_bytes());
        tables.push(0x32); // A: scale code 3, base planes 2
        tables.push(0x50); // B: scale code 5, base planes 0

        // tables end at 8 + 3 * 2 = 14; planes 14..33 and 33..57; signs from 57.
        let words = block(0b101, &tables, 15);
        let s = BlockStructure::parse(&words).unwrap();

        assert_eq!(s.active_count(), 2);
        let a = s.block8(0).unwrap();
        assert_eq!((a.plane_offset, a.plane_bytes), (14, 19));
        assert_eq!((a.base_planes, a.scale_code), (2, 3));
        assert_eq!(a.subblock_planes(0), 4);
        assert_eq!(a.subblock_planes(1), 3);
        assert_eq!(a.subblock_planes(2), 2);
        let b = s.block8(2).unwrap();
        assert_eq!((b.plane_offset, b.plane_bytes), (33, 24));
        assert_eq!((b.base_planes, b.scale_code), (0, 5));
        assert_eq!(b.subblock_planes(7), 3);
        assert!(s.block8(1).is_none());
        assert_eq!(s.sign_offset(), 57);
    }

    #[test]
    fn planes_that_do_not_fit_are_rejected() {
        let mut tables = Vec::new();
        tables.extend_from_slice(&0xFFFFu16.to_le_bytes());
        tables.extend_from_slice(&0xFFFFu16.to_le_bytes());
        tables.push(0x00);
        tables.push(0x00);
        // Tables end at 14 and both 8x8 blocks need 24 bytes of planes, so the
        // block needs 62 bytes but 14 words are only 56.
        assert_eq!(
            BlockStructure::parse(&block(0b101, &tables, 14)),
            Err(Error::BlockContentOverrun {
                needed: 14 + 48,
                available: 56
            })
        );
    }

    #[test]
    fn tables_that_do_not_fit_are_rejected() {
        // 16 active blocks need 8 + 48 bytes of tables, but only 3 words are present.
        let words = block(0xFFFF, &[], 3);
        assert_eq!(
            BlockStructure::parse(&words),
            Err(Error::BlockContentOverrun {
                needed: 56,
                available: 12
            })
        );
    }

    #[test]
    fn too_few_words_for_a_header() {
        assert_eq!(
            BlockStructure::parse(&[0x1234]),
            Err(Error::BlockTooSmall { words: 1 })
        );
        assert_eq!(
            BlockStructure::parse(&[]),
            Err(Error::BlockTooSmall { words: 0 })
        );
    }

    #[test]
    fn empty_ballot_has_no_blocks() {
        let s = BlockStructure::parse(&block(0, &[], 2)).unwrap();
        assert_eq!(s.active_count(), 0);
        assert_eq!(s.sign_offset(), 8);
        assert!((0..16).all(|r| s.block8(r).is_none()));
    }

    #[test]
    fn byte_view_is_little_endian_and_total() {
        let words = [0x4433_2211, 0x8877_6655];
        let b = BlockBytes::new(&words);
        assert_eq!(b.len(), 8);
        assert_eq!(b.u8(0), 0x11);
        assert_eq!(b.u8(3), 0x44);
        assert_eq!(b.u8(4), 0x55);
        assert_eq!(b.u16(1), 0x3322);
        assert_eq!(b.u16(3), 0x5544);
        assert_eq!(b.u8(8), 0);
        assert_eq!(b.u16(7), 0x0088);
        assert_eq!(b.u16(usize::MAX - 1), 0);
    }
}
