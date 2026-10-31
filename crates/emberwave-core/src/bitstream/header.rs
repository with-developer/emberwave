//! Block headers and the start of frame header.
//!
//! Both are 8 bytes of two little-endian `u32` words. Bit-fields start in the
//! low bits, exactly as the C structures in `bitstream.md` are laid out.

use super::consts::{EXTENDED_CODE_START_OF_FRAME, HEADER_BYTES};
use super::error::Error;

/// Chroma subsampling of a stream (`chroma_resolution`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaSubsampling {
    /// 4:2:0: chroma planes have half the luma resolution in both directions.
    Yuv420,
    /// 4:4:4: chroma planes have the luma resolution.
    Yuv444,
}

/// Position of 4:2:0 chroma samples (`chroma_siting`). Signalling only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaSiting {
    /// Chroma sits in the centre of the luma samples it covers.
    Center,
    /// Chroma sits on the left luma sample.
    Left,
}

/// Value range of the Y, Cb and Cr samples (`ycbcr_range`). Signalling only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YcbcrRange {
    /// Full range.
    Full,
    /// Limited ("studio") range.
    Limited,
}

/// Colour primaries (`color_primaries`). Signalling only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorPrimaries {
    /// sRGB / BT.709 primaries.
    Srgb,
    /// BT.2020 primaries.
    Bt2020,
}

/// Y'CbCr matrix (`ycbcr_transform`). Signalling only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YcbcrTransform {
    /// BT.709.
    Bt709,
    /// BT.2020 non-constant luminance.
    Bt2020,
}

/// Transfer function (`transfer_function`). Signalling only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferFunction {
    /// sRGB non-linear.
    Srgb,
    /// SMPTE ST 2084 (PQ).
    Pq,
}

/// Header of a coded 32x32 block (`BitstreamHeader` in `bitstream.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockHeader {
    /// One bit per 8x8 block, row-major from bit 0. A clear bit means the 8x8
    /// block is all zero and carries no data.
    pub ballot: u16,
    /// Size of the whole block in 32-bit words, header included.
    pub payload_words: u16,
    /// Frame counter, modulo 8.
    pub sequence: u8,
    /// How to scale coefficients back to floating point.
    pub quant_code: u8,
    /// Linear index of the block in the frame (see `FrameLayout`).
    pub block_index: u32,
}

impl BlockHeader {
    /// Decodes the two header words.
    pub fn from_words(word0: u32, word1: u32) -> Self {
        Self {
            ballot: (word0 & 0xffff) as u16,
            payload_words: ((word0 >> 16) & 0xfff) as u16,
            sequence: ((word0 >> 28) & 0x7) as u8,
            quant_code: (word1 & 0xff) as u8,
            block_index: word1 >> 8,
        }
    }

    /// Size of the whole block in bytes, header included.
    pub fn payload_bytes(&self) -> usize {
        usize::from(self.payload_words) * 4
    }

    /// Encodes the two header words, the inverse of [`from_words`](Self::from_words).
    /// Fields wider than the bitstream allows are cut to their bits:
    /// `payload_words` to 12, `sequence` to 3 and `block_index` to 24.
    pub fn to_words(&self) -> (u32, u32) {
        let word0 = u32::from(self.ballot)
            | (u32::from(self.payload_words) & 0xfff) << 16
            | (u32::from(self.sequence) & 0x7) << 28;
        let word1 = u32::from(self.quant_code) | (self.block_index & 0x00ff_ffff) << 8;
        (word0, word1)
    }
}

/// The start of frame header (`BitstreamSequenceHeader` in `bitstream.md`).
///
/// The same layout is the stream description in a PWV1 file header, where
/// `sequence`, `total_blocks` and `code` are reserved and zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceHeader {
    /// Luma width in pixels.
    pub width: u32,
    /// Luma height in pixels.
    pub height: u32,
    /// Frame counter, modulo 8.
    pub sequence: u8,
    /// Number of non-zero 32x32 blocks the frame carries.
    pub total_blocks: u32,
    /// Extended header kind. Only start of frame (0) is defined.
    pub code: u8,
    /// Chroma subsampling.
    pub chroma: ChromaSubsampling,
    /// Colour primaries (signalling only).
    pub color_primaries: ColorPrimaries,
    /// Transfer function (signalling only).
    pub transfer_function: TransferFunction,
    /// Y'CbCr matrix (signalling only).
    pub ycbcr_transform: YcbcrTransform,
    /// Sample range (signalling only).
    pub ycbcr_range: YcbcrRange,
    /// Chroma siting (signalling only).
    pub chroma_siting: ChromaSiting,
}

impl SequenceHeader {
    /// Encodes the two words of a start of frame header, the inverse of
    /// [`from_words`](Self::from_words), with the `extended` bit set.
    pub fn start_of_frame_words(&self) -> (u32, u32) {
        let (word0, word1) = self.stream_words();
        (word0 | 1 << 31, word1)
    }

    /// Encodes the two words without the `extended` bit, as the stream
    /// description of a PWV1 file header has them. Width and height must be
    /// in `1..=16384`; fields wider than the bitstream allows are cut to their
    /// bits.
    pub fn stream_words(&self) -> (u32, u32) {
        let word0 = (self.width.wrapping_sub(1) & 0x3fff)
            | (self.height.wrapping_sub(1) & 0x3fff) << 14
            | (u32::from(self.sequence) & 0x7) << 28;
        let bit = |set: bool, shift: u32| u32::from(set) << shift;
        let word1 = (self.total_blocks & 0x00ff_ffff)
            | (u32::from(self.code) & 0x3) << 24
            | bit(self.chroma == ChromaSubsampling::Yuv444, 26)
            | bit(self.color_primaries == ColorPrimaries::Bt2020, 27)
            | bit(self.transfer_function == TransferFunction::Pq, 28)
            | bit(self.ycbcr_transform == YcbcrTransform::Bt2020, 29)
            | bit(self.ycbcr_range == YcbcrRange::Limited, 30)
            | bit(self.chroma_siting == ChromaSiting::Left, 31);
        (word0, word1)
    }

    /// Decodes the two header words. The `extended` bit is not looked at.
    pub fn from_words(word0: u32, word1: u32) -> Self {
        Self {
            width: (word0 & 0x3fff) + 1,
            height: ((word0 >> 14) & 0x3fff) + 1,
            sequence: ((word0 >> 28) & 0x7) as u8,
            total_blocks: word1 & 0x00ff_ffff,
            code: ((word1 >> 24) & 0x3) as u8,
            chroma: if (word1 >> 26) & 1 == 0 {
                ChromaSubsampling::Yuv420
            } else {
                ChromaSubsampling::Yuv444
            },
            color_primaries: if (word1 >> 27) & 1 == 0 {
                ColorPrimaries::Srgb
            } else {
                ColorPrimaries::Bt2020
            },
            transfer_function: if (word1 >> 28) & 1 == 0 {
                TransferFunction::Srgb
            } else {
                TransferFunction::Pq
            },
            ycbcr_transform: if (word1 >> 29) & 1 == 0 {
                YcbcrTransform::Bt709
            } else {
                YcbcrTransform::Bt2020
            },
            ycbcr_range: if (word1 >> 30) & 1 == 0 {
                YcbcrRange::Full
            } else {
                YcbcrRange::Limited
            },
            chroma_siting: if (word1 >> 31) & 1 == 0 {
                ChromaSiting::Center
            } else {
                ChromaSiting::Left
            },
        }
    }
}

/// A parsed 8 byte header, told apart by its `extended` bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Header {
    /// Header of a coded 32x32 block.
    Block(BlockHeader),
    /// Start of frame header.
    StartOfFrame(SequenceHeader),
}

impl Header {
    /// Parses the header at the start of `bytes`.
    ///
    /// Fails if fewer than 8 bytes are available or if an extended header has
    /// an unknown `code`.
    pub fn parse(bytes: &[u8]) -> Result<Self, Error> {
        let (word0, word1) = read_words(bytes)?;
        if (word0 >> 31) & 1 == 0 {
            return Ok(Self::Block(BlockHeader::from_words(word0, word1)));
        }

        let header = SequenceHeader::from_words(word0, word1);
        if header.code != EXTENDED_CODE_START_OF_FRAME {
            return Err(Error::UnknownExtendedCode(header.code));
        }
        Ok(Self::StartOfFrame(header))
    }
}

/// The sequence number (bits 28 to 30 of word 0) of a header of either kind,
/// without interpreting anything else. `None` if fewer than 4 bytes are given.
pub(crate) fn peek_sequence(bytes: &[u8]) -> Option<u8> {
    let first = bytes.first_chunk::<4>()?;
    Some(((u32::from_le_bytes(*first) >> 28) & 0x7) as u8)
}

/// Reads the two little-endian words of a header.
pub(crate) fn read_words(bytes: &[u8]) -> Result<(u32, u32), Error> {
    match bytes.first_chunk::<HEADER_BYTES>() {
        Some(head) => {
            let word0 = u32::from_le_bytes([head[0], head[1], head[2], head[3]]);
            let word1 = u32::from_le_bytes([head[4], head[5], head[6], head[7]]);
            Ok((word0, word1))
        }
        None => Err(Error::TruncatedHeader {
            available: bytes.len(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The expected bytes below are worked out by hand from the C bit-field
    // layout in bitstream.md, independently of the decoding code.

    #[test]
    fn block_header_fields() {
        // ballot 0xA5C3, payload_words 0x123, sequence 5, extended 0,
        // quant_code 0x9D, block_index 0xABCDEF.
        let bytes = [0xC3, 0xA5, 0x23, 0x51, 0x9D, 0xEF, 0xCD, 0xAB];
        let Header::Block(h) = Header::parse(&bytes).unwrap() else {
            panic!("expected a block header");
        };
        assert_eq!(h.ballot, 0xA5C3);
        assert_eq!(h.payload_words, 0x123);
        assert_eq!(h.payload_bytes(), 0x123 * 4);
        assert_eq!(h.sequence, 5);
        assert_eq!(h.quant_code, 0x9D);
        assert_eq!(h.block_index, 0x00AB_CDEF);
    }

    #[test]
    fn start_of_frame_fields() {
        // 1920x1080, sequence 3, 1234 blocks, 4:4:4, BT.2020, PQ, BT.2020
        // matrix, limited range, left siting.
        let bytes = [0x7F, 0xC7, 0x0D, 0xB1, 0xD2, 0x04, 0x00, 0xFC];
        let Header::StartOfFrame(h) = Header::parse(&bytes).unwrap() else {
            panic!("expected a start of frame header");
        };
        assert_eq!((h.width, h.height), (1920, 1080));
        assert_eq!(h.sequence, 3);
        assert_eq!(h.total_blocks, 1234);
        assert_eq!(h.chroma, ChromaSubsampling::Yuv444);
        assert_eq!(h.color_primaries, ColorPrimaries::Bt2020);
        assert_eq!(h.transfer_function, TransferFunction::Pq);
        assert_eq!(h.ycbcr_transform, YcbcrTransform::Bt2020);
        assert_eq!(h.ycbcr_range, YcbcrRange::Limited);
        assert_eq!(h.chroma_siting, ChromaSiting::Left);
    }

    #[test]
    fn headers_encode_to_the_bytes_they_decode_from() {
        let block = [0xC3, 0xA5, 0x23, 0x51, 0x9D, 0xEF, 0xCD, 0xAB];
        let Header::Block(h) = Header::parse(&block).unwrap() else {
            panic!("expected a block header");
        };
        let (word0, word1) = h.to_words();
        assert_eq!([word0.to_le_bytes(), word1.to_le_bytes()].concat(), block);

        let start = [0x7F, 0xC7, 0x0D, 0xB1, 0xD2, 0x04, 0x00, 0xFC];
        let Header::StartOfFrame(h) = Header::parse(&start).unwrap() else {
            panic!("expected a start of frame header");
        };
        let (word0, word1) = h.start_of_frame_words();
        assert_eq!([word0.to_le_bytes(), word1.to_le_bytes()].concat(), start);
        // Without the extended bit, as in a PWV1 file header.
        assert_eq!(h.stream_words(), (word0 & !(1 << 31), word1));
    }

    #[test]
    fn start_of_frame_defaults_are_zero_bits() {
        let bytes = [0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00];
        let Header::StartOfFrame(h) = Header::parse(&bytes).unwrap() else {
            panic!("expected a start of frame header");
        };
        assert_eq!((h.width, h.height), (1, 1));
        assert_eq!(h.chroma, ChromaSubsampling::Yuv420);
        assert_eq!(h.ycbcr_range, YcbcrRange::Full);
        assert_eq!(h.chroma_siting, ChromaSiting::Center);
    }

    #[test]
    fn unknown_extended_code_is_rejected() {
        // extended = 1, code = 2.
        let bytes = [0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x02];
        assert_eq!(Header::parse(&bytes), Err(Error::UnknownExtendedCode(2)));
    }

    #[test]
    fn short_input_is_rejected() {
        assert_eq!(
            Header::parse(&[0; 7]),
            Err(Error::TruncatedHeader { available: 7 })
        );
        assert_eq!(
            Header::parse(&[]),
            Err(Error::TruncatedHeader { available: 0 })
        );
    }
}
