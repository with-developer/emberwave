//! The PWV1 raw disk format.
//!
//! A file is a 24 byte header followed by frames, each a little-endian `u32`
//! size and that many bytes of packet data (a start of frame header followed
//! by 32x32 blocks). Defined in upstream `bitstream/bitstream.md`, section
//! "Raw disk format specification".

use super::error::Error;
use super::header::{SequenceHeader, read_words};

/// First four bytes of a PWV1 file.
pub const MAGIC: [u8; 4] = *b"PWV1";

/// Size of the PWV1 header in bytes.
pub const FILE_HEADER_BYTES: usize = 24;

/// The only header version there is.
pub const HEADER_VERSION: u8 = 1;

/// Offset of the stream description (a `BitstreamSequenceHeader`) in the header.
const STREAM_OFFSET: usize = 4;
/// Offset of the frame rate numerator.
const FRAME_RATE_OFFSET: usize = 12;
/// Offset of the reference bit depth byte.
const BIT_DEPTH_OFFSET: usize = 20;
/// Offset of the header version byte.
const VERSION_OFFSET: usize = 21;

/// The header of a PWV1 file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pwv1Header {
    /// Picture size, chroma format and colour signalling, in the layout of a
    /// start of frame header. `extended`, `sequence`, `total_blocks` and
    /// `code` are reserved here and carry no meaning.
    pub stream: SequenceHeader,
    /// Frame rate numerator. 0 with a denominator of 0 means the frames are
    /// unrelated images.
    pub frame_rate_num: u32,
    /// Frame rate denominator.
    pub frame_rate_den: u32,
    /// Bit depth of the source video: 0 (arbitrary), 8, 10 or 16.
    pub reference_bit_depth: u8,
}

impl Pwv1Header {
    /// Parses the header at the start of `bytes`.
    pub fn parse(bytes: &[u8]) -> Result<Self, Error> {
        let header = bytes
            .first_chunk::<FILE_HEADER_BYTES>()
            .ok_or(Error::TruncatedFile)?;
        if header[..MAGIC.len()] != MAGIC {
            return Err(Error::BadMagic);
        }
        if header[VERSION_OFFSET] != HEADER_VERSION {
            return Err(Error::UnsupportedHeaderVersion(header[VERSION_OFFSET]));
        }

        let (word0, word1) = read_words(&header[STREAM_OFFSET..])?;
        let read_u32 = |offset: usize| {
            u32::from_le_bytes([
                header[offset],
                header[offset + 1],
                header[offset + 2],
                header[offset + 3],
            ])
        };

        Ok(Self {
            stream: SequenceHeader::from_words(word0, word1),
            frame_rate_num: read_u32(FRAME_RATE_OFFSET),
            frame_rate_den: read_u32(FRAME_RATE_OFFSET + 4),
            reference_bit_depth: header[BIT_DEPTH_OFFSET],
        })
    }

    /// The 24 header bytes, the inverse of [`parse`](Self::parse). The
    /// reserved fields of the stream description (`sequence`,
    /// `total_blocks`, `code`) are written as they are; a file has them zero.
    pub fn to_bytes(&self) -> [u8; FILE_HEADER_BYTES] {
        let mut bytes = [0u8; FILE_HEADER_BYTES];
        bytes[..MAGIC.len()].copy_from_slice(&MAGIC);
        let (word0, word1) = self.stream.stream_words();
        bytes[STREAM_OFFSET..STREAM_OFFSET + 4].copy_from_slice(&word0.to_le_bytes());
        bytes[STREAM_OFFSET + 4..STREAM_OFFSET + 8].copy_from_slice(&word1.to_le_bytes());
        bytes[FRAME_RATE_OFFSET..FRAME_RATE_OFFSET + 4]
            .copy_from_slice(&self.frame_rate_num.to_le_bytes());
        bytes[FRAME_RATE_OFFSET + 4..FRAME_RATE_OFFSET + 8]
            .copy_from_slice(&self.frame_rate_den.to_le_bytes());
        bytes[BIT_DEPTH_OFFSET] = self.reference_bit_depth;
        bytes[VERSION_OFFSET] = HEADER_VERSION;
        bytes
    }
}

/// A PWV1 file held in memory.
#[derive(Debug, Clone, Copy)]
pub struct Pwv1File<'a> {
    header: Pwv1Header,
    frames: &'a [u8],
}

impl<'a> Pwv1File<'a> {
    /// Parses the header of `bytes` and prepares to iterate over the frames.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        let header = Pwv1Header::parse(bytes)?;
        Ok(Self {
            header,
            frames: &bytes[FILE_HEADER_BYTES..],
        })
    }

    /// The file header.
    pub fn header(&self) -> &Pwv1Header {
        &self.header
    }

    /// Iterates over the frames. Each item is the packet data of one frame.
    ///
    /// A frame that is cut off yields one error and ends the iteration.
    pub fn frames(&self) -> Frames<'a> {
        Frames { rest: self.frames }
    }
}

/// Iterator over the frames of a [`Pwv1File`].
#[derive(Debug, Clone)]
pub struct Frames<'a> {
    rest: &'a [u8],
}

impl<'a> Iterator for Frames<'a> {
    type Item = Result<&'a [u8], Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.rest.is_empty() {
            return None;
        }

        let Some((size, body)) = self.rest.split_first_chunk::<4>() else {
            self.rest = &[];
            return Some(Err(Error::TruncatedFile));
        };

        let size = u32::from_le_bytes(*size) as usize;
        if size > body.len() {
            self.rest = &[];
            return Some(Err(Error::TruncatedFile));
        }

        let (frame, rest) = body.split_at(size);
        self.rest = rest;
        Some(Ok(frame))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::header::{ChromaSiting, ChromaSubsampling, YcbcrRange};

    fn header_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"PWV1");
        // 1920x1080, 4:2:0, limited range, centre siting.
        bytes.extend_from_slice(&[0x7F, 0xC7, 0x0D, 0x01]);
        bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x40]);
        bytes.extend_from_slice(&60u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&[10, 1, 0, 0]);
        bytes
    }

    #[test]
    fn parses_the_header() {
        let h = Pwv1Header::parse(&header_bytes()).unwrap();
        assert_eq!((h.stream.width, h.stream.height), (1920, 1080));
        assert_eq!(h.stream.chroma, ChromaSubsampling::Yuv420);
        assert_eq!(h.stream.ycbcr_range, YcbcrRange::Limited);
        assert_eq!(h.stream.chroma_siting, ChromaSiting::Center);
        assert_eq!((h.frame_rate_num, h.frame_rate_den), (60, 1));
        assert_eq!(h.reference_bit_depth, 10);
    }

    #[test]
    fn a_header_writes_back_to_its_bytes() {
        let bytes = header_bytes();
        let header = Pwv1Header::parse(&bytes).unwrap();
        assert_eq!(header.to_bytes().as_slice(), bytes.as_slice());
    }

    #[test]
    fn rejects_bad_headers() {
        let mut bytes = header_bytes();
        assert_eq!(Pwv1Header::parse(&bytes[..23]), Err(Error::TruncatedFile));
        assert_eq!(Pwv1Header::parse(&[]), Err(Error::TruncatedFile));

        bytes[VERSION_OFFSET] = 2;
        assert_eq!(
            Pwv1Header::parse(&bytes),
            Err(Error::UnsupportedHeaderVersion(2))
        );

        bytes[VERSION_OFFSET] = 1;
        bytes[0] = b'X';
        assert_eq!(Pwv1Header::parse(&bytes), Err(Error::BadMagic));
    }

    #[test]
    fn iterates_frames() {
        let mut bytes = header_bytes();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&[1, 2, 3]);
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&[4, 5]);

        let file = Pwv1File::parse(&bytes).unwrap();
        let frames: Vec<_> = file.frames().collect();
        assert_eq!(
            frames,
            [Ok(&[1u8, 2, 3][..]), Ok(&[][..]), Ok(&[4u8, 5][..])]
        );
    }

    #[test]
    fn a_cut_off_frame_ends_the_iteration_with_one_error() {
        let mut bytes = header_bytes();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.push(9);
        bytes.extend_from_slice(&5u32.to_le_bytes());
        bytes.extend_from_slice(&[1, 2]);

        let file = Pwv1File::parse(&bytes).unwrap();
        let mut frames = file.frames();
        assert_eq!(frames.next(), Some(Ok(&[9u8][..])));
        assert_eq!(frames.next(), Some(Err(Error::TruncatedFile)));
        assert_eq!(frames.next(), None);

        // Size prefix itself cut off.
        let mut bytes = header_bytes();
        bytes.extend_from_slice(&[1, 0]);
        let file = Pwv1File::parse(&bytes).unwrap();
        let mut frames = file.frames();
        assert_eq!(frames.next(), Some(Err(Error::TruncatedFile)));
        assert_eq!(frames.next(), None);
    }

    #[test]
    fn header_only_file_has_no_frames() {
        let bytes = header_bytes();
        let file = Pwv1File::parse(&bytes).unwrap();
        assert_eq!(file.frames().count(), 0);
    }
}
