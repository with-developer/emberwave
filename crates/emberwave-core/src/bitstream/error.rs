//! Error type of the bitstream parser.

use thiserror::Error;

use super::header::ChromaSubsampling;

/// Everything that can go wrong while reading a PyroWave bitstream.
///
/// Parsing untrusted input never panics; it returns one of these instead.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum Error {
    /// Width or height is outside `1..=16384`, or a 4:2:0 picture has an odd
    /// dimension.
    #[error("invalid dimensions {width}x{height} for {chroma:?}")]
    InvalidDimensions {
        /// Luma width.
        width: u32,
        /// Luma height.
        height: u32,
        /// Chroma subsampling the dimensions were checked against.
        chroma: ChromaSubsampling,
    },

    /// Fewer than 8 bytes are left where a header is expected.
    #[error("truncated header: {available} bytes available, 8 needed")]
    TruncatedHeader {
        /// Bytes that were left.
        available: usize,
    },

    /// Bytes remain after the last complete block or header of a packet.
    #[error("{0} trailing bytes after the last block")]
    TrailingBytes(usize),

    /// A block claims more bytes than the packet has left.
    #[error("block claims {claimed} bytes but only {available} are left in the packet")]
    TruncatedBlock {
        /// Bytes the block header asks for.
        claimed: usize,
        /// Bytes that are left.
        available: usize,
    },

    /// A block is smaller than its own header.
    #[error("block of {words} words is smaller than its 2 word header")]
    BlockTooSmall {
        /// `payload_words` from the header.
        words: usize,
    },

    /// The parts of a block (control words, scales, bit-planes, signs) do not
    /// fit in the size the header declares.
    #[error("block contents need {needed} bytes but the block has {available}")]
    BlockContentOverrun {
        /// Bytes the contents need at least.
        needed: usize,
        /// Bytes the block has.
        available: usize,
    },

    /// An extended header with a `code` that this decoder does not know.
    #[error("unknown extended header code {0}")]
    UnknownExtendedCode(u8),

    /// A start of frame header announces a different chroma format than the
    /// decoder was set up for.
    #[error("chroma format mismatch: stream has {stream:?}, decoder expects {expected:?}")]
    ChromaMismatch {
        /// Format found in the stream.
        stream: ChromaSubsampling,
        /// Format the layout was built for.
        expected: ChromaSubsampling,
    },

    /// A start of frame header announces other dimensions than the decoder
    /// was set up for.
    #[error("dimension mismatch: stream has {}x{}, decoder expects {}x{}", .stream.0, .stream.1, .expected.0, .expected.1)]
    DimensionMismatch {
        /// `(width, height)` found in the stream.
        stream: (u32, u32),
        /// `(width, height)` the layout was built for.
        expected: (u32, u32),
    },

    /// A block index lies outside the frame.
    #[error("block index {index} is out of range (frame has {count} blocks)")]
    BlockIndexOutOfRange {
        /// Index from the header.
        index: u32,
        /// Blocks in the frame.
        count: u32,
    },

    /// The blocks of a frame add up to more than 2^32 words.
    #[error("frame payload is too large")]
    FrameTooLarge,

    /// A PWV1 file does not start with the `PWV1` magic.
    #[error("not a PWV1 file (bad magic)")]
    BadMagic,

    /// A PWV1 header with a version other than 1.
    #[error("unsupported PWV1 header version {0}")]
    UnsupportedHeaderVersion(u8),

    /// A PWV1 file ends in the middle of its header or of a frame.
    #[error("PWV1 data is truncated")]
    TruncatedFile,
}
