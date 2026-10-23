//! The PyroWave v1 bitstream: headers, block layout, packet assembly and the
//! PWV1 file format.
//!
//! Everything in this module is plain Rust with no GPU dependency, and none
//! of it panics on malformed input. Decoding the coefficients inside a block
//! is a separate stage: the GPU does it in the decoder, and [`crate::cpu`] has
//! a slow reference version of it.

pub mod block;
pub mod consts;
pub mod error;
pub mod header;
pub mod layout;
pub mod packet;
pub mod pwv1;

pub use error::Error;
pub use header::{
    BlockHeader, ChromaSiting, ChromaSubsampling, ColorPrimaries, Header, SequenceHeader,
    TransferFunction, YcbcrRange, YcbcrTransform,
};
pub use layout::{Band, BandLayout, BlockLocation, Component, FrameLayout};
pub use packet::{FrameAssembler, FrameView, MISSING_BLOCK};
pub use pwv1::{Pwv1File, Pwv1Header};
