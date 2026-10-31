//! Error type of the GPU decoder and encoder.

use thiserror::Error;

/// Everything that can go wrong while setting up the GPU or decoding or
/// encoding on it.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GpuError {
    /// No adapter matched the requested backends.
    #[error("no suitable GPU adapter found: {0}")]
    NoAdapter(#[from] wgpu::RequestAdapterError),

    /// The adapter refused to create a device.
    #[error("could not create a GPU device: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),

    /// A buffer the frame needs is larger than the device allows.
    #[error("{what} needs {needed} bytes, but the device allows at most {limit}")]
    LimitExceeded {
        /// Which buffer.
        what: &'static str,
        /// Bytes it needs.
        needed: u64,
        /// Bytes the device supports.
        limit: u64,
    },

    /// The frame is so large that 32-bit buffer offsets do not suffice.
    #[error("the frame is too large for 32-bit buffer offsets")]
    TooLarge,

    /// A value the call cannot take, such as a share outside 0 to 1.
    #[error("{0}")]
    InvalidArgument(String),

    /// The GPU or the driver reported an error.
    #[error("the GPU reported an error: {0}")]
    Device(String),

    /// The device was lost: a driver reset, a GPU that went away, or a
    /// destroyed device. Its context and decoders cannot be used any more.
    #[error("the GPU device was lost ({0}); create a new context")]
    DeviceLost(String),

    /// Waiting for the result or mapping it failed.
    #[error("reading the result back failed: {0}")]
    Readback(String),

    /// The packets did not add up to a complete frame.
    #[error("frame is incomplete: {received} of {total} blocks received")]
    IncompleteFrame {
        /// Distinct blocks that arrived.
        received: u32,
        /// Blocks the frame announces.
        total: u32,
    },

    /// The bitstream is malformed.
    #[error(transparent)]
    Bitstream(#[from] crate::bitstream::Error),

    /// The planes or the budget do not suit the encoder.
    #[error(transparent)]
    Encode(#[from] crate::encode::EncodeError),
    /// A texture of a [`TexturePicture`](super::TexturePicture) cannot hold
    /// its plane: its format, size, dimension, samples or usage.
    #[error("unsuitable texture: {0}")]
    Texture(String),
}
