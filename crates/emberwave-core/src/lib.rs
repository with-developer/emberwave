//! Core library of emberwave, a portable (wgpu) port of the PyroWave
//! low-latency intra-frame wavelet video codec.
//!
//! - [`bitstream`]: the PyroWave v1 bitstream (pure Rust, GPU independent).
//! - [`cpu`]: a slow reference decoder used as a test oracle.
//! - [`color`]: YCbCr to RGB conversion, shared by the CPU and GPU paths.
//! - [`gpu`] (feature `gpu`, on by default): the wgpu/WGSL decoder.
//!
//! See `ARCHITECTURE.md` for the design.

pub mod bitstream;
pub mod color;
pub mod cpu;
pub mod encode;
#[cfg(feature = "gpu")]
pub mod gpu;

/// Version of the `emberwave-core` crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::VERSION;

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
    }
}
