//! Core library of emberwave, a portable (wgpu) port of the PyroWave
//! low-latency intra-frame wavelet video codec.
//!
//! - [`bitstream`]: the PyroWave v1 bitstream (pure Rust, GPU independent).
//! - [`cpu`]: a slow reference decoder used as a test oracle.
//! - [`color`]: YCbCr to RGB conversion, shared by the CPU and GPU paths.
//! - [`encode`]: an encoder on the CPU, the reference of the GPU's.
//! - [`gpu`] (feature `gpu`, on by default): the wgpu/WGSL decoder and
//!   encoder.
//!
//! See `ARCHITECTURE.md` for the design.
//!
//! Items hidden from this documentation are there for emberwave's own tests
//! and tools: they are not part of the API, and any release may change them.
//! With the `gpu` feature, [`wgpu`] is the version emberwave is built with,
//! whose types the GPU calls take: use it to share a device or a texture.

// The docs link to the calls that wait for the GPU, which the browser build
// leaves out (decision D70).
#![cfg_attr(target_arch = "wasm32", allow(rustdoc::broken_intra_doc_links))]

#[cfg(feature = "gpu")]
pub use wgpu;

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
