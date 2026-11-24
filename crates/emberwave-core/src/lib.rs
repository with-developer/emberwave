//! Core library of emberwave, a portable (wgpu) port of the PyroWave
//! low-latency intra-frame wavelet video codec: everything that needs no GPU.
//!
//! - [`bitstream`]: the PyroWave v1 bitstream (pure Rust, GPU independent).
//! - [`cpu`]: a slow reference decoder used as a test oracle.
//! - [`color`]: YCbCr to RGB conversion, shared by the CPU and GPU paths.
//! - [`encode`]: an encoder on the CPU, the reference of the GPU's.
//!
//! The wgpu/WGSL decoder and encoder are the crate `emberwave-wgpu`, whose
//! version follows wgpu's: this crate has no wgpu type in its API (decision
//! D75). See `ARCHITECTURE.md` for the design.
//!
//! Items hidden from this documentation are there for emberwave's own crates,
//! tests and tools, `emberwave-wgpu` among them: they are not part of the API,
//! and any release may change them.

pub mod bitstream;
pub mod color;
pub mod cpu;
pub mod encode;

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
