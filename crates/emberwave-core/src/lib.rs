//! Core library of emberwave, a portable (wgpu) port of the PyroWave
//! low-latency intra-frame wavelet video codec.
//!
//! - [`bitstream`]: the PyroWave v1 bitstream (pure Rust, GPU independent).
//!
//! The wgpu/WGSL decode pipeline arrives in milestone M2. See
//! `ARCHITECTURE.md` for the plan.

pub mod bitstream;

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
