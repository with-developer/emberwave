//! Core library of emberwave, a portable (wgpu) port of the PyroWave
//! low-latency intra-frame wavelet video codec.
//!
//! The crate will hold the bitstream parser (pure Rust, GPU independent) and
//! the wgpu/WGSL decode pipeline. See `ARCHITECTURE.md` for the plan.

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
