//! The wgpu decode pipeline.
//!
//! [`GpuContext`] creates the device and [`Decoder`] decodes frames on it. The
//! shaders are plain WGSL that uses only what wgpu guarantees on Vulkan, Metal,
//! DX12 and WebGPU: no subgroup operations, no 8 or 16 bit storage and no
//! optional features.

mod context;
mod decoder;
mod error;
mod layout;
mod pipelines;

pub use context::{BACKEND_ENV, GpuContext, GpuOptions, parse_backends};
pub use decoder::{Decoder, Frame, FramePlane, RgbFrame, StageTimings, TimedOutput, Timings};
pub use error::GpuError;

/// The WGSL sources of the pipeline stages, for offline validation.
pub mod shader_sources {
    /// Coefficient decoding and dequantisation.
    pub const DEQUANT: &str = include_str!("shaders/dequant.wgsl");
    /// Inverse wavelet transform of one level.
    pub const IDWT: &str = include_str!("shaders/idwt.wgsl");
    /// Float planes to 8-bit samples.
    pub const PACK: &str = include_str!("shaders/pack.wgsl");
    /// 8-bit YCbCr planes to RGB.
    pub const YUV2RGB: &str = include_str!("shaders/yuv2rgb.wgsl");
}
