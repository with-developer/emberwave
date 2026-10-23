//! The wgpu pipelines.
//!
//! [`GpuContext`] creates the device, [`Decoder`] decodes frames on it and
//! [`Encoder`] encodes pictures. The shaders are plain WGSL that uses only what
//! wgpu guarantees on Vulkan, Metal, DX12 and WebGPU: no subgroup operations,
//! no 8 or 16 bit storage and no optional features.

mod context;
mod decoder;
mod encoder;
mod error;
mod layout;
mod pipelines;
mod resources;

pub use context::{BACKEND_ENV, GpuContext, GpuOptions, parse_backends};
pub use decoder::{Decoder, Frame, FramePlane, RgbFrame, StageTimings, TimedOutput, Timings};
pub use encoder::Encoder;
pub use error::GpuError;

/// The WGSL sources of the pipeline stages, for offline validation.
pub mod shader_sources {
    /// Coefficient decoding and dequantisation.
    pub const DEQUANT: &str = include_str!("shaders/dequant.wgsl");
    /// Inverse wavelet transform of one level.
    pub const IDWT: &str = include_str!("shaders/idwt.wgsl");
    /// 8-bit YCbCr planes to RGB.
    pub const YUV2RGB: &str = include_str!("shaders/yuv2rgb.wgsl");
    /// Forward wavelet transform of one level, for the encoder.
    pub const FDWT: &str = include_str!("shaders/fdwt.wgsl");
    /// Quantisation and the cost tables of the rate control, behind the
    /// quantisation it shares with the packing.
    pub const EQUANT: &str = concat!(
        include_str!("shaders/quantise.wgsl"),
        include_str!("shaders/equant.wgsl")
    );
    /// The rate control of the encoder.
    pub const RATE: &str = include_str!("shaders/rate.wgsl");
    /// Prefix sums for the rate control and the packing.
    pub const SCAN: &str = include_str!("shaders/scan.wgsl");
    /// The packing of the encoder's blocks, behind the quantisation.
    pub const PACK: &str = concat!(
        include_str!("shaders/quantise.wgsl"),
        include_str!("shaders/pack.wgsl")
    );

    /// Every shader, by name, with its compute entry points.
    pub const ALL: [(&str, &str, &[&str]); 8] = [
        ("dequant", DEQUANT, &["main"]),
        ("idwt", IDWT, &["main"]),
        ("yuv2rgb", YUV2RGB, &["main"]),
        ("fdwt", FDWT, &["main"]),
        ("equant", EQUANT, &["main"]),
        ("rate", RATE, &["analyse", "cutoff", "ties", "choose"]),
        ("scan", SCAN, &["scan_chunks", "scan_totals", "add_totals"]),
        ("pack", PACK, &["main"]),
    ];
}
