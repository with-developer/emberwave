//! The GPU decoder and encoder of emberwave, a portable port of the PyroWave
//! low-latency intra-frame wavelet video codec, in wgpu and WGSL, on top of
//! `emberwave-core`'s bitstream, CPU reference and colour code.
//!
//! [`wgpu`] is the version this crate is built with, whose types the GPU
//! calls take: use it to share a device or a texture. A new major version of
//! wgpu changes them, and this crate's major version with it; the core has no
//! wgpu type in its API (decision D75).
//!
//! [`GpuContext`] creates the device, [`Decoder`] decodes frames on it and
//! [`Encoder`] encodes pictures. The shaders are plain WGSL that uses only what
//! wgpu guarantees on Vulkan, Metal, DX12 and WebGPU: no subgroup operations,
//! no 8 or 16 bit storage and no optional features. One variant of the
//! inverse transform, which stores the planes into 8-bit textures, runs only
//! where the adapter allows it, and the portable one everywhere else
//! (decision D52).
//!
//! In a browser nothing may wait for the GPU, so the calls that do, such as
//! [`Decoder::decode`] and the [`Encoder`]'s encodes, are not built for
//! wasm32; the decoder's `record_*` and `submit_*` calls are.

// The calls left out of the browser build leave their helpers unused there,
// and the docs link to them (decision D70); the native builds check for code
// that is dead everywhere and for broken links.
#![cfg_attr(
    target_arch = "wasm32",
    allow(dead_code, unused_imports, rustdoc::broken_intra_doc_links)
)]

pub use wgpu;

mod context;
mod decoder;
mod encoder;
mod error;
mod layout;
mod pipelines;
mod resources;

pub use context::{
    BACKEND_ENV, CallerQueueLock, GpuContext, GpuOptions, QueueTurn, QueueTurns, SubmissionHooks,
    parse_backends,
};
pub use decoder::{
    Decoder, Frame, FramePlane, PLANE_FORMAT, RgbFrame, StageTimings, TEXTURE_FORMATS, TimedOutput,
    Timings,
};
pub use encoder::{EncodeStageTimings, EncodeTimings, Encoder, TexturePicture};
pub use error::GpuError;

// What the docs and the C API promise: a context is shared by threads, and
// decoders and encoders move between them. In a browser wgpu's objects stay
// on their thread.
#[cfg(not(target_arch = "wasm32"))]
const _: () = {
    const fn shared<T: Send + Sync>() {}
    const fn movable<T: Send>() {}
    shared::<GpuContext>();
    movable::<Decoder>();
    movable::<Encoder>();
};

/// The WGSL sources of the pipeline stages, for offline validation.
#[doc(hidden)]
pub mod shader_sources {
    /// Coefficient decoding and dequantisation.
    pub const DEQUANT: &str = include_str!("shaders/dequant.wgsl");
    /// The blocks of detail a frame drawn in part lacks, after the
    /// dequantisation (decision D66).
    pub const CONCEAL: &str = include_str!("shaders/conceal.wgsl");
    /// Inverse wavelet transform of one level, whose last level packs the
    /// planes into a buffer.
    pub const IDWT: &str = concat!(
        include_str!("shaders/idwt.wgsl"),
        include_str!("shaders/idwt_packed.wgsl")
    );
    /// The inverse transform whose last level stores the planes straight into
    /// three 8-bit textures instead, on adapters that can write them
    /// (decision D52). Not among the portable shaders of [`ALL`]: WebGPU does
    /// not guarantee `r8unorm` storage textures.
    pub const IDWT_PLANES: &str = concat!(
        include_str!("shaders/idwt.wgsl"),
        include_str!("shaders/idwt_planes.wgsl")
    );
    /// 8-bit YCbCr planes to RGB.
    pub const YUV2RGB: &str = include_str!("shaders/yuv2rgb.wgsl");
    /// The RGB picture drawn into a texture: a vertex and a fragment shader,
    /// `vs` and `fs`, so not among the compute shaders of [`ALL`].
    pub const DRAW: &str = include_str!("shaders/draw.wgsl");
    /// A picture in textures packed into the encoder's planes buffer.
    pub const GATHER: &str = include_str!("shaders/gather.wgsl");
    /// A picture in an RGB texture converted to YCbCr into the encoder's
    /// planes buffer (decision D65).
    pub const GATHER_RGB: &str = include_str!("shaders/gather_rgb.wgsl");
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
    pub const ALL: [(&str, &str, &[&str]); 11] = [
        ("dequant", DEQUANT, &["main"]),
        ("conceal", CONCEAL, &["main"]),
        ("idwt", IDWT, &["main"]),
        ("yuv2rgb", YUV2RGB, &["main"]),
        ("gather", GATHER, &["main"]),
        ("gather_rgb", GATHER_RGB, &["yuv444", "yuv420"]),
        ("fdwt", FDWT, &["main"]),
        ("equant", EQUANT, &["main"]),
        ("rate", RATE, &["analyse", "cutoff", "ties", "choose"]),
        ("scan", SCAN, &["scan_chunks", "scan_totals", "add_totals"]),
        ("pack", PACK, &["main"]),
    ];
}
