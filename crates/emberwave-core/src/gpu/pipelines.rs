//! The compute pipelines, built once per device and shared by all decoders
//! and encoders.
//!
//! Compiling the shaders is the slowest part of creating a decoder (on DX12
//! most of all), so a [`GpuContext`](super::GpuContext) builds them on first
//! use and hands out the same ones afterwards.

use super::error::GpuError;
use super::shader_sources;

/// A compute pipeline with the bind group layout it was created with.
#[derive(Debug)]
pub(crate) struct Stage {
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
}

/// The stages of the decoder.
#[derive(Debug)]
pub(crate) struct Pipelines {
    pub dequant: Stage,
    pub idwt: Stage,
    pub yuv2rgb: Stage,
}

impl Pipelines {
    pub(crate) fn new(device: &wgpu::Device) -> Result<Self, GpuError> {
        with_error_scope(device, || Self {
            dequant: create_stage(
                device,
                "dequant",
                shader_sources::DEQUANT,
                &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                    uniform_entry(3),
                ],
            ),
            idwt: create_stage(
                device,
                "idwt",
                shader_sources::IDWT,
                &[
                    storage_entry(0, false),
                    storage_entry(1, false),
                    uniform_entry(2),
                ],
            ),
            yuv2rgb: create_stage(
                device,
                "yuv2rgb",
                shader_sources::YUV2RGB,
                &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    uniform_entry(2),
                ],
            ),
        })
    }
}

/// The entry points of `rate.wgsl`, which share one bind group layout.
#[derive(Debug)]
pub(crate) struct RateStages {
    pub analyse: Stage,
    pub cutoff: Stage,
    pub ties: Stage,
    pub choose: Stage,
}

/// The entry points of `scan.wgsl`, which share one bind group layout.
#[derive(Debug)]
pub(crate) struct ScanStages {
    pub chunks: Stage,
    pub totals: Stage,
    pub add: Stage,
}

/// The stages of the encoder.
#[derive(Debug)]
pub(crate) struct EncoderPipelines {
    pub fdwt: Stage,
    pub equant: Stage,
    pub rate: RateStages,
    pub scan: ScanStages,
    pub pack: Stage,
}

impl EncoderPipelines {
    pub(crate) fn new(device: &wgpu::Device) -> Result<Self, GpuError> {
        with_error_scope(device, || {
            let [analyse, cutoff, ties, choose] = create_stages(
                device,
                "rate",
                shader_sources::RATE,
                ["analyse", "cutoff", "ties", "choose"],
                &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    storage_entry(2, false),
                    storage_entry(3, false),
                    storage_entry(4, false),
                    storage_entry(5, false),
                    uniform_entry(6),
                ],
            );
            let [chunks, totals, add] = create_stages(
                device,
                "scan",
                shader_sources::SCAN,
                ["scan_chunks", "scan_totals", "add_totals"],
                &[
                    storage_entry(0, false),
                    storage_entry(1, false),
                    uniform_entry(2),
                ],
            );
            Self {
                fdwt: create_stage(
                    device,
                    "fdwt",
                    shader_sources::FDWT,
                    &[
                        storage_entry(0, false),
                        storage_entry(1, true),
                        uniform_entry(2),
                        storage_entry(3, true),
                    ],
                ),
                equant: create_stage(
                    device,
                    "equant",
                    shader_sources::EQUANT,
                    &[
                        storage_entry(0, true),
                        storage_entry(1, false),
                        uniform_entry(2),
                    ],
                ),
                rate: RateStages {
                    analyse,
                    cutoff,
                    ties,
                    choose,
                },
                scan: ScanStages {
                    chunks,
                    totals,
                    add,
                },
                pack: create_stage(
                    device,
                    "pack",
                    shader_sources::PACK,
                    &[
                        storage_entry(0, true),
                        storage_entry(1, true),
                        storage_entry(2, true),
                        storage_entry(3, true),
                        storage_entry(4, false),
                        uniform_entry(5),
                        uniform_entry(6),
                    ],
                ),
            }
        })
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// The pipeline of a shader whose entry point is `main`.
fn create_stage(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> Stage {
    let [stage] = create_stages(device, label, source, ["main"], entries);
    stage
}

/// The pipelines of several entry points of one shader, which share the bind
/// group layout made of `entries`: each uses some of its bindings.
fn create_stages<const N: usize>(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    entry_points: [&str; N],
    entries: &[wgpu::BindGroupLayoutEntry],
) -> [Stage; N] {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    entry_points.map(|entry_point| {
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry_point),
            // The shaders write every workgroup variable before they read it,
            // so the zeroing WebGPU requires is wasted work. naga does it with
            // one invocation per workgroup, which made the inverse transform
            // take 0.64 instead of 0.38 ms at 4K (ARCHITECTURE.md, decision
            // D25). Browsers zero the memory regardless.
            compilation_options: wgpu::PipelineCompilationOptions {
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
            cache: None,
        });
        Stage {
            pipeline,
            layout: layout.clone(),
        }
    })
}

/// Runs `body` and turns validation and out-of-memory errors raised inside it
/// into an error value instead of letting the default handler panic.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn with_error_scope<T>(
    device: &wgpu::Device,
    body: impl FnOnce() -> T,
) -> Result<T, GpuError> {
    let memory_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    // Internal errors are what a driver's failure to compile a shader or to
    // build a pipeline becomes.
    let internal_scope = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let value = body();
    // Scopes are popped innermost first.
    let validation = pollster::block_on(validation_scope.pop());
    let internal = pollster::block_on(internal_scope.pop());
    let memory = pollster::block_on(memory_scope.pop());

    match validation.or(internal).or(memory) {
        Some(error) => Err(GpuError::Device(error.to_string())),
        None => Ok(value),
    }
}

/// In a browser nothing may block, so the scopes cannot be waited for. The
/// work runs as it is; WebGPU reports its errors to the browser's console.
#[cfg(target_arch = "wasm32")]
pub(crate) fn with_error_scope<T>(
    _device: &wgpu::Device,
    body: impl FnOnce() -> T,
) -> Result<T, GpuError> {
    Ok(body())
}
