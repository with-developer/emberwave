//! The compute pipelines, built once per device and shared by all decoders.
//!
//! Compiling the shaders is the slowest part of creating a decoder (on DX12
//! most of all), so a [`GpuContext`](super::GpuContext) builds them on first
//! use and hands out the same ones afterwards.

use super::error::GpuError;

const DEQUANT_SHADER: &str = include_str!("shaders/dequant.wgsl");
const IDWT_SHADER: &str = include_str!("shaders/idwt.wgsl");
const YUV2RGB_SHADER: &str = include_str!("shaders/yuv2rgb.wgsl");

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
                DEQUANT_SHADER,
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
                IDWT_SHADER,
                &[
                    storage_entry(0, false),
                    storage_entry(1, false),
                    uniform_entry(2),
                ],
            ),
            yuv2rgb: create_stage(
                device,
                "yuv2rgb",
                YUV2RGB_SHADER,
                &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    uniform_entry(2),
                ],
            ),
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

fn create_stage(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> Stage {
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
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        // The shaders write every workgroup variable before they read it, so
        // the zeroing WebGPU requires is wasted work. naga does it with one
        // invocation per workgroup, which made the inverse transform take 0.64
        // instead of 0.38 ms at 4K (ARCHITECTURE.md, decision D25). Browsers
        // zero the memory regardless.
        compilation_options: wgpu::PipelineCompilationOptions {
            zero_initialize_workgroup_memory: false,
            ..Default::default()
        },
        cache: None,
    });
    Stage { pipeline, layout }
}

/// Runs `body` and turns validation and out-of-memory errors raised inside it
/// into an error value instead of letting the default handler panic.
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
