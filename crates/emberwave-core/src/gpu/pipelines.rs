//! The compute pipelines, built once per device and shared by all decoders
//! and encoders.
//!
//! Compiling the shaders is the slowest part of creating a decoder (on DX12
//! most of all), so a [`GpuContext`](super::GpuContext) builds them on first
//! use and hands out the same ones afterwards.

use super::decoder::PLANE_FORMAT;
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
    /// The inverse transform that stores the planes into textures, on a
    /// device that can (decision D52).
    pub idwt_planes: Option<Stage>,
    pub yuv2rgb: Stage,
}

impl Pipelines {
    /// The decoder's pipelines, with `idwt_planes` if `plane_stores`: the
    /// device must have the adapter's own format features then, and take
    /// [`PLANE_FORMAT`] as a storage texture. A device that fails to build
    /// it decodes with the copies.
    pub(crate) fn new(device: &wgpu::Device, plane_stores: bool) -> Result<Self, GpuError> {
        let mut pipelines = with_error_scope(device, || Self {
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
            idwt_planes: None,
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
        })?;
        if plane_stores {
            pipelines.idwt_planes = with_error_scope(device, || {
                create_stage(
                    device,
                    "idwt planes",
                    shader_sources::IDWT_PLANES,
                    &[
                        storage_entry(0, false),
                        uniform_entry(2),
                        plane_entry(3),
                        plane_entry(4),
                        plane_entry(5),
                    ],
                )
            })
            .ok();
        }
        Ok(pipelines)
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
    pub gather: Stage,
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
                gather: create_stage(
                    device,
                    "gather",
                    shader_sources::GATHER,
                    &[
                        storage_entry(0, false),
                        uniform_entry(1),
                        texture_entry(2),
                        texture_entry(3),
                        texture_entry(4),
                        sampler_entry(5),
                    ],
                ),
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

/// A plane's texture that `idwt_planes.wgsl` stores into.
fn plane_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: PLANE_FORMAT,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

/// A 2D texture of floats, filterable or not, which `gather.wgsl` reads with
/// `textureGather` through a sampler that does not filter.
fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// A sampler that does not filter, for `textureGather`, which takes texels as
/// they are.
fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
        count: None,
    }
}

/// The render pipeline of `draw.wgsl` for targets of one format, with its
/// bind group layout: the RGBA words and the picture's size.
#[derive(Debug)]
pub(crate) struct DrawPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub layout: wgpu::BindGroupLayout,
    pub format: wgpu::TextureFormat,
}

impl DrawPipeline {
    pub(crate) fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> Result<Self, GpuError> {
        let fragment = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        with_error_scope(device, || {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("draw"),
                source: wgpu::ShaderSource::Wgsl(shader_sources::DRAW.into()),
            });
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("draw"),
                entries: &[
                    fragment(0, wgpu::BufferBindingType::Storage { read_only: true }),
                    fragment(1, wgpu::BufferBindingType::Uniform),
                ],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("draw"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("draw"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                multiview_mask: None,
                cache: None,
            });
            Self {
                pipeline,
                layout,
                format,
            }
        })
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
        // Named after the shader, and the entry point if there are several,
        // for validation messages and GPU captures.
        let name = if N == 1 {
            label.to_owned()
        } else {
            format!("{label} {entry_point}")
        };
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&name),
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
