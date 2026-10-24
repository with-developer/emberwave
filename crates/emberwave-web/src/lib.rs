//! emberwave in the browser: decodes PyroWave (`.pwv1`) files with WebGPU and
//! draws them on a canvas, and times the decoding on the browser's GPU
//! ([`Bench`]). The pages are in `web/`, and `scripts/build_web.sh` builds
//! this crate for them.
//!
//! Only the browser build has content. On other targets the crate is empty,
//! so that the workspace builds everywhere.

#![cfg(target_arch = "wasm32")]

use std::sync::{Arc, Mutex};

use emberwave_core::bitstream::Pwv1File;
use emberwave_core::color::ColorParams;
use emberwave_core::gpu::{Decoder, GpuContext, GpuOptions};
use wasm_bindgen::prelude::*;

mod bench;
pub use bench::Bench;

const BLIT_SHADER: &str = include_str!("blit.wgsl");

/// Frame rate for a stream that does not say: PWV1 files without one hold
/// unrelated pictures, shown at this pace.
const DEFAULT_FRAME_RATE: f64 = 30.0;

fn js_error(error: impl std::fmt::Display) -> JsError {
    JsError::new(&error.to_string())
}

/// Milliseconds on the page's clock (`performance.now()`), or 0 without one.
fn now_ms() -> f64 {
    web_sys::window()
        .and_then(|window| window.performance())
        .map_or(0.0, |performance| performance.now())
}

/// A canvas format without sRGB encoding: the samples are gamma encoded
/// already, as they are in the stream.
fn surface_format(surface: &wgpu::Surface<'_>, adapter: &wgpu::Adapter) -> wgpu::TextureFormat {
    let formats = surface.get_capabilities(adapter).formats;
    formats
        .iter()
        .copied()
        .find(|format| !format.is_srgb())
        .or_else(|| formats.first().copied())
        .unwrap_or(wgpu::TextureFormat::Bgra8Unorm)
}

/// Times from submitting a frame until the browser reported it done, summed
/// up by the callbacks of the queue until the page asks for their average.
#[derive(Debug, Default)]
struct GpuTimes {
    total_ms: f64,
    frames: u32,
}

/// The stream being played.
struct Stream {
    frames: Vec<Vec<u8>>,
    decoder: Decoder,
    color: ColorParams,
    width: u32,
    height: u32,
    frame_rate: f64,
    /// Binds the decoder's RGBA words for the blit, once they exist.
    bind_group: Option<wgpu::BindGroup>,
    next: usize,
}

/// Plays PyroWave files on a canvas.
#[wasm_bindgen]
pub struct Player {
    context: GpuContext,
    adapter: wgpu::Adapter,
    surface: wgpu::Surface<'static>,
    blit: wgpu::RenderPipeline,
    /// Width and height of the picture, for the blit shader.
    picture: wgpu::Buffer,
    stream: Option<Stream>,
    gpu_times: Arc<Mutex<GpuTimes>>,
    /// The first GPU error the browser reported since the last frame. Error
    /// scopes cannot be waited for in a browser, so errors arrive here.
    gpu_error: Arc<Mutex<Option<String>>>,
}

#[wasm_bindgen]
impl Player {
    /// Sets up WebGPU for `canvas`. Fails if the browser has no WebGPU.
    pub async fn create(canvas: web_sys::HtmlCanvasElement) -> Result<Player, JsError> {
        // A panic aborts the module; at least its message reaches the console.
        std::panic::set_hook(Box::new(|info| {
            web_sys::console::error_1(&JsValue::from_str(&info.to_string()));
        }));

        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::BROWSER_WEBGPU;
        let instance = wgpu::Instance::new(descriptor);
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(js_error)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(js_error)?;
        let context = GpuContext::from_adapter(adapter.clone(), &GpuOptions::default())
            .await
            .map_err(js_error)?;

        let device = context.device();
        let gpu_error: Arc<Mutex<Option<String>>> = Arc::default();
        let record = Arc::clone(&gpu_error);
        device.on_uncaptured_error(Arc::new(move |error| {
            if let Ok(mut first) = record.lock() {
                first.get_or_insert_with(|| error.to_string());
            }
        }));
        let format = surface_format(&surface, &adapter);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });
        let blit = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: None,
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
        let picture = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("picture size"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            context,
            adapter,
            surface,
            blit,
            picture,
            stream: None,
            gpu_times: Arc::default(),
            gpu_error,
        })
    }

    /// The GPU in use, for the page to show. Browsers often keep its name to
    /// themselves.
    pub fn adapter(&self) -> String {
        let info = self.context.adapter_info();
        if info.name.is_empty() {
            "WebGPU".to_owned()
        } else {
            format!("{} (WebGPU)", info.name)
        }
    }

    /// Loads a PWV1 file and sizes the canvas to its pictures. Returns a line
    /// that describes the stream.
    pub fn load(&mut self, bytes: &[u8]) -> Result<String, JsError> {
        // The old stream goes first, whether or not the new one loads: its
        // frames and GPU memory are not needed while the new ones are made.
        self.stream = None;
        if let Ok(mut times) = self.gpu_times.lock() {
            *times = GpuTimes::default();
        }

        let file = Pwv1File::parse(bytes).map_err(js_error)?;
        let header = *file.header();
        let stream = header.stream;
        let largest = self.context.device().limits().max_texture_dimension_2d;
        if stream.width > largest || stream.height > largest {
            return Err(JsError::new(&format!(
                "the pictures are {}x{}, and this browser draws at most {largest} pixels across",
                stream.width, stream.height
            )));
        }
        let frames = file
            .frames()
            .map(|frame| frame.map(<[u8]>::to_vec))
            .collect::<Result<Vec<_>, _>>()
            .map_err(js_error)?;
        if frames.is_empty() {
            return Err(JsError::new("the file has no frames"));
        }

        let decoder = Decoder::new(&self.context, stream.width, stream.height, stream.chroma)
            .map_err(js_error)?;
        let config = self
            .surface
            .get_default_config(&self.adapter, stream.width, stream.height)
            .ok_or_else(|| JsError::new("this canvas cannot show the picture"))?;
        let config = wgpu::SurfaceConfiguration {
            format: surface_format(&self.surface, &self.adapter),
            ..config
        };
        self.surface.configure(self.context.device(), &config);

        let mut size = [0u8; 16];
        size[..4].copy_from_slice(&stream.width.to_le_bytes());
        size[4..8].copy_from_slice(&stream.height.to_le_bytes());
        self.context.queue().write_buffer(&self.picture, 0, &size);

        let frame_rate = if header.frame_rate_num != 0 && header.frame_rate_den != 0 {
            f64::from(header.frame_rate_num) / f64::from(header.frame_rate_den)
        } else {
            DEFAULT_FRAME_RATE
        };
        let description = format!(
            "{}x{} {:?}, {} frames at {frame_rate:.2} fps",
            stream.width,
            stream.height,
            stream.chroma,
            frames.len()
        );
        self.stream = Some(Stream {
            frames,
            decoder,
            color: ColorParams::from_stream(&stream),
            width: stream.width,
            height: stream.height,
            frame_rate,
            bind_group: None,
            next: 0,
        });
        Ok(description)
    }

    /// Width of the loaded pictures, 0 before a file is loaded.
    pub fn width(&self) -> u32 {
        self.stream.as_ref().map_or(0, |stream| stream.width)
    }

    /// Height of the loaded pictures, 0 before a file is loaded.
    pub fn height(&self) -> u32 {
        self.stream.as_ref().map_or(0, |stream| stream.height)
    }

    /// Frames per second the stream asks for.
    pub fn frame_rate(&self) -> f64 {
        self.stream
            .as_ref()
            .map_or(DEFAULT_FRAME_RATE, |stream| stream.frame_rate)
    }

    /// Decodes the next frame, the file's frames in turn, and draws it.
    /// Returns the milliseconds the CPU spent recording and submitting it.
    pub fn draw_next(&mut self) -> Result<f64, JsError> {
        if let Some(error) = self
            .gpu_error
            .lock()
            .ok()
            .and_then(|mut first| first.take())
        {
            return Err(JsError::new(&format!("the GPU reported an error: {error}")));
        }
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| JsError::new("no file is loaded"))?;
        let start = now_ms();
        let device = self.context.device();
        let queue = self.context.queue();

        let target = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            other => {
                return Err(JsError::new(&format!(
                    "the canvas has no texture to draw on: {other:?}"
                )));
            }
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame"),
        });
        stream
            .decoder
            .encode_rgb(&stream.frames[stream.next], &stream.color, &mut encoder)
            .map_err(js_error)?;
        stream.next = (stream.next + 1) % stream.frames.len();

        if stream.bind_group.is_none() {
            let rgba = stream
                .decoder
                .rgba_buffer()
                .ok_or_else(|| JsError::new("the decoder made no RGB picture"))?;
            stream.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blit"),
                layout: &self.blit.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: rgba.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.picture.as_entire_binding(),
                    },
                ],
            }));
        }

        let view = target.texture.create_view(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.blit);
            pass.set_bind_group(0, stream.bind_group.as_ref(), &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);

        let submitted = now_ms();
        let gpu_times = Arc::clone(&self.gpu_times);
        queue.on_submitted_work_done(move || {
            if let Ok(mut times) = gpu_times.lock() {
                times.total_ms += now_ms() - submitted;
                times.frames += 1;
            }
        });
        queue.present(target);
        Ok(submitted - start)
    }

    /// The average milliseconds from submitting a frame until the browser
    /// reported the GPU done with it (decoding and drawing, plus the delay of
    /// the report), over the frames reported since the last call. 0 if none
    /// was.
    pub fn take_gpu_ms(&self) -> f64 {
        match self.gpu_times.lock() {
            Ok(mut times) => {
                let average = if times.frames == 0 {
                    0.0
                } else {
                    times.total_ms / f64::from(times.frames)
                };
                *times = GpuTimes::default();
                average
            }
            Err(_) => 0.0,
        }
    }
}
