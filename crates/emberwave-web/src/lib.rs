//! emberwave in the browser: decodes PyroWave (`.pwv1`) files with WebGPU and
//! draws them on a canvas, makes a demo stream for the player ([`Demo`]), and
//! times the decoding on the browser's GPU ([`Bench`]). The pages are in
//! `web/`, and `scripts/build_web.sh` builds this crate for them.
//!
//! Only the browser build has content. On other targets the crate is empty,
//! so that the workspace builds everywhere.

#![cfg(target_arch = "wasm32")]

use std::sync::{Arc, Mutex};

use emberwave_core::bitstream::{Pwv1File, consts, pwv1};
use emberwave_core::color::ColorParams;
use emberwave_core::gpu::{Decoder, GpuContext, GpuOptions, TEXTURE_FORMATS};
use wasm_bindgen::prelude::*;

mod bench;
mod demo;
pub use bench::Bench;
pub use demo::Demo;

/// Frame rate for a stream that does not say: PWV1 files without one hold
/// unrelated pictures, shown at this pace.
const DEFAULT_FRAME_RATE: f64 = 30.0;

/// The largest file the player loads, 1 GiB. Its bytes stay in the module's
/// memory while it plays, and a phone may not have more to spare.
const MAX_FILE_BYTES: u32 = 1 << 30;

/// The largest picture the player decodes, in pixels: DCI 8K, 8192x4320. The
/// format allows 16384x16384, whose decoder takes gigabytes of GPU memory,
/// and a file of a few bytes can ask for that.
const MAX_PIXELS: u64 = 8192 * 4320;

/// An error for the page: the first line of the message. An error from the
/// browser formats as `JsValue(Error: ...)` followed by its stack.
fn js_error(error: impl std::fmt::Display) -> JsError {
    let text = error.to_string();
    let first = text.lines().next().unwrap_or_default();
    JsError::new(&first.replace("JsValue(", ""))
}

/// Milliseconds on the page's clock (`performance.now()`), or 0 without one.
fn now_ms() -> f64 {
    web_sys::window()
        .and_then(|window| window.performance())
        .map_or(0.0, |performance| performance.now())
}

/// A canvas format the decoder draws into, in the order the browser
/// prefers them: 8-bit RGBA or BGRA without sRGB encoding, as the samples are
/// gamma encoded already. WebGPU canvases always offer both.
fn surface_format(surface: &wgpu::Surface<'_>, adapter: &wgpu::Adapter) -> wgpu::TextureFormat {
    surface
        .get_capabilities(adapter)
        .formats
        .into_iter()
        .find(|format| TEXTURE_FORMATS.contains(format))
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
    /// The file, whose frames are played in turn up to `end`.
    file: Vec<u8>,
    /// Where the frames end that can be played: the end of the file, or
    /// where the first one starts that cannot.
    end: usize,
    /// Where the size of the next frame to play is.
    next: usize,
    decoder: Decoder,
    color: ColorParams,
    width: u32,
    height: u32,
    frame_rate: f64,
}

/// The frames of a file that can be played, in order: how many there are,
/// where they end, and why the rest of the file cannot be played, if it
/// cannot. A frame that is cut off, or too short for the start of frame
/// header that every frame begins with, ends them, so that playing never
/// meets a frame it cannot read.
fn playable_frames(file: &Pwv1File<'_>) -> (usize, usize, Option<String>) {
    let mut end = pwv1::HEADER_BYTES;
    let mut count = 0;
    for frame in file.frames() {
        let reason = match frame {
            Ok(frame) if frame.len() >= consts::HEADER_BYTES => {
                count += 1;
                end += 4 + frame.len();
                continue;
            }
            Ok(frame) => format!(
                "frame {} has {} bytes, fewer than a start of frame header",
                count + 1,
                frame.len()
            ),
            Err(error) => format!("frame {} is cut off ({error})", count + 1),
        };
        return (count, end, Some(reason));
    }
    (count, end, None)
}

/// Plays PyroWave files on a canvas.
#[wasm_bindgen]
pub struct Player {
    context: GpuContext,
    adapter: wgpu::Adapter,
    surface: wgpu::Surface<'static>,
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

        Ok(Self {
            context,
            adapter,
            surface,
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

    /// Whether the GPU device was lost: a driver reset, or a GPU that went
    /// away. A lost player does not come back; the page makes a new one.
    pub fn is_lost(&self) -> bool {
        self.context.lost().is_some()
    }

    /// The largest file [`load`](Self::load) takes, in bytes, for the page
    /// to check before it hands one over.
    pub fn max_file_bytes() -> u32 {
        MAX_FILE_BYTES
    }

    /// Loads a PWV1 file and sizes the canvas to its pictures. Returns a line
    /// that describes the stream. The frames before one that is cut off or
    /// malformed are played, and the line says why the rest are not.
    pub fn load(&mut self, bytes: Vec<u8>) -> Result<String, JsError> {
        // The old stream goes first, whether or not the new one loads: its
        // frames and GPU memory are not needed while the new ones are made.
        self.stream = None;
        if let Ok(mut times) = self.gpu_times.lock() {
            *times = GpuTimes::default();
        }
        // An error the old stream's work raised is not the new one's.
        if let Ok(mut first) = self.gpu_error.lock() {
            *first = None;
        }

        if bytes.len() > MAX_FILE_BYTES as usize {
            return Err(JsError::new(&format!(
                "the file has {} MiB, and the player loads at most {} MiB",
                bytes.len() >> 20,
                MAX_FILE_BYTES >> 20
            )));
        }
        let file = Pwv1File::parse(&bytes).map_err(js_error)?;
        let header = *file.header();
        let stream = header.stream;
        let largest = self.context.device().limits().max_texture_dimension_2d;
        if stream.width > largest || stream.height > largest {
            return Err(JsError::new(&format!(
                "the pictures are {}x{}, and this browser draws at most {largest} pixels across",
                stream.width, stream.height
            )));
        }
        if u64::from(stream.width) * u64::from(stream.height) > MAX_PIXELS {
            return Err(JsError::new(&format!(
                "the pictures are {}x{}, and the player decodes at most {MAX_PIXELS} pixels, \
                 those of 8192x4320",
                stream.width, stream.height
            )));
        }
        let (count, end, unplayable) = playable_frames(&file);
        if count == 0 {
            return Err(JsError::new(
                unplayable.as_deref().unwrap_or("the file has no frames"),
            ));
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

        let frame_rate = if header.frame_rate_num != 0 && header.frame_rate_den != 0 {
            f64::from(header.frame_rate_num) / f64::from(header.frame_rate_den)
        } else {
            DEFAULT_FRAME_RATE
        };
        let mut description = format!(
            "{}x{} {:?}, {count} frames at {frame_rate:.2} fps",
            stream.width, stream.height, stream.chroma,
        );
        if let Some(reason) = unplayable {
            description.push_str(&format!("; the rest of the file is not played: {reason}"));
        }
        self.stream = Some(Stream {
            file: bytes,
            end,
            next: pwv1::HEADER_BYTES,
            decoder,
            color: ColorParams::from_stream(&stream),
            width: stream.width,
            height: stream.height,
            frame_rate,
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
        // load() found every frame up to `end` whole.
        let frame = stream
            .file
            .get(stream.next..stream.next + 4)
            .and_then(|size| size.try_into().ok())
            .map(|size| stream.next + 4..stream.next + 4 + u32::from_le_bytes(size) as usize)
            .and_then(|range| Some((stream.file.get(range.clone())?, range.end)));
        let Some((frame, after)) = frame else {
            return Err(JsError::new("the frames ran out"));
        };
        // The canvas has the picture's size, so the decoder draws straight
        // into it.
        let view = target.texture.create_view(&Default::default());
        stream
            .decoder
            .encode_rgb_to_texture(frame, &stream.color, &view, &mut encoder)
            .map_err(js_error)?;
        stream.next = if after >= stream.end {
            pwv1::HEADER_BYTES
        } else {
            after
        };
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
