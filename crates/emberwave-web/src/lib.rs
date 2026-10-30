//! emberwave in the browser: decodes PyroWave (`.pwv1`) files with WebGPU and
//! draws them on a canvas, or a stream packet by packet as a client receives
//! it, makes a demo stream for the player ([`Demo`]), and times the decoding
//! on the browser's GPU ([`Bench`]). The pages are in `web/`, and
//! `scripts/build_web.sh` builds this crate for them.
//!
//! Only the browser build has content. On other targets the crate is empty,
//! so that the workspace builds everywhere.

#![cfg(target_arch = "wasm32")]

use std::sync::{Arc, Mutex};

use emberwave_core::bitstream::{ChromaSubsampling, Pwv1File, YcbcrRange, consts, pwv1};
use emberwave_core::color::{ColorMatrix, ColorParams};
use emberwave_core::encode;
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

/// The canvas's texture to draw the next picture on.
fn current_texture(surface: &wgpu::Surface<'_>) -> Result<wgpu::SurfaceTexture, JsError> {
    match surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(texture)
        | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => Ok(texture),
        other => Err(JsError::new(&format!(
            "the canvas has no texture to draw on: {other:?}"
        ))),
    }
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
    /// The file, whose frames are played in turn up to `end`; empty for a
    /// stream of packets opened without one.
    file: Vec<u8>,
    /// Where the frames end that can be played: the end of the file, or
    /// where the first one starts that cannot.
    end: usize,
    /// Where each frame that can be played is in `file`, without its size.
    frames: Vec<std::ops::Range<usize>>,
    /// Where the size of the next frame to play is.
    next: usize,
    decoder: Decoder,
    /// The colours of the file's header, or for a stream opened without a
    /// file those it takes until a frame's start of frame header says
    /// otherwise, or those set by [`Player::set_colors`].
    color: ColorParams,
    /// Whether the colours follow each frame's start of frame header, as for
    /// a stream opened without a file until [`Player::set_colors`].
    follows_frames: bool,
    width: u32,
    height: u32,
    frame_rate: f64,
}

/// Where `frame`, an encoded frame, splits into packets of at most
/// `max_bytes` each, at its blocks, as a host sends them: the end of each
/// packet, in bytes from the start of the frame. The first packet starts with
/// the start of frame header, and a block larger than `max_bytes` gets a
/// packet of its own, as with `emberwave_frame_packets` in the C API.
#[wasm_bindgen]
pub fn packet_ends(frame: &[u8], max_bytes: usize) -> Vec<u32> {
    let start = frame.as_ptr() as usize;
    encode::packets(frame, max_bytes)
        .into_iter()
        .map(|packet| (packet.as_ptr() as usize - start + packet.len()) as u32)
        .collect()
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
        self.check_size(stream.width, stream.height)?;
        let (count, end, unplayable) = playable_frames(&file);
        if count == 0 {
            return Err(JsError::new(
                unplayable.as_deref().unwrap_or("the file has no frames"),
            ));
        }
        let decoder = self.new_decoder(stream.width, stream.height, stream.chroma)?;
        let mut frames = Vec::with_capacity(count);
        let mut at = pwv1::HEADER_BYTES;
        while at < end {
            let size = u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"));
            frames.push(at + 4..at + 4 + size as usize);
            at += 4 + size as usize;
        }

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
            frames,
            next: pwv1::HEADER_BYTES,
            decoder,
            color: ColorParams::from_stream(&stream),
            follows_frames: false,
            width: stream.width,
            height: stream.height,
            frame_rate,
        });
        Ok(description)
    }

    /// Opens a stream of `width` x `height` pictures, in 4:4:4 with
    /// `chroma_444` and 4:2:0 otherwise, `frame_rate` a second (30 for one
    /// that is not a positive finite number), without a file: its frames come
    /// as packets through
    /// [`push_packet`](Self::push_packet), as a client of a live stream
    /// receives them. Sizes the canvas, and returns a line that describes the
    /// stream. The colours follow each frame's start of frame header, unless
    /// [`set_colors`](Self::set_colors) sets them.
    pub fn open_stream(
        &mut self,
        width: u32,
        height: u32,
        chroma_444: bool,
        frame_rate: f64,
    ) -> Result<String, JsError> {
        self.stream = None;
        if let Ok(mut times) = self.gpu_times.lock() {
            *times = GpuTimes::default();
        }
        if let Ok(mut first) = self.gpu_error.lock() {
            *first = None;
        }
        self.check_size(width, height)?;
        let chroma = if chroma_444 {
            ChromaSubsampling::Yuv444
        } else {
            ChromaSubsampling::Yuv420
        };
        let decoder = self.new_decoder(width, height, chroma)?;
        let frame_rate = if frame_rate.is_finite() && frame_rate > 0.0 {
            frame_rate
        } else {
            DEFAULT_FRAME_RATE
        };
        self.stream = Some(Stream {
            file: Vec::new(),
            end: 0,
            frames: Vec::new(),
            next: 0,
            decoder,
            color: ColorParams::default(),
            follows_frames: true,
            width,
            height,
            frame_rate,
        });
        Ok(format!(
            "{width}x{height} {chroma:?}, a stream of packets at {frame_rate:.2} fps"
        ))
    }

    /// Sets the colours to convert the stream's pictures with: `matrix`
    /// `bt601`, `bt709` or `bt2020`, and `range` `full` or `limited`. Then
    /// these two no longer follow the frames' start of frame headers, which
    /// can signal the wrong ones: upstream's encoder signals full range
    /// whatever the pictures are. The chroma siting of a stream opened
    /// without a file still follows them; a loaded file's is its PWV1
    /// header's. An error changes nothing.
    pub fn set_colors(&mut self, matrix: &str, range: &str) -> Result<(), JsError> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| JsError::new("no stream is open"))?;
        let matrix = match matrix {
            "bt601" => ColorMatrix::Bt601,
            "bt709" => ColorMatrix::Bt709,
            "bt2020" => ColorMatrix::Bt2020,
            other => {
                return Err(JsError::new(&format!(
                    "the matrix is {other:?}, not bt601, bt709 or bt2020"
                )));
            }
        };
        let range = match range {
            "full" => YcbcrRange::Full,
            "limited" => YcbcrRange::Limited,
            other => {
                return Err(JsError::new(&format!(
                    "the range is {other:?}, not full or limited"
                )));
            }
        };
        stream.color.matrix = matrix;
        stream.color.range = range;
        stream.follows_frames = false;
        Ok(())
    }

    /// How many frames of the loaded file can be played; 0 for a stream
    /// opened without a file.
    pub fn frames(&self) -> u32 {
        self.stream
            .as_ref()
            .map_or(0, |stream| stream.frames.len() as u32)
    }

    /// A copy of frame `index` of the loaded file, of those that can be
    /// played, for a page that sends it on as packets.
    pub fn frame(&self, index: u32) -> Result<Vec<u8>, JsError> {
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| JsError::new("no file is loaded"))?;
        let range = stream
            .frames
            .get(index as usize)
            .ok_or_else(|| JsError::new(&format!("the file has no frame {index}")))?;
        Ok(stream.file[range.clone()].to_vec())
    }

    /// Adds a packet of the frame being assembled, as it arrives: a start of
    /// frame header with blocks, or blocks alone, in any order; a block that
    /// arrived already is ignored. A malformed packet is an error; the blocks
    /// in front of what is malformed are kept.
    ///
    /// A packet of a frame one to three frames newer by the bitstream's
    /// sequence number starts that frame and drops what arrived of the older
    /// one. As the number has three bits, one of a frame four to seven frames
    /// newer, or one to four frames older, is ignored; one five to seven
    /// frames older passes for a newer frame, and one eight frames off for
    /// the current one. So a transport keeps frame numbers of its own: it ignores the packets of
    /// frames older than the one being assembled, and before it pushes the
    /// first packet of a newer frame it draws the current one if it is
    /// ready, then calls [`drop_packets`](Self::drop_packets).
    ///
    /// A start of frame header of another picture size is an error: the
    /// stream changed size, and the page opens it again. Blocks alone do not
    /// say the size, so one of another size can pass for one of this stream
    /// while its index is in range: after that error, push nothing more to
    /// this stream.
    pub fn push_packet(&mut self, packet: &[u8]) -> Result<(), JsError> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| JsError::new("no stream is open"))?;
        stream.decoder.push_packet(packet).map_err(js_error)
    }

    /// Whether the frame assembled from packets can be drawn: all of it, or
    /// with `allow_partial` the part that arrived, if that includes every
    /// block of the coarsest level and more than 90 percent of the blocks
    /// the start of frame header announced, or the share `set_partial_share`
    /// set. Until that header arrives, the count is of all the picture's
    /// blocks, so a frame without it is ready only if its encoder left out
    /// few blocks.
    pub fn is_ready(&self, allow_partial: bool) -> bool {
        self.stream
            .as_ref()
            .is_some_and(|stream| stream.decoder.is_ready(allow_partial))
    }

    /// Sets the share of a frame's announced blocks that must have arrived,
    /// with every block of the coarsest level, for `is_ready(true)` and
    /// `draw_assembled` to take a frame that lost packets: 0.9 by default,
    /// as upstream's decoder has it. A lower share draws frames that lost
    /// more, with more of their detail blank, where a page would skip them
    /// and leave the last picture; 1 draws whole frames only. It holds for
    /// the stream open. Throws for a share outside 0 to 1, and without a
    /// stream.
    pub fn set_partial_share(&mut self, share: f32) -> Result<(), JsError> {
        if !(0.0..=1.0).contains(&share) {
            return Err(JsError::new(&format!(
                "a share of {share} is not from 0 to 1"
            )));
        }
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| JsError::new("no stream is open"))?;
        stream.decoder.set_partial_share(share);
        Ok(())
    }

    /// Sets whether a frame drawn in part conceals the blocks of detail it
    /// lacks with the last frame's, where the picture is still, rather than
    /// leaving them blank: off by default (decision D66). It holds for the
    /// stream open. Throws without a stream.
    pub fn set_concealment(&mut self, on: bool) -> Result<(), JsError> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| JsError::new("no stream is open"))?;
        stream.decoder.set_concealment(on);
        Ok(())
    }

    /// Drops what arrived of the frame being assembled, and forgets its
    /// sequence number, so that the next packet starts a frame of its own.
    /// Call it before pushing the first packet of a newer frame, never after:
    /// that packet would go too.
    pub fn drop_packets(&mut self) {
        if let Some(stream) = self.stream.as_mut() {
            stream.decoder.clear();
        }
    }

    /// Decodes the frame assembled from packets and draws it: all of it, or
    /// what arrived of it if `is_ready(true)` allows that; the blocks that
    /// did not arrive decode as zeros, or are concealed if `set_concealment`
    /// says so. A page that draws whole frames only
    /// checks `is_ready(false)` first. Packets of the frame that arrive after
    /// it was drawn are kept, and the frame is not drawn again. Returns the
    /// milliseconds the CPU spent on it.
    pub fn draw_assembled(&mut self) -> Result<f64, JsError> {
        self.take_gpu_error()?;
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| JsError::new("no stream is open"))?;
        if !stream.decoder.is_ready(true) {
            return Err(JsError::new("no frame is ready to draw"));
        }
        // A loaded file's colours are its header's, siting included; a
        // stream without a file follows its frames' headers, for the siting
        // also after `set_colors`.
        if let Some(header) = stream.decoder.start_of_frame() {
            if stream.follows_frames {
                stream.color = ColorParams::from_stream(header);
            } else if stream.file.is_empty() {
                stream.color.chroma_siting = header.chroma_siting;
            }
        }
        let start = now_ms();
        let target = current_texture(&self.surface)?;
        let view = target.texture.create_view(&Default::default());
        stream
            .decoder
            .submit_assembled_to_texture(&stream.color, &view)
            .map_err(js_error)?;
        let submitted = now_ms();
        self.time_until_done(submitted);
        self.context.queue().present(target);
        Ok(submitted - start)
    }

    /// An error unless pictures of `width` x `height` fit this browser and
    /// the player.
    fn check_size(&self, width: u32, height: u32) -> Result<(), JsError> {
        let largest = self.context.device().limits().max_texture_dimension_2d;
        if width > largest || height > largest {
            return Err(JsError::new(&format!(
                "the pictures are {width}x{height}, and this browser draws at most {largest} pixels across"
            )));
        }
        if u64::from(width) * u64::from(height) > MAX_PIXELS {
            return Err(JsError::new(&format!(
                "the pictures are {width}x{height}, and the player decodes at most {MAX_PIXELS} pixels, \
                 those of 8192x4320"
            )));
        }
        Ok(())
    }

    /// A decoder for pictures of `width` x `height`, and the canvas sized to
    /// them.
    fn new_decoder(
        &mut self,
        width: u32,
        height: u32,
        chroma: ChromaSubsampling,
    ) -> Result<Decoder, JsError> {
        let decoder = Decoder::new(&self.context, width, height, chroma).map_err(js_error)?;
        let config = self
            .surface
            .get_default_config(&self.adapter, width, height)
            .ok_or_else(|| JsError::new("this canvas cannot show the picture"))?;
        let config = wgpu::SurfaceConfiguration {
            format: surface_format(&self.surface, &self.adapter),
            ..config
        };
        self.surface.configure(self.context.device(), &config);
        Ok(decoder)
    }

    /// The first error the browser reported for the GPU's work since the
    /// last frame, as an error.
    fn take_gpu_error(&self) -> Result<(), JsError> {
        match self
            .gpu_error
            .lock()
            .ok()
            .and_then(|mut first| first.take())
        {
            Some(error) => Err(JsError::new(&format!("the GPU reported an error: {error}"))),
            None => Ok(()),
        }
    }

    /// Adds the time from `submitted` until the browser reports the work
    /// submitted so far done to the times [`take_gpu_ms`](Self::take_gpu_ms)
    /// averages.
    fn time_until_done(&self, submitted: f64) {
        let gpu_times = Arc::clone(&self.gpu_times);
        self.context.queue().on_submitted_work_done(move || {
            if let Ok(mut times) = gpu_times.lock() {
                times.total_ms += now_ms() - submitted;
                times.frames += 1;
            }
        });
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
        self.take_gpu_error()?;
        let stream = self
            .stream
            .as_mut()
            .filter(|stream| !stream.file.is_empty())
            .ok_or_else(|| JsError::new("no file is loaded"))?;
        let start = now_ms();
        let device = self.context.device();
        let queue = self.context.queue();
        let target = current_texture(&self.surface)?;

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
        // The decoder assembled the frame to decode it: let go of it, so that
        // packets pushed next, after a switch to them, start afresh.
        stream.decoder.clear();

        let submitted = now_ms();
        self.time_until_done(submitted);
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
