//! The benchmark of the browser player's decoding (`web/bench.html`): makes
//! test streams with the CPU encoder, then decodes them frame after frame on
//! the player's path, to RGB in GPU memory, and waits for each frame, which is
//! what `emberwave-cli bench --output rgb` times natively.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll, Waker};

use emberwave_core::bitstream::ChromaSubsampling;
use emberwave_core::color::ColorParams;
use emberwave_core::encode::{self, Signalling};
use emberwave_core::gpu::{Decoder, GpuContext, GpuOptions};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::js_sys::Promise;

use crate::{js_error, now_ms};

/// Times the decoding of test streams on this browser's GPU.
#[wasm_bindgen]
pub struct Bench {
    context: GpuContext,
    stream: Option<BenchStream>,
}

/// The frames of a test stream and the decoder for them.
struct BenchStream {
    frames: Vec<Vec<u8>>,
    decoder: Decoder,
    next: usize,
}

#[wasm_bindgen]
impl Bench {
    /// Sets up WebGPU without a canvas. Fails if the browser has no WebGPU.
    pub async fn create() -> Result<Bench, JsError> {
        // A panic aborts the module; at least its message reaches the console.
        std::panic::set_hook(Box::new(|info| {
            web_sys::console::error_1(&JsValue::from_str(&info.to_string()));
        }));
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::BROWSER_WEBGPU;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(js_error)?;
        let context = GpuContext::from_adapter(adapter, &GpuOptions::default())
            .await
            .map_err(js_error)?;
        Ok(Self {
            context,
            stream: None,
        })
    }

    /// The GPU in use. Browsers often keep its name to themselves.
    pub fn adapter(&self) -> String {
        let info = self.context.adapter_info();
        if info.name.is_empty() {
            "WebGPU".to_owned()
        } else {
            format!("{} (WebGPU)", info.name)
        }
    }

    /// Makes a test stream of `count` pictures of `width` x `height` in
    /// 4:2:0, a synthetic scene that changes from picture to picture, each
    /// encoded by the CPU encoder within `budget` bytes, and a decoder for
    /// it. Returns the bytes of the frames together. Takes a while at 4K:
    /// the CPU encoder takes about a quarter of a second for a 4K picture
    /// natively.
    pub fn prepare(
        &mut self,
        width: u32,
        height: u32,
        count: u32,
        budget: u32,
    ) -> Result<f64, JsError> {
        // The previous stream's GPU memory goes before the next is made.
        self.stream = None;
        let chroma = ChromaSubsampling::Yuv420;
        let mut encoder =
            encode::Encoder::new(width, height, chroma, Signalling::default()).map_err(js_error)?;
        let frames = (0..count.max(1))
            .map(|index| {
                let planes = scene(width, height, index);
                encoder
                    .encode([&planes[0], &planes[1], &planes[2]], budget as usize)
                    .map_err(js_error)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let decoder = Decoder::new(&self.context, width, height, chroma).map_err(js_error)?;
        let bytes = frames.iter().map(Vec::len).sum::<usize>();
        self.stream = Some(BenchStream {
            frames,
            decoder,
            next: 0,
        });
        Ok(bytes as f64)
    }

    /// Records the decode of the next frame to RGB in GPU memory, as the
    /// player does, and submits it. Returns the milliseconds the CPU took;
    /// [`work_done`](Self::work_done) says when the GPU is done.
    pub fn decode_next(&mut self) -> Result<f64, JsError> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| JsError::new("no test stream is prepared"))?;
        let start = now_ms();
        let mut encoder =
            self.context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("benchmark frame"),
                });
        stream
            .decoder
            .encode_rgb(
                &stream.frames[stream.next],
                &ColorParams::default(),
                &mut encoder,
            )
            .map_err(js_error)?;
        stream.next = (stream.next + 1) % stream.frames.len();
        self.context.queue().submit([encoder.finish()]);
        Ok(now_ms() - start)
    }

    /// A promise that resolves once the GPU is done with the work submitted
    /// so far.
    pub fn work_done(&self) -> Promise {
        let done = WorkDone::new(self.context.queue());
        wasm_bindgen_futures::future_to_promise(async move {
            done.await;
            Ok(JsValue::UNDEFINED)
        })
    }
}

/// Completes once the GPU is done with the work submitted before it was
/// made. wgpu calls back from the browser's promise; its callback has to be
/// `Send`, which a JavaScript function is not, so it sets a flag and wakes
/// this future rather than resolving a promise itself.
struct WorkDone {
    state: Arc<Mutex<(bool, Option<Waker>)>>,
}

impl WorkDone {
    fn new(queue: &wgpu::Queue) -> Self {
        let state: Arc<Mutex<(bool, Option<Waker>)>> = Arc::default();
        let record = Arc::clone(&state);
        queue.on_submitted_work_done(move || {
            let mut state = record.lock().unwrap_or_else(PoisonError::into_inner);
            state.0 = true;
            if let Some(waker) = state.1.take() {
                waker.wake();
            }
        });
        Self { state }
    }
}

impl Future for WorkDone {
    type Output = ();

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.0 {
            Poll::Ready(())
        } else {
            state.1 = Some(context.waker().clone());
            Poll::Pending
        }
    }
}

/// Picture `index` of the test scene in 4:2:0: a gradient, colour bars, a
/// shaded disc that moves from picture to picture, a fine grid and a little
/// grain, which the encoder finds about as hard as a game's picture.
fn scene(width: u32, height: u32, index: u32) -> [Vec<u8>; 3] {
    let (w, h) = (width as usize, height as usize);
    let mut grain = 0x9e37_79b9_u32 ^ index.wrapping_mul(0x85eb_ca6b);
    let mut noise = move || {
        grain ^= grain << 13;
        grain ^= grain >> 17;
        grain ^= grain << 5;
        (grain >> 29) as f32 - 3.5
    };
    let (cx, cy) = (
        w as f32 * (0.3 + 0.05 * index as f32),
        h as f32 * (0.5 + 0.03 * index as f32),
    );
    let radius = h as f32 * 0.25;
    let luma: Vec<u8> = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            let mut value = if y < h as f32 * 0.2 {
                // Eight bars of the grey levels of colour bars.
                [235.0, 210.0, 170.0, 145.0, 106.0, 81.0, 41.0, 16.0][(i % w) * 8 / w]
            } else {
                40.0 + 160.0 * x / w as f32 * (0.6 + 0.4 * y / h as f32)
            };
            let (dx, dy) = (x - cx, y - cy);
            let distance = (dx * dx + dy * dy).sqrt();
            if distance < radius {
                value = 60.0 + 150.0 * (1.0 - distance / radius) + 20.0 * (dy / radius);
            }
            if (i % w) % 16 == 0 || (i / w) % 16 == 0 {
                value = value * 0.8 + 30.0;
            }
            (value + noise()).clamp(0.0, 255.0) as u8
        })
        .collect();
    let (cw, ch) = (w / 2, h / 2);
    let chroma = |phase: f32| -> Vec<u8> {
        (0..cw * ch)
            .map(|i| {
                let (x, y) = ((i % cw) as f32 / cw as f32, (i / cw) as f32 / ch as f32);
                (128.0 + 60.0 * ((x * 6.0 + phase).sin() * (y * 4.0 - phase).cos())) as u8
            })
            .collect()
    };
    [
        luma,
        chroma(index as f32 * 0.4),
        chroma(1.7 + index as f32 * 0.3),
    ]
}
