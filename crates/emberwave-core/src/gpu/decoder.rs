//! The wgpu decoder: uploads a frame's blocks and runs the GPU stages.
//!
//! The split follows upstream. The CPU assembles packets
//! ([`FrameAssembler`]); the GPU decodes coefficients and dequantises them
//! (`dequant.wgsl`) and runs the inverse transform level by level
//! (`idwt.wgsl`), whose last level writes the 8-bit planes. On request a last
//! stage converts YCbCr to RGB (`yuv2rgb.wgsl`).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::bitstream::consts::DECOMPOSITION_LEVELS;
use crate::bitstream::{
    Band, ChromaSubsampling, Component, FrameAssembler, FrameLayout, SequenceHeader,
};
use crate::color::ColorParams;

use super::context::{GpuContext, notice_loss};
use super::error::GpuError;
use super::layout::{BufferLayout, DequantParams, TransformLevel, YuvParams};
use super::pipelines::{Pipelines, Stage, with_error_scope};
use super::resources::{
    StageQueries, buffer, buffer_limit, check_size, entry, map_for_writing, read_back,
    read_back_with, uniform_buffer,
};

/// Zero words after the payload. The dequantiser may read a little past the
/// end of the last block; this keeps those reads deterministic.
const PAYLOAD_PADDING_WORDS: usize = 4;

/// The payload buffer never shrinks below this many words.
const MIN_PAYLOAD_WORDS: usize = 16 * 1024;

/// One decoded colour plane, 8 bits per sample, rows tightly packed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FramePlane {
    /// Width in samples.
    pub width: u32,
    /// Height in samples.
    pub height: u32,
    /// Row-major samples, `width * height` bytes.
    pub data: Vec<u8>,
}

/// A decoded frame: the Y, Cb and Cr planes at 8 bits per sample.
///
/// Values are in the range of the stream: this is what the reference decoder
/// writes to 8-bit UNORM storage, with no colour conversion applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    /// Chroma subsampling; chroma planes are half size in 4:2:0.
    pub chroma: ChromaSubsampling,
    /// The planes in bitstream order (Y, Cb, Cr).
    pub planes: [FramePlane; 3],
}

/// A decoded frame converted to RGB.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbFrame {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row-major red, green and blue bytes: `3 * width * height` of them.
    pub data: Vec<u8>,
}

/// What [`Decoder::time_frame`] decodes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimedOutput {
    /// 8-bit Y, Cb and Cr planes in GPU memory.
    Yuv,
    /// RGB in GPU memory: the stages above plus the colour conversion.
    Rgb(ColorParams),
}

/// GPU time of each stage, measured with timestamp queries.
///
/// Only available on devices that support them, and then only at the
/// granularity of a pass: all five inverse transform levels are one pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageTimings {
    /// Coefficient decoding and dequantisation.
    pub dequant: Duration,
    /// The inverse transform, all levels, including the conversion of the
    /// planes to 8 bits at the last one.
    pub idwt: Duration,
    /// Colour conversion, if it was part of the decode.
    pub yuv2rgb: Option<Duration>,
}

impl StageTimings {
    /// The sum of the stages.
    pub fn total(&self) -> Duration {
        self.dequant + self.idwt + self.yuv2rgb.unwrap_or_default()
    }
}

/// Where the time of one decode went.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timings {
    /// CPU time to assemble the packets and hand the data to the GPU.
    pub prepare: Duration,
    /// Wall-clock time from encoding the GPU work until it was done. When the
    /// picture is read back this includes that.
    pub gpu_wall: Duration,
    /// GPU time per stage, if the device can measure it.
    pub stages: Option<StageTimings>,
}

/// What a decode reads back to the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Read {
    Nothing,
    Yuv,
    Rgb,
}

/// Buffers of the colour conversion, created when it is first used.
struct ColorResources {
    rgba: wgpu::Buffer,
    readback: wgpu::Buffer,
    params: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
}

/// Stages that are timed: dequantisation, the inverse transform and the
/// colour conversion, one pass each.
const TIMED_STAGES: u32 = 3;

/// Where the CPU writes a frame's block table and payload for the GPU to
/// copy, in the decodes that wait for their frame (decision D34): kept from
/// frame to frame, so that a frame goes into memory that is there already,
/// where `Queue::write_buffer` allocates new memory for every frame and the
/// first writes to it cost more than the copy.
struct FrameUpload {
    /// The block table, then, from the table's full size on, the payload.
    buffer: wgpu::Buffer,
    /// Whether `buffer` is mapped for writing.
    mapped: bool,
}

/// A dispatch of the inverse transform for one decomposition level.
struct IdwtStage {
    bind_group: wgpu::BindGroup,
    grid: (u32, u32, u32),
}

/// Decodes PyroWave frames of one size and chroma format on the GPU.
///
/// All GPU memory is allocated up front and reused for every frame, apart from
/// the payload buffer which grows when a frame needs more room.
pub struct Decoder {
    device: wgpu::Device,
    queue: wgpu::Queue,
    assembler: FrameAssembler,
    buffers: BufferLayout,

    payload: wgpu::Buffer,
    payload_capacity_words: usize,
    block_offsets: wgpu::Buffer,
    /// Made on the first decode that waits for its frame, and made again
    /// when the payload buffer grows; `encode_rgb`, which does not wait,
    /// writes through the queue instead, as a browser has to.
    upload: Option<FrameUpload>,
    pyramid: wgpu::Buffer,
    packed: wgpu::Buffer,
    readback: wgpu::Buffer,

    pipelines: Arc<Pipelines>,

    dequant_params: wgpu::Buffer,
    dequant_bind_group: wgpu::BindGroup,
    dequant_grid: (u32, u32),

    idwt_stages: Vec<IdwtStage>,

    color: Option<ColorResources>,
    timestamps: Option<StageQueries>,

    /// Why the device was lost, once it has been (see [`GpuContext::lost`]).
    lost: Arc<Mutex<Option<String>>>,
}

impl Decoder {
    /// Creates a decoder for pictures of `width` x `height` luma pixels.
    ///
    /// Allocates all buffers and builds the pipelines.
    pub fn new(
        context: &GpuContext,
        width: u32,
        height: u32,
        chroma: ChromaSubsampling,
    ) -> Result<Self, GpuError> {
        // A lost device stays lost: nothing is made on it.
        if let Some(reason) = context.lost() {
            return Err(GpuError::DeviceLost(reason));
        }
        let layout = FrameLayout::new(width, height, chroma)?;
        let buffers = BufferLayout::new(&layout)?;
        let device = context.device().clone();
        let queue = context.queue().clone();
        // Compiled once per context, however many decoders are created.
        let pipelines = context
            .pipelines()
            .map_err(|error| context.explain(error))?;

        check_size(&device, "the wavelet pyramid", buffers.pyramid_floats * 4)?;
        check_size(&device, "the packed planes", buffers.packed_words * 4)?;

        with_error_scope(&device, || {
            let storage = |label: &str, bytes: u64, usage| buffer(&device, label, bytes, usage);

            let payload_capacity_words = MIN_PAYLOAD_WORDS;
            let payload = storage(
                "payload",
                payload_capacity_words as u64 * 4,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let block_offsets = storage(
                "block offsets",
                u64::from(layout.block_count()) * 4,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let pyramid = storage(
                "wavelet pyramid",
                buffers.pyramid_floats * 4,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            // COPY_DST lets a test fill the planes before a decode, to see
            // that the decode writes every word of them.
            let packed = storage(
                "packed planes",
                buffers.packed_words * 4,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            );
            let readback = storage(
                "readback",
                buffers.packed_words * 4,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );

            let dequant_params_data = DequantParams::new(&layout, &buffers);
            let dequant_grid = dequant_params_data.grid();
            let dequant_params =
                uniform_buffer(&device, &queue, "dequant params", &dequant_params_data);
            let dequant_bind_group = dequant_bind_group(
                &device,
                &pipelines.dequant,
                &payload,
                &block_offsets,
                &pyramid,
                &dequant_params,
            );

            // Levels from the coarsest to the finest: each consumes the LL band
            // its predecessor wrote.
            let idwt_stages = (0..DECOMPOSITION_LEVELS)
                .rev()
                .map(|level| {
                    let stage = TransformLevel::new(&layout, &buffers, level);
                    let params = uniform_buffer(&device, &queue, "idwt params", &stage.params);
                    IdwtStage {
                        bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("idwt"),
                            layout: &pipelines.idwt.layout,
                            entries: &[entry(0, &pyramid), entry(1, &packed), entry(2, &params)],
                        }),
                        grid: stage.grid(),
                    }
                })
                .collect();

            let timestamps = context
                .supports_timestamps()
                .then(|| StageQueries::new(&device, &queue, TIMED_STAGES));

            Self {
                device: device.clone(),
                queue: queue.clone(),
                assembler: FrameAssembler::new(layout.clone()),
                buffers: buffers.clone(),
                payload,
                payload_capacity_words,
                block_offsets,
                upload: None,
                pyramid,
                packed,
                readback,
                pipelines: Arc::clone(&pipelines),
                dequant_params,
                dequant_bind_group,
                dequant_grid,
                idwt_stages,
                color: None,
                timestamps,
                lost: context.lost_record(),
            }
        })
        .map_err(|error| context.explain(error))
    }

    /// Why the device of this decoder was lost, if it was. See
    /// [`GpuContext::lost`].
    pub fn lost(&self) -> Option<String> {
        self.lost.lock().ok().and_then(|lost| lost.clone())
    }

    /// An error if the device has been lost, so that nothing is decoded on it
    /// and no stale result is returned.
    fn check_device(&self) -> Result<(), GpuError> {
        match self.lost() {
            Some(reason) => Err(GpuError::DeviceLost(reason)),
            None => Ok(()),
        }
    }

    /// The block layout of the frames this decoder decodes.
    pub fn layout(&self) -> &FrameLayout {
        self.assembler.layout()
    }

    /// Decodes one complete frame: a start of frame header followed by its
    /// blocks, as stored in a PWV1 file.
    ///
    /// Whatever was assembled before is discarded, so the same frame may be
    /// decoded repeatedly.
    pub fn decode(&mut self, frame: &[u8]) -> Result<Frame, GpuError> {
        self.assemble_complete(frame)?;
        self.decode_assembled()
    }

    /// Decodes one complete frame like [`decode`](Self::decode) and converts
    /// it to RGB on the GPU.
    pub fn decode_rgb(&mut self, frame: &[u8], color: &ColorParams) -> Result<RgbFrame, GpuError> {
        self.assemble_complete(frame)?;
        self.decode_assembled_rgb(color)
    }

    /// Decodes one complete frame and waits for the GPU without reading the
    /// picture back, which is what a player does: the picture stays in GPU
    /// memory. Returns where the time went, for benchmarks.
    ///
    /// The per-stage times are `Some` only if the context was created with
    /// [`GpuOptions::timestamps`](super::GpuOptions::timestamps) and the device
    /// supports them.
    pub fn time_frame(&mut self, frame: &[u8], output: TimedOutput) -> Result<Timings, GpuError> {
        let start = Instant::now();
        self.assemble_complete(frame)?;
        let assembled = start.elapsed();

        let color = match output {
            TimedOutput::Yuv => None,
            TimedOutput::Rgb(color) => Some(color),
        };
        self.run_gpu(color.as_ref(), Read::Nothing, true, assembled, &mut |_| {})
    }

    /// Records the decode of one complete frame, and its conversion to RGB,
    /// into `encoder`, without submitting it or waiting for anything. Once the
    /// caller has submitted `encoder`, [`rgba_buffer`](Self::rgba_buffer)
    /// holds the picture: `width * height` words, one per pixel, red in the
    /// low byte. This is the path for a player that draws the picture on the
    /// GPU, and the only one in a browser, where nothing may block.
    ///
    /// The frame's data goes to the GPU with the next submission on this
    /// decoder's queue, so record one frame per submission.
    pub fn encode_rgb(
        &mut self,
        frame: &[u8],
        color: &ColorParams,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), GpuError> {
        self.check_device()?;
        self.assemble_complete(frame)?;
        self.ensure_payload_capacity()?;
        with_error_scope(&self.device, || self.upload())?;
        self.prepare_color(color)?;
        self.encode_passes(encoder, true, None);
        self.assembler.mark_frame_decoded();
        Ok(())
    }

    /// The RGBA words that [`encode_rgb`](Self::encode_rgb) and
    /// [`decode_rgb`](Self::decode_rgb) write, once one of them has run:
    /// `width * height` words, one per pixel, red in the low byte, alpha 255.
    /// The buffer belongs to the decoder: dropping the decoder destroys it,
    /// clones of the handle included.
    pub fn rgba_buffer(&self) -> Option<&wgpu::Buffer> {
        self.color.as_ref().map(|resources| &resources.rgba)
    }

    /// Feeds one packet of the frame being received. See
    /// [`FrameAssembler::push_packet`].
    pub fn push_packet(&mut self, packet: &[u8]) -> Result<(), GpuError> {
        Ok(self.assembler.push_packet(packet)?)
    }

    /// Whether the frame being received can be decoded. See
    /// [`FrameAssembler::is_ready`].
    pub fn is_ready(&self, allow_partial_frame: bool) -> bool {
        self.assembler.is_ready(allow_partial_frame)
    }

    /// Discards what was received, and forgets the order of the frames: the
    /// next packet starts a frame, whatever its sequence number. See
    /// [`FrameAssembler::clear`].
    pub fn clear(&mut self) {
        self.assembler.clear();
    }

    /// The start of frame header of the frame being received, once it has
    /// arrived: among other things, the colour signalling of the picture.
    pub fn start_of_frame(&self) -> Option<&SequenceHeader> {
        self.assembler.start_of_frame()
    }

    /// Decodes the frame assembled from [`push_packet`](Self::push_packet)
    /// calls. Blocks that did not arrive decode as zeros.
    pub fn decode_assembled(&mut self) -> Result<Frame, GpuError> {
        let layout = self.layout();
        let chroma = layout.chroma();
        let mut planes = Component::ALL.map(|component| {
            let (width, height) = layout.plane_size(component);
            FramePlane {
                width,
                height,
                data: vec![0; width as usize * height as usize],
            }
        });
        self.decode_assembled_rows(|component, row, samples| {
            let plane = &mut planes[component.index()];
            let width = plane.width as usize;
            plane.data[row as usize * width..][..width].copy_from_slice(samples);
        })?;
        Ok(Frame { chroma, planes })
    }

    /// Like [`decode_assembled`](Self::decode_assembled), converted to RGB.
    pub fn decode_assembled_rgb(&mut self, color: &ColorParams) -> Result<RgbFrame, GpuError> {
        let (width, height) = (self.layout().width(), self.layout().height());
        let mut data = vec![0; width as usize * height as usize * 3];
        self.run_gpu(Some(color), Read::Rgb, false, Duration::ZERO, &mut |rgba| {
            rgba_to_rgb(rgba, &mut data);
        })?;
        Ok(RgbFrame {
            width,
            height,
            data,
        })
    }

    /// Decodes the frame assembled from [`push_packet`](Self::push_packet)
    /// calls, like [`decode_assembled`](Self::decode_assembled), and hands
    /// the picture to `row` one row at a time, rather than in a new frame:
    /// the plane (Y, Cb and Cr in turn), the row's index from the top, and
    /// its samples, as many as the plane is wide. The rows come straight from
    /// the memory the GPU wrote, so a caller can put them where it wants
    /// them, with its own row strides, at the cost of one copy. A panic in
    /// `row` goes on to the caller, and the decoder can still be used.
    pub fn decode_assembled_rows(
        &mut self,
        mut row: impl FnMut(Component, u32, &[u8]),
    ) -> Result<(), GpuError> {
        let planes = self.packed_planes();
        self.run_gpu(None, Read::Yuv, false, Duration::ZERO, &mut |packed| {
            for (component, plane) in Component::ALL.into_iter().zip(&planes) {
                for (index, samples) in plane.rows(packed).enumerate() {
                    row(component, index as u32, samples);
                }
            }
        })?;
        Ok(())
    }

    /// Like [`decode_assembled_rows`](Self::decode_assembled_rows), converted
    /// to RGB: `row` gets the row's index from the top and its pixels, three
    /// bytes each with red first.
    pub fn decode_assembled_rgb_rows(
        &mut self,
        color: &ColorParams,
        mut row: impl FnMut(u32, &[u8]),
    ) -> Result<(), GpuError> {
        let width = self.layout().width() as usize;
        let mut line = vec![0; width * 3];
        self.run_gpu(Some(color), Read::Rgb, false, Duration::ZERO, &mut |rgba| {
            for (index, pixels) in rgba.chunks_exact(width * 4).enumerate() {
                rgba_to_rgb(pixels, &mut line);
                row(index as u32, &line);
            }
        })?;
        Ok(())
    }

    /// Discards what was assembled and takes `frame` as a complete frame.
    fn assemble_complete(&mut self, frame: &[u8]) -> Result<(), GpuError> {
        self.assembler.clear();
        self.assembler.push_packet(frame)?;
        if self.assembler.is_ready(false) {
            Ok(())
        } else {
            Err(GpuError::IncompleteFrame {
                received: self.assembler.received_blocks(),
                total: self.assembler.total_blocks(),
            })
        }
    }

    /// Runs the GPU stages for the assembled frame and waits for them.
    ///
    /// `color` adds the colour conversion. `read` says what is copied back to
    /// the CPU, which `consume` then gets while the buffer is mapped, unless
    /// the device was lost; [`Read::Nothing`] calls it not at all.
    /// `assembled_in` is the CPU time already spent assembling the frame. With
    /// `timed` the stages are timed, if the device can.
    fn run_gpu(
        &mut self,
        color: Option<&ColorParams>,
        read: Read,
        timed: bool,
        assembled_in: Duration,
        consume: &mut dyn FnMut(&[u8]),
    ) -> Result<Timings, GpuError> {
        self.check_device()?;
        let outcome = self.run_gpu_on_device(color, read, timed, assembled_in, consume);
        // The device can be lost while the frame is on it, and wgpu tells
        // during the wait, or for a destroyed device when it is next polled.
        // Whatever failed then is the loss, and whatever came back is not to
        // be trusted.
        if outcome.is_err() {
            notice_loss(&self.device, &self.lost);
        }
        self.check_device()?;
        outcome
    }

    /// The work of [`run_gpu`](Self::run_gpu), without the checks for a lost
    /// device.
    fn run_gpu_on_device(
        &mut self,
        color: Option<&ColorParams>,
        read: Read,
        timed: bool,
        assembled_in: Duration,
        consume: &mut dyn FnMut(&[u8]),
    ) -> Result<Timings, GpuError> {
        let prepare_start = Instant::now();
        self.ensure_payload_capacity()?;
        let (table_bytes, payload_bytes) = self.upload_mapped()?;
        if let Some(color) = color {
            self.prepare_color(color)?;
        }
        let prepare = assembled_in + prepare_start.elapsed();

        let queries = self.timestamps.as_ref().filter(|_| timed);
        let gpu_start = Instant::now();
        with_error_scope(&self.device, || {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("decode"),
                });

            if let Some(upload) = &self.upload {
                let payload_start = self.table_capacity_bytes();
                encoder.copy_buffer_to_buffer(
                    &upload.buffer,
                    0,
                    &self.block_offsets,
                    0,
                    table_bytes,
                );
                encoder.copy_buffer_to_buffer(
                    &upload.buffer,
                    payload_start,
                    &self.payload,
                    0,
                    payload_bytes,
                );
            }
            self.encode_passes(&mut encoder, color.is_some(), queries);

            match (read, self.color.as_ref()) {
                (Read::Yuv, _) => encoder.copy_buffer_to_buffer(
                    &self.packed,
                    0,
                    &self.readback,
                    0,
                    self.buffers.packed_words * 4,
                ),
                (Read::Rgb, Some(resources)) => encoder.copy_buffer_to_buffer(
                    &resources.rgba,
                    0,
                    &resources.readback,
                    0,
                    resources.rgba.size(),
                ),
                _ => {}
            }

            self.queue.submit([encoder.finish()]);
        })?;

        // The wait for the frame polls the device, which records a loss. A
        // device lost by then ran the frame in part or not at all, and its
        // bytes are not the picture: run_gpu reports the loss instead.
        let lost = &self.lost;
        let mut deliver = |bytes: &[u8]| {
            if lost.lock().is_ok_and(|lost| lost.is_none()) {
                consume(bytes);
            }
        };
        match (read, self.color.as_ref()) {
            (Read::Yuv, _) => read_back_with(
                &self.device,
                &self.readback,
                self.buffers.packed_words * 4,
                &mut deliver,
            )?,
            (Read::Rgb, Some(resources)) => read_back_with(
                &self.device,
                &resources.readback,
                resources.rgba.size(),
                &mut deliver,
            )?,
            _ => {
                self.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .map_err(|error| GpuError::Readback(error.to_string()))?;
            }
        }
        let gpu_wall = gpu_start.elapsed();

        let stages = match queries {
            Some(q) => self.read_stage_timings(q, color.is_some())?,
            None => None,
        };

        self.assembler.mark_frame_decoded();
        Ok(Timings {
            prepare,
            gpu_wall,
            stages,
        })
    }

    /// Records the GPU stages of a frame into `encoder`: dequantisation, the
    /// inverse transform and, with `rgb`, the colour conversion. With
    /// `queries` each pass writes its timestamps.
    fn encode_passes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        rgb: bool,
        queries: Option<&StageQueries>,
    ) {
        // Each pass is a stage of its own.
        let writes = |pass: u32| queries.map(|q| q.writes(Some(pass), Some(pass)));

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dequant"),
                timestamp_writes: writes(0),
            });
            pass.set_pipeline(&self.pipelines.dequant.pipeline);
            pass.set_bind_group(0, &self.dequant_bind_group, &[]);
            pass.dispatch_workgroups(self.dequant_grid.0, self.dequant_grid.1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("idwt"),
                timestamp_writes: writes(1),
            });
            pass.set_pipeline(&self.pipelines.idwt.pipeline);
            for stage in &self.idwt_stages {
                pass.set_bind_group(0, &stage.bind_group, &[]);
                pass.dispatch_workgroups(stage.grid.0, stage.grid.1, stage.grid.2);
            }
        }

        if let (true, Some(resources)) = (rgb, self.color.as_ref()) {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("yuv2rgb"),
                timestamp_writes: writes(2),
            });
            pass.set_pipeline(&self.pipelines.yuv2rgb.pipeline);
            pass.set_bind_group(0, &resources.bind_group, &[]);
            pass.dispatch_workgroups(resources.grid.0, resources.grid.1, 1);
        }
    }

    /// Creates the buffers of the colour conversion on first use and writes
    /// the parameters, which may change from call to call.
    fn prepare_color(&mut self, color: &ColorParams) -> Result<(), GpuError> {
        let params = YuvParams::new(self.assembler.layout(), &self.buffers, color);

        if self.color.is_none() {
            let bytes = u64::from(params.width) * u64::from(params.height) * 4;
            check_size(&self.device, "the RGB image", bytes)?;

            let resources = with_error_scope(&self.device, || {
                let rgba = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("rgba"),
                    size: bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("rgba readback"),
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("yuv2rgb params"),
                    size: std::mem::size_of::<YuvParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("yuv2rgb"),
                    layout: &self.pipelines.yuv2rgb.layout,
                    entries: &[
                        entry(0, &self.packed),
                        entry(1, &rgba),
                        entry(2, &params_buffer),
                    ],
                });
                ColorResources {
                    rgba,
                    readback,
                    params: params_buffer,
                    bind_group,
                    grid: params.grid(),
                }
            })?;
            self.color = Some(resources);
        }

        if let Some(resources) = &self.color {
            with_error_scope(&self.device, || {
                self.queue
                    .write_buffer(&resources.params, 0, bytemuck::bytes_of(&params));
            })?;
        }
        Ok(())
    }

    /// Reads the timestamps the last decode wrote and turns them into stage
    /// times. `None` if the device wrote nothing usable.
    fn read_stage_timings(
        &self,
        queries: &StageQueries,
        with_rgb: bool,
    ) -> Result<Option<StageTimings>, GpuError> {
        let Some(stages) = queries.read(&self.device, &self.queue)? else {
            return Ok(None);
        };
        Ok(match (stages[0], stages[1]) {
            (Some(dequant), Some(idwt)) => Some(StageTimings {
                dequant,
                idwt,
                yuv2rgb: if with_rgb { stages[2] } else { None },
            }),
            _ => None,
        })
    }

    /// Reads one band of the wavelet pyramid back, row-major with the width
    /// of the band as stride. Intended for tests and debugging; it waits for
    /// the GPU.
    ///
    /// After a decode the LL bands of levels 3 to 0 hold the intermediate
    /// results of the inverse transform, and the other bands hold the
    /// dequantised coefficients. Returns `None` if the component has no such
    /// level.
    pub fn read_band(
        &self,
        component: Component,
        level: usize,
        band: Band,
    ) -> Result<Option<Vec<f32>>, GpuError> {
        let Some(offset) = self.buffers.band_offset(component, level, band) else {
            return Ok(None);
        };
        let (width, height) = self.layout().level_size(level);
        let bytes = u64::from(width) * u64::from(height) * 4;

        self.check_device()?;
        let copied = with_error_scope(&self.device, || {
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("band readback"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("band readback"),
                });
            encoder.copy_buffer_to_buffer(&self.pyramid, u64::from(offset) * 4, &staging, 0, bytes);
            self.queue.submit([encoder.finish()]);
            staging
        })
        .and_then(|staging| read_back(&self.device, &staging, bytes));
        // As in run_gpu: a failure on a device that was lost meanwhile is the loss.
        self.check_device()?;
        let data = copied?;
        Ok(Some(
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ))
    }

    /// Grows the payload buffer if the assembled frame does not fit.
    ///
    /// A frame whose payload is larger than the device can bind is an error,
    /// not a crash. This can only happen with a huge stream: a 4K frame whose
    /// every block is as large as the format allows is about 200 MiB, and the
    /// WebGPU default limit is 128 MiB.
    fn ensure_payload_capacity(&mut self) -> Result<(), GpuError> {
        let needed_words = self.assembler.frame().payload().len() + PAYLOAD_PADDING_WORDS;
        if needed_words <= self.payload_capacity_words {
            return Ok(());
        }

        let limit_bytes = buffer_limit(&self.device);
        let capacity_bytes =
            payload_capacity_bytes(needed_words, limit_bytes).ok_or(GpuError::LimitExceeded {
                what: "the payload of this frame",
                needed: needed_words as u64 * 4,
                limit: limit_bytes & !3,
            })?;
        let (payload, bind_group) = with_error_scope(&self.device, || {
            let payload = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("payload"),
                size: capacity_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = dequant_bind_group(
                &self.device,
                &self.pipelines.dequant,
                &payload,
                &self.block_offsets,
                &self.pyramid,
                &self.dequant_params,
            );
            (payload, bind_group)
        })?;

        // Work already submitted keeps the old buffer until it is done.
        std::mem::replace(&mut self.payload, payload).destroy();
        self.payload_capacity_words = (capacity_bytes / 4) as usize;
        self.dequant_bind_group = bind_group;
        Ok(())
    }

    /// Bytes of the block table: where the payload starts in the upload
    /// buffer.
    fn table_capacity_bytes(&self) -> u64 {
        u64::from(self.layout().block_count()) * 4
    }

    /// Writes the block table and the payload of the assembled frame into
    /// the upload buffer, which is made, or made larger, first if it has to
    /// be, and returns the bytes of each for the copies at the start of the
    /// frame's command buffer. The previous frame is done, so mapping the
    /// buffer waits for nothing.
    fn upload_mapped(&mut self) -> Result<(u64, u64), GpuError> {
        let payload_start = self.table_capacity_bytes();
        let capacity_bytes = payload_start + self.payload_capacity_words as u64 * 4;
        let upload = match &mut self.upload {
            Some(upload) if upload.buffer.size() >= capacity_bytes => upload,
            slot => {
                let buffer = with_error_scope(&self.device, || {
                    self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("frame upload"),
                        size: capacity_bytes,
                        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: true,
                    })
                })?;
                // Work already submitted keeps the old buffer until it is done.
                if let Some(old) = slot.take() {
                    old.buffer.destroy();
                }
                slot.insert(FrameUpload {
                    buffer,
                    mapped: true,
                })
            }
        };
        if !upload.mapped {
            map_for_writing(&self.device, &upload.buffer)?;
            upload.mapped = true;
        }

        let frame = self.assembler.frame();
        let table: &[u8] = bytemuck::cast_slice(frame.block_offsets());
        let payload: &[u8] = bytemuck::cast_slice(frame.payload());
        let payload_bytes = (payload.len() + PAYLOAD_PADDING_WORDS * 4) as u64;
        {
            let mut view = upload
                .buffer
                .slice(..payload_start + payload_bytes)
                .get_mapped_range_mut()
                .map_err(|error| GpuError::Device(error.to_string()))?;
            let start = payload_start as usize;
            view.slice(..table.len()).copy_from_slice(table);
            view.slice(start..start + payload.len())
                .copy_from_slice(payload);
            view.slice(start + payload.len()..start + payload_bytes as usize)
                .fill(0);
        }
        with_error_scope(&self.device, || upload.buffer.unmap())?;
        upload.mapped = false;
        Ok((table.len() as u64, payload_bytes))
    }

    /// Uploads the block table and the payload of the assembled frame through
    /// the queue, for `encode_rgb`, which does not wait for its frame and so
    /// cannot map the upload buffer.
    fn upload(&self) {
        let frame = self.assembler.frame();
        self.queue.write_buffer(
            &self.block_offsets,
            0,
            bytemuck::cast_slice(frame.block_offsets()),
        );

        let payload = frame.payload();
        self.queue
            .write_buffer(&self.payload, 0, bytemuck::cast_slice(payload));
        self.queue.write_buffer(
            &self.payload,
            payload.len() as u64 * 4,
            bytemuck::cast_slice(&[0u32; PAYLOAD_PADDING_WORDS]),
        );
    }

    /// Where the planes are in the packed buffer, Y, Cb and Cr.
    fn packed_planes(&self) -> [PackedPlane; 3] {
        let layout = self.layout();
        Component::ALL.map(|component| {
            let i = component.index();
            let (width, height) = layout.plane_size(component);
            PackedPlane {
                width: width as usize,
                height: height as usize,
                offset: self.buffers.packed_offsets[i] as usize * 4,
                stride: self.buffers.packed_strides[i] as usize * 4,
            }
        })
    }
}

/// A plane in the packed buffer, in bytes: its rows are padded to whole
/// words.
#[derive(Debug, Clone, Copy)]
struct PackedPlane {
    width: usize,
    height: usize,
    offset: usize,
    stride: usize,
}

impl PackedPlane {
    /// The plane's rows in `packed`, without their padding.
    fn rows<'a>(&self, packed: &'a [u8]) -> impl Iterator<Item = &'a [u8]> {
        let (width, stride) = (self.width, self.stride);
        packed[self.offset..]
            .chunks(stride)
            .take(self.height)
            .map(move |row| &row[..width])
    }
}

/// Copies the red, green and blue bytes of RGBA pixels, dropping alpha, until
/// either slice runs out. Written so that the compiler vectorises it: 0.5 ms
/// for a 4K frame on an Apple M3 Max, where a copy per pixel took 4.5 ms.
fn rgba_to_rgb(rgba: &[u8], rgb: &mut [u8]) {
    for (to, from) in rgb.chunks_exact_mut(3).zip(rgba.chunks_exact(4)) {
        to[0] = from[0];
        to[1] = from[1];
        to[2] = from[2];
    }
}

impl std::fmt::Debug for Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decoder")
            .field("layout", self.assembler.layout())
            .field("payload_capacity_words", &self.payload_capacity_words)
            .finish_non_exhaustive()
    }
}

impl Drop for Decoder {
    /// Frees the GPU memory of the large buffers at once. Native wgpu does
    /// that when a buffer is dropped anyway, but in a browser dropping it only
    /// lets go of the JavaScript object, and the memory waits for the garbage
    /// collector: about 140 MiB for a 4K decoder. Work already submitted keeps
    /// its buffers until it is done.
    fn drop(&mut self) {
        let mut buffers = vec![
            &self.payload,
            &self.block_offsets,
            &self.pyramid,
            &self.packed,
            &self.readback,
        ];
        if let Some(upload) = &self.upload {
            buffers.push(&upload.buffer);
        }
        if let Some(color) = &self.color {
            buffers.extend([&color.rgba, &color.readback]);
        }
        for buffer in buffers {
            buffer.destroy();
        }
    }
}

/// The size in bytes of a payload buffer that holds `needed_words` words on a
/// device whose buffers can be at most `limit_bytes` large, or `None` if even
/// the exact size is too much.
///
/// The size is rounded up to a power of two, so that a stream of slowly
/// growing frames does not reallocate every time, but never beyond the limit.
fn payload_capacity_bytes(needed_words: usize, limit_bytes: u64) -> Option<u64> {
    let limit = limit_bytes & !3;
    let needed = needed_words as u64 * 4;
    if needed > limit {
        return None;
    }
    let rounded = needed_words.next_power_of_two().max(MIN_PAYLOAD_WORDS) as u64 * 4;
    Some(rounded.min(limit))
}

fn dequant_bind_group(
    device: &wgpu::Device,
    stage: &Stage,
    payload: &wgpu::Buffer,
    block_offsets: &wgpu::Buffer,
    pyramid: &wgpu::Buffer,
    params: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dequant"),
        layout: &stage.layout,
        entries: &[
            entry(0, payload),
            entry(1, block_offsets),
            entry(2, pyramid),
            entry(3, params),
        ],
    })
}

#[cfg(test)]
mod tests {
    use wgpu::util::DeviceExt;

    use super::*;
    use crate::gpu::context::test_context;

    /// Runs `to_unorm8` of `idwt.wgsl`, the rounding of the last level, on
    /// `values` on the GPU and returns the 8-bit levels. The function is cut out
    /// of the shader's source, so the test runs exactly the code the decoder runs.
    fn to_unorm8_on_gpu(context: &GpuContext, values: &[f32]) -> Vec<u8> {
        let source = super::super::shader_sources::IDWT;
        let start = source.find("fn to_unorm8").expect("to_unorm8 in idwt.wgsl");
        // The function ends with the first line that starts with a brace; "\n}"
        // finds it whatever the line endings.
        let end = start + source[start..].find("\n}").expect("end of to_unorm8") + 2;
        let shader = format!(
            "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n{}\n\
             @compute @workgroup_size(64)\n\
             fn main(@builtin(global_invocation_id) id: vec3<u32>) {{\n\
                 if (id.x < arrayLength(&data)) {{ data[id.x] = to_unorm8(data[id.x]); }}\n\
             }}\n",
            &source[start..end]
        );

        let device = context.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("to_unorm8 test"),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("to_unorm8 test"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bytes = (values.len() * 4) as u64;
        let data = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("values"),
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("to_unorm8 test"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[entry(0, &data)],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((values.len() as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&data, 0, &readback, 0, bytes);
        context.queue().submit([encoder.finish()]);

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| result.expect("map"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let levels: Vec<f32> =
            bytemuck::cast_slice(&slice.get_mapped_range().expect("range")).to_vec();
        levels.iter().map(|&level| level as u8).collect()
    }

    /// A frame without blocks: all coefficients zero, every sample mid-grey.
    fn empty_frame(width: u32, height: u32, chroma: ChromaSubsampling) -> Vec<u8> {
        let word0 = (width - 1) | ((height - 1) << 14) | (1 << 28) | (1 << 31);
        let word1 = u32::from(chroma == ChromaSubsampling::Yuv444) << 26;
        [word0.to_le_bytes(), word1.to_le_bytes()].concat()
    }

    #[test]
    fn the_last_level_writes_every_word_with_zero_padding() {
        let Some(context) = test_context() else {
            return;
        };

        // Odd widths leave padding in the last word of a row; 4:2:0 chroma
        // ends at level 1, whose size need not be a multiple of the tile.
        let mut shapes = Vec::new();
        for width in [1u32, 2, 3, 5, 31, 33, 127, 129, 131] {
            for height in [1u32, 2, 3, 33, 131] {
                shapes.push((width, height, ChromaSubsampling::Yuv444));
            }
        }
        for width in [2u32, 6, 30, 34, 98, 130, 194, 226] {
            for height in [2u32, 6, 34, 130] {
                shapes.push((width, height, ChromaSubsampling::Yuv420));
            }
        }

        for (width, height, chroma) in shapes {
            let mut decoder = Decoder::new(context, width, height, chroma).unwrap();
            // Anything the decode leaves unwritten keeps this pattern.
            let size = decoder.buffers.packed_words as usize * 4;
            decoder
                .queue
                .write_buffer(&decoder.packed, 0, &vec![0xA5; size]);
            decoder
                .assemble_complete(&empty_frame(width, height, chroma))
                .unwrap();
            let mut packed = Vec::new();
            decoder
                .run_gpu(None, Read::Yuv, false, Duration::ZERO, &mut |bytes| {
                    packed = bytes.to_vec();
                })
                .unwrap();

            for component in Component::ALL {
                let i = component.index();
                let (w, h) = decoder.layout().plane_size(component);
                let stride = decoder.buffers.packed_strides[i] as usize * 4;
                let start = decoder.buffers.packed_offsets[i] as usize * 4;
                for row in 0..h as usize {
                    let line = &packed[start + row * stride..start + (row + 1) * stride];
                    let (samples, padding) = line.split_at(w as usize);
                    assert!(
                        samples.iter().all(|&s| s == 128),
                        "{width}x{height} {chroma:?} {component:?} row {row}: {samples:?}"
                    );
                    assert!(
                        padding.iter().all(|&p| p == 0),
                        "{width}x{height} {chroma:?} {component:?} row {row}, padding {padding:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn no_row_is_handed_out_once_the_device_is_lost() {
        // A context of its own, whose record of a loss is written by hand.
        if test_context().is_none() {
            return;
        }
        let context = GpuContext::new(&crate::gpu::GpuOptions::default()).expect("a context");
        let chroma = ChromaSubsampling::Yuv420;
        let mut decoder = Decoder::new(&context, 64, 32, chroma).unwrap();
        decoder
            .assemble_complete(&empty_frame(64, 32, chroma))
            .unwrap();

        // As the callback of a device lost while the frame is on the GPU
        // writes it during the wait. run_gpu_on_device does not look before
        // the work, so only the check after the wait can hold the rows back.
        *context.lost_record().lock().unwrap() = Some("lost during the frame".to_owned());
        let mut handed_out = 0;
        for read in [Read::Yuv, Read::Rgb] {
            let color = (read == Read::Rgb).then(ColorParams::default);
            decoder
                .run_gpu_on_device(color.as_ref(), read, false, Duration::ZERO, &mut |_| {
                    handed_out += 1;
                })
                .expect("the work itself succeeds");
        }
        assert_eq!(handed_out, 0);

        // And the methods report the loss.
        let outcome = decoder.decode_assembled_rows(|_, _, _| handed_out += 1);
        assert!(
            matches!(outcome, Err(GpuError::DeviceLost(_))),
            "{outcome:?}"
        );
        assert_eq!(handed_out, 0);
    }

    #[test]
    fn rgba_to_rgb_drops_alpha_until_either_side_runs_out() {
        let rgba: Vec<u8> = (0..4 * 37).map(|i| (i * 7 + 3) as u8).collect();
        let expected: Vec<u8> = rgba
            .chunks_exact(4)
            .flat_map(|pixel| pixel[..3].to_vec())
            .collect();
        let mut rgb = vec![0; 3 * 37];
        rgba_to_rgb(&rgba, &mut rgb);
        assert_eq!(rgb, expected);

        // A shorter output takes the first pixels and leaves the input be; a
        // shorter input leaves the rest of the output alone.
        let mut rgb = vec![0; 3 * 5];
        rgba_to_rgb(&rgba, &mut rgb);
        assert_eq!(rgb, expected[..15]);
        let mut rgb = vec![9; 3 * 40];
        rgba_to_rgb(&rgba, &mut rgb);
        assert_eq!(rgb[..3 * 37], expected);
        assert!(rgb[3 * 37..].iter().all(|&byte| byte == 9));
    }

    #[test]
    fn the_payload_buffer_grows_by_doubling_but_never_past_the_limit() {
        const MIB: u64 = 1024 * 1024;

        // Small frames get the minimum, then powers of two.
        assert_eq!(
            payload_capacity_bytes(10, 128 * MIB),
            Some(MIN_PAYLOAD_WORDS as u64 * 4)
        );
        assert_eq!(payload_capacity_bytes(20_000, 128 * MIB), Some(32_768 * 4));

        // A payload that fits the limit but whose power of two does not gets
        // exactly its size.
        let needed_words = (70 * MIB / 4) as usize;
        assert_eq!(
            payload_capacity_bytes(needed_words, 100 * MIB),
            Some(100 * MIB)
        );
        // A limit that is not a multiple of four is rounded down to whole words.
        assert_eq!(
            payload_capacity_bytes(needed_words, 100 * MIB + 3),
            Some(100 * MIB)
        );

        // About 200 MiB, a 4K frame of blocks of the largest size the format
        // allows, does not fit the WebGPU default of 128 MiB but fits a larger limit.
        let huge_words = (12_429u64 * 4095) as usize;
        assert_eq!(payload_capacity_bytes(huge_words, 128 * MIB), None);
        assert_eq!(
            payload_capacity_bytes(huge_words, 4095 * MIB),
            Some(256 * MIB)
        );

        // Exactly the limit fits, one word more does not.
        let words = (128 * MIB / 4) as usize;
        assert_eq!(payload_capacity_bytes(words, 128 * MIB), Some(128 * MIB));
        assert_eq!(payload_capacity_bytes(words + 1, 128 * MIB), None);
    }

    #[test]
    fn the_last_level_rounds_ties_to_even_like_the_cpu() {
        let Some(context) = test_context() else {
            return;
        };

        // (k + 0.5) / 255 for k in 0..255: scaled back by 255 it lands on k + 0.5
        // whenever the float arithmetic allows, which is a tie. Then values
        // outside [0, 1], which are clamped.
        let mut values: Vec<f32> = (0..255).map(|k| (k as f32 + 0.5) / 255.0).collect();
        values.extend([1.5, -0.25, 1.0, 0.0, f32::MIN_POSITIVE, 0.999_999]);

        let expected: Vec<u8> = values
            .iter()
            .map(|v| (v.clamp(0.0, 1.0) * 255.0).round_ties_even() as u8)
            .collect();
        let ties = values
            .iter()
            .take(255)
            .enumerate()
            .filter(|(k, v)| **v * 255.0 == *k as f32 + 0.5)
            .count();
        // Round-half-up would get half of the ties wrong, so there must be plenty.
        assert!(ties >= 40, "only {ties} exact ties");

        assert_eq!(to_unorm8_on_gpu(context, &values), expected);
    }
}
