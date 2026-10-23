//! The wgpu decoder: uploads a frame's blocks and runs the GPU stages.
//!
//! The split follows upstream. The CPU assembles packets
//! ([`FrameAssembler`]); the GPU decodes coefficients and dequantises them
//! (`dequant.wgsl`), runs the inverse transform level by level (`idwt.wgsl`),
//! and converts the planes to 8 bits (`pack.wgsl`). On request a last stage
//! converts YCbCr to RGB (`yuv2rgb.wgsl`).

use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use wgpu::util::DeviceExt;

use crate::bitstream::consts::DECOMPOSITION_LEVELS;
use crate::bitstream::{Band, ChromaSubsampling, Component, FrameAssembler, FrameLayout};
use crate::color::ColorParams;

use super::context::GpuContext;
use super::error::GpuError;
use super::layout::{BufferLayout, DequantParams, IdwtLevel, PackParams, YuvParams};
use super::pipelines::{Pipelines, Stage, with_error_scope};

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
    /// The inverse transform, all levels.
    pub idwt: Duration,
    /// Conversion of the planes to 8 bits.
    pub pack: Duration,
    /// Colour conversion, if it was part of the decode.
    pub yuv2rgb: Option<Duration>,
}

impl StageTimings {
    /// The sum of the stages.
    pub fn total(&self) -> Duration {
        self.dequant + self.idwt + self.pack + self.yuv2rgb.unwrap_or_default()
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

/// Number of timestamps: a begin and an end for each of four passes.
const TIMESTAMP_QUERIES: u32 = 8;

/// What it takes to read timestamps.
struct TimestampResources {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// Nanoseconds per timestamp tick.
    period_ns: f64,
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
    pyramid: wgpu::Buffer,
    packed: wgpu::Buffer,
    readback: wgpu::Buffer,

    pipelines: Arc<Pipelines>,

    dequant_params: wgpu::Buffer,
    dequant_bind_group: wgpu::BindGroup,
    dequant_grid: (u32, u32),

    idwt_stages: Vec<IdwtStage>,

    pack_bind_group: wgpu::BindGroup,
    pack_grid: (u32, u32, u32),

    color: Option<ColorResources>,
    timestamps: Option<TimestampResources>,

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
        let layout = FrameLayout::new(width, height, chroma)?;
        let buffers = BufferLayout::new(&layout)?;
        let device = context.device().clone();
        let queue = context.queue().clone();
        // Compiled once per context, however many decoders are created.
        let pipelines = context.pipelines()?;

        let check = |what: &'static str, bytes: u64| {
            let limit = buffer_limit(&device);
            if bytes > limit {
                Err(GpuError::LimitExceeded {
                    what,
                    needed: bytes,
                    limit,
                })
            } else {
                Ok(())
            }
        };
        check("the wavelet pyramid", buffers.pyramid_floats * 4)?;
        check("the float planes", buffers.plane_floats * 4)?;
        check("the packed planes", buffers.packed_words * 4)?;

        with_error_scope(&device, || {
            let storage = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: bytes.max(16),
                    usage,
                    mapped_at_creation: false,
                })
            };

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
            let planes = storage(
                "float planes",
                buffers.plane_floats * 4,
                wgpu::BufferUsages::STORAGE,
            );
            let packed = storage(
                "packed planes",
                buffers.packed_words * 4,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            let readback = storage(
                "readback",
                buffers.packed_words * 4,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );

            let dequant_params_data = DequantParams::new(&layout, &buffers);
            let dequant_grid = dequant_params_data.grid();
            let dequant_params = uniform_buffer(&device, "dequant params", &dequant_params_data);
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
                    let stage = IdwtLevel::new(&layout, &buffers, level);
                    let params = uniform_buffer(&device, "idwt params", &stage.params);
                    IdwtStage {
                        bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("idwt"),
                            layout: &pipelines.idwt.layout,
                            entries: &[entry(0, &pyramid), entry(1, &planes), entry(2, &params)],
                        }),
                        grid: stage.grid(),
                    }
                })
                .collect();

            let pack_params_data = PackParams::new(&layout, &buffers);
            let pack_grid = pack_params_data.grid();
            let pack_params = uniform_buffer(&device, "pack params", &pack_params_data);
            let pack_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pack"),
                layout: &pipelines.pack.layout,
                entries: &[entry(0, &planes), entry(1, &packed), entry(2, &pack_params)],
            });

            let timestamps = context.supports_timestamps().then(|| TimestampResources {
                query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("stage timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: TIMESTAMP_QUERIES,
                }),
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("timestamps"),
                    size: u64::from(TIMESTAMP_QUERIES) * 8,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                readback: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("timestamp readback"),
                    size: u64::from(TIMESTAMP_QUERIES) * 8,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                period_ns: f64::from(queue.get_timestamp_period()),
            });

            Self {
                device: device.clone(),
                queue: queue.clone(),
                assembler: FrameAssembler::new(layout.clone()),
                buffers: buffers.clone(),
                payload,
                payload_capacity_words,
                block_offsets,
                pyramid,
                packed,
                readback,
                pipelines: Arc::clone(&pipelines),
                dequant_params,
                dequant_bind_group,
                dequant_grid,
                idwt_stages,
                pack_bind_group,
                pack_grid,
                color: None,
                timestamps,
                lost: context.lost_record(),
            }
        })
    }

    /// An error if the device has been lost, so that nothing is decoded on it
    /// and no stale result is returned.
    fn check_device(&self) -> Result<(), GpuError> {
        match self.lost.lock().ok().and_then(|lost| lost.clone()) {
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
        let (_, timings) = self.run_gpu(color.as_ref(), Read::Nothing, true, assembled)?;
        Ok(timings)
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

    /// Decodes the frame assembled from [`push_packet`](Self::push_packet)
    /// calls. Blocks that did not arrive decode as zeros.
    pub fn decode_assembled(&mut self) -> Result<Frame, GpuError> {
        let (bytes, _) = self.run_gpu(None, Read::Yuv, false, Duration::ZERO)?;
        Ok(self.unpack(&bytes))
    }

    /// Like [`decode_assembled`](Self::decode_assembled), converted to RGB.
    pub fn decode_assembled_rgb(&mut self, color: &ColorParams) -> Result<RgbFrame, GpuError> {
        let (bytes, _) = self.run_gpu(Some(color), Read::Rgb, false, Duration::ZERO)?;
        Ok(self.unpack_rgb(&bytes))
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
    /// the CPU: the bytes are returned, empty for [`Read::Nothing`].
    /// `assembled_in` is the CPU time already spent assembling the frame. With
    /// `timed` the stages are timed, if the device can.
    fn run_gpu(
        &mut self,
        color: Option<&ColorParams>,
        read: Read,
        timed: bool,
        assembled_in: Duration,
    ) -> Result<(Vec<u8>, Timings), GpuError> {
        self.check_device()?;
        let outcome = self.run_gpu_on_device(color, read, timed, assembled_in);
        // The device can be lost while the frame is on it, and wgpu tells
        // during the wait. Whatever failed then is the loss, and whatever came
        // back is not to be trusted.
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
    ) -> Result<(Vec<u8>, Timings), GpuError> {
        let prepare_start = Instant::now();
        self.ensure_payload_capacity()?;
        // In an error scope like all GPU work, so that a failed copy comes back
        // as an error instead of reaching wgpu's default handler, which panics.
        with_error_scope(&self.device, || self.upload())?;
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

            // Timestamps go at the boundaries of a pass: queries 2n and 2n + 1 for pass n.
            let writes = |pass: u32| {
                queries.map(|q| wgpu::ComputePassTimestampWrites {
                    query_set: &q.query_set,
                    beginning_of_pass_write_index: Some(2 * pass),
                    end_of_pass_write_index: Some(2 * pass + 1),
                })
            };

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

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pack"),
                    timestamp_writes: writes(2),
                });
                pass.set_pipeline(&self.pipelines.pack.pipeline);
                pass.set_bind_group(0, &self.pack_bind_group, &[]);
                pass.dispatch_workgroups(self.pack_grid.0, self.pack_grid.1, self.pack_grid.2);
            }

            if let (Some(_), Some(resources)) = (color, self.color.as_ref()) {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("yuv2rgb"),
                    timestamp_writes: writes(3),
                });
                pass.set_pipeline(&self.pipelines.yuv2rgb.pipeline);
                pass.set_bind_group(0, &resources.bind_group, &[]);
                pass.dispatch_workgroups(resources.grid.0, resources.grid.1, 1);
            }

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

            if let Some(q) = queries {
                encoder.resolve_query_set(&q.query_set, 0..TIMESTAMP_QUERIES, &q.resolve, 0);
                encoder.copy_buffer_to_buffer(
                    &q.resolve,
                    0,
                    &q.readback,
                    0,
                    u64::from(TIMESTAMP_QUERIES) * 8,
                );
            }
            self.queue.submit([encoder.finish()]);
        })?;

        let bytes = match (read, self.color.as_ref()) {
            (Read::Yuv, _) => self.read_back(&self.readback, self.buffers.packed_words * 4)?,
            (Read::Rgb, Some(resources)) => {
                self.read_back(&resources.readback, resources.rgba.size())?
            }
            _ => {
                self.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .map_err(|error| GpuError::Readback(error.to_string()))?;
                Vec::new()
            }
        };
        let gpu_wall = gpu_start.elapsed();

        let stages = match queries {
            Some(q) => self.read_stage_timings(q, color.is_some())?,
            None => None,
        };

        self.assembler.mark_frame_decoded();
        Ok((
            bytes,
            Timings {
                prepare,
                gpu_wall,
                stages,
            },
        ))
    }

    /// Creates the buffers of the colour conversion on first use and writes
    /// the parameters, which may change from call to call.
    fn prepare_color(&mut self, color: &ColorParams) -> Result<(), GpuError> {
        let params = YuvParams::new(self.assembler.layout(), &self.buffers, color);

        if self.color.is_none() {
            let bytes = u64::from(params.width) * u64::from(params.height) * 4;
            let limit = buffer_limit(&self.device);
            if bytes > limit {
                return Err(GpuError::LimitExceeded {
                    what: "the RGB image",
                    needed: bytes,
                    limit,
                });
            }

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
        queries: &TimestampResources,
        with_rgb: bool,
    ) -> Result<Option<StageTimings>, GpuError> {
        let bytes = self.read_back(&queries.readback, u64::from(TIMESTAMP_QUERIES) * 8)?;
        let ticks: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();

        // A device that wrote nothing leaves zeros. A pass shorter than one tick
        // of the timer is a valid measurement of zero, not a missing one.
        if ticks.iter().all(|&tick| tick == 0) {
            return Ok(None);
        }
        let stage = |pass: usize| {
            let (begin, end) = (ticks[2 * pass], ticks[2 * pass + 1]);
            (end >= begin)
                .then(|| Duration::from_secs_f64((end - begin) as f64 * queries.period_ns * 1e-9))
        };
        Ok(match (stage(0), stage(1), stage(2)) {
            (Some(dequant), Some(idwt), Some(pack)) => Some(StageTimings {
                dequant,
                idwt,
                pack,
                yuv2rgb: if with_rgb { stage(3) } else { None },
            }),
            _ => None,
        })
    }

    /// Turns RGBA words into tight RGB bytes.
    fn unpack_rgb(&self, rgba: &[u8]) -> RgbFrame {
        let layout = self.layout();
        let mut data = Vec::with_capacity(rgba.len() / 4 * 3);
        for pixel in rgba.chunks_exact(4) {
            data.extend_from_slice(&pixel[..3]);
        }
        RgbFrame {
            width: layout.width(),
            height: layout.height(),
            data,
        }
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
        .and_then(|staging| self.read_back(&staging, bytes));
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

        self.payload = payload;
        self.payload_capacity_words = (capacity_bytes / 4) as usize;
        self.dequant_bind_group = bind_group;
        Ok(())
    }

    /// Uploads the block table and the payload of the assembled frame.
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

    /// Waits for the GPU and copies a mappable buffer's first `bytes` bytes out.
    fn read_back(&self, buffer: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>, GpuError> {
        // Mapping and unmapping report errors through the error scopes like
        // the other calls. Waiting cannot: wgpu treats a device error while
        // polling as fatal (decision D22).
        with_error_scope(&self.device, || {
            let slice = buffer.slice(..bytes);
            let (sender, receiver) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                // The receiver is only gone if this function already returned.
                let _ = sender.send(result);
            });

            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|error| GpuError::Readback(error.to_string()))?;
            receiver
                .recv()
                .map_err(|error| GpuError::Readback(error.to_string()))?
                .map_err(|error| GpuError::Readback(error.to_string()))?;

            let data = slice
                .get_mapped_range()
                .map_err(|error| GpuError::Readback(error.to_string()))?
                .to_vec();
            buffer.unmap();
            Ok(data)
        })?
    }

    /// Cuts the packed planes (rows padded to whole words) into tight planes.
    fn unpack(&self, packed: &[u8]) -> Frame {
        let layout = self.layout();
        let planes = Component::ALL.map(|component| {
            let i = component.index();
            let (width, height) = layout.plane_size(component);
            let stride = self.buffers.packed_strides[i] as usize * 4;
            let start = self.buffers.packed_offsets[i] as usize * 4;

            let mut data = Vec::with_capacity(width as usize * height as usize);
            for row in 0..height as usize {
                let begin = start + row * stride;
                data.extend_from_slice(&packed[begin..begin + width as usize]);
            }
            FramePlane {
                width,
                height,
                data,
            }
        });

        Frame {
            chroma: layout.chroma(),
            planes,
        }
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

/// The most bytes one buffer can have and still be bound as a storage buffer.
fn buffer_limit(device: &wgpu::Device) -> u64 {
    let limits = device.limits();
    limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size)
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn uniform_buffer<T: bytemuck::Pod>(device: &wgpu::Device, label: &str, value: &T) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(value),
        usage: wgpu::BufferUsages::UNIFORM,
    })
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
    use super::*;
    use crate::gpu::GpuOptions;

    /// A context, or `None` if there is no GPU and none is required.
    fn context() -> Option<GpuContext> {
        match GpuContext::new(&GpuOptions::default()) {
            Ok(context) => Some(context),
            Err(error) => {
                let required =
                    std::env::var_os("EMBERWAVE_REQUIRE_GPU").is_some_and(|v| !v.is_empty());
                assert!(
                    !required,
                    "a GPU is required but none is available: {error}"
                );
                eprintln!("no GPU available, skipping: {error}");
                None
            }
        }
    }

    /// Runs the packing stage on `values` as the first samples of the luma
    /// plane of a 256x4 picture and returns the bytes it produces for them.
    fn pack(context: &GpuContext, values: &[f32]) -> Vec<u8> {
        let layout = FrameLayout::new(256, 4, ChromaSubsampling::Yuv444).unwrap();
        let buffers = BufferLayout::new(&layout).unwrap();
        let pipelines = context.pipelines().unwrap();
        let device = context.device();

        let mut planes = vec![0.0f32; buffers.plane_floats as usize];
        planes[..values.len()].copy_from_slice(values);
        let planes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planes"),
            contents: bytemuck::cast_slice(&planes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let packed_bytes = buffers.packed_words * 4;
        let packed = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("packed"),
            size: packed_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: packed_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = PackParams::new(&layout, &buffers);
        let params_buffer = uniform_buffer(device, "pack params", &params);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pack"),
            layout: &pipelines.pack.layout,
            entries: &[
                entry(0, &planes_buffer),
                entry(1, &packed),
                entry(2, &params_buffer),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipelines.pack.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let (x, y, z) = params.grid();
            pass.dispatch_workgroups(x, y, z);
        }
        encoder.copy_buffer_to_buffer(&packed, 0, &readback, 0, packed_bytes);
        context.queue().submit([encoder.finish()]);

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| result.expect("map"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let data = slice.get_mapped_range().expect("range");
        data[..values.len()].to_vec()
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
    fn packing_rounds_ties_to_even_like_the_cpu() {
        let Some(context) = context() else {
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

        assert_eq!(pack(&context, &values), expected);
    }
}
