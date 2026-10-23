//! The wgpu encoder: the CPU encoder's stages ([`crate::encode`]) as compute
//! shaders, so that a picture goes up once and its frame comes back once.
//!
//! `fdwt.wgsl` transforms the planes level by level, from the finest.
//! `equant.wgsl` quantises every block and measures it with 0 to 15
//! bit-planes dropped. `rate.wgsl`, with the prefix sums of `scan.wgsl`,
//! chooses how many each block drops so that the frame fits its budget, and
//! `pack.wgsl` writes the blocks one after the other. The CPU adds the start
//! of frame header.
//!
//! Given the CPU encoder's coefficients, every stage after the transform
//! makes the CPU encoder's choices; the transform itself may differ in the
//! last place of a coefficient, so a frame is the CPU encoder's frame or
//! one that differs in a few of its blocks (decision D29).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::bitstream::consts::{DECOMPOSITION_LEVELS, HEADER_BYTES};
use crate::bitstream::{ChromaSubsampling, Component, FrameLayout, SequenceHeader};
use crate::encode::fdwt::sample_level;
use crate::encode::{
    EncodeError, Signalling, check_planes, next_sequence, start_of_frame, stream_header,
};

use super::context::GpuContext;
use super::error::GpuError;
use super::layout::{
    BufferLayout, CostRecord, FrameParams, MAX_BLOCK_WORDS, QuantParams, RATE_KEYS,
    RATE_STATE_BYTES, RateSummary, ScanParams, TransformLevel, rate_grid,
};
use super::pipelines::{EncoderPipelines, with_error_scope};
use super::resources::{
    StageQueries, buffer, buffer_limit, check_size, entry, read_back, uniform_buffer,
};

/// A dispatch of the forward transform for one decomposition level.
struct FdwtStage {
    bind_group: wgpu::BindGroup,
    grid: (u32, u32, u32),
}

/// GPU time of each stage of an encode, measured with timestamp queries at
/// the boundaries of its compute passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeStageTimings {
    /// The forward transform, all levels.
    pub transform: Duration,
    /// Quantisation and the cost tables.
    pub quantisation: Duration,
    /// The rate control, its prefix sums included.
    pub rate_control: Duration,
    /// The packing of the blocks.
    pub packing: Duration,
}

impl EncodeStageTimings {
    /// The sum of the stages.
    pub fn total(&self) -> Duration {
        self.transform + self.quantisation + self.rate_control + self.packing
    }
}

/// Where the time of one encode went.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeTimings {
    /// CPU time to check the planes and put them where wgpu uploads them
    /// from, and to prepare the GPU work.
    pub prepare: Duration,
    /// Wall-clock time from submitting the GPU work until the frame was back
    /// in CPU memory: the upload, the stages and the read-back.
    pub gpu_wall: Duration,
    /// GPU time per stage, if the context was created with
    /// [`GpuOptions::timestamps`](super::GpuOptions::timestamps) and the
    /// device supports them.
    pub stages: Option<EncodeStageTimings>,
}

/// The stages that are timed, in order: transform, quantisation, rate
/// control and packing.
const TIMED_STAGES: u32 = 4;
const TRANSFORM: u32 = 0;
const QUANTISATION: u32 = 1;
const RATE_CONTROL: u32 = 2;
const PACKING: u32 = 3;

/// The payload buffer never holds fewer words than this.
const MIN_PAYLOAD_WORDS: u64 = 16 * 1024;

/// Bytes at the start of the read-back buffer: the rate control's summary,
/// the payload's size in words and padding to 8 bytes.
const SUMMARY_BYTES: u64 = std::mem::size_of::<RateSummary>() as u64 + 8;

/// Encodes pictures of one size and chroma format into PyroWave frames on
/// the GPU, as [`crate::encode::Encoder`] does on the CPU.
///
/// All GPU memory is allocated up front and reused for every picture, apart
/// from the payload buffer, which grows when a frame's budget needs more.
pub struct Encoder {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: FrameLayout,
    buffers: BufferLayout,
    signalling: Signalling,
    /// Sequence number of the last frame, modulo 8.
    sequence: u8,
    pipelines: Arc<EncoderPipelines>,

    /// The 8-bit planes, rows padded to whole words, as the decoder's.
    planes: wgpu::Buffer,
    /// [`sample_level`] of every 8-bit value, for `fdwt.wgsl`.
    sample_levels: wgpu::Buffer,
    pyramid: wgpu::Buffer,
    /// A [`CostRecord`] per block, in block index order.
    costs: wgpu::Buffer,
    /// The rate control's histogram and [`RateSummary`].
    rate_state: wgpu::Buffer,
    /// Per block, its first drop and its steps (`rate.wgsl`).
    steps: wgpu::Buffer,
    /// Per block, what its steps of the cutoff key save, then the sum of
    /// those of the blocks before it.
    tie_savings: wgpu::Buffer,
    /// Per block, the planes it drops.
    drops: wgpu::Buffer,
    /// Per block, its words with that drop, then its offset in the payload.
    block_words: wgpu::Buffer,
    /// The totals of the chunks of a scan, then the sum of everything.
    chunk_totals: wgpu::Buffer,
    frame_params: wgpu::Buffer,
    quant_params: wgpu::Buffer,
    /// The blocks of a frame, one after the other.
    payload: wgpu::Buffer,
    payload_capacity_words: u64,
    /// The summary, then the payload.
    readback: wgpu::Buffer,

    fdwt_stages: Vec<FdwtStage>,
    quant_bind_group: wgpu::BindGroup,
    /// One workgroup per block: `equant.wgsl` and `pack.wgsl`.
    block_grid: (u32, u32),
    rate_bind_group: wgpu::BindGroup,
    rate_grid: (u32, u32),
    scan_ties: wgpu::BindGroup,
    scan_words: wgpu::BindGroup,
    scan_params: ScanParams,
    pack_bind_group: wgpu::BindGroup,

    timestamps: Option<StageQueries>,

    /// Why the device was lost, once it has been (see [`GpuContext::lost`]).
    lost: Arc<Mutex<Option<String>>>,
}

impl Encoder {
    /// Creates an encoder for pictures of `width` x `height` pixels: from 1
    /// to 16384 each, and even for 4:2:0. `signalling` is what the frames say
    /// about their colours.
    ///
    /// Allocates the buffers and builds the pipelines.
    pub fn new(
        context: &GpuContext,
        width: u32,
        height: u32,
        chroma: ChromaSubsampling,
        signalling: Signalling,
    ) -> Result<Self, GpuError> {
        let layout = FrameLayout::new(width, height, chroma)?;
        let buffers = BufferLayout::new(&layout)?;
        let scan_params = ScanParams::new(layout.block_count())?;
        let device = context.device().clone();
        let queue = context.queue().clone();
        // Compiled once per context, however many encoders are created.
        let pipelines = context.encoder_pipelines()?;

        let blocks = u64::from(layout.block_count());
        let cost_bytes = blocks * std::mem::size_of::<CostRecord>() as u64;
        let step_bytes = blocks * 16 * 4;
        check_size(&device, "the wavelet pyramid", buffers.pyramid_floats * 4)?;
        check_size(&device, "the packed planes", buffers.packed_words * 4)?;
        check_size(&device, "the cost tables", cost_bytes)?;
        check_size(&device, "the rate control's steps", step_bytes)?;

        with_error_scope(&device, || {
            let storage = |label: &str, bytes: u64, usage| buffer(&device, label, bytes, usage);
            // Tests write and read the buffers between the stages.
            let usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST;
            let planes = storage("planes", buffers.packed_words * 4, usage);
            let levels: Vec<f32> = (0..=u8::MAX).map(sample_level).collect();
            let sample_levels = storage("sample levels", levels.len() as u64 * 4, usage);
            queue.write_buffer(&sample_levels, 0, bytemuck::cast_slice(&levels));
            let pyramid = storage("wavelet pyramid", buffers.pyramid_floats * 4, usage);
            let costs = storage("cost tables", cost_bytes, usage);
            let rate_state = storage("rate control", RATE_STATE_BYTES, usage);
            let steps = storage("steps", step_bytes, usage);
            let tie_savings = storage("tie savings", blocks * 4, usage);
            let drops = storage("drops", blocks * 4, usage);
            let block_words = storage("block words", blocks * 4, usage);
            let chunk_totals = storage(
                "chunk totals",
                (u64::from(scan_params.chunk_count) + 1) * 4,
                usage,
            );
            let frame_params = storage(
                "frame params",
                std::mem::size_of::<FrameParams>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            let quant_params_data = QuantParams::new(&layout, &buffers);
            let quant_params = uniform_buffer(&device, "equant params", &quant_params_data);
            let scan_params_buffer = uniform_buffer(&device, "scan params", &scan_params);
            let payload_capacity_words = MIN_PAYLOAD_WORDS;
            let (payload, readback) = payload_buffers(&device, payload_capacity_words);

            // Levels from the finest to the coarsest: each splits the LL band
            // its predecessor wrote.
            let fdwt_stages = (0..DECOMPOSITION_LEVELS)
                .map(|level| {
                    let stage = TransformLevel::new(&layout, &buffers, level);
                    let params = uniform_buffer(&device, "fdwt params", &stage.params);
                    FdwtStage {
                        bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("fdwt"),
                            layout: &pipelines.fdwt.layout,
                            entries: &[
                                entry(0, &pyramid),
                                entry(1, &planes),
                                entry(2, &params),
                                entry(3, &sample_levels),
                            ],
                        }),
                        grid: stage.grid(),
                    }
                })
                .collect();

            let quant_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("equant"),
                layout: &pipelines.equant.layout,
                entries: &[
                    entry(0, &pyramid),
                    entry(1, &costs),
                    entry(2, &quant_params),
                ],
            });
            let rate_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rate"),
                layout: &pipelines.rate.analyse.layout,
                entries: &[
                    entry(0, &costs),
                    entry(1, &rate_state),
                    entry(2, &steps),
                    entry(3, &tie_savings),
                    entry(4, &drops),
                    entry(5, &block_words),
                    entry(6, &frame_params),
                ],
            });
            let scan_bind_group = |values: &wgpu::Buffer| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("scan"),
                    layout: &pipelines.scan.chunks.layout,
                    entries: &[
                        entry(0, values),
                        entry(1, &chunk_totals),
                        entry(2, &scan_params_buffer),
                    ],
                })
            };
            let scan_ties = scan_bind_group(&tie_savings);
            let scan_words = scan_bind_group(&block_words);
            let pack_bind_group = PackBuffers {
                pyramid: &pyramid,
                costs: &costs,
                drops: &drops,
                offsets: &block_words,
                payload: &payload,
                quant_params: &quant_params,
                frame_params: &frame_params,
            }
            .bind_group(&device, &pipelines);

            Self {
                device: device.clone(),
                queue: queue.clone(),
                layout: layout.clone(),
                buffers: buffers.clone(),
                signalling,
                sequence: 0,
                pipelines: Arc::clone(&pipelines),
                planes,
                sample_levels,
                pyramid,
                costs,
                rate_state,
                steps,
                tie_savings,
                drops,
                block_words,
                chunk_totals,
                frame_params,
                quant_params,
                payload,
                payload_capacity_words,
                readback,
                fdwt_stages,
                quant_bind_group,
                block_grid: quant_params_data.grid(),
                rate_bind_group,
                rate_grid: rate_grid(&layout),
                scan_ties,
                scan_words,
                scan_params,
                pack_bind_group,
                timestamps: context
                    .supports_timestamps()
                    .then(|| StageQueries::new(&device, &queue, TIMED_STAGES)),
                lost: context.lost_record(),
            }
        })
    }

    /// The block layout of the frames.
    pub fn layout(&self) -> &FrameLayout {
        &self.layout
    }

    /// The stream as a PWV1 file header describes it: size, chroma format
    /// and signalling, with the per-frame fields zero.
    pub fn stream(&self) -> SequenceHeader {
        stream_header(&self.layout, &self.signalling)
    }

    /// Why the device of this encoder was lost, if it was. See
    /// [`GpuContext::lost`].
    pub fn lost(&self) -> Option<String> {
        self.lost.lock().ok().and_then(|lost| lost.clone())
    }

    /// Encodes one picture into a frame of at most `budget` bytes: a start of
    /// frame header and the coded blocks in block index order, as a PWV1 file
    /// stores a frame and as [`crate::encode::Encoder::encode`] makes it.
    ///
    /// `planes` are the Y, Cb and Cr samples, row-major without padding, at
    /// the sizes of [`FrameLayout::plane_size`]. A picture that cannot fit
    /// the budget is [`GpuError::Encode`]. It waits for the GPU, so it cannot
    /// run in a browser, where nothing may block.
    pub fn encode(&mut self, planes: [&[u8]; 3], budget: usize) -> Result<Vec<u8>, GpuError> {
        self.run_encode(planes, budget, false)
            .map(|(frame, _)| frame)
    }

    /// Encodes like [`encode`](Self::encode) and says where the time went,
    /// for benchmarks. The per-stage times are `Some` only if the context was
    /// created with [`GpuOptions::timestamps`](super::GpuOptions::timestamps)
    /// and the device supports them.
    pub fn encode_timed(
        &mut self,
        planes: [&[u8]; 3],
        budget: usize,
    ) -> Result<(Vec<u8>, EncodeTimings), GpuError> {
        self.run_encode(planes, budget, true)
    }

    fn run_encode(
        &mut self,
        planes: [&[u8]; 3],
        budget: usize,
        timed: bool,
    ) -> Result<(Vec<u8>, EncodeTimings), GpuError> {
        self.check_device()?;
        let outcome = self.encode_on_device(planes, budget, timed);
        // As in the decoder: a failure on a device that was lost meanwhile
        // is the loss, and whatever came back is not to be trusted.
        self.check_device()?;
        outcome
    }

    /// The work of [`run_encode`](Self::run_encode), without the checks for
    /// a lost device.
    fn encode_on_device(
        &mut self,
        planes: [&[u8]; 3],
        budget: usize,
        timed: bool,
    ) -> Result<(Vec<u8>, EncodeTimings), GpuError> {
        let prepare_start = Instant::now();
        self.upload_planes(planes)?;
        // No frame takes more than every block at its largest, which fits
        // 32 bits; neither need the budget.
        let most_words = u64::from(self.layout.block_count()) * MAX_BLOCK_WORDS;
        let budget_words = ((budget.saturating_sub(HEADER_BYTES) / 4) as u64).min(most_words);
        self.ensure_payload_capacity(budget_words)?;
        let capacity_words = budget_words.min(self.payload_capacity_words);
        let sequence = next_sequence(self.sequence);
        let params = FrameParams {
            block_count: self.layout.block_count(),
            grid_width: self.rate_grid.0,
            budget_words: budget_words as u32,
            sequence: u32::from(sequence),
            capacity_words: capacity_words as u32,
            pad: [0; 3],
        };

        with_error_scope(&self.device, || {
            self.queue
                .write_buffer(&self.frame_params, 0, bytemuck::bytes_of(&params));
        })?;
        let prepare = prepare_start.elapsed();

        let queries = self.timestamps.as_ref().filter(|_| timed);
        let gpu_start = Instant::now();
        with_error_scope(&self.device, || {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encode"),
                });
            encoder.clear_buffer(&self.rate_state, 0, None);
            self.encode_transform(&mut encoder, queries);
            self.encode_quantisation(&mut encoder, queries);
            self.encode_rate_control(&mut encoder, queries);
            self.encode_packing(&mut encoder, queries);
            self.encode_read_back(&mut encoder, capacity_words);
            self.queue.submit([encoder.finish()]);
        })?;
        let data = read_back(
            &self.device,
            &self.readback,
            SUMMARY_BYTES + capacity_words * 4,
        )?;
        let gpu_wall = gpu_start.elapsed();
        let stages = match queries {
            Some(q) => q.read(&self.device, &self.queue)?.and_then(|stages| {
                Some(EncodeStageTimings {
                    transform: stages[TRANSFORM as usize]?,
                    quantisation: stages[QUANTISATION as usize]?,
                    rate_control: stages[RATE_CONTROL as usize]?,
                    packing: stages[PACKING as usize]?,
                })
            }),
            None => None,
        };

        let summary: RateSummary =
            bytemuck::pod_read_unaligned(&data[..std::mem::size_of::<RateSummary>()]);
        if summary.fits == 0 || budget < HEADER_BYTES {
            return Err(EncodeError::BudgetTooSmall {
                budget,
                minimum: HEADER_BYTES + summary.minimum as usize * 4,
            }
            .into());
        }
        let at = std::mem::size_of::<RateSummary>();
        let payload_words = u64::from(u32::from_le_bytes([
            data[at],
            data[at + 1],
            data[at + 2],
            data[at + 3],
        ]));
        if payload_words > budget_words {
            // The rate control never lets this happen; a frame over its
            // budget is not returned, whatever the GPU did.
            return Err(GpuError::Device(format!(
                "the GPU made a frame of {payload_words} words for a budget of {budget_words}"
            )));
        }
        if payload_words > capacity_words {
            // Only when the device cannot hold a buffer of the budget's size.
            return Err(GpuError::LimitExceeded {
                what: "the frame",
                needed: payload_words * 4,
                limit: capacity_words * 4,
            });
        }

        self.sequence = sequence;
        let payload = &data[SUMMARY_BYTES as usize..][..payload_words as usize * 4];
        let mut frame = Vec::with_capacity(HEADER_BYTES + payload.len());
        frame.extend_from_slice(&start_of_frame(&self.stream(), sequence, summary.sent));
        frame.extend_from_slice(payload);
        debug_assert!(frame.len() <= budget);
        Ok((
            frame,
            EncodeTimings {
                prepare,
                gpu_wall,
                stages,
            },
        ))
    }

    /// An error if the device has been lost, so that nothing is encoded on it
    /// and no stale result is returned.
    fn check_device(&self) -> Result<(), GpuError> {
        match self.lost() {
            Some(reason) => Err(GpuError::DeviceLost(reason)),
            None => Ok(()),
        }
    }

    /// Grows the payload buffer, if need be, so that it holds `words` words
    /// or as many as the device allows. It grows in powers of two, so that
    /// budgets that creep up do not reallocate every frame.
    fn ensure_payload_capacity(&mut self, words: u64) -> Result<(), GpuError> {
        if words <= self.payload_capacity_words {
            return Ok(());
        }
        let limit_words = buffer_limit(&self.device).saturating_sub(SUMMARY_BYTES) / 4;
        let most_words = u64::from(self.layout.block_count()) * MAX_BLOCK_WORDS;
        let capacity = words
            .next_power_of_two()
            .min(most_words)
            .min(limit_words)
            .max(self.payload_capacity_words);
        if capacity == self.payload_capacity_words {
            return Ok(());
        }
        let (payload, readback, bind_group) = with_error_scope(&self.device, || {
            let (payload, readback) = payload_buffers(&self.device, capacity);
            let bind_group = PackBuffers {
                pyramid: &self.pyramid,
                costs: &self.costs,
                drops: &self.drops,
                offsets: &self.block_words,
                payload: &payload,
                quant_params: &self.quant_params,
                frame_params: &self.frame_params,
            }
            .bind_group(&self.device, &self.pipelines);
            (payload, readback, bind_group)
        })?;
        // Work already submitted keeps the old buffers until it is done.
        std::mem::replace(&mut self.payload, payload).destroy();
        std::mem::replace(&mut self.readback, readback).destroy();
        self.payload_capacity_words = capacity;
        self.pack_bind_group = bind_group;
        Ok(())
    }

    /// Writes the planes into the planes buffer, rows padded to whole words,
    /// or returns an error if they do not have the sizes of the pictures.
    /// They go straight into wgpu's staging memory: one copy, not two.
    fn upload_planes(&mut self, planes: [&[u8]; 3]) -> Result<(), GpuError> {
        check_planes(&self.layout, planes)?;
        // Never zero: a picture has a sample at least.
        let Some(size) = wgpu::BufferSize::new(self.buffers.packed_words * 4) else {
            return Ok(());
        };
        with_error_scope(&self.device, || {
            // None only with a validation error, which the scope reports.
            let Some(mut view) = self.queue.write_buffer_with(&self.planes, 0, size) else {
                return;
            };
            for (component, plane) in Component::ALL.into_iter().zip(planes) {
                let i = component.index();
                let width = self.layout.plane_size(component).0 as usize;
                let stride = self.buffers.packed_strides[i] as usize * 4;
                let start = self.buffers.packed_offsets[i] as usize * 4;
                if width == stride {
                    view.slice(start..start + plane.len())
                        .copy_from_slice(plane);
                    continue;
                }
                for (row, samples) in plane.chunks_exact(width).enumerate() {
                    let at = start + row * stride;
                    view.slice(at..at + width).copy_from_slice(samples);
                    view.slice(at + width..at + stride).fill(0);
                }
            }
        })
    }

    /// Records the forward transform of the uploaded planes; with `queries`,
    /// timed.
    fn encode_transform(&self, encoder: &mut wgpu::CommandEncoder, queries: Option<&StageQueries>) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fdwt"),
            timestamp_writes: queries.map(|q| q.writes(Some(TRANSFORM), Some(TRANSFORM))),
        });
        pass.set_pipeline(&self.pipelines.fdwt.pipeline);
        for stage in &self.fdwt_stages {
            pass.set_bind_group(0, &stage.bind_group, &[]);
            pass.dispatch_workgroups(stage.grid.0, stage.grid.1, stage.grid.2);
        }
    }

    /// Records the quantisation of the coefficients in the pyramid, which
    /// fills the cost tables.
    fn encode_quantisation(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queries: Option<&StageQueries>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("equant"),
            timestamp_writes: queries.map(|q| q.writes(Some(QUANTISATION), Some(QUANTISATION))),
        });
        pass.set_pipeline(&self.pipelines.equant.pipeline);
        pass.set_bind_group(0, &self.quant_bind_group, &[]);
        pass.dispatch_workgroups(self.block_grid.0, self.block_grid.1, 1);
    }

    /// Records the rate control, which needs its state cleared: the drops
    /// and, in `block_words`, where each block goes in the payload.
    fn encode_rate_control(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queries: Option<&StageQueries>,
    ) {
        let rate = &self.pipelines.rate;
        let (x, y) = self.rate_grid;
        {
            // The stage begins with this pass and ends with the last scan.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rate"),
                timestamp_writes: queries.map(|q| q.writes(Some(RATE_CONTROL), None)),
            });
            pass.set_bind_group(0, &self.rate_bind_group, &[]);
            pass.set_pipeline(&rate.analyse.pipeline);
            pass.dispatch_workgroups(x, y, 1);
            pass.set_pipeline(&rate.cutoff.pipeline);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&rate.ties.pipeline);
            pass.dispatch_workgroups(x, y, 1);
        }
        self.encode_scan(encoder, &self.scan_ties, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rate"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.rate_bind_group, &[]);
            pass.set_pipeline(&rate.choose.pipeline);
            pass.dispatch_workgroups(x, y, 1);
        }
        self.encode_scan(
            encoder,
            &self.scan_words,
            queries.map(|q| q.writes(None, Some(RATE_CONTROL))),
        );
    }

    /// Records the exclusive prefix sum of the values `bind_group` binds.
    fn encode_scan(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let scan = &self.pipelines.scan;
        let (x, y) = self.scan_params.grid();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scan"),
            timestamp_writes,
        });
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_pipeline(&scan.chunks.pipeline);
        pass.dispatch_workgroups(x, y, 1);
        pass.set_pipeline(&scan.totals.pipeline);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&scan.add.pipeline);
        pass.dispatch_workgroups(x, y, 1);
    }

    /// Records the packing of the blocks into the payload.
    fn encode_packing(&self, encoder: &mut wgpu::CommandEncoder, queries: Option<&StageQueries>) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pack"),
            timestamp_writes: queries.map(|q| q.writes(Some(PACKING), Some(PACKING))),
        });
        pass.set_pipeline(&self.pipelines.pack.pipeline);
        pass.set_bind_group(0, &self.pack_bind_group, &[]);
        pass.dispatch_workgroups(self.block_grid.0, self.block_grid.1, 1);
    }

    /// Records the copies into the read-back buffer: the summary of the rate
    /// control, the payload's size from the last scan, and `words` words of
    /// the payload, all a frame within the budget can have.
    fn encode_read_back(&self, encoder: &mut wgpu::CommandEncoder, words: u64) {
        let summary_bytes = std::mem::size_of::<RateSummary>() as u64;
        encoder.copy_buffer_to_buffer(
            &self.rate_state,
            RATE_KEYS * 4,
            &self.readback,
            0,
            summary_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &self.chunk_totals,
            u64::from(self.scan_params.chunk_count) * 4,
            &self.readback,
            summary_bytes,
            4,
        );
        if words > 0 {
            encoder.copy_buffer_to_buffer(
                &self.payload,
                0,
                &self.readback,
                SUMMARY_BYTES,
                words * 4,
            );
        }
    }
}

/// What `pack.wgsl` binds.
struct PackBuffers<'a> {
    pyramid: &'a wgpu::Buffer,
    costs: &'a wgpu::Buffer,
    drops: &'a wgpu::Buffer,
    offsets: &'a wgpu::Buffer,
    payload: &'a wgpu::Buffer,
    quant_params: &'a wgpu::Buffer,
    frame_params: &'a wgpu::Buffer,
}

impl PackBuffers<'_> {
    fn bind_group(&self, device: &wgpu::Device, pipelines: &EncoderPipelines) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pack"),
            layout: &pipelines.pack.layout,
            entries: &[
                entry(0, self.pyramid),
                entry(1, self.costs),
                entry(2, self.drops),
                entry(3, self.offsets),
                entry(4, self.payload),
                entry(5, self.quant_params),
                entry(6, self.frame_params),
            ],
        })
    }
}

/// The payload buffer for `words` words and the read-back buffer that takes
/// it with the summary.
fn payload_buffers(device: &wgpu::Device, words: u64) -> (wgpu::Buffer, wgpu::Buffer) {
    let payload = buffer(
        device,
        "payload",
        words * 4,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let readback = buffer(
        device,
        "encoder readback",
        SUMMARY_BYTES + words * 4,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    (payload, readback)
}

impl std::fmt::Debug for Encoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Encoder")
            .field("layout", &self.layout)
            .field("signalling", &self.signalling)
            .field("payload_capacity_words", &self.payload_capacity_words)
            .finish_non_exhaustive()
    }
}

impl Drop for Encoder {
    /// Frees the GPU memory of the buffers at once, as the decoder does: in a
    /// browser dropping a buffer leaves its memory to the garbage collector.
    /// Work already submitted keeps its buffers until it is done.
    fn drop(&mut self) {
        for buffer in [
            &self.planes,
            &self.sample_levels,
            &self.pyramid,
            &self.costs,
            &self.rate_state,
            &self.steps,
            &self.tie_savings,
            &self.drops,
            &self.block_words,
            &self.chunk_totals,
            &self.payload,
            &self.readback,
        ] {
            buffer.destroy();
        }
    }
}

#[cfg(test)]
mod tests;
