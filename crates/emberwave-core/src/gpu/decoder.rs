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
use crate::bitstream::packet::{DEFAULT_MINIMUM_PACKET_RATIO, DEFAULT_PRISTINE_BANDS};
use crate::bitstream::{
    Band, ChromaSubsampling, Component, FrameAssembler, FrameLayout, SequenceHeader,
};
use crate::color::ColorParams;

use super::context::{GpuContext, notice_loss};
use super::error::GpuError;
use super::layout::{BufferLayout, DequantParams, TransformLevel, YuvParams};
use super::pipelines::{DrawPipeline, Pipelines, Stage, with_error_scope};
use super::resources::{
    StageQueries, buffer, buffer_limit, check_size, entry, map_for_writing, read_back,
    read_back_with, texture, uniform_buffer,
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

/// Where a decode puts the picture beyond the decoder's own buffers: back to
/// the CPU, or drawn into a texture, or its planes into three.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Read<'a> {
    Nothing,
    Yuv,
    Rgb,
    Texture(&'a wgpu::TextureView),
    Planes([&'a wgpu::TextureView; 3]),
}

/// Buffers of the colour conversion, created when it is first used.
struct ColorResources {
    rgba: wgpu::Buffer,
    /// Made by the first decode that reads RGB back: a player never does.
    readback: Option<wgpu::Buffer>,
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
    /// The level's parameters, which the bind groups of decodes that store
    /// the planes into textures take as well.
    params: wgpu::Buffer,
    /// Whether the level is the last of a component, which writes its plane.
    writes_plane: bool,
    grid: (u32, u32, u32),
}

/// How the planes of a decode into plane textures get there.
enum PlaneFill {
    /// Copies of the packed planes (decision D41).
    Copy,
    /// Stores of the inverse transform's last levels (decision D52): a bind
    /// group with the textures for each level that writes a plane, in the
    /// order of `idwt_stages`, and none for the others.
    Store(Vec<Option<wgpu::BindGroup>>),
}

/// What a decode puts into textures besides its passes, which
/// [`Decoder::prepare_output`] makes ready.
enum Output {
    /// Nothing.
    None,
    /// The RGB picture.
    Picture(Fill),
    /// The Y, Cb and Cr planes.
    Planes(PlaneFill),
}

/// Decodes PyroWave frames of one size and chroma format on the GPU.
///
/// All GPU memory is allocated up front and reused for every frame, apart from
/// the payload buffer which grows when a frame needs more room.
pub struct Decoder {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// [`GpuContext::id`] of the context the decoder was made on.
    context_id: u64,
    assembler: FrameAssembler,
    /// The share of a frame's announced blocks a frame that lost packets
    /// needs ([`set_partial_share`](Self::set_partial_share)).
    partial_share: f32,
    buffers: BufferLayout,

    payload: wgpu::Buffer,
    payload_capacity_words: usize,
    block_offsets: wgpu::Buffer,
    /// Made on the first decode that waits for its frame, and made again
    /// when the payload buffer grows; `encode_rgb`, which does not wait,
    /// writes through the queue instead, as a browser has to. Never made
    /// where `shared_uploads` is.
    upload: Option<FrameUpload>,
    /// Whether the decodes that wait for their frame write it straight into
    /// `payload` and `block_offsets`, which the CPU can map then (decision
    /// D53).
    shared_uploads: bool,
    pyramid: wgpu::Buffer,
    packed: wgpu::Buffer,
    /// Made by the first decode that reads the planes back: a player never
    /// does, and the browser's never may.
    readback: Option<wgpu::Buffer>,

    pipelines: Arc<Pipelines>,

    dequant_params: wgpu::Buffer,
    dequant_bind_group: wgpu::BindGroup,
    dequant_grid: (u32, u32),

    idwt_stages: Vec<IdwtStage>,

    color: Option<ColorResources>,
    /// What draws the RGB picture into a texture, made by the first such
    /// draw, and made again for a target of another format.
    draw: Option<DrawResources>,
    timestamps: Option<StageQueries>,

    /// Why the device was lost, once it has been (see [`GpuContext::lost`]).
    lost: Arc<Mutex<Option<String>>>,
    record_lock: Arc<Mutex<()>>,
}

/// The pipeline of `draw.wgsl` for one target format, and its bind group:
/// the RGBA words and the picture's size, a buffer the bind group keeps.
struct DrawResources {
    pipeline: DrawPipeline,
    bind_group: wgpu::BindGroup,
}

/// How the RGB picture gets into a texture (decision D42).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fill {
    /// A copy of the colour conversion's words, written in the texture's
    /// channel order: blue first with `bgra`.
    Copy { bgra: bool },
    /// A draw of `draw.wgsl`, for the textures a copy cannot fill.
    Draw,
}

impl Fill {
    /// How the picture gets into `texture`: a copy when the texture takes one
    /// and its rows are a multiple of 256 bytes, as a copy's must be, a draw
    /// otherwise. On an Apple M3 Max the copy takes 0.12 to 0.16 ms less a
    /// frame.
    fn of(texture: &wgpu::Texture) -> Self {
        let row_bytes = texture.width() * 4;
        if texture.usage().contains(wgpu::TextureUsages::COPY_DST)
            && row_bytes.is_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        {
            Self::Copy {
                bgra: texture.format() == wgpu::TextureFormat::Bgra8Unorm,
            }
        } else {
            Self::Draw
        }
    }
}

/// The texture formats the decoder draws into: 8-bit RGBA or BGRA, as most
/// graphics APIs and displays take them. Not their sRGB forms: the samples
/// are gamma encoded already, and an sRGB target would encode them again.
pub const TEXTURE_FORMATS: [wgpu::TextureFormat; 2] = [
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Bgra8Unorm,
];

/// The format of the textures the decoder puts the planes into: one
/// unsigned normalised 8-bit channel, which holds the samples of the stream
/// exactly (decision D41).
pub const PLANE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

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
        let buffers = BufferLayout::for_decoder(&layout)?;
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
            let frame_usage = frame_buffer_usage(context.shares_uploads());
            let payload = storage("payload", payload_capacity_words as u64 * 4, frame_usage);
            let block_offsets = storage(
                "block offsets",
                u64::from(layout.block_count()) * 4,
                frame_usage,
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
                        writes_plane: stage.params.comps.iter().any(|comp| comp.is_plane != 0),
                        params,
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
                partial_share: DEFAULT_MINIMUM_PACKET_RATIO,
                buffers: buffers.clone(),
                payload,
                payload_capacity_words,
                block_offsets,
                upload: None,
                shared_uploads: context.shares_uploads(),
                pyramid,
                packed,
                readback: None,
                pipelines: Arc::clone(&pipelines),
                dequant_params,
                dequant_bind_group,
                dequant_grid,
                idwt_stages,
                color: None,
                draw: None,
                timestamps,
                lost: context.lost_record(),
                record_lock: Arc::clone(&context.record_lock),
                context_id: context.id(),
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
        self.record_assembled(Some(color), Read::Nothing, encoder)?;
        self.assembler.mark_frame_decoded();
        Ok(())
    }

    /// The RGBA words that [`encode_rgb`](Self::encode_rgb) and
    /// [`decode_rgb`](Self::decode_rgb) write, once one of them has run:
    /// `width * height` words, one per pixel, red in the low byte, alpha 255.
    /// After a decode copied into a BGRA texture they are BGRA words, blue in
    /// the low byte, until the next decode. The buffer belongs to the
    /// decoder: dropping the decoder destroys it, clones of the handle
    /// included.
    pub fn rgba_buffer(&self) -> Option<&wgpu::Buffer> {
        self.color.as_ref().map(|resources| &resources.rgba)
    }

    /// Records the decode of one complete frame and its conversion to RGB
    /// into `encoder`, as [`encode_rgb`](Self::encode_rgb) does, and then
    /// what puts the picture into `target`. Once the caller has submitted
    /// `encoder`, the texture holds the picture.
    ///
    /// `target` is a view of a 2D texture of the picture's size, of one mip
    /// level and one layer, in one of [`TEXTURE_FORMATS`], such as a canvas's
    /// or one from [`create_texture`](Self::create_texture). An unsuitable
    /// texture is [`GpuError::Texture`], before the frame is used.
    ///
    /// A texture with the `COPY_DST` usage and a width of a multiple of 64
    /// pixels, so that its rows are a multiple of 256 bytes as a copy's are,
    /// takes the picture with a copy, which costs less than a draw (decision
    /// D42). Any other texture takes a draw, and needs the
    /// `RENDER_ATTACHMENT` usage for it. A copy does not go through the view:
    /// whatever its format, the texture gets the picture's bytes, and a view
    /// in an sRGB format then decodes them once more when it is sampled. A
    /// draw needs the view in the texture's own format, which cannot be
    /// checked: a view in another format the texture allows, such as its sRGB
    /// form, fails wgpu's validation when the picture is drawn.
    pub fn encode_rgb_to_texture(
        &mut self,
        frame: &[u8],
        color: &ColorParams,
        target: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), GpuError> {
        self.check_device()?;
        self.check_target(target)?;
        self.assemble_complete(frame)?;
        self.record_assembled(Some(color), Read::Texture(target), encoder)?;
        self.assembler.mark_frame_decoded();
        Ok(())
    }

    /// Decodes the frame assembled from [`push_packet`](Self::push_packet)
    /// calls, converts it to RGB and draws it into `target`, as
    /// [`encode_rgb_to_texture`](Self::encode_rgb_to_texture) does, and waits
    /// for the GPU. When this returns the texture holds the picture, for any
    /// queue of the device to read; a texture another queue may still be
    /// reading must not be passed. Blocks that did not arrive decode as zeros.
    pub fn decode_assembled_to_texture(
        &mut self,
        color: &ColorParams,
        target: &wgpu::TextureView,
    ) -> Result<(), GpuError> {
        self.check_target(target)?;
        self.run_gpu(
            Some(color),
            Read::Texture(target),
            false,
            Duration::ZERO,
            &mut |_| {},
        )?;
        Ok(())
    }

    /// Decodes the frame assembled from [`push_packet`](Self::push_packet)
    /// calls into `target`, as
    /// [`decode_assembled_to_texture`](Self::decode_assembled_to_texture)
    /// does, but returns once the work is submitted to the queue, without
    /// waiting for it. Work submitted to the queue later runs after it. On
    /// Metal, which orders the command buffers of a queue that use a texture
    /// by when they were committed, that is all the ordering a client needs
    /// whose queue the context shares (decision D43). The frame goes to the
    /// GPU through the queue's staging memory: the decoder's own upload
    /// buffer can only take a frame once the previous one is done.
    pub fn submit_assembled_to_texture(
        &mut self,
        color: &ColorParams,
        target: &wgpu::TextureView,
    ) -> Result<(), GpuError> {
        self.check_target(target)?;
        self.submit_assembled(Some(color), Read::Texture(target))
    }

    /// A texture that [`encode_rgb_to_texture`](Self::encode_rgb_to_texture)
    /// and [`decode_assembled_to_texture`](Self::decode_assembled_to_texture)
    /// can put the picture into: the picture's size, in `format`, one of
    /// [`TEXTURE_FORMATS`], for drawing and copying into, sampling and copying
    /// from (`RENDER_ATTACHMENT`, `COPY_DST`, `TEXTURE_BINDING` and
    /// `COPY_SRC`). BGRA suits a copy onto a display's surface, which usually
    /// is BGRA.
    pub fn create_texture(&self, format: wgpu::TextureFormat) -> Result<wgpu::Texture, GpuError> {
        self.check_device()?;
        if !TEXTURE_FORMATS.contains(&format) {
            return Err(GpuError::Texture(format!(
                "{format:?} is not one of {TEXTURE_FORMATS:?}"
            )));
        }
        let layout = self.layout();
        self.new_texture(
            "decoded picture",
            (layout.width(), layout.height()),
            format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
        )
    }

    /// Records the decode of one complete frame into `encoder`, as
    /// [`encode_rgb_to_texture`](Self::encode_rgb_to_texture) does but
    /// without the colour conversion, with its Y, Cb and Cr planes going
    /// into `targets`, in that order. Once the caller has submitted
    /// `encoder`, the textures hold the planes: the samples of the stream, as
    /// [`decode`](Self::decode) gives them, for a caller that converts the
    /// colours itself (decision D41).
    ///
    /// `targets` are views of three different 2D textures of the planes'
    /// sizes ([`FrameLayout::plane_size`]), of one mip level and one layer, in
    /// [`PLANE_FORMAT`] and with the `COPY_DST` usage, such as those of
    /// [`create_plane_texture`](Self::create_plane_texture). An unsuitable
    /// texture is [`GpuError::Texture`], before the frame is used. Only the
    /// views' textures matter, except that where the decoder
    /// [stores the planes](Self::stores_planes) and all three textures have
    /// the `STORAGE_BINDING` usage, the inverse transform stores the planes
    /// into the views, which must then allow that usage. Otherwise the planes
    /// are copied into the textures.
    pub fn encode_to_plane_textures(
        &mut self,
        frame: &[u8],
        targets: [&wgpu::TextureView; 3],
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), GpuError> {
        self.check_device()?;
        self.check_plane_targets(targets)?;
        self.assemble_complete(frame)?;
        self.record_assembled(None, Read::Planes(targets), encoder)?;
        self.assembler.mark_frame_decoded();
        Ok(())
    }

    /// Decodes the frame assembled from [`push_packet`](Self::push_packet)
    /// calls into its planes in `targets`, as
    /// [`encode_to_plane_textures`](Self::encode_to_plane_textures) does, and
    /// waits for the GPU. When this returns the textures hold the planes, for
    /// any queue of the device to read; textures another queue may still be
    /// reading must not be passed. Blocks that did not arrive decode as zeros.
    pub fn decode_assembled_to_plane_textures(
        &mut self,
        targets: [&wgpu::TextureView; 3],
    ) -> Result<(), GpuError> {
        self.check_plane_targets(targets)?;
        self.run_gpu(
            None,
            Read::Planes(targets),
            false,
            Duration::ZERO,
            &mut |_| {},
        )?;
        Ok(())
    }

    /// Decodes the frame assembled from [`push_packet`](Self::push_packet)
    /// calls into the plane textures `targets`, as
    /// [`decode_assembled_to_plane_textures`](Self::decode_assembled_to_plane_textures)
    /// does, but returns once the work is submitted, as
    /// [`submit_assembled_to_texture`](Self::submit_assembled_to_texture)
    /// does.
    pub fn submit_assembled_to_plane_textures(
        &mut self,
        targets: [&wgpu::TextureView; 3],
    ) -> Result<(), GpuError> {
        self.check_plane_targets(targets)?;
        self.submit_assembled(None, Read::Planes(targets))
    }

    /// A texture that [`encode_to_plane_textures`](Self::encode_to_plane_textures)
    /// and [`decode_assembled_to_plane_textures`](Self::decode_assembled_to_plane_textures)
    /// can put the plane of `component` into: the plane's size, in
    /// [`PLANE_FORMAT`], for copying into, sampling and copying from
    /// (`COPY_DST`, `TEXTURE_BINDING` and `COPY_SRC`), and where the decoder
    /// [stores the planes](Self::stores_planes) for storing into as well
    /// (`STORAGE_BINDING`).
    pub fn create_plane_texture(&self, component: Component) -> Result<wgpu::Texture, GpuError> {
        self.check_device()?;
        let size = self.layout().plane_size(component);
        let mut usage = wgpu::TextureUsages::COPY_DST;
        usage.set(wgpu::TextureUsages::STORAGE_BINDING, self.stores_planes());
        self.new_texture("decoded plane", size, PLANE_FORMAT, usage)
    }

    /// Whether decodes into plane textures store the planes straight into
    /// textures with the `STORAGE_BINDING` usage, from the inverse transform,
    /// rather than copy them in: on Metal, where a 4K frame takes 0.08 to
    /// 0.12 ms less on an Apple M3 Max, frame after frame (decision D52).
    /// See [`GpuContext::stores_planes`].
    pub fn stores_planes(&self) -> bool {
        self.pipelines.idwt_planes.is_some()
    }

    /// A texture of `size` and `format` that the decoder can write with
    /// `usage`, and that can be sampled and copied from.
    fn new_texture(
        &self,
        label: &str,
        (width, height): (u32, u32),
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> Result<wgpu::Texture, GpuError> {
        with_error_scope(&self.device, || {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: usage | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        })
    }

    /// The GPU device the decoder runs on.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// [`GpuContext::id`] of the context the decoder was made on: an object
    /// made on a context of another id is on another device, whatever
    /// comparing devices says. The methods that draw into textures, such as
    /// [`decode_assembled_to_texture`](Self::decode_assembled_to_texture),
    /// cannot tell from a texture view which device it is on, and a view of
    /// another context's makes wgpu fail, or names an unrelated view of this
    /// device: the caller keeps track.
    pub fn context_id(&self) -> u64 {
        self.context_id
    }

    /// Feeds one packet of the frame being received. See
    /// [`FrameAssembler::push_packet`].
    pub fn push_packet(&mut self, packet: &[u8]) -> Result<(), GpuError> {
        Ok(self.assembler.push_packet(packet)?)
    }

    /// Whether the frame being received can be decoded: all its blocks
    /// arrived, or, with `allow_partial_frame`, every block of the coarsest
    /// level and more than [`partial_share`](Self::partial_share) of the
    /// blocks its start of frame header announced. See
    /// [`FrameAssembler::is_ready_with`].
    pub fn is_ready(&self, allow_partial_frame: bool) -> bool {
        self.assembler.is_ready_with(
            allow_partial_frame,
            DEFAULT_PRISTINE_BANDS,
            self.partial_share,
            None,
        )
    }

    /// Sets the share of a frame's announced blocks that must have arrived,
    /// with every block of the coarsest level, for [`is_ready`](Self::is_ready)
    /// to take a frame that lost packets: 0.9 by default, as upstream's
    /// decoder has it. A lower share takes frames that lost more, with more
    /// of their detail blank, where a client would skip them; 1 takes whole
    /// frames only. Below 0 counts as 0 and above 1 as 1; NaN sets the
    /// default again. The share stays until it is set again.
    pub fn set_partial_share(&mut self, share: f32) {
        self.partial_share = if share.is_nan() {
            DEFAULT_MINIMUM_PACKET_RATIO
        } else {
            share.clamp(0.0, 1.0)
        };
    }

    /// The share of [`set_partial_share`](Self::set_partial_share).
    pub fn partial_share(&self) -> f32 {
        self.partial_share
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
        read: Read<'_>,
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
        read: Read<'_>,
        timed: bool,
        assembled_in: Duration,
        consume: &mut dyn FnMut(&[u8]),
    ) -> Result<Timings, GpuError> {
        let record_lock = Arc::clone(&self.record_lock);
        let _record = record_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prepare_start = Instant::now();
        self.ensure_payload_capacity()?;
        let (table_bytes, payload_bytes) = self.upload_mapped()?;
        let output = self.prepare_output(color, read)?;
        let prepare = assembled_in + prepare_start.elapsed();

        self.make_readback(read)?;
        // What the frame copies back and where to: the planes, or the RGBA
        // words, into the buffer just made sure of.
        let copy_back = match read {
            Read::Nothing | Read::Texture(_) | Read::Planes(_) => None,
            Read::Yuv => self
                .readback
                .as_ref()
                .map(|target| (&self.packed, target, self.buffers.packed_words * 4)),
            Read::Rgb => self.color.as_ref().and_then(|resources| {
                let target = resources.readback.as_ref()?;
                Some((&resources.rgba, target, resources.rgba.size()))
            }),
        };

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
            self.encode_passes(&mut encoder, color.is_some(), queries, &output);
            self.encode_output(&mut encoder, read, &output);
            if let Some((source, target, bytes)) = copy_back {
                encoder.copy_buffer_to_buffer(source, 0, target, 0, bytes);
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
        match copy_back {
            Some((_, target, bytes)) => read_back_with(&self.device, target, bytes, &mut deliver)?,
            None => {
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

    /// Makes ready what a decode with `color` and `read` needs besides its
    /// passes: the colour conversion's parameters, in the channel order a
    /// copy into the texture of `read` needs, and the draw, if the texture
    /// takes one, or the bind groups that store the planes into textures.
    /// Says how the picture or the planes get into the textures.
    fn prepare_output(
        &mut self,
        color: Option<&ColorParams>,
        read: Read<'_>,
    ) -> Result<Output, GpuError> {
        let output = match read {
            Read::Texture(target) => Output::Picture(Fill::of(target.texture())),
            Read::Planes(targets) => Output::Planes(self.plane_fill(targets)),
            Read::Nothing | Read::Yuv | Read::Rgb => Output::None,
        };
        if let Some(color) = color {
            let bgra = matches!(output, Output::Picture(Fill::Copy { bgra: true }));
            self.prepare_color(color, bgra)?;
        }
        if let (Read::Texture(target), Output::Picture(Fill::Draw)) = (read, &output) {
            self.prepare_draw(target.texture().format())?;
        }
        Ok(output)
    }

    /// How the planes get into `targets`: stored by the inverse transform
    /// where this decoder can store them and every texture allows it, the
    /// `STORAGE_BINDING` usage, and copied otherwise (decisions D41, D52).
    /// A view that does not allow stores, which its texture's usage cannot
    /// tell, fails the bind groups and takes the copy, which the checks of
    /// the targets made sure of.
    fn plane_fill(&self, targets: [&wgpu::TextureView; 3]) -> PlaneFill {
        let Some(stage) = &self.pipelines.idwt_planes else {
            return PlaneFill::Copy;
        };
        let storage = wgpu::TextureUsages::STORAGE_BINDING;
        if !targets
            .iter()
            .all(|target| target.texture().usage().contains(storage))
        {
            return PlaneFill::Copy;
        }
        let [y, cb, cr] = targets;
        let stores = with_error_scope(&self.device, || {
            let bind_groups = self
                .idwt_stages
                .iter()
                .map(|level| {
                    level.writes_plane.then(|| {
                        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("idwt planes"),
                            layout: &stage.layout,
                            entries: &[
                                entry(0, &self.pyramid),
                                entry(2, &level.params),
                                texture(3, y),
                                texture(4, cb),
                                texture(5, cr),
                            ],
                        })
                    })
                })
                .collect();
            PlaneFill::Store(bind_groups)
        });
        stores.unwrap_or(PlaneFill::Copy)
    }

    /// Records, after the passes, what `read` asks of textures, which
    /// [`prepare_output`](Self::prepare_output) made ready.
    fn encode_output(&self, encoder: &mut wgpu::CommandEncoder, read: Read<'_>, output: &Output) {
        match (read, output) {
            (Read::Texture(target), Output::Picture(fill)) => {
                self.encode_picture(encoder, target, *fill);
            }
            (Read::Planes(targets), Output::Planes(PlaneFill::Copy)) => {
                self.encode_plane_copies(encoder, targets);
            }
            _ => {}
        }
    }

    /// Records the decode of the frame the assembler holds into `encoder`,
    /// its data uploaded through the queue: the passes, with the colour
    /// conversion if `color` is given, and then what `read` asks of
    /// textures. The caller marks the frame decoded once that cannot fail
    /// any more.
    fn record_assembled(
        &mut self,
        color: Option<&ColorParams>,
        read: Read<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), GpuError> {
        self.ensure_payload_capacity()?;
        with_error_scope(&self.device, || self.upload())?;
        let output = self.prepare_output(color, read)?;
        self.encode_passes(encoder, color.is_some(), None, &output);
        self.encode_output(encoder, read, &output);
        Ok(())
    }

    /// Records the decode of the frame the assembler holds, as
    /// [`record_assembled`](Self::record_assembled) does, and submits it
    /// without waiting for it. The frame counts as decoded only once it is
    /// submitted: finishing the commands can still fail, as for a view in a
    /// format the draw does not take, and then the frame is still there.
    fn submit_assembled(
        &mut self,
        color: Option<&ColorParams>,
        read: Read<'_>,
    ) -> Result<(), GpuError> {
        self.check_device()?;
        let record_lock = Arc::clone(&self.record_lock);
        let _record = record_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("decode"),
            });
        let outcome = self
            .record_assembled(color, read, &mut encoder)
            .and_then(|()| {
                with_error_scope(&self.device, || {
                    self.queue.submit([encoder.finish()]);
                })
            });
        if outcome.is_ok() {
            self.assembler.mark_frame_decoded();
        }
        // Nothing waits for the work, so nothing else polls the device: one
        // poll that does not wait hands back what finished work held, and
        // tells of a lost device.
        notice_loss(&self.device, &self.lost);
        self.check_device()?;
        outcome
    }

    /// An error unless `target` shows a texture the picture can be put into:
    /// a copy needs `COPY_DST` and rows of a multiple of 256 bytes, anything
    /// else a draw, which needs `RENDER_ATTACHMENT` (decisions D42, D44).
    fn check_target(&self, target: &wgpu::TextureView) -> Result<(), GpuError> {
        let texture = target.texture();
        let size = (self.layout().width(), self.layout().height());
        let any = wgpu::TextureUsages::empty();
        if let Some(problem) = target_problem(texture, &TEXTURE_FORMATS, any, "the picture's", size)
        {
            return Err(GpuError::Texture(format!("the target texture {problem}")));
        }
        if Fill::of(texture) == Fill::Draw
            && !texture
                .usage()
                .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
        {
            return Err(GpuError::Texture(
                "the target texture lacks the RENDER_ATTACHMENT usage that a draw needs: \
                 only a texture with the COPY_DST usage and a width of a multiple of 64 \
                 pixels takes a copy"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    /// An error unless `targets` show three different textures that the Y,
    /// Cb and Cr planes, in that order, can be copied into.
    fn check_plane_targets(&self, targets: [&wgpu::TextureView; 3]) -> Result<(), GpuError> {
        for (component, target) in Component::ALL.into_iter().zip(targets) {
            let size = self.layout().plane_size(component);
            let usage = wgpu::TextureUsages::COPY_DST;
            if let Some(problem) = target_problem(
                target.texture(),
                &[PLANE_FORMAT],
                usage,
                "the plane's",
                size,
            ) {
                return Err(GpuError::Texture(format!(
                    "the texture of the {component:?} plane {problem}"
                )));
            }
        }
        let [y, cb, cr] = targets.map(|target| target.texture());
        if y == cb || y == cr || cb == cr {
            return Err(GpuError::Texture(
                "two of the planes have the same texture".to_owned(),
            ));
        }
        Ok(())
    }

    /// Makes the pipeline that draws into textures of `format`, and its bind
    /// group, unless they are there. The colour conversion's buffers come
    /// first, from [`prepare_color`](Self::prepare_color).
    fn prepare_draw(&mut self, format: wgpu::TextureFormat) -> Result<(), GpuError> {
        if self
            .draw
            .as_ref()
            .is_some_and(|draw| draw.pipeline.format == format)
        {
            return Ok(());
        }
        let Some(color) = &self.color else {
            return Err(GpuError::Device(
                "there is no RGB picture to draw".to_owned(),
            ));
        };
        let pipeline = DrawPipeline::new(&self.device, format)?;
        let size = [self.layout().width(), self.layout().height(), 0, 0];
        let bind_group = with_error_scope(&self.device, || {
            let picture = uniform_buffer(&self.device, &self.queue, "picture size", &size);
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("draw"),
                layout: &pipeline.layout,
                entries: &[entry(0, &color.rgba), entry(1, &picture)],
            })
        })?;
        self.draw = Some(DrawResources {
            pipeline,
            bind_group,
        });
        Ok(())
    }

    /// Records what puts the RGB picture into `target`: a copy of the colour
    /// conversion's words, which [`prepare_color`](Self::prepare_color) wrote
    /// in the texture's channel order, or a draw.
    fn encode_picture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        fill: Fill,
    ) {
        match (fill, &self.color) {
            (Fill::Copy { .. }, Some(color)) => {
                let texture = target.texture();
                encoder.copy_buffer_to_texture(
                    wgpu::TexelCopyBufferInfo {
                        buffer: &color.rgba,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(texture.width() * 4),
                            rows_per_image: None,
                        },
                    },
                    texture.as_image_copy(),
                    texture.size(),
                );
            }
            (Fill::Copy { .. }, None) => {}
            (Fill::Draw, _) => self.encode_draw(encoder, target),
        }
    }

    /// Records the draw of the RGB picture into `target`, which
    /// [`prepare_draw`](Self::prepare_draw) made ready.
    fn encode_draw(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let Some(draw) = &self.draw else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("draw"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
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
        pass.set_pipeline(&draw.pipeline.pipeline);
        pass.set_bind_group(0, &draw.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Records copies of the Y, Cb and Cr planes into the textures of
    /// `targets`. The rows of the packed planes are a multiple of 256 bytes
    /// long ([`BufferLayout::for_decoder`]), as a copy takes them, and a copy
    /// leaves out what pads them; the planes start at multiples of 512 bytes,
    /// where Direct3D 12 copies from. A render pass per plane, the other way
    /// to take the padding out, cost about 0.3 ms more on an Apple M3 Max,
    /// 1080p and 4K alike (decision D41).
    fn encode_plane_copies(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        targets: [&wgpu::TextureView; 3],
    ) {
        for (component, target) in Component::ALL.into_iter().zip(targets) {
            let i = component.index();
            let texture = target.texture();
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.packed,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: u64::from(self.buffers.packed_offsets[i]) * 4,
                        bytes_per_row: Some(self.buffers.packed_strides[i] * 4),
                        rows_per_image: None,
                    },
                },
                texture.as_image_copy(),
                texture.size(),
            );
        }
    }

    /// Makes the buffer that `read` copies the picture back into, the first
    /// time a decode needs it.
    fn make_readback(&mut self, read: Read<'_>) -> Result<(), GpuError> {
        let (slot, bytes, label) = match read {
            Read::Nothing | Read::Texture(_) | Read::Planes(_) => return Ok(()),
            Read::Yuv => (
                &mut self.readback,
                self.buffers.packed_words * 4,
                "readback",
            ),
            Read::Rgb => match self.color.as_mut() {
                Some(resources) => (
                    &mut resources.readback,
                    resources.rgba.size(),
                    "rgba readback",
                ),
                None => return Ok(()),
            },
        };
        if slot.is_none() {
            let usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
            *slot = Some(with_error_scope(&self.device, || {
                buffer(&self.device, label, bytes, usage)
            })?);
        }
        Ok(())
    }

    /// Records the GPU stages of a frame into `encoder`: dequantisation, the
    /// inverse transform and, with `rgb`, the colour conversion. With
    /// `queries` each pass writes its timestamps. When `output` stores the
    /// planes into textures, the levels that write planes store them there.
    fn encode_passes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        rgb: bool,
        queries: Option<&StageQueries>,
        output: &Output,
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
            let stores = match output {
                Output::Planes(PlaneFill::Store(bind_groups)) => bind_groups.as_slice(),
                _ => &[],
            };
            for (index, stage) in self.idwt_stages.iter().enumerate() {
                match (stores.get(index), &self.pipelines.idwt_planes) {
                    (Some(Some(bind_group)), Some(planes)) => {
                        pass.set_pipeline(&planes.pipeline);
                        pass.set_bind_group(0, bind_group, &[]);
                    }
                    _ => {
                        pass.set_pipeline(&self.pipelines.idwt.pipeline);
                        pass.set_bind_group(0, &stage.bind_group, &[]);
                    }
                }
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
    /// the parameters, which may change from call to call: with `bgra` the
    /// words have blue in the low byte, for a copy into a BGRA texture.
    fn prepare_color(&mut self, color: &ColorParams, bgra: bool) -> Result<(), GpuError> {
        let mut params = YuvParams::new(self.assembler.layout(), &self.buffers, color);
        params.bgra = u32::from(bgra);

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
                    readback: None,
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
                usage: frame_buffer_usage(self.shared_uploads),
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
        if self.shared_uploads {
            return self.upload_in_place();
        }
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
            map_for_writing(&self.device, &[&upload.buffer])?;
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

    /// Writes the block table and the payload of the assembled frame where
    /// the shaders read them, on a GPU that shares its memory with the CPU
    /// (decision D53): no upload buffer, and no copies at the start of the
    /// frame's command buffer, whose sizes this returns as zero. The previous
    /// frame is done, so mapping the buffers waits for nothing.
    fn upload_in_place(&mut self) -> Result<(u64, u64), GpuError> {
        let written = map_for_writing(&self.device, &[&self.block_offsets, &self.payload])
            .and_then(|()| self.write_in_place());
        // Unmapped whatever happened: a buffer left mapped would fail every
        // later decode and every write through the queue.
        let unmapped = with_error_scope(&self.device, || {
            self.block_offsets.unmap();
            self.payload.unmap();
        });
        written?;
        unmapped?;
        Ok((0, 0))
    }

    /// The writing of [`upload_in_place`](Self::upload_in_place), into the
    /// mapped buffers.
    fn write_in_place(&self) -> Result<(), GpuError> {
        let frame = self.assembler.frame();
        let table: &[u8] = bytemuck::cast_slice(frame.block_offsets());
        let payload: &[u8] = bytemuck::cast_slice(frame.payload());
        let padded = payload.len() + PAYLOAD_PADDING_WORDS * 4;
        let mapped = |buffer: &wgpu::Buffer, bytes: usize| {
            buffer
                .slice(..bytes as u64)
                .get_mapped_range_mut()
                .map_err(|error| GpuError::Device(error.to_string()))
        };
        mapped(&self.block_offsets, table.len())?
            .slice(..)
            .copy_from_slice(table);
        let mut view = mapped(&self.payload, padded)?;
        view.slice(..payload.len()).copy_from_slice(payload);
        view.slice(payload.len()..).fill(0);
        Ok(())
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

/// What keeps `texture` from being a target of `size` in one of `formats`
/// that the decoder writes with `usage`, if anything, said after the texture:
/// the size is `whose`.
fn target_problem(
    texture: &wgpu::Texture,
    formats: &[wgpu::TextureFormat],
    usage: wgpu::TextureUsages,
    whose: &str,
    (width, height): (u32, u32),
) -> Option<String> {
    let problem = if texture.dimension() != wgpu::TextureDimension::D2 {
        format!("is {:?} rather than 2D", texture.dimension())
    } else if !formats.contains(&texture.format()) {
        format!("is {:?}, not one of {formats:?}", texture.format())
    } else if (texture.width(), texture.height()) != (width, height) {
        format!(
            "is {}x{} rather than {whose} {width}x{height}",
            texture.width(),
            texture.height()
        )
    } else if texture.sample_count() != 1 {
        format!(
            "has {} samples per texel rather than 1",
            texture.sample_count()
        )
    } else if (texture.mip_level_count(), texture.depth_or_array_layers()) != (1, 1) {
        // A view cannot be checked, but every view of a texture of one level
        // and one layer shows all of it.
        format!(
            "has {} mip levels and {} layers rather than one of each",
            texture.mip_level_count(),
            texture.depth_or_array_layers()
        )
    } else if !texture.usage().contains(usage) {
        format!("lacks the {usage:?} usage")
    } else {
        return None;
    };
    Some(problem)
}

/// A plane in the packed buffer, in bytes: its rows are padded to whole
/// words, and in a decoder to a multiple of 256 bytes.
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
        ];
        buffers.extend(&self.readback);
        if let Some(upload) = &self.upload {
            buffers.push(&upload.buffer);
        }
        if let Some(color) = &self.color {
            buffers.push(&color.rgba);
            buffers.extend(&color.readback);
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

/// The usage of the buffers the shaders read a frame from: written through
/// the queue, or with `shared_uploads` also mapped by the CPU (decision D53).
fn frame_buffer_usage(shared_uploads: bool) -> wgpu::BufferUsages {
    let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    usage.set(wgpu::BufferUsages::MAP_WRITE, shared_uploads);
    usage
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
    fn a_copy_fills_the_textures_it_can_and_a_draw_the_others() {
        // The texture tests see the same picture either way: this pins which
        // way each texture takes (decision D42).
        let Some(context) = test_context() else {
            return;
        };
        let texture = |width, format, usage| {
            context.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("target"),
                size: wgpu::Extent3d {
                    width,
                    height: 8,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | usage,
                view_formats: &[],
            })
        };
        let (rgba, bgra) = (
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Bgra8Unorm,
        );
        let copies = wgpu::TextureUsages::COPY_DST;
        let none = wgpu::TextureUsages::empty();
        for (width, format, usage, fill) in [
            (64, rgba, copies, Fill::Copy { bgra: false }),
            (1920, bgra, copies, Fill::Copy { bgra: true }),
            (3840, rgba, copies, Fill::Copy { bgra: false }),
            (1920, bgra, none, Fill::Draw),
            (200, rgba, copies, Fill::Draw),
            (1000, bgra, copies, Fill::Draw),
        ] {
            assert_eq!(
                Fill::of(&texture(width, format, usage)),
                fill,
                "{width} wide, {format:?}, {usage:?}"
            );
        }
    }

    #[test]
    fn frames_go_where_the_shaders_read_them_where_memory_is_shared() {
        // The decode tests see the same frames either way: this pins which
        // way a frame goes (decision D53).
        let Some(context) = test_context() else {
            return;
        };
        let mut decoder =
            Decoder::new(context, 64, 32, ChromaSubsampling::Yuv420).expect("decoder");
        assert_eq!(decoder.shared_uploads, context.shares_uploads());
        decoder
            .decode(&empty_frame(64, 32, ChromaSubsampling::Yuv420))
            .expect("decode");
        // No upload buffer and so no copies from it, or the buffer.
        assert_eq!(decoder.upload.is_none(), context.shares_uploads());
        let usage = decoder.payload.usage();
        assert_eq!(
            usage.contains(wgpu::BufferUsages::MAP_WRITE),
            context.shares_uploads()
        );
        assert_eq!(usage, decoder.block_offsets.usage());
    }

    #[test]
    fn the_planes_are_stored_where_all_three_textures_take_stores() {
        // The plane tests see the same planes either way: this pins which way
        // each set of textures takes (decision D52).
        let Some(context) = test_context() else {
            return;
        };
        let decoder = Decoder::new(context, 64, 32, ChromaSubsampling::Yuv420).expect("decoder");
        assert_eq!(decoder.stores_planes(), context.stores_planes());
        // Textures a client samples, which decodes copy into, or store into.
        let copy = wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING;
        let store = copy | wgpu::TextureUsages::STORAGE_BINDING;
        let textures = |usages: [wgpu::TextureUsages; 3]| {
            Component::ALL.map(|component| {
                let (width, height) = decoder.layout().plane_size(component);
                context.device().create_texture(&wgpu::TextureDescriptor {
                    label: Some("plane"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: PLANE_FORMAT,
                    usage: usages[component.index()],
                    view_formats: &[],
                })
            })
        };
        let fill = |textures: &[wgpu::Texture; 3], view: &wgpu::TextureViewDescriptor| {
            let views = textures.each_ref().map(|texture| texture.create_view(view));
            decoder.plane_fill(views.each_ref())
        };
        let plain = wgpu::TextureViewDescriptor::default();
        assert!(matches!(
            fill(&textures([copy; 3]), &plain),
            PlaneFill::Copy
        ));
        if !context.stores_planes() {
            // Textures of R8 storage need the adapter's own format features.
            return;
        }
        // In 4:2:0 the planes end at levels 1 and 0, the last two of five.
        match fill(&textures([store; 3]), &plain) {
            PlaneFill::Store(bind_groups) => assert_eq!(
                bind_groups.iter().map(Option::is_some).collect::<Vec<_>>(),
                [false, false, false, true, true]
            ),
            PlaneFill::Copy => panic!("textures that take stores are copied into"),
        }
        assert!(matches!(
            fill(&textures([store, copy, store]), &plain),
            PlaneFill::Copy
        ));
        let sampled = wgpu::TextureViewDescriptor {
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            ..Default::default()
        };
        assert!(matches!(
            fill(&textures([store; 3]), &sampled),
            PlaneFill::Copy
        ));
    }

    #[test]
    fn the_last_level_writes_every_word_with_zero_padding() {
        let Some(context) = test_context() else {
            return;
        };

        // Odd widths leave padding in the last word of a row; 4:2:0 chroma
        // ends at level 1, whose size need not be a multiple of the tile. The
        // words after that, up to a multiple of 256 bytes, are there for
        // copies only (BufferLayout::for_decoder): the decode leaves them
        // alone, though its tiles reach into them, and nothing reads them.
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
                let words = w.div_ceil(4) as usize * 4;
                for row in 0..h as usize {
                    let line = &packed[start + row * stride..][..stride];
                    let (written, unwritten) = line.split_at(words);
                    let (samples, padding) = written.split_at(w as usize);
                    assert!(
                        samples.iter().all(|&s| s == 128),
                        "{width}x{height} {chroma:?} {component:?} row {row}: {samples:?}"
                    );
                    assert!(
                        padding.iter().all(|&p| p == 0),
                        "{width}x{height} {chroma:?} {component:?} row {row}, padding {padding:?}"
                    );
                    assert!(
                        unwritten.iter().all(|&p| p == 0xA5),
                        "{width}x{height} {chroma:?} {component:?} row {row}: written past the \
                         samples"
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
