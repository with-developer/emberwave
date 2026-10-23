//! `emberwave-cli bench-encode`: measure the encode time per frame.

use std::path::PathBuf;
use std::time::Instant;

use emberwave_core::bitstream::consts::HEADER_BYTES;
use emberwave_core::bitstream::{ChromaSubsampling, Component};
use emberwave_core::encode::{self, Signalling};
use emberwave_core::gpu;

use crate::bench::{Series, json_object, ms, parse_seconds, print_table, quote, warm_up};
use crate::{Backend, GpuArgs, Result, chroma_name, y4m};

#[derive(Debug, clap::Args)]
pub struct Args {
    /// The Y4M video to encode, 8-bit 4:2:0 or 4:4:4; its frames are read
    /// into memory first and then encoded in turn, as often as needed
    input: PathBuf,

    /// The most bytes a frame may take, its start of frame header included
    #[arg(long)]
    bytes_per_frame: usize,

    /// Number of frames to measure (default: one pass over the video)
    #[arg(long)]
    frames: Option<usize>,

    /// Frames encoded first and not counted, at least
    #[arg(long, default_value_t = 10)]
    warmup: usize,

    /// Seconds of encoding before the counted frames, at least: a GPU runs
    /// slower for the first moments of a load
    #[arg(long, default_value_t = 1.0, value_parser = parse_seconds)]
    warmup_seconds: f64,

    /// Encode on the GPU rather than on the CPU
    #[arg(long)]
    gpu: bool,

    /// With --gpu: encode pictures that are on the GPU already, an R8
    /// texture per plane made from each frame before the measurement, as a
    /// screen capture would hand them over
    #[arg(long, requires = "gpu")]
    textures: bool,

    /// Print the result as JSON
    #[arg(long)]
    json: bool,

    /// With --gpu: which GPU
    #[command(flatten)]
    gpu_args: GpuArgs,
}

/// The CPU or the GPU encoder.
enum Encoder {
    Cpu(Box<encode::Encoder>),
    Gpu(Box<gpu::Encoder>),
}

/// A picture to encode: planes in CPU memory, or textures on the GPU.
enum Picture<'a> {
    Planes([&'a [u8]; 3]),
    Textures(gpu::TexturePicture<'a>),
}

/// The series of a measurement: `total` always, the others on a GPU.
struct Measurement {
    total: Series,
    prepare: Series,
    gpu_wall: Series,
    gpu_sum: Series,
    transform: Series,
    quantisation: Series,
    rate_control: Series,
    packing: Series,
}

impl Measurement {
    fn new() -> Self {
        Self {
            total: Series::new("total"),
            prepare: Series::new("cpu prepare"),
            gpu_wall: Series::new("gpu wall"),
            gpu_sum: Series::new("gpu (timestamps)"),
            transform: Series::new("transform"),
            quantisation: Series::new("quantisation"),
            rate_control: Series::new("rate control"),
            packing: Series::new("packing"),
        }
    }

    /// Encodes one frame, and records its times.
    fn encode(&mut self, encoder: &mut Encoder, picture: Picture<'_>, budget: usize) -> Result<()> {
        match (encoder, picture) {
            (Encoder::Cpu(encoder), Picture::Planes(planes)) => {
                let start = Instant::now();
                encoder.encode(planes, budget)?;
                self.total.samples.push(ms(start.elapsed()));
            }
            (Encoder::Cpu(_), Picture::Textures(_)) => {
                return Err("the CPU encoder takes pictures in CPU memory".into());
            }
            (Encoder::Gpu(encoder), picture) => {
                let (_, timings) = match picture {
                    Picture::Planes(planes) => encoder.encode_timed(planes, budget)?,
                    Picture::Textures(textures) => {
                        encoder.encode_textures_timed(textures, budget)?
                    }
                };
                self.total
                    .samples
                    .push(ms(timings.prepare + timings.gpu_wall));
                self.prepare.samples.push(ms(timings.prepare));
                self.gpu_wall.samples.push(ms(timings.gpu_wall));
                if let Some(stages) = timings.stages {
                    self.gpu_sum.samples.push(ms(stages.total()));
                    self.transform.samples.push(ms(stages.transform));
                    self.quantisation.samples.push(ms(stages.quantisation));
                    self.rate_control.samples.push(ms(stages.rate_control));
                    self.packing.samples.push(ms(stages.packing));
                }
            }
        }
        Ok(())
    }

    fn series(&self) -> Vec<&Series> {
        [
            &self.total,
            &self.prepare,
            &self.gpu_wall,
            &self.gpu_sum,
            &self.transform,
            &self.quantisation,
            &self.rate_control,
            &self.packing,
        ]
        .into_iter()
        .filter(|s| !s.samples.is_empty())
        .collect()
    }
}

pub fn run(args: &Args) -> Result<()> {
    if args.frames == Some(0) {
        return Err("--frames must be at least 1".into());
    }
    if !args.gpu && (args.gpu_args.backend != Backend::Auto || args.gpu_args.software) {
        return Err("--backend and --software choose the GPU of --gpu".into());
    }
    if args.bytes_per_frame < HEADER_BYTES {
        return Err(format!(
            "--bytes-per-frame {} is less than the {HEADER_BYTES} bytes of a frame header",
            args.bytes_per_frame
        )
        .into());
    }
    let mut reader = y4m::open(&args.input)?;
    let header = *reader.header();
    if header.chroma == ChromaSubsampling::Yuv420 && (header.width | header.height) % 2 != 0 {
        return Err(format!(
            "{}x{} is odd, and 4:2:0 needs an even width and height",
            header.width, header.height
        )
        .into());
    }
    let (width, height) = (u32::try_from(header.width)?, u32::try_from(header.height)?);
    // In memory first, as `bench` has its file, so that the encodes follow
    // each other as they would in a stream, with no reading in between: a
    // GPU that waits for the CPU slows down, and not every GPU alike.
    let mut frames = Vec::new();
    while let Some(frame) = reader.next_frame()? {
        frames.push(frame);
    }
    if frames.is_empty() {
        return Err("the Y4M file has no frames".into());
    }

    // With --textures, a texture per plane of every frame, on the device.
    let mut textures = Vec::new();
    let (mut encoder, adapter, backend) = if args.gpu {
        let context = args.gpu_args.create_context(true)?;
        let info = context.adapter_info();
        let encoder = gpu::Encoder::new(
            &context,
            width,
            height,
            header.chroma,
            Signalling::default(),
        )?;
        if args.textures {
            let layout = encoder.layout().clone();
            textures = frames
                .iter()
                .map(|frame| {
                    std::array::from_fn::<_, 3, _>(|i| {
                        let size = layout.plane_size(Component::ALL[i]);
                        plane_texture(&context, size, &frame[i])
                    })
                })
                .collect();
            // Uploaded before the first encode is timed.
            context.device().poll(wgpu::PollType::wait_indefinitely())?;
        }
        (
            Encoder::Gpu(Box::new(encoder)),
            info.name,
            format!("{:?}", info.backend),
        )
    } else {
        let encoder = encode::Encoder::new(width, height, header.chroma, Signalling::default())?;
        (
            Encoder::Cpu(Box::new(encoder)),
            "CPU".to_owned(),
            "CPU".to_owned(),
        )
    };

    // One pass over the video unless --frames says otherwise.
    let count = args.frames.unwrap_or(frames.len());
    let picture = |i: usize| {
        let i = i % frames.len();
        match textures.get(i) {
            Some([y, cb, cr]) => Picture::Textures(gpu::TexturePicture::Planes([y, cb, cr])),
            None => {
                let frame = &frames[i];
                Picture::Planes([&frame[0][..], &frame[1][..], &frame[2][..]])
            }
        }
    };

    let mut measurement = Measurement::new();
    let warmup = warm_up(args.warmup, args.warmup_seconds, |i| {
        measurement.encode(&mut encoder, picture(i), args.bytes_per_frame)
    })?;
    let mut measurement = Measurement::new();
    for i in 0..count {
        measurement.encode(&mut encoder, picture(i), args.bytes_per_frame)?;
    }

    let description = format!(
        "{width}x{height} {}, {} bytes per frame, encoded on the {}{}",
        chroma_name(header.chroma),
        args.bytes_per_frame,
        if args.gpu { "GPU" } else { "CPU" },
        if args.textures { " from textures" } else { "" }
    );
    let series = measurement.series();
    if args.json {
        println!(
            "{}",
            json_object(
                &[
                    ("input", quote(&args.input.display().to_string())),
                    ("adapter", quote(&adapter)),
                    ("backend", quote(&backend)),
                    ("description", quote(&description)),
                    ("frames", count.to_string()),
                    ("warmup", warmup.to_string()),
                    ("bytes_per_frame", args.bytes_per_frame.to_string()),
                    (
                        "input",
                        quote(if args.textures { "textures" } else { "planes" }),
                    ),
                ],
                &series,
            )
        );
    } else {
        println!("adapter:  {adapter} ({backend})");
        println!("encoding: {description}");
        println!("frames:   {count} measured after {warmup} warm-up frames");
        println!();
        print_table(&series);
    }
    Ok(())
}

/// A view of an `R8Unorm` texture holding `plane`, of `width` x `height`
/// samples, on the context's device.
fn plane_texture(
    context: &gpu::GpuContext,
    (width, height): (u32, u32),
    plane: &[u8],
) -> wgpu::TextureView {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = context.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("picture plane"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    context.queue().write_texture(
        texture.as_image_copy(),
        plane,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width),
            rows_per_image: Some(height),
        },
        size,
    );
    texture.create_view(&Default::default())
}
