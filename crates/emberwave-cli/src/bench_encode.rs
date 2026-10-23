//! `emberwave-cli bench-encode`: measure the encode time per frame.

use std::path::{Path, PathBuf};
use std::time::Instant;

use emberwave_core::bitstream::ChromaSubsampling;
use emberwave_core::encode::{self, Signalling};
use emberwave_core::gpu;

use crate::bench::{Series, json_object, ms, parse_seconds, print_table, quote, warm_up};
use crate::{Backend, GpuArgs, Result, chroma_name, y4m};

#[derive(Debug, clap::Args)]
pub struct Args {
    /// The Y4M video to encode, 8-bit 4:2:0 or 4:4:4; its frames are encoded
    /// in turn, as often as needed
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

    /// Print the result as JSON
    #[arg(long)]
    json: bool,

    /// With --gpu: which GPU
    #[command(flatten)]
    gpu_args: GpuArgs,
}

/// The frames of a Y4M file in turn, from the start again after the last,
/// read as they are needed so that a long video does not fill the memory.
struct Frames {
    path: PathBuf,
    reader: y4m::Reader<std::io::BufReader<std::fs::File>>,
}

impl Frames {
    /// The number of frames of the video at `path`.
    fn count(path: &Path) -> Result<usize> {
        let mut reader = y4m::open(path)?;
        let mut count = 0;
        while reader.next_frame()?.is_some() {
            count += 1;
        }
        Ok(count)
    }

    fn next(&mut self) -> Result<[Vec<u8>; 3]> {
        if let Some(frame) = self.reader.next_frame()? {
            return Ok(frame);
        }
        self.reader = y4m::open(&self.path)?;
        self.reader
            .next_frame()?
            .ok_or_else(|| "the Y4M file has no frames".into())
    }
}

/// The CPU or the GPU encoder.
enum Encoder {
    Cpu(Box<encode::Encoder>),
    Gpu(Box<gpu::Encoder>),
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
    fn encode(&mut self, encoder: &mut Encoder, planes: [&[u8]; 3], budget: usize) -> Result<()> {
        match encoder {
            Encoder::Cpu(encoder) => {
                let start = Instant::now();
                encoder.encode(planes, budget)?;
                self.total.samples.push(ms(start.elapsed()));
            }
            Encoder::Gpu(encoder) => {
                let (_, timings) = encoder.encode_timed(planes, budget)?;
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
    let reader = y4m::open(&args.input)?;
    let header = *reader.header();
    if header.chroma == ChromaSubsampling::Yuv420 && (header.width | header.height) % 2 != 0 {
        return Err(format!(
            "{}x{} is odd, and 4:2:0 needs an even width and height",
            header.width, header.height
        )
        .into());
    }
    let (width, height) = (u32::try_from(header.width)?, u32::try_from(header.height)?);
    let mut frames = Frames {
        path: args.input.clone(),
        reader,
    };

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
    let count = match args.frames {
        Some(count) => count,
        None => Frames::count(&args.input)?,
    };
    if count == 0 {
        return Err("the Y4M file has no frames".into());
    }

    // Only the encoder's call is timed, not the reading of the video.
    let mut measurement = Measurement::new();
    let warmup = warm_up(args.warmup, args.warmup_seconds, |_| {
        let planes = frames.next()?;
        measurement.encode(
            &mut encoder,
            [&planes[0], &planes[1], &planes[2]],
            args.bytes_per_frame,
        )
    })?;
    let mut measurement = Measurement::new();
    for _ in 0..count {
        let planes = frames.next()?;
        measurement.encode(
            &mut encoder,
            [&planes[0], &planes[1], &planes[2]],
            args.bytes_per_frame,
        )?;
    }

    let description = format!(
        "{width}x{height} {}, {} bytes per frame, encoded on the {}",
        chroma_name(header.chroma),
        args.bytes_per_frame,
        if args.gpu { "GPU" } else { "CPU" }
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
