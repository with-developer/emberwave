//! `emberwave-cli decode`: decode a file to PNG images or a Y4M video.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::ValueEnum;
use emberwave_core::bitstream::Pwv1File;
use emberwave_core::gpu::Decoder;

use crate::stats::Summary;
use crate::{ColorArgs, GpuArgs, MatrixChoice, RangeChoice, Result, read_file, y4m};

/// Output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Format {
    /// One RGB PNG per frame
    Png,
    /// One Y4M file with the YUV planes of all frames, without colour conversion
    Y4m,
}

#[derive(Debug, clap::Args)]
pub struct Args {
    /// The file to decode
    input: PathBuf,

    /// Directory for the output; created if it does not exist
    #[arg(long)]
    out_dir: PathBuf,

    /// Output format
    #[arg(long, value_enum, default_value_t = Format::Png)]
    format: Format,

    /// Decode at most this many frames
    #[arg(long)]
    frames: Option<usize>,

    #[command(flatten)]
    gpu: GpuArgs,

    #[command(flatten)]
    color: ColorArgs,
}

pub fn run(args: &Args) -> Result<()> {
    if args.frames == Some(0) {
        return Err("--frames must be at least 1".into());
    }
    // The conversion options do nothing to planes that are written as they
    // are; accepting them would let the file say something the data does not.
    if args.format == Format::Y4m
        && (args.color.matrix != MatrixChoice::Auto || args.color.range != RangeChoice::Auto)
    {
        return Err(
            "--matrix and --range set the RGB conversion; Y4M output keeps the planes as decoded"
                .into(),
        );
    }

    let bytes = read_file(&args.input)?;
    let file = Pwv1File::parse(&bytes)?;
    let header = *file.header();
    let stream = header.stream;
    // Before anything is written, as bench and compare refuse it too.
    if file.frames().next().is_none() {
        return Err("the file has no frames".into());
    }

    // Before the GPU: a directory that cannot be made fails at once.
    std::fs::create_dir_all(&args.out_dir)
        .map_err(|error| format!("{}: {error}", args.out_dir.display()))?;
    let context = args.gpu.create_context(false)?;
    let mut decoder = Decoder::new(&context, stream.width, stream.height, stream.chroma)?;
    let color = args.color.resolve(&stream);

    let stem = args
        .input
        .file_stem()
        .map_or_else(|| "frame".to_owned(), |s| s.to_string_lossy().into_owned());

    let mut video = match args.format {
        Format::Y4m => {
            let info = y4m::StreamInfo {
                width: stream.width,
                height: stream.height,
                chroma: stream.chroma,
                siting: stream.chroma_siting,
                frame_rate: (header.frame_rate_num, header.frame_rate_den),
                range: stream.ycbcr_range,
            };
            if !info.has_frame_rate() {
                let (num, den) = y4m::DEFAULT_FRAME_RATE;
                err!("warning: the file has no frame rate, the Y4M file says {num}:{den}");
            }
            Some(y4m::Writer::create(
                &args.out_dir.join(format!("{stem}.y4m")),
                &info,
            )?)
        }
        Format::Png => None,
    };

    let limit = args.frames.unwrap_or(usize::MAX);
    let mut times_ms = Vec::new();
    for (index, frame) in file.frames().enumerate().take(limit) {
        let frame = frame?;

        // The time covers everything up to having the picture on the CPU.
        let start = Instant::now();
        match args.format {
            Format::Png => {
                let rgb = decoder.decode_rgb(frame, &color)?;
                let elapsed = start.elapsed();
                let path = args.out_dir.join(format!("{stem}_{index:06}.png"));
                write_png(&path, rgb.width, rgb.height, &rgb.data)
                    .map_err(|error| format!("{}: {error}", path.display()))?;
                times_ms.push(elapsed.as_secs_f64() * 1000.0);
            }
            Format::Y4m => {
                let yuv = decoder.decode(frame)?;
                times_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                if let Some(video) = video.as_mut() {
                    let [y, cb, cr] = &yuv.planes;
                    video.write_frame([&y.data, &cb.data, &cr.data])?;
                }
            }
        }
        err!("frame {index:06}: {:.3} ms", times_ms[index]);
    }

    if let Some(video) = video {
        video.finish()?;
    }

    match Summary::of(&times_ms) {
        Some(summary) => out!(
            "decoded {} frames to {}: mean {:.3} ms, p50 {:.3} ms, p99 {:.3} ms per frame",
            times_ms.len(),
            args.out_dir.display(),
            summary.mean,
            summary.p50,
            summary.p99
        ),
        None => return Err("the file has no frames".into()),
    }
    Ok(())
}

fn write_png(path: &Path, width: u32, height: u32, rgb: &[u8]) -> Result<()> {
    let mut encoder = png::Encoder::new(BufWriter::new(File::create(path)?), width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    // Speed over size: these images are for looking at and for tests.
    encoder.set_compression(png::Compression::Fast);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(rgb)?;
    // Writes the end of the file and flushes it, and says if that failed,
    // which dropping the writer would not.
    writer.finish()?;
    Ok(())
}
