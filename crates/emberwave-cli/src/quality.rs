//! `emberwave-cli quality`: measure a stream against the pictures it was
//! encoded from, in PSNR and in the error the encoder's rate control
//! minimises.

use std::path::PathBuf;

use emberwave_core::bitstream::pwv1::Pwv1File;
use emberwave_core::bitstream::{FrameAssembler, FrameLayout};
use emberwave_core::cpu::decode_frame;
use emberwave_core::encode::{CoefficientError, coefficient_error};

use crate::compare::format_db;
use crate::{Result, chroma_name, read_file, y4m};

#[derive(Debug, clap::Args)]
pub struct Args {
    /// The pictures the stream was encoded from: an 8-bit Y4M file
    source: PathBuf,

    /// The stream: a PWV1 file from any encoder, upstream's too
    stream: PathBuf,

    /// Measure the first N frames; the files may have more. Without it, they
    /// must have as many as each other
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u64).range(1..))]
    frames: Option<u64>,
}

pub fn run(args: &Args) -> Result<()> {
    let bytes = read_file(&args.stream)?;
    let file = Pwv1File::parse(&bytes).map_err(|error| in_stream(args, error))?;
    let stream = &file.header().stream;
    let layout = FrameLayout::new(stream.width, stream.height, stream.chroma)?;
    let mut source = y4m::open(&args.source)?;
    let header = *source.header();
    let size = (stream.width as usize, stream.height as usize, stream.chroma);
    if (header.width, header.height, header.chroma) != size {
        return Err(format!(
            "the pictures are {}x{} {} and the stream {}x{} {}",
            header.width,
            header.height,
            chroma_name(header.chroma),
            stream.width,
            stream.height,
            chroma_name(stream.chroma)
        )
        .into());
    }

    let limit = args.frames.unwrap_or(u64::MAX);
    let mut assembler = FrameAssembler::new(layout);
    let mut frames = file.frames();
    let mut count = 0usize;
    let mut bytes_total = 0usize;
    // Per plane, over all frames: squared differences and samples.
    let mut squared_total = [0u64; 3];
    let mut samples_total = [0u64; 3];
    let mut error_total = CoefficientError::default();
    while (count as u64) < limit {
        let picture = source
            .next_frame()
            .map_err(|error| format!("{}: {error}", args.source.display()))?;
        let frame = frames
            .next()
            .transpose()
            .map_err(|error| in_stream(args, error))?;
        let (picture, frame) = match (picture, frame) {
            (Some(picture), Some(frame)) => (picture, frame),
            (None, None) => break,
            (None, Some(_)) | (Some(_), None) => {
                return Err(format!(
                    "the pictures and the stream have different numbers of frames: one ends after \
                     {count}; --frames measures the first ones"
                )
                .into());
            }
        };
        assembler.clear();
        assembler
            .push_packet(frame)
            .map_err(|error| in_frame(args, count, error))?;
        if !assembler.is_ready(false) {
            return Err(in_frame(
                args,
                count,
                format!(
                    "incomplete: {} of {} blocks",
                    assembler.received_blocks(),
                    assembler.total_blocks()
                ),
            ));
        }
        let view = assembler.frame();
        let planes = [&picture[0][..], &picture[1][..], &picture[2][..]];
        let error =
            coefficient_error(planes, &view).map_err(|error| in_frame(args, count, error))?;
        let decoded = decode_frame(&view).map_err(|error| in_frame(args, count, error))?;

        let mut line = format!("frame {count:>4}: {:>9} bytes", frame.len());
        for (i, name) in ["Y", "Cb", "Cr"].iter().enumerate() {
            let squared = squared_difference(&decoded.planes[i].to_unorm8(), &picture[i]);
            let samples = picture[i].len() as u64;
            squared_total[i] += squared;
            samples_total[i] += samples;
            line.push_str(&format!(
                "  {name} {} dB",
                format_db(psnr(squared, samples))
            ));
        }
        line.push_str(&format!("  {}", describe(&error)));
        out!("{line}");

        bytes_total += frame.len();
        for i in 0..3 {
            error_total.weighted[i] += error.weighted[i];
            error_total.squared[i] += error.squared[i];
        }
        count += 1;
    }
    if count == 0 {
        return Err("there are no frames to measure".into());
    }

    // The PSNR of all frames together, from their mean squared difference,
    // which a frame without any keeps finite.
    let frames = count as f64;
    let mut line = format!("mean:       {:>9} bytes", bytes_total / count);
    for (i, name) in ["Y", "Cb", "Cr"].iter().enumerate() {
        let db = psnr(squared_total[i], samples_total[i]);
        line.push_str(&format!("  {name} {} dB", format_db(db)));
    }
    let mean = CoefficientError {
        weighted: error_total.weighted.map(|e| e / frames),
        squared: error_total.squared.map(|e| e / frames),
    };
    line.push_str(&format!("  {}", describe(&mean)));
    out!("{line}");
    Ok(())
}

/// The sum of the squared differences of two 8-bit planes.
fn squared_difference(a: &[u8], b: &[u8]) -> u64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| u64::from(x.abs_diff(y)).pow(2))
        .sum()
}

/// PSNR in dB of `squared` over `samples` 8-bit samples; infinite for none.
fn psnr(squared: u64, samples: u64) -> f64 {
    if squared == 0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / (squared as f64 / samples as f64)).log10()
}

/// The weighted coefficient error, in total and per component.
fn describe(error: &CoefficientError) -> String {
    let [y, cb, cr] = error.weighted;
    format!(
        "weighted error {:.4e} (Y {y:.3e}, Cb {cb:.3e}, Cr {cr:.3e})",
        error.weighted_total()
    )
}

/// `error` with the stream it came from.
fn in_stream(args: &Args, error: impl std::fmt::Display) -> Box<dyn std::error::Error> {
    format!("{}: {error}", args.stream.display()).into()
}

/// `error` with the stream and the frame it came from.
fn in_frame(
    args: &Args,
    frame: usize,
    error: impl std::fmt::Display,
) -> Box<dyn std::error::Error> {
    format!("{}: frame {frame}: {error}", args.stream.display()).into()
}
