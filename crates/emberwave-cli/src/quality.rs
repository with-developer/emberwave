//! `emberwave-cli quality`: measure a stream against the pictures it was
//! encoded from, in PSNR and in the error the encoder's rate control
//! minimises.

use std::path::PathBuf;

use emberwave_core::bitstream::pwv1::Pwv1File;
use emberwave_core::bitstream::{FrameAssembler, FrameLayout};
use emberwave_core::cpu::decode_frame;
use emberwave_core::encode::{CoefficientError, coefficient_error};

use crate::compare::{format_db, psnr};
use crate::{Result, chroma_name, read_file, y4m};

#[derive(Debug, clap::Args)]
pub struct Args {
    /// The pictures the stream was encoded from: an 8-bit Y4M file
    source: PathBuf,

    /// The stream: a PWV1 file from any encoder, upstream's too
    stream: PathBuf,
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

    let mut assembler = FrameAssembler::new(layout);
    let mut frames = file.frames();
    let mut count = 0usize;
    let mut bytes_total = 0usize;
    let mut psnr_total = [0.0f64; 3];
    let mut error_total = CoefficientError::default();
    loop {
        let picture = source.next_frame()?;
        let frame = frames
            .next()
            .transpose()
            .map_err(|error| in_stream(args, error))?;
        let (picture, frame) = match (picture, frame) {
            (Some(picture), Some(frame)) => (picture, frame),
            (None, None) => break,
            (Some(_), None) | (None, Some(_)) => {
                return Err(format!(
                    "the pictures and the stream have different numbers of frames: one ends after {count}"
                )
                .into());
            }
        };
        assembler.clear();
        assembler
            .push_packet(frame)
            .map_err(|error| in_stream(args, error))?;
        let view = assembler.frame();
        let planes = [&picture[0][..], &picture[1][..], &picture[2][..]];
        let error = coefficient_error(planes, &view)?;
        let decoded = decode_frame(&view).map_err(|error| in_stream(args, error))?;

        let mut line = format!("frame {count:>4}: {:>9} bytes", frame.len());
        for (i, name) in ["Y", "Cb", "Cr"].iter().enumerate() {
            let db = psnr(&decoded.planes[i].to_unorm8(), &picture[i]);
            psnr_total[i] += db;
            line.push_str(&format!("  {name} {} dB", format_db(db)));
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

    let frames = count as f64;
    let mut line = format!("mean:       {:>9} bytes", bytes_total / count);
    for (i, name) in ["Y", "Cb", "Cr"].iter().enumerate() {
        line.push_str(&format!(
            "  {name} {} dB",
            format_db(psnr_total[i] / frames)
        ));
    }
    let mean = CoefficientError {
        weighted: error_total.weighted.map(|e| e / frames),
        squared: error_total.squared.map(|e| e / frames),
    };
    line.push_str(&format!("  {}", describe(&mean)));
    out!("{line}");
    Ok(())
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
