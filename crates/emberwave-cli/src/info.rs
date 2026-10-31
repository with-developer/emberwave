//! `emberwave-cli info`: print the header and frame statistics of a file.

use std::path::PathBuf;

use emberwave_core::bitstream::{
    ColorPrimaries, FrameLayout, Header, Pwv1File, TransferFunction, YcbcrRange, YcbcrTransform,
};

use crate::stats::Summary;
use crate::{Result, chroma_name, read_file, siting_name};

#[derive(Debug, clap::Args)]
pub struct Args {
    /// The file to inspect
    input: PathBuf,

    /// List every frame
    #[arg(long)]
    verbose: bool,
}

pub fn run(args: &Args) -> Result<()> {
    let bytes = read_file(&args.input)?;
    let file = Pwv1File::parse(&bytes)?;
    let header = file.header();
    let stream = &header.stream;
    let layout = FrameLayout::new(stream.width, stream.height, stream.chroma)?;

    out!(
        "file:          {} ({} bytes)",
        args.input.display(),
        bytes.len()
    );
    out!("format:        PWV1");
    out!(
        "picture:       {}x{}, {} (padded to {}x{} for the transform)",
        stream.width,
        stream.height,
        chroma_name(stream.chroma),
        layout.aligned_width(),
        layout.aligned_height()
    );
    out!(
        "colour:        {} matrix, {} range, {} chroma siting",
        match stream.ycbcr_transform {
            YcbcrTransform::Bt709 => "BT.709",
            YcbcrTransform::Bt2020 => "BT.2020",
        },
        match stream.ycbcr_range {
            YcbcrRange::Full => "full",
            YcbcrRange::Limited => "limited",
        },
        siting_name(stream.chroma_siting)
    );
    out!(
        "signalling:    {} primaries, {} transfer",
        match stream.color_primaries {
            ColorPrimaries::Srgb => "sRGB",
            ColorPrimaries::Bt2020 => "BT.2020",
        },
        match stream.transfer_function {
            TransferFunction::Srgb => "sRGB",
            TransferFunction::Pq => "PQ",
        }
    );

    let frame_rate = if header.frame_rate_num == 0 || header.frame_rate_den == 0 {
        None
    } else {
        Some(f64::from(header.frame_rate_num) / f64::from(header.frame_rate_den))
    };
    match frame_rate {
        Some(fps) => out!(
            "frame rate:    {}/{} ({fps:.3} fps)",
            header.frame_rate_num,
            header.frame_rate_den
        ),
        None => out!("frame rate:    none (unrelated images)"),
    }
    match header.reference_bit_depth {
        0 => out!("bit depth:     arbitrary"),
        bits => out!("bit depth:     {bits} (of the source video)"),
    }
    out!(
        "block layout:  {} blocks of 32x32 coefficients at most per frame",
        layout.block_count()
    );

    let mut sizes = Vec::new();
    let mut truncated = false;
    for (index, frame) in file.frames().enumerate() {
        let frame = match frame {
            Ok(frame) => frame,
            Err(error) => {
                out!("frame {index:>6}: {error}");
                truncated = true;
                break;
            }
        };
        sizes.push(frame.len() as f64);

        if args.verbose {
            match Header::parse(frame) {
                Ok(Header::StartOfFrame(sof)) => out!(
                    "frame {index:>6}: {:>9} bytes, sequence {}, {} blocks announced",
                    frame.len(),
                    sof.sequence,
                    sof.total_blocks
                ),
                Ok(Header::Block(_)) => out!(
                    "frame {index:>6}: {:>9} bytes, no start of frame header first",
                    frame.len()
                ),
                Ok(_) => out!(
                    "frame {index:>6}: {:>9} bytes, a header this version does not know",
                    frame.len()
                ),
                Err(error) => out!("frame {index:>6}: {:>9} bytes, {error}", frame.len()),
            }
        }
    }

    match Summary::of(&sizes) {
        Some(summary) => {
            out!(
                "frames:        {}{} (bytes per frame: min {:.0}, mean {:.0}, max {:.0})",
                sizes.len(),
                if truncated {
                    ", the last one is cut off"
                } else {
                    ""
                },
                summary.min,
                summary.mean,
                summary.max
            );
            if let Some(fps) = frame_rate {
                out!(
                    "bit rate:      {:.1} Mbit/s on average at {fps:.3} fps",
                    summary.mean * 8.0 * fps / 1e6
                );
            }
        }
        None => out!("frames:        0"),
    }
    Ok(())
}
