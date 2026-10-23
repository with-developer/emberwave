//! `emberwave-cli encode`: encode a Y4M file to a PWV1 file, each frame within
//! a byte budget.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::ValueEnum;
use emberwave_core::bitstream::{ColorPrimaries, Pwv1Header, TransferFunction, YcbcrTransform};
use emberwave_core::encode::{Encoder, Signalling};

use crate::{Result, chroma_name, siting_name, y4m};

/// The Y'CbCr matrix to signal, which a Y4M file does not say.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SignalledMatrix {
    /// ITU-R BT.709, with sRGB primaries and transfer
    Bt709,
    /// ITU-R BT.2020, with BT.2020 primaries and the PQ transfer of HDR10
    Bt2020,
}

#[derive(Debug, clap::Args)]
pub struct Args {
    /// The Y4M file to encode: 8-bit 4:2:0 or 4:4:4
    input: PathBuf,

    /// The PWV1 file to write
    #[arg(long)]
    out: PathBuf,

    /// The most bytes a frame may take, its start of frame header included
    #[arg(long)]
    bytes_per_frame: usize,

    /// Encode at most this many frames
    #[arg(long)]
    frames: Option<usize>,

    /// The colours to signal; the range and the chroma siting come from the
    /// Y4M header
    #[arg(long, value_enum, default_value_t = SignalledMatrix::Bt709)]
    matrix: SignalledMatrix,
}

pub fn run(args: &Args) -> Result<()> {
    let video = y4m::read(&args.input)?;
    let (primaries, transfer, transform) = match args.matrix {
        SignalledMatrix::Bt709 => (
            ColorPrimaries::Srgb,
            TransferFunction::Srgb,
            YcbcrTransform::Bt709,
        ),
        SignalledMatrix::Bt2020 => (
            ColorPrimaries::Bt2020,
            TransferFunction::Pq,
            YcbcrTransform::Bt2020,
        ),
    };
    let signalling = Signalling {
        color_primaries: primaries,
        transfer_function: transfer,
        ycbcr_transform: transform,
        ycbcr_range: video.range,
        chroma_siting: video.siting,
    };
    let width = u32::try_from(video.width)?;
    let height = u32::try_from(video.height)?;
    let mut encoder = Encoder::new(width, height, video.chroma, signalling)?;
    eprintln!(
        "{}x{} {}, {} chroma siting, {:?} range; {} bytes per frame at most",
        width,
        height,
        chroma_name(video.chroma),
        siting_name(video.siting),
        video.range,
        args.bytes_per_frame
    );

    let header = Pwv1Header {
        stream: encoder.stream(),
        frame_rate_num: video.frame_rate.0,
        frame_rate_den: video.frame_rate.1,
        reference_bit_depth: 8,
    };
    let mut out = BufWriter::new(File::create(&args.out)?);
    out.write_all(&header.to_bytes())?;

    let limit = args.frames.unwrap_or(usize::MAX);
    let (mut bytes, mut seconds, mut count) = (0usize, 0.0f64, 0usize);
    for (index, [y, cb, cr]) in video.frames.iter().take(limit).enumerate() {
        let start = Instant::now();
        let frame = encoder.encode([y, cb, cr], args.bytes_per_frame)?;
        let elapsed = start.elapsed().as_secs_f64();
        out.write_all(&u32::try_from(frame.len())?.to_le_bytes())?;
        out.write_all(&frame)?;
        eprintln!(
            "frame {index:6}: {} bytes, {:.1} ms",
            frame.len(),
            elapsed * 1e3
        );
        bytes += frame.len();
        seconds += elapsed;
        count += 1;
    }
    out.flush()?;

    if count == 0 {
        return Err("the Y4M file has no frames".into());
    }
    println!(
        "encoded {count} frames to {}: {} bytes per frame and {:.1} ms on average",
        args.out.display(),
        bytes / count,
        seconds / count as f64 * 1e3
    );
    Ok(())
}
