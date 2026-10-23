//! `emberwave-cli encode`: encode a Y4M file to a PWV1 file, each frame within
//! a byte budget.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::ValueEnum;
use emberwave_core::bitstream::consts::HEADER_BYTES;
use emberwave_core::bitstream::{
    ChromaSubsampling, ColorPrimaries, Pwv1Header, TransferFunction, YcbcrTransform,
};
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

    /// The most bytes a frame may take, its start of frame header included;
    /// at least 8
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
    // Everything that can be refused is refused before a file is written.
    if args.bytes_per_frame < HEADER_BYTES {
        return Err(format!(
            "--bytes-per-frame {} is less than the {HEADER_BYTES} bytes of a frame header",
            args.bytes_per_frame
        )
        .into());
    }
    if args.frames == Some(0) {
        return Err("--frames 0 leaves nothing to encode".into());
    }
    let mut video = y4m::open(&args.input)?;
    let header = *video.header();
    if header.chroma == ChromaSubsampling::Yuv420 && (header.width | header.height) % 2 != 0 {
        return Err(format!(
            "{}x{} is odd, and 4:2:0 needs an even width and height",
            header.width, header.height
        )
        .into());
    }

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
        ycbcr_range: header.range,
        chroma_siting: header.siting,
    };
    let width = u32::try_from(header.width)?;
    let height = u32::try_from(header.height)?;
    let mut encoder = Encoder::new(width, height, header.chroma, signalling)?;
    eprintln!(
        "{}x{} {}, {} chroma siting, {:?} range; {} bytes per frame at most",
        width,
        height,
        chroma_name(header.chroma),
        siting_name(header.siting),
        header.range,
        args.bytes_per_frame
    );

    // A PWV1 header has no frame rate when either number is zero.
    let (num, den) = header.frame_rate;
    let frame_rate = if num == 0 || den == 0 {
        (0, 0)
    } else {
        (num, den)
    };
    let file_header = Pwv1Header {
        stream: encoder.stream(),
        frame_rate_num: frame_rate.0,
        frame_rate_den: frame_rate.1,
        reference_bit_depth: 8,
    };

    // The file appears under its name only once it is complete; a failure
    // leaves whatever was there before.
    let partial = partial_path(&args.out);
    let result = (|| -> Result<(usize, usize, f64)> {
        let mut out = BufWriter::new(File::create(&partial)?);
        out.write_all(&file_header.to_bytes())?;
        let limit = args.frames.unwrap_or(usize::MAX);
        let (mut count, mut bytes, mut seconds) = (0usize, 0usize, 0.0f64);
        while count < limit {
            let Some([y, cb, cr]) = video.next_frame()? else {
                break;
            };
            let start = Instant::now();
            let frame = encoder.encode([&y, &cb, &cr], args.bytes_per_frame)?;
            let elapsed = start.elapsed().as_secs_f64();
            out.write_all(&u32::try_from(frame.len())?.to_le_bytes())?;
            out.write_all(&frame)?;
            eprintln!(
                "frame {count:6}: {} bytes, {:.1} ms",
                frame.len(),
                elapsed * 1e3
            );
            count += 1;
            bytes += frame.len();
            seconds += elapsed;
        }
        if count == 0 {
            return Err("the Y4M file has no frames".into());
        }
        out.into_inner()
            .map_err(|error| error.into_error())?
            .sync_all()?;
        Ok((count, bytes, seconds))
    })();
    let (count, bytes, seconds) = match result {
        Ok(totals) => totals,
        Err(error) => {
            let _ = std::fs::remove_file(&partial);
            return Err(error);
        }
    };
    std::fs::rename(&partial, &args.out)?;

    println!(
        "encoded {count} frames to {}: {} bytes per frame and {:.1} ms on average",
        args.out.display(),
        bytes / count,
        seconds / count as f64 * 1e3
    );
    Ok(())
}

/// Where the file is written until it is complete: next to it, so that the
/// rename does not cross file systems.
fn partial_path(out: &Path) -> PathBuf {
    let mut name = out.file_name().unwrap_or_default().to_os_string();
    name.push(".partial");
    out.with_file_name(name)
}
