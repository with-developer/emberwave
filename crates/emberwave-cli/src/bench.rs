//! `emberwave-cli bench`: measure the decode time per frame.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::ValueEnum;
use emberwave_core::bitstream::Pwv1File;
use emberwave_core::gpu::{Decoder, TimedOutput, Timings};

use crate::stats::Summary;
use crate::{ColorArgs, GpuArgs, Result, chroma_name};

/// What the benchmark decodes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Output {
    /// Y, Cb and Cr planes (8-bit) in GPU memory
    Yuv,
    /// RGB in GPU memory: also runs the colour conversion
    Rgb,
}

#[derive(Debug, clap::Args)]
pub struct Args {
    /// The file to decode; its frames are decoded in turn, as often as needed
    input: PathBuf,

    /// Number of frames to measure (default: one pass over the file)
    #[arg(long)]
    frames: Option<usize>,

    /// Frames decoded first and not counted, at least
    #[arg(long, default_value_t = 10)]
    warmup: usize,

    /// Seconds of decoding before the counted frames, at least: a GPU runs
    /// slower for the first moments of a load
    #[arg(long, default_value_t = 1.0, value_parser = parse_seconds)]
    warmup_seconds: f64,

    /// What to decode to
    #[arg(long, value_enum, default_value_t = Output::Yuv)]
    output: Output,

    /// Also copy the picture back to the CPU, and time all of it
    #[arg(long)]
    readback: bool,

    /// Print the result as JSON
    #[arg(long)]
    json: bool,

    #[command(flatten)]
    gpu: GpuArgs,

    #[command(flatten)]
    color: ColorArgs,
}

/// Parses a number of seconds: finite, and between 0 and an hour.
fn parse_seconds(text: &str) -> std::result::Result<f64, String> {
    let seconds: f64 = text.parse().map_err(|error| format!("{error}"))?;
    if (0.0..=3600.0).contains(&seconds) {
        Ok(seconds)
    } else {
        Err("must be between 0 and 3600".to_owned())
    }
}

/// A named series of timings in milliseconds.
struct Series {
    name: &'static str,
    samples: Vec<f64>,
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

pub fn run(args: &Args) -> Result<()> {
    if args.frames == Some(0) {
        return Err("--frames must be at least 1".into());
    }

    let bytes = std::fs::read(&args.input)?;
    let file = Pwv1File::parse(&bytes)?;
    let stream = file.header().stream;
    let frames: Vec<&[u8]> = file.frames().collect::<std::result::Result<_, _>>()?;
    if frames.is_empty() {
        return Err("the file has no frames".into());
    }

    let context = args.gpu.create_context(true)?;
    let adapter = context.adapter_info();
    let mut decoder = Decoder::new(&context, stream.width, stream.height, stream.chroma)?;
    let color = args.color.resolve(&stream);
    let output = match args.output {
        Output::Yuv => TimedOutput::Yuv,
        Output::Rgb => TimedOutput::Rgb(color),
    };

    let count = args.frames.unwrap_or(frames.len());

    // Decodes frame `i` (the file's frames in turn) and returns the timings.
    let mut decode = |i: usize| -> Result<Timings> {
        let frame = frames[i % frames.len()];
        if args.readback {
            let start = Instant::now();
            match args.output {
                Output::Yuv => {
                    decoder.decode(frame)?;
                }
                Output::Rgb => {
                    decoder.decode_rgb(frame, &color)?;
                }
            }
            Ok(Timings {
                prepare: start.elapsed(),
                gpu_wall: Duration::ZERO,
                stages: None,
            })
        } else {
            Ok(decoder.time_frame(frame, output)?)
        }
    };

    // Warm up for a number of frames and for a while: a GPU that has been idle
    // clocks down, and runs the first moments of a load noticeably slower.
    let warmup_time = Duration::from_secs_f64(args.warmup_seconds);
    let warmup_start = Instant::now();
    let mut warmup = 0;
    while warmup < args.warmup || warmup_start.elapsed() < warmup_time {
        decode(warmup)?;
        warmup += 1;
    }

    let mut total = Series {
        name: "total",
        samples: Vec::new(),
    };
    let mut prepare = Series {
        name: "cpu prepare",
        samples: Vec::new(),
    };
    let mut gpu_wall = Series {
        name: "gpu wall",
        samples: Vec::new(),
    };
    let mut dequant = Series {
        name: "dequant",
        samples: Vec::new(),
    };
    let mut idwt = Series {
        name: "idwt",
        samples: Vec::new(),
    };
    let mut yuv2rgb = Series {
        name: "yuv2rgb",
        samples: Vec::new(),
    };
    let mut gpu_sum = Series {
        name: "gpu (timestamps)",
        samples: Vec::new(),
    };

    for i in 0..count {
        let timings = decode(i)?;
        if args.readback {
            total.samples.push(ms(timings.prepare));
            continue;
        }

        total.samples.push(ms(timings.prepare + timings.gpu_wall));
        prepare.samples.push(ms(timings.prepare));
        gpu_wall.samples.push(ms(timings.gpu_wall));
        if let Some(stages) = timings.stages {
            dequant.samples.push(ms(stages.dequant));
            idwt.samples.push(ms(stages.idwt));
            if let Some(duration) = stages.yuv2rgb {
                yuv2rgb.samples.push(ms(duration));
            }
            gpu_sum.samples.push(ms(stages.total()));
        }
    }

    let series: Vec<&Series> = [
        &total, &prepare, &gpu_wall, &gpu_sum, &dequant, &idwt, &yuv2rgb,
    ]
    .into_iter()
    .filter(|s| !s.samples.is_empty())
    .collect();

    let description = format!(
        "{}x{} {}, decoded to {}{}",
        stream.width,
        stream.height,
        chroma_name(stream.chroma),
        match args.output {
            Output::Yuv => "YUV",
            Output::Rgb => "RGB",
        },
        if args.readback {
            " and read back to the CPU"
        } else {
            " in GPU memory"
        },
    );

    if args.json {
        println!(
            "{}",
            json(
                args,
                &adapter.name,
                &format!("{:?}", adapter.backend),
                &description,
                (count, warmup),
                &series
            )
        );
    } else {
        println!("adapter:  {} ({:?})", adapter.name, adapter.backend);
        println!("decoding: {description}");
        println!("frames:   {count} measured after {warmup} warm-up frames");
        println!();
        println!(
            "{:<20}{:>9}{:>9}{:>9}{:>9}{:>9}   (ms per frame)",
            "", "mean", "p50", "p99", "min", "max"
        );
        for s in series {
            if let Some(summary) = Summary::of(&s.samples) {
                println!(
                    "{:<20}{:>9.3}{:>9.3}{:>9.3}{:>9.3}{:>9.3}",
                    s.name, summary.mean, summary.p50, summary.p99, summary.min, summary.max
                );
            }
        }
    }
    Ok(())
}

/// The result as JSON. Built by hand: the shape is small and flat. `frames` is
/// the number of measured frames and of warm-up frames that were decoded.
fn json(
    args: &Args,
    adapter: &str,
    backend: &str,
    description: &str,
    (frames, warmup): (usize, usize),
    series: &[&Series],
) -> String {
    let mut out = String::new();
    let _ = write!(
        out,
        "{{\"input\":{},\"adapter\":{},\"backend\":{},\"description\":{},\"frames\":{},\"warmup\":{},\"readback\":{},\"unit\":\"ms\",\"series\":{{",
        quote(&args.input.display().to_string()),
        quote(adapter),
        quote(backend),
        quote(description),
        frames,
        warmup,
        args.readback,
    );
    for (i, s) in series.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        if let Some(summary) = Summary::of(&s.samples) {
            let _ = write!(
                out,
                "{}:{{\"mean\":{:.6},\"p50\":{:.6},\"p99\":{:.6},\"min\":{:.6},\"max\":{:.6}}}",
                quote(s.name),
                summary.mean,
                summary.p50,
                summary.p99,
                summary.min,
                summary.max
            );
        }
    }
    out.push_str("}}");
    out
}

/// A JSON string literal.
fn quote(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_warm_up_time_must_be_a_sensible_number_of_seconds() {
        assert_eq!(parse_seconds("0"), Ok(0.0));
        assert_eq!(parse_seconds("1.5"), Ok(1.5));
        for bad in ["-1", "nan", "inf", "3601", "soon"] {
            assert!(parse_seconds(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn json_strings_are_escaped() {
        assert_eq!(quote("plain"), "\"plain\"");
        assert_eq!(quote("a\"b\\c"), "\"a\\\"b\\\\c\"");
        assert_eq!(quote("line\nbreak\ttab"), "\"line\\nbreak\\ttab\"");
        assert_eq!(quote("\u{1}"), "\"\\u0001\"");
        assert_eq!(quote("한국어"), "\"한국어\"");
    }
}
