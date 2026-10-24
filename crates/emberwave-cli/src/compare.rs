//! `emberwave-cli compare`: compare two Y4M files plane by plane.

use std::path::{Path, PathBuf};

use crate::{Result, chroma_name, y4m};

#[derive(Debug, clap::Args)]
pub struct Args {
    /// The decoded file
    a: PathBuf,

    /// The reference to compare it with
    b: PathBuf,

    /// Fail unless every plane of every frame reaches this PSNR in dB
    #[arg(long, value_parser = parse_finite)]
    min_psnr: Option<f64>,
}

/// A number that is not infinite or NaN: no PSNR is below NaN, so
/// `--min-psnr nan` would pass anything.
fn parse_finite(text: &str) -> std::result::Result<f64, String> {
    let value: f64 = text.parse().map_err(|error| format!("{error}"))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err("must be a finite number".to_owned())
    }
}

pub fn run(args: &Args) -> Result<()> {
    // Frame by frame, so that two large files do not have to fit in memory.
    let mut a = y4m::open(&args.a)?;
    let mut b = y4m::open(&args.b)?;
    let (ha, hb) = (*a.header(), *b.header());
    if (ha.width, ha.height, ha.chroma) != (hb.width, hb.height, hb.chroma) {
        return Err(format!(
            "the files differ in format: {}x{} {} against {}x{} {}",
            ha.width,
            ha.height,
            chroma_name(ha.chroma),
            hb.width,
            hb.height,
            chroma_name(hb.chroma)
        )
        .into());
    }

    let mut worst = f64::INFINITY;
    let mut index = 0;
    loop {
        let next_a = a.next_frame().map_err(|error| in_file(&args.a, error))?;
        let next_b = b.next_frame().map_err(|error| in_file(&args.b, error))?;
        let (frame_a, frame_b) = match (next_a, next_b) {
            (Some(frame_a), Some(frame_b)) => (frame_a, frame_b),
            (None, None) => break,
            (Some(_), None) | (None, Some(_)) => {
                return Err(format!(
                    "the files have different numbers of frames: one ends after {index}"
                )
                .into());
            }
        };
        let mut line = format!("frame {index:>4}:");
        for (name, (plane_a, plane_b)) in ["Y", "Cb", "Cr"].iter().zip(frame_a.iter().zip(&frame_b))
        {
            let db = psnr(plane_a, plane_b);
            worst = worst.min(db);
            line.push_str(&format!(
                "  {name} {} dB (max diff {})",
                format_db(db),
                max_diff(plane_a, plane_b)
            ));
        }
        out!("{line}");
        index += 1;
    }
    // Nothing compared is not a pass: `worst` would stay infinite.
    if index == 0 {
        return Err("the files have no frames to compare".into());
    }
    out!("worst PSNR: {} dB", format_db(worst));

    match args.min_psnr {
        Some(min) if worst < min => Err(format!(
            "worst PSNR {} dB is below the required {min} dB",
            format_db(worst)
        )
        .into()),
        _ => Ok(()),
    }
}

/// `error` with the file it came from.
fn in_file(path: &Path, error: Box<dyn std::error::Error>) -> Box<dyn std::error::Error> {
    format!("{}: {error}", path.display()).into()
}

/// PSNR in dB of two 8-bit planes; infinite if they are identical.
pub fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let squared_error: u64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = i64::from(x) - i64::from(y);
            (d * d) as u64
        })
        .sum();
    if squared_error == 0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / (squared_error as f64 / a.len() as f64)).log10()
}

fn max_diff(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x.abs_diff(y))
        .max()
        .unwrap_or(0)
}

fn format_db(db: f64) -> String {
    if db.is_infinite() {
        "   inf".to_owned()
    } else {
        format!("{db:6.2}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_planes_have_infinite_psnr() {
        assert_eq!(psnr(&[1, 2, 3], &[1, 2, 3]), f64::INFINITY);
    }

    #[test]
    fn psnr_of_a_known_error() {
        // One sample of four off by 16: MSE 64, PSNR 10 log10(255^2 / 64).
        let db = psnr(&[0, 0, 0, 0], &[16, 0, 0, 0]);
        assert!(
            (db - 10.0 * (255.0f64 * 255.0 / 64.0).log10()).abs() < 1e-9,
            "{db}"
        );
        assert_eq!(max_diff(&[0, 0, 0, 0], &[16, 0, 0, 0]), 16);
    }
}
