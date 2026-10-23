//! `emberwave-cli compare`: compare two Y4M files plane by plane.

use std::path::PathBuf;

use crate::{Result, chroma_name, y4m};

#[derive(Debug, clap::Args)]
pub struct Args {
    /// The decoded file
    a: PathBuf,

    /// The reference to compare it with
    b: PathBuf,

    /// Fail unless every plane of every frame reaches this PSNR in dB
    #[arg(long)]
    min_psnr: Option<f64>,
}

pub fn run(args: &Args) -> Result<()> {
    let a = y4m::read(&args.a)?;
    let b = y4m::read(&args.b)?;
    if (a.width, a.height, a.chroma) != (b.width, b.height, b.chroma) {
        return Err(format!(
            "the files differ in format: {}x{} {} against {}x{} {}",
            a.width,
            a.height,
            chroma_name(a.chroma),
            b.width,
            b.height,
            chroma_name(b.chroma)
        )
        .into());
    }
    if a.frames.len() != b.frames.len() {
        return Err(format!(
            "the files have different numbers of frames: {} against {}",
            a.frames.len(),
            b.frames.len()
        )
        .into());
    }

    // Nothing compared is not a pass: `worst` would stay infinite.
    if a.frames.is_empty() {
        return Err("the files have no frames to compare".into());
    }

    let mut worst = f64::INFINITY;
    for (index, (frame_a, frame_b)) in a.frames.iter().zip(&b.frames).enumerate() {
        let mut line = format!("frame {index:>4}:");
        for (name, (plane_a, plane_b)) in ["Y", "Cb", "Cr"].iter().zip(frame_a.iter().zip(frame_b))
        {
            let db = psnr(plane_a, plane_b);
            worst = worst.min(db);
            line.push_str(&format!(
                "  {name} {} dB (max diff {})",
                format_db(db),
                max_diff(plane_a, plane_b)
            ));
        }
        println!("{line}");
    }
    println!("worst PSNR: {} dB", format_db(worst));

    match args.min_psnr {
        Some(min) if worst < min => Err(format!(
            "worst PSNR {} dB is below the required {min} dB",
            format_db(worst)
        )
        .into()),
        _ => Ok(()),
    }
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
