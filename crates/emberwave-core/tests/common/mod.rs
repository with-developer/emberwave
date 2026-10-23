//! Helpers shared by the integration tests: locating the committed test
//! vectors, reading reference decodes (Y4M) and measuring PSNR.

#![allow(dead_code)] // each test binary uses a different subset

use std::path::{Path, PathBuf};

/// Directory with the committed vectors and their reference decodes.
pub fn testdata_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

/// Names (without extension) of every committed `*.pwv1` vector, sorted.
pub fn vector_names() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(testdata_dir())
        .expect("testdata directory")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension()? == "pwv1" {
                Some(path.file_stem()?.to_str()?.to_owned())
            } else {
                None
            }
        })
        .collect();
    names.sort();
    assert!(
        !names.is_empty(),
        "no *.pwv1 vectors in {:?}",
        testdata_dir()
    );
    names
}

/// A vector file.
pub fn read_vector(name: &str) -> Vec<u8> {
    std::fs::read(testdata_dir().join(format!("{name}.pwv1"))).expect("vector file")
}

/// A decoded Y4M file with 8-bit planar frames.
pub struct Y4m {
    pub width: usize,
    pub height: usize,
    pub chroma_444: bool,
    /// Per frame: the Y, Cb and Cr planes.
    pub frames: Vec<[Vec<u8>; 3]>,
}

/// Directory the reference decodes are read from: `testdata/` unless the
/// `EMBERWAVE_REFERENCE_DIR` environment variable points elsewhere, for
/// example at decodes made by another upstream build or precision mode.
pub fn reference_dir() -> PathBuf {
    std::env::var_os("EMBERWAVE_REFERENCE_DIR").map_or_else(testdata_dir, PathBuf::from)
}

/// Which decode by the original decoder a test compares against.
#[derive(Debug, Clone, Copy)]
pub enum Reference {
    /// The decoder's default precision, which keeps the wavelet pyramid in
    /// FP16 (`NAME.ref.y4m`).
    Default,
    /// The decoder's FP32 precision, `PYROWAVE_PRECISION=2`
    /// (`NAME.ref-fp32.y4m`).
    Fp32,
}

impl Reference {
    /// Both kinds of reference.
    pub const ALL: [Reference; 2] = [Reference::Default, Reference::Fp32];

    fn file_suffix(self) -> &'static str {
        match self {
            Reference::Default => "ref",
            Reference::Fp32 => "ref-fp32",
        }
    }

    /// PSNR a decoder must reach against this reference.
    ///
    /// 60 dB is the project's acceptance bar against the original decoder
    /// (`CLAUDE.md`, scope). The original decoder's own FP16 storage already
    /// costs a few dB below FP32 arithmetic, so against its FP32 output the
    /// bar is higher: it catches small errors that would still pass at 60 dB.
    /// Neither number is ever lowered to make a test pass.
    pub fn min_psnr(self) -> f64 {
        match self {
            Reference::Default => 60.0,
            Reference::Fp32 => 80.0,
        }
    }
}

/// Reads the reference decode of a vector.
pub fn read_reference(name: &str, which: Reference) -> Y4m {
    let suffix = which.file_suffix();
    read_y4m(&reference_dir().join(format!("{name}.{suffix}.y4m")))
}

pub fn read_y4m(path: &Path) -> Y4m {
    let data = std::fs::read(path).expect("y4m file");
    let newline = data.iter().position(|&b| b == b'\n').expect("y4m header");
    let header = std::str::from_utf8(&data[..newline]).expect("y4m header text");
    assert!(header.starts_with("YUV4MPEG2"), "not a y4m file: {path:?}");

    let (mut width, mut height, mut chroma_444) = (0, 0, false);
    for token in header.split_whitespace().skip(1) {
        let (tag, value) = token.split_at(1);
        match tag {
            "W" => width = value.parse().expect("width"),
            "H" => height = value.parse().expect("height"),
            "C" => {
                assert!(
                    value == "420" || value == "444",
                    "unsupported y4m colourspace {value}"
                );
                chroma_444 = value == "444";
            }
            _ => {}
        }
    }

    let (chroma_width, chroma_height) = if chroma_444 {
        (width, height)
    } else {
        (width / 2, height / 2)
    };
    let sizes = [
        width * height,
        chroma_width * chroma_height,
        chroma_width * chroma_height,
    ];

    let mut frames = Vec::new();
    let mut position = newline + 1;
    while position < data.len() {
        assert_eq!(
            &data[position..position + 6],
            b"FRAME\n",
            "frame marker in {path:?}"
        );
        position += 6;
        let mut planes: [Vec<u8>; 3] = Default::default();
        for (plane, size) in planes.iter_mut().zip(sizes) {
            *plane = data[position..position + size].to_vec();
            position += size;
        }
        frames.push(planes);
    }

    Y4m {
        width,
        height,
        chroma_444,
        frames,
    }
}

/// PSNR in dB between two 8-bit planes. Infinite when they are identical.
pub fn psnr(reference: &[u8], test: &[u8]) -> f64 {
    assert_eq!(reference.len(), test.len(), "plane sizes differ");
    let squared_error: u64 = reference
        .iter()
        .zip(test)
        .map(|(&a, &b)| {
            let d = i64::from(a) - i64::from(b);
            (d * d) as u64
        })
        .sum();
    if squared_error == 0 {
        return f64::INFINITY;
    }
    let mse = squared_error as f64 / reference.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// Largest absolute difference between two 8-bit planes.
pub fn max_abs_diff(reference: &[u8], test: &[u8]) -> u8 {
    reference
        .iter()
        .zip(test)
        .map(|(&a, &b)| a.abs_diff(b))
        .max()
        .unwrap_or(0)
}
