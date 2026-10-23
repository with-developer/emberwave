//! Forward CDF 9/7 wavelet transform, the inverse of [`crate::cpu::idwt`].
//!
//! `bitstream.md` ("Wavelet transform", "Image dimension alignment and
//! padding") fixes the filter, the gains and the whole-sample symmetric
//! extension at the edges of every level; how the picture is padded up to the
//! aligned size is the encoder's choice.

use crate::bitstream::consts::DC_SHIFT;
use crate::cpu::idwt::{ALPHA, BETA, DELTA, GAMMA, INV_K, K, PADDING, mirror};

/// Forward 1-D transform of one line, in place: afterwards even positions
/// hold the low-pass and odd positions the high-pass samples, as the inverse
/// transform takes them. The line length must be even and larger than the
/// padding, which holds for every level of a valid frame.
pub(crate) fn forward_line(line: &mut [f32], extended: &mut Vec<f32>) {
    let len = line.len();
    debug_assert!(
        len > PADDING && len.is_multiple_of(2),
        "line of {len} samples"
    );

    extended.clear();
    extended
        .extend((0..len + 2 * PADDING).map(|i| line[mirror(i as isize - PADDING as isize, len)]));

    // Four lifting steps, the inverse's in reverse order, each shrinking the
    // valid region by one sample at both ends. The padding is even, so the
    // parity of a position is that of the sample it holds.
    let end = extended.len();
    for i in (1..end - 1).step_by(2) {
        extended[i] += ALPHA * (extended[i - 1] + extended[i + 1]);
    }
    for i in (2..end - 2).step_by(2) {
        extended[i] += BETA * (extended[i - 1] + extended[i + 1]);
    }
    for i in (3..end - 3).step_by(2) {
        extended[i] += GAMMA * (extended[i - 1] + extended[i + 1]);
    }
    for i in (4..end - 4).step_by(2) {
        extended[i] += DELTA * (extended[i - 1] + extended[i + 1]);
    }

    // The band gains: the inverse multiplies low-pass samples by K and
    // high-pass samples by 1/K.
    for (i, sample) in line.iter_mut().enumerate() {
        let lifted = extended[i + PADDING];
        *sample = if i % 2 == 0 {
            lifted * INV_K
        } else {
            lifted * K
        };
    }
}

/// The four bands of one level, each half the width and height of its input.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Level {
    /// Low-pass in both directions; the next level's input.
    pub ll: Vec<f32>,
    /// High-pass horizontally (odd columns).
    pub hl: Vec<f32>,
    /// High-pass vertically (odd rows).
    pub lh: Vec<f32>,
    /// High-pass in both directions.
    pub hh: Vec<f32>,
}

/// One level of the 2-D forward transform of a `width` x `height` image,
/// row-major: rows first, then columns, then the samples sorted into the four
/// bands, as [`crate::cpu::idwt::inverse_2d`] expects them.
pub(crate) fn forward_2d(mut image: Vec<f32>, width: usize, height: usize) -> Level {
    let mut extended = Vec::new();
    for row in image.chunks_exact_mut(width) {
        forward_line(row, &mut extended);
    }

    let mut column = vec![0.0f32; height];
    for x in 0..width {
        for (y, sample) in column.iter_mut().enumerate() {
            *sample = image[y * width + x];
        }
        forward_line(&mut column, &mut extended);
        for (y, sample) in column.iter().enumerate() {
            image[y * width + x] = *sample;
        }
    }

    let (band_width, band_height) = (width / 2, height / 2);
    let band = |dx: usize, dy: usize| -> Vec<f32> {
        let mut samples = Vec::with_capacity(band_width * band_height);
        for y in 0..band_height {
            let row = (2 * y + dy) * width;
            samples.extend((0..band_width).map(|x| image[row + 2 * x + dx]));
        }
        samples
    };
    Level {
        ll: band(0, 0),
        hl: band(1, 0),
        lh: band(0, 1),
        hh: band(1, 1),
    }
}

/// An 8-bit sample as the transform takes it: scaled to `[0, 1]`, less the DC
/// shift that the decoder adds back.
pub(crate) fn sample_level(sample: u8) -> f32 {
    f32::from(sample) / 255.0 - DC_SHIFT
}

/// The samples of an 8-bit plane as the transform takes them
/// ([`sample_level`]), padded to `aligned_width` x `aligned_height` by
/// repeating the last column and the last row. Repeating the edge keeps the
/// padding from adding detail that would cost bits.
pub(crate) fn pad_plane(
    samples: &[u8],
    width: usize,
    height: usize,
    aligned_width: usize,
    aligned_height: usize,
) -> Vec<f32> {
    let mut padded = Vec::with_capacity(aligned_width * aligned_height);
    for y in 0..aligned_height {
        let row = &samples[y.min(height - 1) * width..][..width];
        padded.extend((0..aligned_width).map(|x| sample_level(row[x.min(width - 1)])));
    }
    padded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::idwt::inverse_2d;

    fn picture(width: usize, height: usize) -> Vec<f32> {
        (0..width * height)
            .map(|i| {
                let (x, y) = ((i % width) as f32, (i / width) as f32);
                (x * 0.21).sin() * (y * 0.13).cos() * 0.4 + 0.08 * ((i * 31) % 11) as f32 / 11.0
            })
            .collect()
    }

    fn largest_difference(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn the_inverse_transform_undoes_five_levels() {
        // A whole pyramid down and up again, at an aligned size.
        let (width, height) = (128usize, 160usize);
        let image = picture(width, height);

        let mut levels = Vec::new();
        let mut current = image.clone();
        let (mut w, mut h) = (width, height);
        for _ in 0..5 {
            let level = forward_2d(current, w, h);
            current = level.ll.clone();
            levels.push(level);
            w /= 2;
            h /= 2;
        }
        for level in levels.iter().rev() {
            current = inverse_2d(&current, &level.hl, &level.lh, &level.hh, w, h);
            w *= 2;
            h *= 2;
        }
        assert!(largest_difference(&current, &image) < 1e-5);
    }

    #[test]
    fn a_constant_stays_in_the_low_pass_band_at_every_level() {
        let (mut w, mut h) = (128usize, 128usize);
        let mut current = vec![0.3f32; w * h];
        for level in 0..5 {
            let bands = forward_2d(current, w, h);
            for high in [&bands.hl, &bands.lh, &bands.hh] {
                assert!(
                    largest_difference(high, &vec![0.0; high.len()]) < 1e-5,
                    "level {level}"
                );
            }
            // The gains keep a constant at its value.
            assert!(largest_difference(&bands.ll, &vec![0.3; bands.ll.len()]) < 1e-5);
            current = bands.ll;
            w /= 2;
            h /= 2;
        }
    }

    #[test]
    fn padding_repeats_the_edges_and_shifts_the_level() {
        let samples = [0u8, 255, 51, 102];
        let padded = pad_plane(&samples, 2, 2, 4, 3);
        let expected = |v: u8| f32::from(v) / 255.0 - 0.5;
        assert_eq!(padded.len(), 12);
        assert_eq!(padded[0], expected(0));
        assert_eq!(padded[1], expected(255));
        assert_eq!(padded[3], expected(255), "last column repeated");
        assert_eq!(padded[4], expected(51));
        assert_eq!(padded[8..12], padded[4..8], "last row repeated");
    }
}
