//! Inverse CDF 9/7 wavelet transform in JPEG 2000 lifting form.
//!
//! `bitstream.md` ("Wavelet transform") defines the transform as the CDF 9/7
//! filter of JPEG 2000 (ITU-T T.800, F.4.8.2) with symmetric signal extension
//! at the picture edges. Like the reference decoder this uses `f32`.

/// Lifting step constants of the CDF 9/7 filter (ITU-T T.800, section F.4.8.2,
/// which `bitstream.md` points to).
///
/// Kept at the published double precision and rounded to `f32` once, the way
/// the reference shaders convert their literals.
pub(crate) const ALPHA: f32 = (-1.586_134_342_059_924_f64) as f32;
pub(crate) const BETA: f32 = (-0.052_980_118_572_961_f64) as f32;
pub(crate) const GAMMA: f32 = 0.882_911_075_530_934_f64 as f32;
pub(crate) const DELTA: f32 = 0.443_506_852_043_971_f64 as f32;
/// Gain of the low-pass band; the high-pass band is scaled by its inverse.
pub(crate) const K: f32 = 1.230_174_104_914_001_f64 as f32;
pub(crate) const INV_K: f32 = (1.0_f64 / 1.230_174_104_914_001_f64) as f32;

/// Samples of signal extension needed on each side: one per lifting step.
pub(crate) const PADDING: usize = 4;

/// Maps an index outside `0..len` to its whole-sample symmetric mirror image:
/// `-1 -> 1`, `len -> len - 2`. Needs `len > PADDING`.
pub(crate) fn mirror(index: isize, len: usize) -> usize {
    let last = len as isize - 1;
    let mut index = index;
    if index < 0 {
        index = -index;
    }
    if index > last {
        index = 2 * last - index;
    }
    index.clamp(0, last) as usize
}

/// Inverse 1-D transform of one line, in place.
///
/// Even positions hold low-pass samples and odd positions high-pass samples,
/// i.e. the two bands are interleaved. The line length must be even and
/// larger than the padding, which holds for every band of a valid frame.
fn inverse_line(line: &mut [f32], extended: &mut Vec<f32>) {
    let len = line.len();
    if len <= PADDING {
        return;
    }

    // Extend by mirroring and undo the band gains. Mirroring keeps the
    // parity of an index (the length is even), so the gain follows the index.
    extended.clear();
    extended.extend((0..len + 2 * PADDING).map(|i| {
        let source = mirror(i as isize - PADDING as isize, len);
        let gain = if source.is_multiple_of(2) { K } else { INV_K };
        line[source] * gain
    }));

    // Four lifting steps, each shrinking the valid region by one sample at
    // both ends, which the padding pays for.
    let end = extended.len();
    for i in (2..end - 1).step_by(2) {
        extended[i] -= DELTA * (extended[i - 1] + extended[i + 1]);
    }
    for i in (3..end - 2).step_by(2) {
        extended[i] -= GAMMA * (extended[i - 1] + extended[i + 1]);
    }
    for i in (4..end - 3).step_by(2) {
        extended[i] -= BETA * (extended[i - 1] + extended[i + 1]);
    }
    for i in (5..end - 4).step_by(2) {
        extended[i] -= ALPHA * (extended[i - 1] + extended[i + 1]);
    }

    line.copy_from_slice(&extended[PADDING..PADDING + len]);
}

/// Combines the four bands of one level into the next finer LL band.
///
/// Each band is `width` x `height` samples, row-major. The result is
/// `2 * width` x `2 * height`: LL at even columns and rows, HL at odd
/// columns, LH at odd rows, HH at odd columns and rows, then filtered
/// horizontally and vertically.
pub fn inverse_2d(
    ll: &[f32],
    hl: &[f32],
    lh: &[f32],
    hh: &[f32],
    width: usize,
    height: usize,
) -> Vec<f32> {
    let out_width = 2 * width;
    let out_height = 2 * height;
    let mut image = vec![0.0f32; out_width * out_height];

    for y in 0..height {
        for x in 0..width {
            let src = y * width + x;
            let top = 2 * y * out_width + 2 * x;
            image[top] = ll.get(src).copied().unwrap_or(0.0);
            image[top + 1] = hl.get(src).copied().unwrap_or(0.0);
            image[top + out_width] = lh.get(src).copied().unwrap_or(0.0);
            image[top + out_width + 1] = hh.get(src).copied().unwrap_or(0.0);
        }
    }

    let mut extended = Vec::new();
    for row in image.chunks_exact_mut(out_width) {
        inverse_line(row, &mut extended);
    }

    let mut column = vec![0.0f32; out_height];
    for x in 0..out_width {
        for (y, sample) in column.iter_mut().enumerate() {
            *sample = image[y * out_width + x];
        }
        inverse_line(&mut column, &mut extended);
        for (y, sample) in column.iter().enumerate() {
            image[y * out_width + x] = *sample;
        }
    }

    image
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Forward 1-D CDF 9/7 transform, the exact inverse of `inverse_line`
    /// (T.800 F.4.8.2 forward lifting), used to build test signals.
    fn forward_line(signal: &[f32]) -> Vec<f32> {
        let len = signal.len();
        let mut x: Vec<f32> = (0..len + 2 * PADDING)
            .map(|i| signal[mirror(i as isize - PADDING as isize, len)])
            .collect();
        let end = x.len();
        for i in (1..end - 1).step_by(2) {
            x[i] += ALPHA * (x[i - 1] + x[i + 1]);
        }
        for i in (2..end - 2).step_by(2) {
            x[i] += BETA * (x[i - 1] + x[i + 1]);
        }
        for i in (3..end - 3).step_by(2) {
            x[i] += GAMMA * (x[i - 1] + x[i + 1]);
        }
        for i in (4..end - 4).step_by(2) {
            x[i] += DELTA * (x[i - 1] + x[i + 1]);
        }
        (0..len)
            .map(|i| {
                let v = x[i + PADDING];
                if i % 2 == 0 { v / K } else { v * K }
            })
            .collect()
    }

    fn rms_error(a: &[f32], b: &[f32]) -> f32 {
        let sum: f32 = a.iter().zip(b).map(|(a, b)| (a - b) * (a - b)).sum();
        (sum / a.len() as f32).sqrt()
    }

    #[test]
    fn mirror_is_whole_sample_symmetric() {
        assert_eq!(mirror(-1, 8), 1);
        assert_eq!(mirror(-4, 8), 4);
        assert_eq!(mirror(8, 8), 6);
        assert_eq!(mirror(11, 8), 3);
        assert_eq!(mirror(3, 8), 3);
        // Never panics, even far outside.
        assert!(mirror(-1000, 8) < 8);
        assert!(mirror(1000, 8) < 8);
    }

    #[test]
    fn inverse_undoes_forward() {
        // A smooth signal plus some texture, a few lengths.
        for len in [8usize, 10, 16, 40, 130] {
            let signal: Vec<f32> = (0..len)
                .map(|i| (i as f32 * 0.37).sin() * 0.4 + ((i * 7) % 5) as f32 * 0.05)
                .collect();
            let mut coded = forward_line(&signal);
            inverse_line(&mut coded, &mut Vec::new());
            assert!(rms_error(&coded, &signal) < 1e-5, "length {len}");
        }
    }

    #[test]
    fn constant_signal_is_in_the_low_pass_band_only() {
        // The forward transform of a constant leaves (almost) nothing in the
        // high-pass samples, and inverting restores the constant.
        let signal = vec![0.25f32; 32];
        let coded = forward_line(&signal);
        for (i, v) in coded.iter().enumerate() {
            if i % 2 == 1 {
                assert!(v.abs() < 1e-6, "high-pass sample {i} = {v}");
            }
        }
        let mut restored = coded;
        inverse_line(&mut restored, &mut Vec::new());
        assert!(restored.iter().all(|v| (v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn short_lines_are_left_alone() {
        let mut line = [1.0, 2.0, 3.0, 4.0];
        inverse_line(&mut line, &mut Vec::new());
        assert_eq!(line, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn two_dimensional_round_trip() {
        // Forward 2-D: rows then columns, then de-interleave into bands.
        let (w, h) = (16usize, 12usize);
        let image: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                (x * 0.3).sin() * (y * 0.2).cos() * 0.4 + 0.1 * ((i * 13) % 7) as f32 / 7.0
            })
            .collect();

        let mut data = image.clone();
        for row in data.chunks_exact_mut(w) {
            let coded = forward_line(row);
            row.copy_from_slice(&coded);
        }
        for x in 0..w {
            let column: Vec<f32> = (0..h).map(|y| data[y * w + x]).collect();
            let coded = forward_line(&column);
            for (y, v) in coded.iter().enumerate() {
                data[y * w + x] = *v;
            }
        }

        let (bw, bh) = (w / 2, h / 2);
        let band = |dx: usize, dy: usize| -> Vec<f32> {
            (0..bw * bh)
                .map(|i| data[(2 * (i / bw) + dy) * w + 2 * (i % bw) + dx])
                .collect()
        };
        let restored = inverse_2d(&band(0, 0), &band(1, 0), &band(0, 1), &band(1, 1), bw, bh);
        assert!(rms_error(&restored, &image) < 1e-5);
    }
}
