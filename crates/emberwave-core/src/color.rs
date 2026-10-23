//! YCbCr to RGB conversion: the definitions shared by the CPU reference and
//! the GPU shader (`yuv2rgb.wgsl`).
//!
//! The decoder produces Y, Cb and Cr planes. Turning them into RGB is a
//! separate step that the bitstream leaves to the application; this module
//! gives a complete, tested one. It follows what upstream's viewer shader does
//! (`shaders/yuv2rgb.frag`): undo the video range, convert with a matrix and
//! upsample 4:2:0 chroma bilinearly at the position the siting says. The
//! matrices are the standard ones for the coefficient sets named in the stream
//! (see `ARCHITECTURE.md`, decision D17 for where upstream's shader differs).
//!
//! 4:2:0 chroma sits at multiples of a quarter chroma sample from each pixel,
//! so the bilinear weights are multiples of 1/16 and the interpolation is done
//! in integers. That keeps the CPU and the GPU in agreement to the last bit
//! up to the few float operations that follow.

use crate::bitstream::{
    ChromaSiting, ChromaSubsampling, SequenceHeader, YcbcrRange, YcbcrTransform,
};

/// Reciprocal of 255, so 8-bit samples are scaled by a multiplication (a
/// division could be turned into a less exact multiplication by a GPU).
pub const INV_255: f32 = 1.0 / 255.0;

/// The 8-bit video black level (limited range luma).
pub const LUMA_BLACK: f32 = 16.0 / 255.0;

/// Stretches limited range luma (219 steps) to `[0, 1]`.
pub const LUMA_SCALE: f32 = 255.0 / 219.0;

/// Stretches limited range chroma (224 steps) to `[-0.5, 0.5]`.
pub const CHROMA_SCALE: f32 = 255.0 / 224.0;

/// The 8-bit value of zero chroma.
pub const CHROMA_CENTRE: f32 = 128.0 / 255.0;

/// The set of coefficients that relates Y'CbCr to R'G'B'.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMatrix {
    /// ITU-R BT.601.
    Bt601,
    /// ITU-R BT.709 (the default for HD video and for sRGB content).
    Bt709,
    /// ITU-R BT.2020, non-constant luminance.
    Bt2020,
}

impl ColorMatrix {
    /// The luma weights `(Kr, Kb)` of the standard.
    pub fn luma_weights(self) -> (f64, f64) {
        match self {
            ColorMatrix::Bt601 => (0.299, 0.114),
            ColorMatrix::Bt709 => (0.2126, 0.0722),
            ColorMatrix::Bt2020 => (0.2627, 0.0593),
        }
    }

    /// The factors of the inverse matrix, `[R from Cr, G from Cb, G from Cr,
    /// B from Cb]`, derived from the luma weights in `f64` and rounded once.
    pub fn coefficients(self) -> [f32; 4] {
        let (kr, kb) = self.luma_weights();
        let kg = 1.0 - kr - kb;
        [
            (2.0 * (1.0 - kr)) as f32,
            (2.0 * kb * (1.0 - kb) / kg) as f32,
            (2.0 * kr * (1.0 - kr) / kg) as f32,
            (2.0 * (1.0 - kb)) as f32,
        ]
    }
}

impl From<YcbcrTransform> for ColorMatrix {
    fn from(transform: YcbcrTransform) -> Self {
        match transform {
            YcbcrTransform::Bt709 => ColorMatrix::Bt709,
            YcbcrTransform::Bt2020 => ColorMatrix::Bt2020,
        }
    }
}

/// How to turn decoded planes into RGB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorParams {
    /// Matrix coefficients.
    pub matrix: ColorMatrix,
    /// Whether the planes use the full 0 to 255 range or the video range
    /// (16 to 235 for luma, 16 to 240 for chroma).
    pub range: YcbcrRange,
    /// Where 4:2:0 chroma samples sit relative to the luma samples.
    pub chroma_siting: ChromaSiting,
}

impl Default for ColorParams {
    /// BT.709, video range, chroma in the centre: what most video is.
    fn default() -> Self {
        Self {
            matrix: ColorMatrix::Bt709,
            range: YcbcrRange::Limited,
            chroma_siting: ChromaSiting::Center,
        }
    }
}

impl ColorParams {
    /// Takes the signalling of a stream, from a PWV1 file header or a start
    /// of frame header. The upstream encoder only fills these in in the PWV1
    /// header, so prefer that one.
    pub fn from_stream(stream: &SequenceHeader) -> Self {
        Self {
            matrix: stream.ycbcr_transform.into(),
            range: stream.ycbcr_range,
            chroma_siting: stream.chroma_siting,
        }
    }
}

/// Bilinear weights, in quarters, of the two chroma samples around luma
/// position `p` along one axis, and the index of the first of them.
///
/// Returns `(base, fraction)` where the value is
/// `(4 - fraction) * chroma[base] + fraction * chroma[base + 1]`, divided by 4.
/// `base` may be -1, which the caller clamps to the edge.
pub fn chroma_taps(p: u32, siting: AxisSiting) -> (i32, u32) {
    let half = (p / 2) as i32;
    match (siting, p % 2) {
        // Chroma centred between two luma samples: chroma coordinate p/2 - 1/4.
        (AxisSiting::Centre, 0) => (half - 1, 3),
        (AxisSiting::Centre, _) => (half, 1),
        // Chroma on the even luma sample: chroma coordinate p/2.
        (AxisSiting::Left, 0) => (half, 0),
        (AxisSiting::Left, _) => (half, 2),
    }
}

/// Siting along one axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisSiting {
    /// Chroma sits between the two luma samples it covers.
    Centre,
    /// Chroma sits on the first (left or top) of them.
    Left,
}

/// Converts 8-bit Y, Cb and Cr planes to RGB, three bytes per pixel.
///
/// The luma plane is `width` x `height`; chroma planes are the same size for
/// 4:4:4 and half of it, rounded up, for 4:2:0 (the bitstream only has even
/// sizes, but any picture converts). Vertical chroma siting is always
/// centred, the horizontal siting follows `params`.
///
/// # Panics
///
/// If a plane is smaller than its dimensions say.
pub fn yuv_to_rgb8(
    planes: [&[u8]; 3],
    width: u32,
    height: u32,
    chroma: ChromaSubsampling,
    params: &ColorParams,
) -> Vec<u8> {
    let [luma, cb, cr] = planes;
    let (width_u, height_u) = (width as usize, height as usize);
    let (chroma_width, chroma_height) = match chroma {
        ChromaSubsampling::Yuv444 => (width_u, height_u),
        ChromaSubsampling::Yuv420 => (width_u.div_ceil(2), height_u.div_ceil(2)),
    };
    let [r_cr, g_cb, g_cr, b_cb] = params.matrix.coefficients();
    let limited = params.range == YcbcrRange::Limited;
    let horizontal = match params.chroma_siting {
        ChromaSiting::Center => AxisSiting::Centre,
        ChromaSiting::Left => AxisSiting::Left,
    };

    let mut rgb = Vec::with_capacity(width_u * height_u * 3);
    for y in 0..height {
        for x in 0..width {
            let luma8 = luma[y as usize * width_u + x as usize];

            // Chroma in 8-bit units, as an exact float.
            let (cb8, cr8) = match chroma {
                ChromaSubsampling::Yuv444 => {
                    let i = y as usize * width_u + x as usize;
                    (f32::from(cb[i]), f32::from(cr[i]))
                }
                ChromaSubsampling::Yuv420 => {
                    let (base_x, frac_x) = chroma_taps(x, horizontal);
                    let (base_y, frac_y) = chroma_taps(y, AxisSiting::Centre);
                    let sample = |plane: &[u8]| {
                        let tap = |dx: i32, dy: i32| {
                            let cx = (base_x + dx).clamp(0, chroma_width as i32 - 1) as usize;
                            let cy = (base_y + dy).clamp(0, chroma_height as i32 - 1) as usize;
                            u32::from(plane[cy * chroma_width + cx])
                        };
                        let sum = (4 - frac_x) * (4 - frac_y) * tap(0, 0)
                            + frac_x * (4 - frac_y) * tap(1, 0)
                            + (4 - frac_x) * frac_y * tap(0, 1)
                            + frac_x * frac_y * tap(1, 1);
                        sum as f32 * 0.0625
                    };
                    (sample(cb), sample(cr))
                }
            };

            let mut y_f = f32::from(luma8) * INV_255;
            let mut cb_f = cb8 * INV_255 - CHROMA_CENTRE;
            let mut cr_f = cr8 * INV_255 - CHROMA_CENTRE;
            if limited {
                y_f = (y_f - LUMA_BLACK) * LUMA_SCALE;
                cb_f *= CHROMA_SCALE;
                cr_f *= CHROMA_SCALE;
                y_f = y_f.clamp(0.0, 1.0);
                cb_f = cb_f.clamp(-0.5, 0.5);
                cr_f = cr_f.clamp(-0.5, 0.5);
            }

            let r = y_f + r_cr * cr_f;
            let g = y_f - g_cb * cb_f - g_cr * cr_f;
            let b = y_f + b_cb * cb_f;
            for value in [r, g, b] {
                rgb.push((value.clamp(0.0, 1.0) * 255.0).round_ties_even() as u8);
            }
        }
    }
    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficients_match_the_published_matrices() {
        let close = |a: [f32; 4], b: [f64; 4]| {
            a.iter()
                .zip(b)
                .all(|(x, y)| (f64::from(*x) - y).abs() < 1e-5)
        };
        // R_Cr, G_Cb, G_Cr, B_Cb.
        assert!(close(
            ColorMatrix::Bt601.coefficients(),
            [1.402, 0.344136, 0.714136, 1.772]
        ));
        assert!(close(
            ColorMatrix::Bt709.coefficients(),
            [1.5748, 0.187324, 0.468124, 1.8556]
        ));
        assert!(close(
            ColorMatrix::Bt2020.coefficients(),
            [1.4746, 0.164553, 0.571353, 1.8814]
        ));
    }

    #[test]
    fn chroma_taps_follow_the_siting() {
        // Centre: chroma coordinate p/2 - 1/4.
        let centre: Vec<_> = (0..6).map(|p| chroma_taps(p, AxisSiting::Centre)).collect();
        assert_eq!(centre, [(-1, 3), (0, 1), (0, 3), (1, 1), (1, 3), (2, 1)]);
        // Left: chroma coordinate p/2.
        let left: Vec<_> = (0..6).map(|p| chroma_taps(p, AxisSiting::Left)).collect();
        assert_eq!(left, [(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2)]);
    }

    /// Forward conversion in f64 straight from the definition of the
    /// standards, to check the inverse independently of its own formulas.
    fn forward(rgb: [f64; 3], matrix: ColorMatrix, limited: bool) -> [u8; 3] {
        let (kr, kb) = matrix.luma_weights();
        let kg = 1.0 - kr - kb;
        let y = kr * rgb[0] + kg * rgb[1] + kb * rgb[2];
        let cb = (rgb[2] - y) / (2.0 * (1.0 - kb));
        let cr = (rgb[0] - y) / (2.0 * (1.0 - kr));
        let (y8, cb8, cr8) = if limited {
            (16.0 + 219.0 * y, 128.0 + 224.0 * cb, 128.0 + 224.0 * cr)
        } else {
            (255.0 * y, 128.0 + 255.0 * cb, 128.0 + 255.0 * cr)
        };
        [y8, cb8, cr8].map(|v| v.round().clamp(0.0, 255.0) as u8)
    }

    #[test]
    fn colours_survive_a_round_trip_through_each_matrix_and_range() {
        let colours = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.2, 0.6, 0.9],
        ];
        for matrix in [ColorMatrix::Bt601, ColorMatrix::Bt709, ColorMatrix::Bt2020] {
            for limited in [true, false] {
                let params = ColorParams {
                    matrix,
                    range: if limited {
                        YcbcrRange::Limited
                    } else {
                        YcbcrRange::Full
                    },
                    chroma_siting: ChromaSiting::Center,
                };
                for colour in colours {
                    // A flat 4:4:4 picture of one colour, 4x4 pixels.
                    let [y, cb, cr] = forward(colour, matrix, limited);
                    let planes = [vec![y; 16], vec![cb; 16], vec![cr; 16]];
                    let rgb = yuv_to_rgb8(
                        [&planes[0], &planes[1], &planes[2]],
                        4,
                        4,
                        ChromaSubsampling::Yuv444,
                        &params,
                    );
                    for (channel, expected) in colour.iter().enumerate() {
                        let got = f64::from(rgb[channel]);
                        // 8-bit quantisation of Y, Cb and Cr costs up to about two steps in RGB.
                        assert!(
                            (got - expected * 255.0).abs() <= 2.0,
                            "{matrix:?} limited={limited} colour {colour:?} channel {channel}: got {got}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn video_black_and_white_map_to_full_range_black_and_white() {
        let params = ColorParams::default();
        let planes = [vec![16u8, 235], vec![128u8; 2], vec![128u8; 2]];
        let rgb = yuv_to_rgb8(
            [&planes[0], &planes[1], &planes[2]],
            2,
            1,
            ChromaSubsampling::Yuv444,
            &params,
        );
        assert_eq!(rgb, [0, 0, 0, 255, 255, 255]);
    }

    #[test]
    fn out_of_range_video_levels_are_clamped() {
        // Video range BT.709. Pixel 0: Y 0 is below black and clamps to 0, Cb 0
        // and Cr 255 clamp to -0.5 and +0.5. Worked out by hand:
        //   R = 0 + 1.5748 * 0.5                      = 0.787  -> 201
        //   G = 0 + 0.187324 * 0.5 - 0.468124 * 0.5   = -0.140 -> 0
        //   B = 0 - 1.8556 * 0.5                      = -0.928 -> 0
        // Pixel 1: Y 255 is above white and clamps to 1, Cb 255 and Cr 0
        // clamp to +0.5 and -0.5:
        //   R = 1 - 1.5748 * 0.5                      = 0.213  -> 54
        //   G = 1 - 0.187324 * 0.5 + 0.468124 * 0.5   = 1.140  -> 255
        //   B = 1 + 1.8556 * 0.5                      = 1.928  -> 255
        let params = ColorParams::default();
        let planes = [vec![0u8, 255], vec![0u8, 255], vec![255u8, 0]];
        let rgb = yuv_to_rgb8(
            [&planes[0], &planes[1], &planes[2]],
            2,
            1,
            ChromaSubsampling::Yuv444,
            &params,
        );
        assert_eq!(rgb, [201, 0, 0, 54, 255, 255]);
    }

    #[test]
    fn flat_four_two_zero_chroma_needs_no_interpolation() {
        // Flat chroma stays flat whatever the siting, including at the edges.
        for siting in [ChromaSiting::Center, ChromaSiting::Left] {
            let params = ColorParams {
                chroma_siting: siting,
                ..ColorParams::default()
            };
            let (w, h) = (8usize, 6usize);
            let luma = vec![120u8; w * h];
            let cb = vec![90u8; (w / 2) * (h / 2)];
            let cr = vec![200u8; (w / 2) * (h / 2)];
            let rgb = yuv_to_rgb8(
                [&luma, &cb, &cr],
                w as u32,
                h as u32,
                ChromaSubsampling::Yuv420,
                &params,
            );

            let flat = yuv_to_rgb8(
                [&luma, &vec![90u8; w * h], &vec![200u8; w * h]],
                w as u32,
                h as u32,
                ChromaSubsampling::Yuv444,
                &params,
            );
            assert_eq!(rgb, flat);
        }
    }

    #[test]
    fn pictures_of_odd_size_convert_too() {
        // Chroma planes are half the luma size rounded up, so even a single
        // pixel has a chroma sample (half of one pixel would round down to
        // none, and clamping an index to an empty plane panics). Neutral
        // chroma gives grey, whatever the siting.
        let full_range = ColorParams {
            range: YcbcrRange::Full,
            ..ColorParams::default()
        };
        for (w, h) in [(1usize, 1usize), (3, 1), (1, 3), (5, 3)] {
            let luma = vec![100u8; w * h];
            let chroma = vec![128u8; w.div_ceil(2) * h.div_ceil(2)];
            for siting in [ChromaSiting::Center, ChromaSiting::Left] {
                let params = ColorParams {
                    chroma_siting: siting,
                    ..full_range
                };
                let rgb = yuv_to_rgb8(
                    [&luma, &chroma, &chroma],
                    w as u32,
                    h as u32,
                    ChromaSubsampling::Yuv420,
                    &params,
                );
                assert_eq!(rgb, vec![100u8; w * h * 3], "{w}x{h} {siting:?}");
            }
        }
    }

    #[test]
    fn centred_chroma_interpolates_between_neighbours() {
        // One chroma row of two samples, 0 and 255, under four luma columns.
        // Centre siting: the luma pixels see 0, 63.75, 191.25 and 255 (bilinear
        // weights 1, 3/4, 1/4 and 0 of the left sample, edges clamped).
        let params = ColorParams {
            matrix: ColorMatrix::Bt709,
            range: YcbcrRange::Full,
            chroma_siting: ChromaSiting::Center,
        };
        let luma = [128u8; 8];
        let cb = [0u8, 255];
        let cr = [128u8; 2];
        let rgb = yuv_to_rgb8([&luma, &cb, &cr], 4, 2, ChromaSubsampling::Yuv420, &params);

        // Blue grows with Cb: it must rise monotonically along the row.
        let blue: Vec<u8> = (0..4).map(|x| rgb[x * 3 + 2]).collect();
        assert!(blue.windows(2).all(|w| w[0] <= w[1]), "{blue:?}");
        assert!(blue[0] < blue[1] && blue[2] < blue[3], "{blue:?}");
        // Both rows of a 4:2:0 picture with one chroma row are the same.
        assert_eq!(rgb[..12], rgb[12..]);
    }
}
