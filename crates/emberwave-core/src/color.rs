//! YCbCr to RGB conversion: the definitions shared by the CPU reference and
//! the GPU shader (`yuv2rgb.wgsl`). And the other way, RGB to YCbCr for the
//! encoder, in integers that the CPU and the GPU (`gather_rgb.wgsl`) compute
//! alike to the last bit ([`rgb_to_ycbcr8`], decision D65).
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

/// Why [`yuv_to_rgb8`] or [`rgb_to_ycbcr8`] refused its input.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ColorError {
    /// A plane holds fewer samples than the picture size needs.
    #[error("plane {plane} has {actual} samples, but the picture needs {needed}")]
    PlaneTooSmall {
        /// 0 for Y, 1 for Cb, 2 for Cr.
        plane: usize,
        /// Samples the picture size needs.
        needed: u64,
        /// Samples the plane has.
        actual: usize,
    },
    /// An RGB picture holds fewer bytes than its size and stride need.
    #[error("the RGB picture has {actual} bytes, but its size and stride need {needed}")]
    PictureTooSmall {
        /// Bytes the picture needs: its rows but the last at the stride,
        /// and the last row's pixels.
        needed: u64,
        /// Bytes given.
        actual: usize,
    },
    /// An RGB picture's rows are closer together than a row's pixels take.
    #[error(
        "the rows of the RGB picture are {stride} bytes apart, less than the {row} bytes of a row"
    )]
    StrideTooSmall {
        /// Bytes from the start of one row to the start of the next.
        stride: usize,
        /// Bytes of a row's pixels, four each.
        row: u64,
    },
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
/// # Errors
///
/// [`ColorError::PlaneTooSmall`] if a plane is smaller than the picture size
/// says. Larger planes are fine; the samples past the picture are ignored.
pub fn yuv_to_rgb8(
    planes: [&[u8]; 3],
    width: u32,
    height: u32,
    chroma: ChromaSubsampling,
    params: &ColorParams,
) -> Result<Vec<u8>, ColorError> {
    let [luma, cb, cr] = planes;
    let (width_u, height_u) = (width as usize, height as usize);
    let (chroma_width, chroma_height) = match chroma {
        ChromaSubsampling::Yuv444 => (width_u, height_u),
        ChromaSubsampling::Yuv420 => (width_u.div_ceil(2), height_u.div_ceil(2)),
    };

    // Checked once here, in 64 bits, so that the indexing below cannot fail.
    let luma_samples = u64::from(width) * u64::from(height);
    let chroma_samples = chroma_width as u64 * chroma_height as u64;
    for (plane, (data, needed)) in planes
        .iter()
        .zip([luma_samples, chroma_samples, chroma_samples])
        .enumerate()
    {
        if (data.len() as u64) < needed {
            return Err(ColorError::PlaneTooSmall {
                plane,
                needed,
                actual: data.len(),
            });
        }
    }
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
    Ok(rgb)
}

/// The order of a pixel's four bytes in an RGB picture: R', G' and B', and
/// a fourth that is not read, such as alpha.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RgbOrder {
    /// R', G', B', then the fourth byte.
    Rgba,
    /// B', G', R', then the fourth byte, as screen captures mostly hold them.
    Bgra,
}

/// Fractional bits of [`ForwardMatrix`]'s factors.
pub const FORWARD_BITS: u32 = 16;

/// RGB to YCbCr in integers, as [`rgb_to_ycbcr8`] and the encoder's
/// `gather_rgb.wgsl` compute it (decision D65): each of Y, Cb and Cr is a
/// row's factors times the 8-bit R', G' and B' of a pixel, or their sums over
/// the pixels a chroma sample covers, with the row's offset and half a step,
/// shifted right and clamped to a byte ([`ForwardMatrix::apply`]).
///
/// The factors are the standard's, scaled by `2^FORWARD_BITS` and rounded
/// once; G's factor of each row is what makes the row's sum exact, so that
/// grey has Cb and Cr of 128 and the full range's white a Y of 255 (219 and
/// 224 steps of 255 in the video range). Sums over up to 8 pixels stay
/// well within 32 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForwardMatrix {
    /// Factors of R', G' and B' for Y, Cb and Cr.
    pub rows: [[i32; 3]; 3],
    /// What each row adds for one pixel, scaled alike: the black level for
    /// Y, 128 for Cb and Cr.
    pub offsets: [i32; 3],
}

impl ForwardMatrix {
    /// The matrix of `matrix`'s coefficients into `range`.
    pub fn new(matrix: ColorMatrix, range: YcbcrRange) -> Self {
        let (kr, kb) = matrix.luma_weights();
        let kg = 1.0 - kr - kb;
        let one = f64::from(1u32 << FORWARD_BITS);
        let (luma_scale, chroma_scale, black) = match range {
            YcbcrRange::Full => (1.0, 1.0, 0),
            YcbcrRange::Limited => (219.0 / 255.0, 224.0 / 255.0, 16),
        };
        let fixed = |value: f64| (value * one).round() as i32;
        // R' and B' as the standard has them; G' takes the rest of the
        // row's sum.
        let luma_sum = fixed(luma_scale);
        let (y_r, y_b) = (fixed(luma_scale * kr), fixed(luma_scale * kb));
        let (cb_r, cb_b) = (
            fixed(-chroma_scale * kr / (2.0 * (1.0 - kb))),
            fixed(chroma_scale * 0.5),
        );
        let (cr_r, cr_b) = (
            fixed(chroma_scale * 0.5),
            fixed(-chroma_scale * kb / (2.0 * (1.0 - kr))),
        );
        debug_assert!(fixed(luma_scale * kg).abs_diff(luma_sum - y_r - y_b) <= 1);
        Self {
            rows: [
                [y_r, luma_sum - y_r - y_b, y_b],
                [cb_r, -cb_r - cb_b, cb_b],
                [cr_r, -cr_r - cr_b, cr_b],
            ],
            offsets: [
                black << FORWARD_BITS,
                128 << FORWARD_BITS,
                128 << FORWARD_BITS,
            ],
        }
    }

    /// Component `row` (0 for Y, 1 for Cb, 2 for Cr) of pixels whose R', G'
    /// and B', each weighed, add up to `sum`, the weights to `1 << shift`:
    /// one pixel with a shift of 0, four of 4:2:0's centred chroma with 2.
    pub fn apply(&self, row: usize, sum: [i32; 3], shift: u32) -> u8 {
        let [r, g, b] = self.rows[row];
        let total = r * sum[0]
            + g * sum[1]
            + b * sum[2]
            + (self.offsets[row] << shift)
            + (1 << (FORWARD_BITS + shift - 1));
        (total >> (FORWARD_BITS + shift)).clamp(0, 255) as u8
    }
}

/// An error if `bytes` of an RGB picture of `width` x `height` pixels, four
/// bytes each, with rows `stride` bytes apart, do not hold it.
pub(crate) fn check_rgb(
    bytes: usize,
    stride: usize,
    width: u32,
    height: u32,
) -> Result<(), ColorError> {
    let row = u64::from(width) * 4;
    if (stride as u64) < row && height > 1 {
        return Err(ColorError::StrideTooSmall { stride, row });
    }
    let needed = match height {
        0 => 0,
        _ => (u64::from(height) - 1) * stride as u64 + row,
    };
    if (bytes as u64) < needed {
        return Err(ColorError::PictureTooSmall {
            needed,
            actual: bytes,
        });
    }
    Ok(())
}

/// Converts an RGB picture to 8-bit Y, Cb and Cr planes, as the GPU encoder
/// does for a picture in a texture and into the same bytes (decision D65).
///
/// `rgb` holds `height` rows of `width` pixels, four bytes each in `order`,
/// the rows `stride` bytes apart. R', G' and B' are as the picture stores
/// them, with the transfer function applied, as a screen capture has them.
/// The chroma planes are the same size as the luma plane for 4:4:4 and half
/// of it, rounded up, for 4:2:0. A 4:2:0 chroma sample is the average of the
/// pixels it covers: those of two rows, and across two columns where
/// `params` sites chroma in the centre, or with weights 1, 2 and 1 across
/// three where it sites chroma on the left one, as [`yuv_to_rgb8`] reads it
/// back. Pixels past an edge repeat the edge.
///
/// # Errors
///
/// [`ColorError::StrideTooSmall`] if a row's pixels do not fit the stride,
/// and [`ColorError::PictureTooSmall`] if `rgb` is smaller than the picture
/// size and stride say.
pub fn rgb_to_ycbcr8(
    rgb: &[u8],
    stride: usize,
    order: RgbOrder,
    width: u32,
    height: u32,
    chroma: ChromaSubsampling,
    params: &ColorParams,
) -> Result<[Vec<u8>; 3], ColorError> {
    check_rgb(rgb.len(), stride, width, height)?;
    let (width, height) = (width as usize, height as usize);
    let matrix = ForwardMatrix::new(params.matrix, params.range);
    let (red, blue) = match order {
        RgbOrder::Rgba => (0, 2),
        RgbOrder::Bgra => (2, 0),
    };
    // R', G' and B' of the pixel at (x, y), clamped to the picture.
    let pixel = |x: isize, y: usize| {
        let x = x.clamp(0, width as isize - 1) as usize;
        let at = y.min(height - 1) * stride + 4 * x;
        [rgb[at + red], rgb[at + 1], rgb[at + blue]].map(i32::from)
    };
    let mut planes: [Vec<u8>; 3] = Default::default();
    planes[0] = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            planes[0].push(matrix.apply(0, pixel(x as isize, y), 0));
        }
    }
    match chroma {
        ChromaSubsampling::Yuv444 => {
            for (plane, row) in planes[1..].iter_mut().zip(1..) {
                *plane = Vec::with_capacity(width * height);
                for y in 0..height {
                    for x in 0..width {
                        plane.push(matrix.apply(row, pixel(x as isize, y), 0));
                    }
                }
            }
        }
        ChromaSubsampling::Yuv420 => {
            let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
            // Columns and weights, from the chroma sample's left luma column.
            let (taps, shift): (&[(isize, i32)], u32) = match params.chroma_siting {
                ChromaSiting::Center => (&[(0, 1), (1, 1)], 2),
                ChromaSiting::Left => (&[(-1, 1), (0, 2), (1, 1)], 3),
            };
            for plane in &mut planes[1..] {
                *plane = Vec::with_capacity(chroma_width * chroma_height);
            }
            for cy in 0..chroma_height {
                for cx in 0..chroma_width {
                    let mut sum = [0i32; 3];
                    for y in [2 * cy, 2 * cy + 1] {
                        for &(dx, weight) in taps {
                            let rgb = pixel(2 * cx as isize + dx, y);
                            for (total, value) in sum.iter_mut().zip(rgb) {
                                *total += weight * value;
                            }
                        }
                    }
                    planes[1].push(matrix.apply(1, sum, shift));
                    planes[2].push(matrix.apply(2, sum, shift));
                }
            }
        }
    }
    Ok(planes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `yuv_to_rgb8` for planes that match the picture size.
    fn convert(
        planes: [&[u8]; 3],
        width: u32,
        height: u32,
        chroma: ChromaSubsampling,
        params: &ColorParams,
    ) -> Vec<u8> {
        yuv_to_rgb8(planes, width, height, chroma, params).expect("the planes match the size")
    }

    #[test]
    fn planes_smaller_than_the_picture_are_an_error() {
        let params = ColorParams::default();
        // 4x2 in 4:2:0 needs 8 luma samples and 2 of each chroma.
        let (luma, chroma) = (vec![0u8; 8], vec![128u8; 2]);
        assert!(
            yuv_to_rgb8(
                [&luma, &chroma, &chroma],
                4,
                2,
                ChromaSubsampling::Yuv420,
                &params
            )
            .is_ok()
        );
        assert_eq!(
            yuv_to_rgb8(
                [&luma[..7], &chroma, &chroma],
                4,
                2,
                ChromaSubsampling::Yuv420,
                &params
            ),
            Err(ColorError::PlaneTooSmall {
                plane: 0,
                needed: 8,
                actual: 7
            })
        );
        assert_eq!(
            yuv_to_rgb8(
                [&luma, &chroma, &chroma[..1]],
                4,
                2,
                ChromaSubsampling::Yuv420,
                &params
            ),
            Err(ColorError::PlaneTooSmall {
                plane: 2,
                needed: 2,
                actual: 1
            })
        );
        // 4:4:4 needs full-size chroma planes.
        assert!(matches!(
            yuv_to_rgb8(
                [&luma, &chroma, &chroma],
                4,
                2,
                ChromaSubsampling::Yuv444,
                &params
            ),
            Err(ColorError::PlaneTooSmall { plane: 1, .. })
        ));
        // The largest picture the format allows is checked without overflow.
        assert!(
            yuv_to_rgb8(
                [&[], &[], &[]],
                u32::MAX,
                u32::MAX,
                ChromaSubsampling::Yuv420,
                &params
            )
            .is_err()
        );
    }

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
                    let rgb = convert(
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
        let rgb = convert(
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
        let rgb = convert(
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
            let rgb = convert(
                [&luma, &cb, &cr],
                w as u32,
                h as u32,
                ChromaSubsampling::Yuv420,
                &params,
            );

            let flat = convert(
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
                let rgb = convert(
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
        let rgb = convert([&luma, &cb, &cr], 4, 2, ChromaSubsampling::Yuv420, &params);

        // Blue grows with Cb: it must rise monotonically along the row.
        let blue: Vec<u8> = (0..4).map(|x| rgb[x * 3 + 2]).collect();
        assert!(blue.windows(2).all(|w| w[0] <= w[1]), "{blue:?}");
        assert!(blue[0] < blue[1] && blue[2] < blue[3], "{blue:?}");
        // Both rows of a 4:2:0 picture with one chroma row are the same.
        assert_eq!(rgb[..12], rgb[12..]);
    }

    const RANGES: [YcbcrRange; 2] = [YcbcrRange::Limited, YcbcrRange::Full];
    const MATRICES: [ColorMatrix; 3] =
        [ColorMatrix::Bt601, ColorMatrix::Bt709, ColorMatrix::Bt2020];

    #[test]
    fn forward_rows_add_up_exactly() {
        for matrix in MATRICES {
            for range in RANGES {
                let forward = ForwardMatrix::new(matrix, range);
                let luma: i32 = forward.rows[0].iter().sum();
                let expected = match range {
                    YcbcrRange::Full => 1 << FORWARD_BITS,
                    YcbcrRange::Limited => 56_284, // 219/255 of 2^16, rounded
                };
                assert_eq!(luma, expected, "{matrix:?} {range:?}");
                assert_eq!(forward.rows[1].iter().sum::<i32>(), 0);
                assert_eq!(forward.rows[2].iter().sum::<i32>(), 0);
                // Grey, and the levels of black and white, for one pixel and
                // for the sums of 4:2:0's chroma.
                for value in [0, 1, 77, 128, 254, 255] {
                    let grey = [value; 3];
                    let y = forward.apply(0, grey, 0);
                    let expected = match range {
                        YcbcrRange::Full => value as u8,
                        YcbcrRange::Limited => {
                            (16.0 + 219.0 * f64::from(value) / 255.0).round() as u8
                        }
                    };
                    assert_eq!(y, expected, "{matrix:?} {range:?} grey {value}");
                    for (shift, weight) in [(0, 1), (2, 4), (3, 8)] {
                        let sum = grey.map(|v| v * weight);
                        assert_eq!(forward.apply(1, sum, shift), 128);
                        assert_eq!(forward.apply(2, sum, shift), 128);
                    }
                }
            }
        }
    }

    #[test]
    fn forward_integers_round_as_the_definition_does() {
        // The factors' rounding moves a result by less than 0.006 of a step:
        // it rounds as the exact value does, but where that is within 0.01
        // of half a step.
        let exact = |rgb: [i32; 3], matrix: ColorMatrix, range: YcbcrRange| {
            let (kr, kb) = matrix.luma_weights();
            let kg = 1.0 - kr - kb;
            let [r, g, b] = rgb.map(|v| f64::from(v) / 255.0);
            let y = kr * r + kg * g + kb * b;
            let (cb, cr) = ((b - y) / (2.0 * (1.0 - kb)), (r - y) / (2.0 * (1.0 - kr)));
            match range {
                YcbcrRange::Limited => [16.0 + 219.0 * y, 128.0 + 224.0 * cb, 128.0 + 224.0 * cr],
                YcbcrRange::Full => [255.0 * y, 128.0 + 255.0 * cb, 128.0 + 255.0 * cr],
            }
        };
        let mut state = 0x2545_f491_u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state % 256) as i32
        };
        let mut checked = 0;
        for matrix in MATRICES {
            for range in RANGES {
                let forward = ForwardMatrix::new(matrix, range);
                for _ in 0..20_000 {
                    let rgb = [next(), next(), next()];
                    for (row, value) in exact(rgb, matrix, range).into_iter().enumerate() {
                        let got = forward.apply(row, rgb, 0);
                        let rounded = value.round().clamp(0.0, 255.0) as u8;
                        if ((value - value.floor()) - 0.5).abs() > 0.01 {
                            assert_eq!(
                                got, rounded,
                                "{matrix:?} {range:?} {rgb:?} row {row}: {value}"
                            );
                            checked += 1;
                        } else {
                            assert!(got.abs_diff(rounded) <= 1);
                        }
                    }
                }
            }
        }
        assert!(checked > 340_000, "{checked}");
    }

    /// A `width` x `height` RGB picture in `order` with rows `stride` bytes
    /// apart, its pixels from `colour` and the bytes between rows 0xA5.
    fn picture(
        width: usize,
        height: usize,
        stride: usize,
        order: RgbOrder,
        colour: impl Fn(usize, usize) -> [u8; 3],
    ) -> Vec<u8> {
        let mut bytes = vec![0xA5; stride * (height - 1) + 4 * width];
        for y in 0..height {
            for x in 0..width {
                let [r, g, b] = colour(x, y);
                let pixel = match order {
                    RgbOrder::Rgba => [r, g, b, 0x5A],
                    RgbOrder::Bgra => [b, g, r, 0x5A],
                };
                bytes[y * stride + 4 * x..][..4].copy_from_slice(&pixel);
            }
        }
        bytes
    }

    #[test]
    fn rgb_pictures_convert_pixel_by_pixel_in_four_four_four() {
        let colour = |x: usize, y: usize| [(x * 37) as u8, (y * 53) as u8, ((x + y) * 11) as u8];
        let params = ColorParams::default();
        let forward = ForwardMatrix::new(params.matrix, params.range);
        let rgba = picture(7, 5, 7 * 4, RgbOrder::Rgba, colour);
        let planes = rgb_to_ycbcr8(
            &rgba,
            28,
            RgbOrder::Rgba,
            7,
            5,
            ChromaSubsampling::Yuv444,
            &params,
        )
        .unwrap();
        for (i, plane) in planes.iter().enumerate() {
            assert_eq!(plane.len(), 35);
            for y in 0..5 {
                for x in 0..7 {
                    let rgb = colour(x, y).map(i32::from);
                    assert_eq!(
                        plane[y * 7 + x],
                        forward.apply(i, rgb, 0),
                        "plane {i} at {x},{y}"
                    );
                }
            }
        }
        // The same picture as BGRA, with bytes between the rows that are not
        // read, gives the same planes.
        let bgra = picture(7, 5, 7 * 4 + 6, RgbOrder::Bgra, colour);
        let again = rgb_to_ycbcr8(
            &bgra,
            34,
            RgbOrder::Bgra,
            7,
            5,
            ChromaSubsampling::Yuv444,
            &params,
        )
        .unwrap();
        assert_eq!(again, planes);
    }

    #[test]
    fn four_two_zero_chroma_averages_what_it_covers_as_it_is_sited() {
        // Blue rising across the columns, one value a column, in a picture
        // whose rows differ in red, which chroma averages over two rows.
        let colour = |x: usize, y: usize| [(y * 40) as u8, 0, (x * 30) as u8];
        let bytes = picture(6, 4, 24, RgbOrder::Rgba, colour);
        for (siting, columns) in [
            (
                ChromaSiting::Center,
                [[0, 1], [2, 3], [4, 5]].map(|c| c.to_vec()),
            ),
            (
                ChromaSiting::Left,
                [[0, 0, 0, 1], [1, 2, 2, 3], [3, 4, 4, 5]].map(|c| c.to_vec()),
            ),
        ] {
            let params = ColorParams {
                chroma_siting: siting,
                ..ColorParams::default()
            };
            let forward = ForwardMatrix::new(params.matrix, params.range);
            let [luma, cb, cr] = rgb_to_ycbcr8(
                &bytes,
                24,
                RgbOrder::Rgba,
                6,
                4,
                ChromaSubsampling::Yuv420,
                &params,
            )
            .unwrap();
            assert_eq!((luma.len(), cb.len(), cr.len()), (24, 6, 6));
            let shift = columns[0].len().ilog2() + 1;
            for cy in 0..2 {
                for cx in 0..3 {
                    let mut sum = [0; 3];
                    for y in [2 * cy, 2 * cy + 1] {
                        for &x in &columns[cx] {
                            for (total, value) in sum.iter_mut().zip(colour(x, y)) {
                                *total += i32::from(value);
                            }
                        }
                    }
                    assert_eq!(
                        cb[cy * 3 + cx],
                        forward.apply(1, sum, shift),
                        "{siting:?} Cb {cx},{cy}"
                    );
                    assert_eq!(
                        cr[cy * 3 + cx],
                        forward.apply(2, sum, shift),
                        "{siting:?} Cr {cx},{cy}"
                    );
                }
            }
        }
    }

    #[test]
    fn rgb_pictures_of_odd_size_repeat_their_edges() {
        // 3x3 in 4:2:0: the last chroma column and row cover the last
        // column and row twice.
        let flat = picture(3, 3, 12, RgbOrder::Rgba, |_, _| [200, 100, 50]);
        let [luma, cb, cr] = rgb_to_ycbcr8(
            &flat,
            12,
            RgbOrder::Rgba,
            3,
            3,
            ChromaSubsampling::Yuv420,
            &ColorParams::default(),
        )
        .unwrap();
        assert_eq!(luma.len(), 9);
        assert_eq!((cb.len(), cr.len()), (4, 4));
        assert!(cb.iter().all(|&v| v == cb[0]) && cr.iter().all(|&v| v == cr[0]));
    }

    #[test]
    fn rgb_pictures_smaller_than_their_size_are_errors() {
        let params = ColorParams::default();
        let bytes = vec![0u8; 4 * 4 * 2];
        let convert = |bytes: &[u8], stride| {
            rgb_to_ycbcr8(
                bytes,
                stride,
                RgbOrder::Bgra,
                4,
                2,
                ChromaSubsampling::Yuv420,
                &params,
            )
        };
        assert!(convert(&bytes, 16).is_ok());
        assert_eq!(
            convert(&bytes, 15),
            Err(ColorError::StrideTooSmall {
                stride: 15,
                row: 16
            })
        );
        assert_eq!(
            convert(&bytes[..31], 16),
            Err(ColorError::PictureTooSmall {
                needed: 32,
                actual: 31
            })
        );
        // The last row needs only its pixels, not the whole stride.
        assert!(convert(&[0u8; 20 + 16], 20).is_ok());
        assert!(matches!(
            convert(&[0u8; 20 + 15], 20),
            Err(ColorError::PictureTooSmall { .. })
        ));
    }

    #[test]
    fn rgb_survives_a_round_trip_through_ycbcr() {
        // A smooth picture: 4:2:0 loses little of it, and the integers and
        // the floats back meet within a few steps.
        let colour =
            |x: usize, y: usize| [(40 + 3 * x) as u8, (200 - 2 * y) as u8, (90 + x + y) as u8];
        for range in RANGES {
            for chroma in [ChromaSubsampling::Yuv444, ChromaSubsampling::Yuv420] {
                for siting in [ChromaSiting::Center, ChromaSiting::Left] {
                    let params = ColorParams {
                        matrix: ColorMatrix::Bt709,
                        range,
                        chroma_siting: siting,
                    };
                    let bytes = picture(16, 8, 64, RgbOrder::Rgba, colour);
                    let [y, cb, cr] =
                        rgb_to_ycbcr8(&bytes, 64, RgbOrder::Rgba, 16, 8, chroma, &params).unwrap();
                    let rgb = convert([&y, &cb, &cr], 16, 8, chroma, &params);
                    for row in 1..7 {
                        for x in 1..15 {
                            let back = &rgb[(row * 16 + x) * 3..][..3];
                            for (got, want) in back.iter().zip(colour(x, row)) {
                                assert!(
                                    got.abs_diff(want) <= 3,
                                    "{range:?} {chroma:?} {siting:?} at {x},{row}: {back:?} for {:?}",
                                    colour(x, row)
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
