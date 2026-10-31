//! A stream the player makes itself (`web/index.html`): an animated scene
//! drawn and encoded in the page with the CPU encoder, so that a visitor
//! without a `.pwv1` file sees a stream decoded at once. The scene is the
//! README's banner (`docs/images/banner.svg`) set in motion: a wavelet whose
//! crests travel under its envelope, and embers rising from it. Its pictures
//! loop without a seam.

use std::f32::consts::{PI, TAU};

use emberwave_core::bitstream::{ChromaSubsampling, Pwv1Header, pwv1};
use emberwave_core::encode::{self, Signalling};
use wasm_bindgen::prelude::*;

use crate::js_error;

/// The demo's pictures: 960x540, small enough for the CPU encoder to make
/// them in a few seconds in a browser, at 30 frames a second, two seconds
/// that loop.
const WIDTH: u32 = 960;
const HEIGHT: u32 = 540;
const FRAMES: u32 = 60;
const FRAME_RATE: u32 = 30;

/// Bytes per frame, 9.6 Mbit/s at 30 frames a second: the scene is mostly a
/// dark, smooth background, which costs little.
const BUDGET: usize = 40_000;

/// How far the wave's glow reaches across it, in rows of a 540-row picture.
const REACH: f32 = 90.0;

/// Draws and encodes the demo stream a picture at a time, so that the page
/// can show its progress in between, into a PWV1 file for
/// [`Player::load`](crate::Player::load).
#[wasm_bindgen]
pub struct Demo {
    encoder: encode::Encoder,
    file: Vec<u8>,
    next: u32,
}

#[wasm_bindgen]
impl Demo {
    /// Starts the stream: the encoder and the file's header.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<Demo, JsError> {
        let encoder = encode::Encoder::new(
            WIDTH,
            HEIGHT,
            ChromaSubsampling::Yuv420,
            Signalling::default(),
        )
        .map_err(js_error)?;
        let header = Pwv1Header {
            stream: encoder.stream(),
            frame_rate_num: FRAME_RATE,
            frame_rate_den: 1,
            reference_bit_depth: 8,
        };
        let mut file = Vec::with_capacity(pwv1::FILE_HEADER_BYTES + FRAMES as usize * (4 + BUDGET));
        file.extend_from_slice(&header.to_bytes());
        Ok(Self {
            encoder,
            file,
            next: 0,
        })
    }

    /// How many pictures the stream has.
    pub fn frames() -> u32 {
        FRAMES
    }

    /// The most bytes a frame takes.
    pub fn frame_budget() -> u32 {
        BUDGET as u32
    }

    /// Draws and encodes the next picture, if any is left. Returns how many
    /// are encoded.
    pub fn encode_next(&mut self) -> Result<u32, JsError> {
        if self.next < FRAMES {
            let [y, cb, cr] = picture(WIDTH, HEIGHT, self.next, FRAMES);
            let frame = self
                .encoder
                .encode([&y, &cb, &cr], BUDGET)
                .map_err(js_error)?;
            let size = u32::try_from(frame.len()).map_err(js_error)?;
            self.file.extend_from_slice(&size.to_le_bytes());
            self.file.extend_from_slice(&frame);
            self.next += 1;
        }
        Ok(self.next)
    }

    /// The PWV1 file of the pictures encoded so far, which the demo no
    /// longer holds afterwards.
    pub fn take_file(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.file)
    }
}

/// The colour of the wave at `along`, 0 at its left end and 1 at its right:
/// yellow, then orange, then red, as the banner's.
fn fire(along: f32) -> [f32; 3] {
    const STOPS: [[f32; 3]; 3] = [[1.0, 0.82, 0.40], [1.0, 0.48, 0.10], [0.91, 0.19, 0.16]];
    let at = along.clamp(0.0, 1.0) * 2.0;
    let (from, to, mix) = if at < 1.0 {
        (STOPS[0], STOPS[1], at)
    } else {
        (STOPS[1], STOPS[2], at - 1.0)
    };
    [0, 1, 2].map(|i| from[i] + (to[i] - from[i]) * mix)
}

/// Picture `index` of `count` of the scene, `width` x `height` in 4:2:0:
/// the Y, Cb and Cr planes, BT.709 in video range, as the stream signals.
fn picture(width: u32, height: u32, index: u32, count: u32) -> [Vec<u8>; 3] {
    let (w, h) = (width as usize, height as usize);
    // Sizes below are in rows of a 540-row picture.
    let scale = h as f32 / 540.0;
    let loop_at = index as f32 / count as f32;

    // Gamma-encoded RGB, 0 to 1: a dark, warm background, lighter towards
    // the middle.
    let mut rgb = vec![[0.0f32; 3]; w * h];
    for (i, pixel) in rgb.iter_mut().enumerate() {
        let u = (i % w) as f32 / w as f32 - 0.5;
        let v = (i / w) as f32 / h as f32 - 0.5;
        let light = 1.0 - (u * u * 1.6 + v * v * 2.4).min(1.0);
        *pixel = [
            0.05 + 0.06 * light,
            0.035 + 0.035 * light,
            0.035 + 0.03 * light,
        ];
    }

    // A Morlet wavelet across the middle, its crests moving by one cycle of
    // the phase per loop, and two fainter echoes of it: one wave at three
    // scales.
    let centre = h as f32 * 0.54;
    let amplitude = h as f32 * 0.26;
    let half = w as f32 * 0.36;
    let phase = TAU * loop_at;
    let envelope = |x: f32| -> f32 {
        let u = (x - w as f32 / 2.0) / half;
        (-(u / 0.42).powi(2)).exp()
    };
    let curve = |x: f32, a: f32| -> f32 {
        let u = (x - w as f32 / 2.0) / half;
        centre - a * (PI * 4.2 * u - phase).cos() * envelope(x)
    };
    for x in 0..w {
        let at = x as f32 + 0.5;
        let colour = fire((at / w as f32 - 0.14) / 0.72);
        for (a, strength, core_width) in [
            (amplitude, 1.0, 2.0),
            (amplitude * 0.55, 0.32, 1.3),
            (amplitude * 0.28, 0.2, 1.1),
        ] {
            let y0 = curve(at, a);
            let slope = (curve(at + 1.0, a) - curve(at - 1.0, a)) / 2.0;
            let stretch = (1.0 + slope * slope).sqrt();
            let reach = REACH * scale * stretch;
            let top = (y0 - reach).max(0.0) as usize;
            let bottom = ((y0 + reach) as usize).min(h - 1);
            for y in top..=bottom {
                // The distance to the curve, across it, in 540-row pixels.
                let d = ((y as f32 + 0.5 - y0) / stretch).abs() / scale;
                let core = 1.0 / (1.0 + (d / core_width).powi(4));
                // The glow fades out by the reach, so that no edge shows.
                let fade = (1.0 - d / REACH).max(0.0).powi(2);
                let glow =
                    fade * (0.45 / (1.0 + (d / 8.0).powi(2)) + 0.16 / (1.0 + (d / 28.0).powi(2)));
                let pixel = &mut rgb[y * w + x];
                for c in 0..3 {
                    pixel[c] += strength * (colour[c] * (core + glow) + 0.45 * core * core);
                }
            }
        }
    }

    // Embers rising from the crests. Each rises once or twice per loop, so
    // the loop has no seam; a fixed sequence places them the same in every
    // picture.
    let mut seed = 0x2545_f491_u32;
    let mut random = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed as f32 / u32::MAX as f32
    };
    for _ in 0..40 {
        let u = random() * 1.6 - 0.8;
        let start = random();
        let laps = 1.0 + (random() * 2.0).floor();
        let size = (1.0 + random() * 1.8) * scale;
        let sway = random() * TAU;
        let risen = (start + laps * loop_at).fract();
        let x0 = w as f32 / 2.0 + u * half;
        let x = x0 + 10.0 * scale * (TAU * risen + sway).sin();
        let y = centre - amplitude * envelope(x0) - 10.0 * scale - risen * 170.0 * scale;
        let bright = (1.0 - risen) * (risen * 6.0).min(1.0);
        let reach = (size * 4.0).ceil() as i32;
        for dy in -reach..=reach {
            for dx in -reach..=reach {
                let (px, py) = (x as i32 + dx, y as i32 + dy);
                if px < 0 || py < 0 || px >= w as i32 || py >= h as i32 {
                    continue;
                }
                let d = ((px as f32 + 0.5 - x).powi(2) + (py as f32 + 0.5 - y).powi(2)).sqrt();
                let fade = (1.0 - d / reach as f32).max(0.0).powi(2);
                let k = bright / (1.0 + (d / size).powi(4))
                    + 0.25 * bright * fade / (1.0 + (d / (3.0 * size)).powi(2));
                let pixel = &mut rgb[py as usize * w + px as usize];
                for (c, ember) in [1.0, 0.62, 0.26].into_iter().enumerate() {
                    pixel[c] += ember * k;
                }
            }
        }
    }

    // BT.709 in video range; chroma as the mean of each 2x2 block, centred.
    let to_byte = |value: f32| (value + 0.5).clamp(0.0, 255.0) as u8;
    let mut luma = vec![0u8; w * h];
    let (cw, ch) = (w / 2, h / 2);
    let mut blue = vec![0.0f32; cw * ch];
    let mut red = vec![0.0f32; cw * ch];
    for (i, pixel) in rgb.iter().enumerate() {
        let [r, g, b] = pixel.map(|c| c.clamp(0.0, 1.0));
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        luma[i] = to_byte(16.0 + 219.0 * y);
        let block = (i / w / 2).min(ch - 1) * cw + (i % w / 2).min(cw - 1);
        blue[block] += (b - y) / 1.8556 / 4.0;
        red[block] += (r - y) / 1.5748 / 4.0;
    }
    let chroma = |plane: Vec<f32>| -> Vec<u8> {
        plane
            .into_iter()
            .map(|c| to_byte(128.0 + 224.0 * c))
            .collect()
    };
    [luma, chroma(blue), chroma(red)]
}
