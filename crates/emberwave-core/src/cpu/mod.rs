//! Slow, single-threaded CPU decoder that follows `bitstream.md` literally.
//!
//! The real decoder runs these stages on the GPU. This module exists so that
//! the bitstream parser can be checked against real encoder output without a
//! GPU, and so that shader stages can be compared against something simple.
//! It is not meant to be fast.

pub mod dequant;
pub mod idwt;

use crate::bitstream::consts::{DC_SHIFT, DECOMPOSITION_LEVELS, NUM_COMPONENTS};
use crate::bitstream::{Band, Component, Error, FrameLayout, FrameView};
use dequant::Bands;

/// One decoded colour plane with samples in `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Plane {
    /// Width in samples.
    pub width: u32,
    /// Height in samples.
    pub height: u32,
    /// Row-major samples.
    pub samples: Vec<f32>,
}

impl Plane {
    /// Converts to 8-bit UNORM the way a GPU storage write does: scale to
    /// 255 and round to nearest, ties to even.
    pub fn to_unorm8(&self) -> Vec<u8> {
        self.samples
            .iter()
            .map(|v| (v * 255.0).round_ties_even() as u8)
            .collect()
    }

    /// Converts to 16-bit UNORM, rounding to nearest, ties to even.
    pub fn to_unorm16(&self) -> Vec<u16> {
        self.samples
            .iter()
            .map(|v| (v * 65535.0).round_ties_even() as u16)
            .collect()
    }
}

/// A decoded frame: Y, Cb and Cr planes.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedFrame {
    /// The planes in bitstream order (Y, Cb, Cr). Chroma planes are half
    /// size for 4:2:0.
    pub planes: [Plane; 3],
}

/// The LL band that enters the inverse transform of each level, per
/// component: level 4 comes from the bitstream and the others are produced by
/// the level above. Row-major with the width of the level as stride.
#[derive(Debug, Clone, Default)]
pub struct LlBands {
    levels: [[Option<Vec<f32>>; DECOMPOSITION_LEVELS]; NUM_COMPONENTS],
}

impl LlBands {
    /// The LL band of `level`, or `None` if the component has no such level.
    pub fn get(&self, component: Component, level: usize) -> Option<&[f32]> {
        self.levels.get(component.index())?.get(level)?.as_deref()
    }
}

/// Decodes a frame: dequantisation, inverse DWT and the DC shift.
///
/// Blocks that were not received decode as zero coefficients.
pub fn decode_frame(frame: &FrameView<'_>) -> Result<DecodedFrame, Error> {
    let layout = frame.layout();
    let mut bands = dequant::dequantize(frame)?;
    let planes = Component::ALL.map(|component| reconstruct(layout, &mut bands, component, None));
    Ok(DecodedFrame { planes })
}

/// Like [`decode_frame`], and also returns the intermediate LL bands.
pub fn decode_frame_with_ll(frame: &FrameView<'_>) -> Result<(DecodedFrame, LlBands), Error> {
    let layout = frame.layout();
    let mut bands = dequant::dequantize(frame)?;
    let mut ll_bands = LlBands::default();
    let planes = Component::ALL.map(|component| {
        reconstruct(
            layout,
            &mut bands,
            component,
            Some(&mut ll_bands.levels[component.index()]),
        )
    });
    Ok((DecodedFrame { planes }, ll_bands))
}

/// Runs the inverse transform from the coarsest level up for one component,
/// optionally keeping a copy of the LL band that enters each level.
fn reconstruct(
    layout: &FrameLayout,
    bands: &mut Bands,
    component: Component,
    mut trace: Option<&mut [Option<Vec<f32>>; DECOMPOSITION_LEVELS]>,
) -> Plane {
    let coarsest = DECOMPOSITION_LEVELS - 1;
    let (coarsest_width, coarsest_height) = layout.level_size(coarsest);
    let (mut width, mut height) = (coarsest_width as usize, coarsest_height as usize);

    let mut take = |level: usize, band: Band, width: usize, height: usize| {
        bands
            .take(component, level, band)
            .unwrap_or_else(|| vec![0.0; width * height])
    };

    let mut ll = take(coarsest, Band::Ll, width, height);
    for level in (layout.finest_level(component)..=coarsest).rev() {
        if let Some(trace) = trace.as_deref_mut() {
            trace[level] = Some(ll.clone());
        }
        let hl = take(level, Band::Hl, width, height);
        let lh = take(level, Band::Lh, width, height);
        let hh = take(level, Band::Hh, width, height);
        ll = idwt::inverse_2d(&ll, &hl, &lh, &hh, width, height);
        width *= 2;
        height *= 2;
    }

    // Crop the padding away and undo the DC shift.
    let (plane_width, plane_height) = layout.plane_size(component);
    let mut samples = Vec::with_capacity(plane_width as usize * plane_height as usize);
    for y in 0..plane_height as usize {
        let row = &ll[y * width..y * width + plane_width as usize];
        samples.extend(row.iter().map(|v| (v + DC_SHIFT).clamp(0.0, 1.0)));
    }

    Plane {
        width: plane_width,
        height: plane_height,
        samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unorm_conversion_rounds_ties_to_even() {
        let plane = Plane {
            width: 4,
            height: 1,
            samples: vec![0.0, 1.0, 0.5, 2.5 / 255.0],
        };
        // 0.5 * 255 = 127.5 -> 128 (even); 2.5 -> 2 (even).
        assert_eq!(plane.to_unorm8(), [0, 255, 128, 2]);
        assert_eq!(plane.to_unorm16()[..2], [0, 65535]);
    }
}
