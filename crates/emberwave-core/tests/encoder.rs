//! emberwave's encoder against its committed vectors in `testdata/encoder/`
//! (made by `scripts/gen_encoder_vectors.sh`): the same bytes again on every
//! system, frames within their budgets, and streams that upstream's decoder
//! decodes the way emberwave's does.
//!
//! The decoder's conformance vectors stay those of upstream's encoder, in
//! `testdata/`; these test the encoder (decision D28).

mod common;

use common::{Reference, Y4m, encoder_dir, encoder_vectors, psnr, read_y4m};
use emberwave_core::bitstream::{ChromaSubsampling, FrameAssembler, FrameLayout, Pwv1File};
use emberwave_core::cpu::decode_frame;
use emberwave_core::encode::{Encoder, Signalling};

/// A line of `testdata/encoder/vectors.txt`.
struct Vector {
    name: String,
    budget: usize,
}

fn vectors() -> Vec<Vector> {
    encoder_vectors()
        .into_iter()
        .map(|(name, budget)| Vector { name, budget })
        .collect()
}

fn source(vector: &Vector) -> Y4m {
    read_y4m(&encoder_dir().join(format!("{}.source.y4m", vector.name)))
}

/// The committed stream: its layout and frames.
fn stream(vector: &Vector) -> (FrameLayout, Vec<Vec<u8>>) {
    let bytes = std::fs::read(encoder_dir().join(format!("{}.pwv1", vector.name))).expect("stream");
    let file = Pwv1File::parse(&bytes).expect("a PWV1 file");
    let header = file.header().stream;
    let layout = FrameLayout::new(header.width, header.height, header.chroma).expect("layout");
    let frames = file
        .frames()
        .map(|frame| frame.expect("a frame").to_vec())
        .collect();
    (layout, frames)
}

fn decode(layout: &FrameLayout, frame: &[u8]) -> [Vec<u8>; 3] {
    let mut assembler = FrameAssembler::new(layout.clone());
    assembler.push_packet(frame).expect("the frame assembles");
    assert!(assembler.is_ready(false), "the frame is complete");
    decode_frame(&assembler.frame())
        .expect("the frame decodes")
        .planes
        .map(|plane| plane.to_unorm8())
}

fn worst(a: &[[Vec<u8>; 3]], b: &[[Vec<u8>; 3]]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| psnr(a, b)))
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn encoding_again_gives_the_committed_bytes() {
    // The encoder is deterministic: fixed order of every sum, no fused
    // multiply-adds, weights as literals (decision D28). If this fails after
    // a deliberate change to the encoder, run scripts/gen_encoder_vectors.sh.
    for vector in vectors() {
        let source = source(&vector);
        let (layout, committed) = stream(&vector);
        let chroma = if source.chroma_444 {
            ChromaSubsampling::Yuv444
        } else {
            ChromaSubsampling::Yuv420
        };
        assert_eq!(layout.chroma(), chroma);
        let mut encoder = Encoder::new(
            source.width as u32,
            source.height as u32,
            chroma,
            Signalling::default(),
        )
        .expect("an encoder");
        assert_eq!(source.frames.len(), committed.len(), "{}", vector.name);
        for (index, ([y, cb, cr], expected)) in source.frames.iter().zip(&committed).enumerate() {
            let frame = encoder
                .encode([y, cb, cr], vector.budget)
                .expect("the frame fits");
            assert!(
                frame == *expected,
                "{} frame {index}: {} bytes, {} committed",
                vector.name,
                frame.len(),
                expected.len()
            );
        }
    }
}

#[test]
fn every_frame_fits_its_budget() {
    for vector in vectors() {
        let (_, frames) = stream(&vector);
        for (index, frame) in frames.iter().enumerate() {
            assert!(
                frame.len() <= vector.budget,
                "{} frame {index}",
                vector.name
            );
        }
    }
}

#[test]
fn upstreams_decoder_decodes_the_streams_as_emberwaves_does() {
    // Upstream's FP32 decode of emberwave's streams, made when the vectors
    // were, against emberwave's decode: the bar of decision D9.
    for vector in vectors() {
        let (layout, frames) = stream(&vector);
        let ours: Vec<[Vec<u8>; 3]> = frames.iter().map(|frame| decode(&layout, frame)).collect();
        let upstream = read_y4m(&encoder_dir().join(format!("{}.ref-fp32.y4m", vector.name)));
        let quality = worst(&ours, &upstream.frames);
        assert!(
            quality >= Reference::Fp32.min_psnr(),
            "{}: {quality:.2} dB",
            vector.name
        );
    }
}

#[test]
fn the_streams_show_their_pictures() {
    // Floors a little below what the vectors reach, to catch an encoder that
    // still makes valid streams but bad pictures. The tight budgets lose more.
    for vector in vectors() {
        let floor = match vector.name.as_str() {
            // Every plane fits the budget: lossless at 8 bits.
            "gradient_200x120_420" | "solid_160x96_420" | "solid_64x64_420_wrap" => f64::INFINITY,
            "scene_200x120_420" => 38.0,
            "scene_131x67_444" => 33.0,
            "noise_192x136_420_lowrate" => 12.0,
            // 300 bytes: what is left of the coarsest bands (23.7 dB).
            "scene_200x120_420_tiny" => 20.0,
            // The smallest frames: the coarsest level, at its smallest,
            // against noise (10.7 dB).
            "noise_64x64_444_smallest" => 9.0,
            other => panic!("no floor for {other}"),
        };
        let (layout, frames) = stream(&vector);
        let decoded: Vec<[Vec<u8>; 3]> =
            frames.iter().map(|frame| decode(&layout, frame)).collect();
        let quality = worst(&decoded, &source(&vector).frames);
        assert!(quality >= floor, "{}: {quality:.2} dB", vector.name);
    }
}
