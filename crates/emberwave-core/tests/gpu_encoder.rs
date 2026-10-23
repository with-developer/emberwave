//! The GPU encoder through its public API: on the sources of the encoder's
//! vectors in `testdata/encoder/`, against the CPU encoder's committed
//! streams; on a device limited to what WebGPU guarantees; frame after frame;
//! from textures as from CPU memory; and on a lost device (decisions D29 and
//! D32).
//!
//! Like `tests/gpu.rs`, they skip without a GPU unless
//! `EMBERWAVE_REQUIRE_GPU` is set.

#![cfg(feature = "gpu")]

mod common;

use common::{
    Y4m, create_context, encoder_dir, encoder_vectors, live_resources, psnr, read_y4m,
    shared_context,
};
use emberwave_core::bitstream::{
    ChromaSubsampling, Component, FrameAssembler, FrameLayout, Header, Pwv1File,
};
use emberwave_core::cpu::decode_frame;
use emberwave_core::encode::{self, EncodeError, Signalling};
use emberwave_core::gpu::{Decoder, Encoder, GpuContext, GpuError, GpuOptions, TexturePicture};

/// Skips the current test when there is no GPU.
macro_rules! gpu_or_return {
    () => {
        match shared_context() {
            Some(context) => context,
            None => return,
        }
    };
}

/// A vector of `testdata/encoder/`: its source, budget and committed frames.
struct Vector {
    name: String,
    budget: usize,
    source: Y4m,
    layout: FrameLayout,
    committed: Vec<Vec<u8>>,
}

fn vectors() -> Vec<Vector> {
    encoder_vectors()
        .into_iter()
        .map(|(name, budget)| {
            let source = read_y4m(&encoder_dir().join(format!("{name}.source.y4m")));
            let bytes = std::fs::read(encoder_dir().join(format!("{name}.pwv1"))).expect("stream");
            let file = Pwv1File::parse(&bytes).expect("a PWV1 file");
            let stream = file.header().stream;
            let layout =
                FrameLayout::new(stream.width, stream.height, stream.chroma).expect("layout");
            let committed = file
                .frames()
                .map(|frame| frame.expect("a frame").to_vec())
                .collect();
            Vector {
                name,
                budget,
                source,
                layout,
                committed,
            }
        })
        .collect()
}

fn encoder_for(context: &GpuContext, layout: &FrameLayout) -> Encoder {
    Encoder::new(
        context,
        layout.width(),
        layout.height(),
        layout.chroma(),
        Signalling::default(),
    )
    .expect("an encoder")
}

fn planes(frame: &[Vec<u8>; 3]) -> [&[u8]; 3] {
    [&frame[0], &frame[1], &frame[2]]
}

/// A texture of `format` and `usage` on `context`'s device, filled with
/// `data` if there is any: rows of texels without padding.
fn texture(
    context: &GpuContext,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    (width, height): (u32, u32),
    data: &[u8],
) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = context.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("test picture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: usage | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    if !data.is_empty() {
        let texel_bytes = format
            .block_copy_size(None)
            .expect("an uncompressed format");
        context.queue().write_texture(
            texture.as_image_copy(),
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * texel_bytes),
                rows_per_image: Some(height),
            },
            size,
        );
    }
    texture
}

/// A picture in textures on a context's device: a texture per plane, and Cb
/// and Cr interleaved in another, as NV12 keeps them.
struct Textures {
    planes: [wgpu::TextureView; 3],
    chroma: wgpu::TextureView,
}

impl Textures {
    fn new(context: &GpuContext, layout: &FrameLayout, source: &[Vec<u8>; 3]) -> Self {
        let usage = wgpu::TextureUsages::TEXTURE_BINDING;
        let view = |texture: wgpu::Texture| texture.create_view(&Default::default());
        let planes = std::array::from_fn(|i| {
            let size = layout.plane_size(Component::ALL[i]);
            view(texture(
                context,
                wgpu::TextureFormat::R8Unorm,
                usage,
                size,
                &source[i],
            ))
        });
        let interleaved: Vec<u8> = source[1]
            .iter()
            .zip(&source[2])
            .flat_map(|(&cb, &cr)| [cb, cr])
            .collect();
        let size = layout.plane_size(Component::Cb);
        let chroma = view(texture(
            context,
            wgpu::TextureFormat::Rg8Unorm,
            usage,
            size,
            &interleaved,
        ));
        Self { planes, chroma }
    }

    fn planes(&self) -> TexturePicture<'_> {
        TexturePicture::Planes([&self.planes[0], &self.planes[1], &self.planes[2]])
    }

    fn luma_chroma(&self) -> TexturePicture<'_> {
        TexturePicture::LumaChroma {
            luma: &self.planes[0],
            chroma: &self.chroma,
        }
    }
}

/// The sequence number of a frame, from its start of frame header.
fn sequence_of(frame: &[u8]) -> u8 {
    match Header::parse(frame).expect("a start of frame header") {
        Header::StartOfFrame(header) => header.sequence,
        other => panic!("{other:?}"),
    }
}

/// The PSNR of the CPU decode of `frame` against `source`, of all samples.
fn quality(layout: &FrameLayout, source: &[Vec<u8>; 3], frame: &[u8]) -> f64 {
    let mut assembler = FrameAssembler::new(layout.clone());
    assembler.push_packet(frame).expect("the frame assembles");
    assert!(assembler.is_ready(false), "the frame is complete");
    let decoded: Vec<u8> = decode_frame(&assembler.frame())
        .expect("the frame decodes")
        .planes
        .iter()
        .flat_map(|plane| plane.to_unorm8())
        .collect();
    psnr(&source.concat(), &decoded)
}

#[test]
fn the_encoders_vectors_come_out_as_the_cpu_made_them_or_as_good() {
    // The GPU's coefficients may differ from the CPU's in the last place, so
    // a frame is the CPU encoder's or differs in a few blocks, never for the
    // worse by more than noise (decision D29).
    let context = gpu_or_return!();
    let (mut frames, mut identical) = (0, 0);
    for vector in vectors() {
        let mut encoder = encoder_for(context, &vector.layout);
        assert_eq!(vector.source.frames.len(), vector.committed.len());
        for (index, (source, expected)) in vector
            .source
            .frames
            .iter()
            .zip(&vector.committed)
            .enumerate()
        {
            let frame = encoder
                .encode(planes(source), vector.budget)
                .expect("the frame fits");
            assert!(
                frame.len() <= vector.budget,
                "{} frame {index}",
                vector.name
            );
            frames += 1;
            if frame == *expected {
                identical += 1;
                continue;
            }
            let (cpu, gpu) = (
                quality(&vector.layout, source, expected),
                quality(&vector.layout, source, &frame),
            );
            println!(
                "{} frame {index}: CPU {} B, {cpu:.3} dB; GPU {} B, {gpu:.3} dB",
                vector.name,
                expected.len(),
                frame.len()
            );
            assert!(gpu >= cpu - 0.05, "{} frame {index}", vector.name);
        }
    }
    println!("{identical} of {frames} frames identical to the CPU encoder's");
    assert!(frames > 10);
}

#[test]
fn the_gpu_decoder_shows_the_gpu_encoders_pictures() {
    let context = gpu_or_return!();
    for (width, height, chroma) in [
        (200, 120, ChromaSubsampling::Yuv420),
        (131, 67, ChromaSubsampling::Yuv444),
    ] {
        let layout = FrameLayout::new(width, height, chroma).unwrap();
        // A smooth picture with an edge.
        let source: [Vec<u8>; 3] = std::array::from_fn(|plane| {
            let (w, h) = layout.plane_size(emberwave_core::bitstream::Component::ALL[plane]);
            (0..w * h)
                .map(|i| {
                    let (x, y) = (i % w, i / w);
                    (40 + (x * 120 / w + y * 50 / h) as usize + if x > w / 2 { 40 } else { 0 })
                        as u8
                })
                .collect()
        });
        let mut encoder = encoder_for(context, &layout);
        let frame = encoder
            .encode(planes(&source), 200_000)
            .expect("the frame fits");
        let mut decoder = Decoder::new(context, width, height, chroma).expect("a decoder");
        let decoded = decoder.decode(&frame).expect("the GPU decodes it");
        for (plane, (expected, actual)) in source.iter().zip(&decoded.planes).enumerate() {
            let db = psnr(expected, &actual.data);
            assert!(
                db >= 50.0,
                "{width}x{height} {chroma:?} plane {plane}: {db:.2} dB"
            );
        }
    }
}

#[test]
fn errors_are_the_cpu_encoders() {
    let context = gpu_or_return!();
    let vector = vectors()
        .into_iter()
        .find(|v| v.name == "scene_200x120_420")
        .unwrap();
    let mut gpu = encoder_for(context, &vector.layout);
    let mut cpu = encode::Encoder::new(
        vector.layout.width(),
        vector.layout.height(),
        vector.layout.chroma(),
        Signalling::default(),
    )
    .unwrap();
    let source = &vector.source.frames[0];
    let Err(EncodeError::BudgetTooSmall { minimum, .. }) = cpu.encode(planes(source), 0) else {
        panic!("a frame of 0 bytes");
    };
    for budget in [0, 7, minimum - 1] {
        let expected = cpu.encode(planes(source), budget).expect_err("too small");
        match gpu.encode(planes(source), budget) {
            Err(GpuError::Encode(actual)) => assert_eq!(actual, expected, "{budget} B"),
            other => panic!("{budget} B: {other:?}"),
        }
    }
    let short = &source[1][1..];
    let expected = cpu
        .encode([&source[0], short, &source[2]], 10_000)
        .expect_err("a short plane");
    assert!(matches!(expected, EncodeError::PlaneSize { .. }));
    match gpu.encode([&source[0], short, &source[2]], 10_000) {
        Err(GpuError::Encode(actual)) => assert_eq!(actual, expected),
        other => panic!("{other:?}"),
    }
    // A failure leaves the numbering alone: the next frame is still frame 1.
    let frame = gpu.encode(planes(source), vector.budget).unwrap();
    let sequence = match Header::parse(&frame).expect("a start of frame header") {
        Header::StartOfFrame(header) => header.sequence,
        other => panic!("{other:?}"),
    };
    assert_eq!(sequence, 1, "the first frame after the errors");
}

#[test]
fn a_device_limited_to_webgpu_guarantees_encodes_the_same() {
    let context = gpu_or_return!();
    let options = GpuOptions {
        strict_webgpu: true,
        ..GpuOptions::default()
    };
    let Some(strict) = create_context(&options) else {
        return;
    };
    for vector in vectors().into_iter().take(3) {
        let mut ours = encoder_for(context, &vector.layout);
        let mut limited = encoder_for(&strict, &vector.layout);
        for source in &vector.source.frames {
            let expected = ours.encode(planes(source), vector.budget).unwrap();
            let frame = limited.encode(planes(source), vector.budget).unwrap();
            assert!(frame == expected, "{}", vector.name);
        }
    }
}

#[test]
fn encoding_frame_after_frame_allocates_nothing() {
    // A context of its own: the counters belong to the device.
    let Some(context) = create_context(&GpuOptions::default()) else {
        return;
    };
    let vector = vectors()
        .into_iter()
        .find(|v| v.name == "scene_200x120_420")
        .unwrap();
    let mut encoder = encoder_for(&context, &vector.layout);
    let source = &vector.source.frames[0];
    // The largest budget first: the payload buffer, 64 KiB at first, grows
    // to what this picture can need at most, and stays.
    let budgets = [1_000_000, 300, 70_000, 20_000];
    for &budget in &budgets {
        encoder.encode(planes(source), budget).unwrap();
    }
    assert!(
        context.device().get_internal_counters().hal.buffers.read() > 0,
        "wgpu's counters read zero: are they enabled for the tests?"
    );
    let before = live_resources(&context);
    for round in 0..24 {
        let frame = &vector.source.frames[round % vector.source.frames.len()];
        encoder
            .encode(planes(frame), budgets[round % budgets.len()])
            .unwrap();
    }
    assert_eq!(
        live_resources(&context),
        before,
        "live GPU objects and memory"
    );
}

#[test]
fn a_lost_device_is_an_error_rather_than_a_stale_result() {
    // A context of its own, since destroying the device loses it for good.
    let Some(context) = create_context(&GpuOptions::default()) else {
        return;
    };
    let vector = vectors()
        .into_iter()
        .find(|v| v.name == "scene_200x120_420")
        .unwrap();
    let mut encoder = encoder_for(&context, &vector.layout);
    let source = &vector.source.frames[0];
    encoder
        .encode(planes(source), vector.budget)
        .expect("an encode on a working device");
    assert_eq!(encoder.lost(), None);

    context.device().destroy();
    let result = encoder.encode(planes(source), vector.budget);
    assert!(matches!(result, Err(GpuError::DeviceLost(_))), "{result:?}");
    assert!(encoder.lost().is_some());
}

#[test]
fn timing_an_encode_reports_where_the_time_goes() {
    // Timestamp queries are optional, so ask for them and accept their absence.
    let options = GpuOptions {
        timestamps: true,
        ..GpuOptions::default()
    };
    let Some(context) = create_context(&options) else {
        return;
    };
    let vector = vectors()
        .into_iter()
        .find(|v| v.name == "scene_200x120_420")
        .unwrap();
    let mut timed = encoder_for(&context, &vector.layout);
    let mut plain = encoder_for(&context, &vector.layout);
    for source in &vector.source.frames {
        let (frame, timings) = timed.encode_timed(planes(source), vector.budget).unwrap();
        // Timing changes nothing in the frame.
        assert!(frame == plain.encode(planes(source), vector.budget).unwrap());
        assert!(timings.gpu_wall > std::time::Duration::ZERO);
        assert!(timings.prepare > std::time::Duration::ZERO);
        match timings.stages {
            Some(stages) => {
                assert!(context.supports_timestamps());
                // A short stage can be below the timer's resolution and read
                // as zero, but the whole encode cannot, nor the stages of
                // several dispatches: quantisation runs a workgroup per block
                // and rate control four passes and two prefix sums.
                assert!(stages.total() > std::time::Duration::ZERO);
                assert!(stages.quantisation > std::time::Duration::ZERO);
                assert!(stages.rate_control > std::time::Duration::ZERO);
                // The GPU cannot take longer than the wall clock around it.
                assert!(stages.total() <= timings.gpu_wall + timings.prepare);
            }
            // A device that offers timestamps has to deliver them.
            None => assert!(
                !context.supports_timestamps(),
                "the device supports timestamp queries but returned no stage times"
            ),
        }
    }
}

#[test]
fn nothing_is_made_on_a_lost_device() {
    // Making a decoder or an encoder on a lost device used to panic in wgpu
    // (issue #35). Each on a context of its own: the first constructor that
    // fails on a destroyed device records the loss, and a later one would
    // stop at the record before making anything. Before as well as after
    // the context compiled its pipelines, whose failure is then the first.
    for compiled in [false, true] {
        let lost = || {
            let context = create_context(&GpuOptions::default())?;
            if compiled {
                Decoder::new(&context, 64, 64, ChromaSubsampling::Yuv420).expect("a decoder");
                Encoder::new(
                    &context,
                    64,
                    64,
                    ChromaSubsampling::Yuv420,
                    Signalling::default(),
                )
                .expect("an encoder");
            }
            context.device().destroy();
            Some(context)
        };
        let Some(context) = lost() else {
            return;
        };
        let decoder = Decoder::new(&context, 64, 64, ChromaSubsampling::Yuv420);
        assert!(
            matches!(decoder, Err(GpuError::DeviceLost(_))),
            "compiled {compiled}: {:?}",
            decoder.map(drop)
        );
        let context = lost().expect("a GPU, as before");
        let encoder = Encoder::new(
            &context,
            64,
            64,
            ChromaSubsampling::Yuv420,
            Signalling::default(),
        );
        assert!(
            matches!(encoder, Err(GpuError::DeviceLost(_))),
            "compiled {compiled}: {:?}",
            encoder.map(drop)
        );
    }
}

#[test]
fn pictures_in_textures_encode_as_the_same_planes_do() {
    // gather.wgsl packs the textures into the planes that the upload from
    // CPU memory fills, so the frames are the same bytes, also on a device
    // limited to what WebGPU guarantees (decision D32).
    let context = gpu_or_return!();
    let mut contexts = vec![(context.clone(), usize::MAX)];
    let strict = GpuOptions {
        strict_webgpu: true,
        ..GpuOptions::default()
    };
    if let Some(strict) = create_context(&strict) {
        contexts.push((strict, 3));
    }
    for (context, count) in &contexts {
        for vector in vectors().into_iter().take(*count) {
            let mut from_planes = encoder_for(context, &vector.layout);
            let mut from_textures = encoder_for(context, &vector.layout);
            let mut from_luma_chroma = encoder_for(context, &vector.layout);
            for (index, source) in vector.source.frames.iter().enumerate() {
                let expected = from_planes.encode(planes(source), vector.budget).unwrap();
                let textures = Textures::new(context, &vector.layout, source);
                let frame = from_textures
                    .encode_textures(textures.planes(), vector.budget)
                    .unwrap();
                assert!(frame == expected, "{} frame {index}: planes", vector.name);
                let frame = from_luma_chroma
                    .encode_textures(textures.luma_chroma(), vector.budget)
                    .unwrap();
                assert!(
                    frame == expected,
                    "{} frame {index}: luma and chroma",
                    vector.name
                );
            }
        }
    }
}

#[test]
fn textures_that_cannot_hold_their_planes_are_errors() {
    let context = gpu_or_return!();
    let layout = FrameLayout::new(64, 32, ChromaSubsampling::Yuv420).unwrap();
    let mut encoder = encoder_for(context, &layout);
    let source: [Vec<u8>; 3] = std::array::from_fn(|i| {
        let (width, height) = layout.plane_size(Component::ALL[i]);
        vec![100 + 20 * i as u8; (width * height) as usize]
    });
    let good = Textures::new(context, &layout, &source);
    let bound = wgpu::TextureUsages::TEXTURE_BINDING;
    let view = |texture: wgpu::Texture| texture.create_view(&Default::default());
    let rgba = view(texture(
        context,
        wgpu::TextureFormat::Rgba8Unorm,
        bound,
        (64, 32),
        &[],
    ));
    let small = view(texture(
        context,
        wgpu::TextureFormat::R8Unorm,
        bound,
        (32, 32),
        &[],
    ));
    let unbound = view(texture(
        context,
        wgpu::TextureFormat::R8Unorm,
        wgpu::TextureUsages::empty(),
        (64, 32),
        &[],
    ));
    let flat_chroma = view(texture(
        context,
        wgpu::TextureFormat::R8Unorm,
        bound,
        (32, 16),
        &[],
    ));
    let cases = [
        (
            "format",
            TexturePicture::Planes([&rgba, &good.planes[1], &good.planes[2]]),
            "the Y texture is Rgba8Unorm rather than R8Unorm",
        ),
        (
            "size",
            TexturePicture::Planes([&small, &good.planes[1], &good.planes[2]]),
            "the Y texture is 32x32 rather than 64x32",
        ),
        (
            "usage",
            TexturePicture::Planes([&unbound, &good.planes[1], &good.planes[2]]),
            "the Y texture lacks the TEXTURE_BINDING usage",
        ),
        (
            "chroma",
            TexturePicture::LumaChroma {
                luma: &good.planes[0],
                chroma: &flat_chroma,
            },
            "the chroma texture is R8Unorm rather than Rg8Unorm",
        ),
    ];
    for (what, picture, message) in cases {
        match encoder.encode_textures(picture, 100_000) {
            Err(GpuError::Texture(actual)) => assert_eq!(actual, message, "{what}"),
            other => panic!("{what}: {:?}", other.map(drop)),
        }
    }
    // A failure leaves the numbering alone: the next frame is still frame 1.
    let frame = encoder.encode_textures(good.planes(), 100_000).unwrap();
    assert_eq!(sequence_of(&frame), 1, "the first frame after the errors");
}

#[test]
fn encoding_from_textures_frame_after_frame_allocates_nothing() {
    // A context of its own: the counters belong to the device. The bind
    // group of each frame's textures goes when the frame is done.
    let Some(context) = create_context(&GpuOptions::default()) else {
        return;
    };
    let vector = vectors()
        .into_iter()
        .find(|v| v.name == "scene_200x120_420")
        .unwrap();
    let mut encoder = encoder_for(&context, &vector.layout);
    let textures: Vec<Textures> = vector
        .source
        .frames
        .iter()
        .map(|source| Textures::new(&context, &vector.layout, source))
        .collect();
    let mut encode = |round: usize| {
        let textures = &textures[round % textures.len()];
        let picture = if round.is_multiple_of(2) {
            textures.planes()
        } else {
            textures.luma_chroma()
        };
        encoder.encode_textures(picture, vector.budget).unwrap();
    };
    for round in 0..4 {
        encode(round);
    }
    let before = live_resources(&context);
    for round in 0..24 {
        encode(round);
    }
    assert_eq!(
        live_resources(&context),
        before,
        "live GPU objects and memory"
    );
}
