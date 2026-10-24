//! Tests of the GPU decoder against the CPU oracle and the original decoder.
//!
//! They need a GPU or a software Vulkan driver (lavapipe). Without one they
//! skip, unless `EMBERWAVE_REQUIRE_GPU` is set, which turns a missing adapter
//! into a failure; CI sets it on the job that installs lavapipe.

#![cfg(feature = "gpu")]

mod common;

use common::{
    Reference, create_context, encoder_dir, encoder_vectors, frames_of, live_resources,
    max_abs_diff, psnr, read_reference, read_y4m, shared_context, testdata_dir, vector_frames,
    vector_names,
};
use emberwave_core::bitstream::consts::DECOMPOSITION_LEVELS;
use emberwave_core::bitstream::{
    Band, BlockHeader, ChromaSiting, ChromaSubsampling, Component, FrameAssembler, FrameLayout,
    YcbcrRange,
};
use emberwave_core::color::{ColorMatrix, ColorParams, yuv_to_rgb8};
use emberwave_core::cpu;
use emberwave_core::gpu::{Decoder, Frame, GpuContext, GpuError, GpuOptions, TimedOutput};

/// Skips the current test when there is no GPU.
macro_rules! gpu_or_return {
    () => {
        match shared_context() {
            Some(context) => context,
            None => return,
        }
    };
}

fn assembled(layout: &FrameLayout, frame: &[u8]) -> FrameAssembler {
    let mut assembler = FrameAssembler::new(layout.clone());
    assembler.push_packet(frame).expect("packet parses");
    assembler
}

fn decoder_for(context: &GpuContext, layout: &FrameLayout) -> Decoder {
    let start = std::time::Instant::now();
    let decoder =
        Decoder::new(context, layout.width(), layout.height(), layout.chroma()).expect("decoder");
    // The first decoder of a context compiles the shaders; later ones reuse them.
    eprintln!(
        "decoder for {}x{} created in {:?}",
        layout.width(),
        layout.height(),
        start.elapsed()
    );
    decoder
}

/// Compares the planes of a GPU frame with a reference frame, plane by plane.
fn assert_matches_reference(
    name: &str,
    index: usize,
    frame: &Frame,
    reference: &[Vec<u8>; 3],
    which: Reference,
    worst: &mut f64,
) {
    for (plane_index, (plane, expected)) in frame.planes.iter().zip(reference).enumerate() {
        let db = psnr(expected, &plane.data);
        println!(
            "{which:?} {name:28} frame {index} plane {plane_index}: PSNR {db:8.2} dB, max |diff| {}",
            max_abs_diff(expected, &plane.data)
        );
        *worst = worst.min(db);
    }
}

#[test]
fn the_encoders_streams_decode_as_upstreams_decoder_decodes_them() {
    // emberwave's encoder's vectors, against upstream's FP32 decode of them
    // (testdata/encoder/, decision D28).
    let context = gpu_or_return!();
    let mut worst = f64::INFINITY;
    for (name, _) in encoder_vectors() {
        let bytes = std::fs::read(encoder_dir().join(format!("{name}.pwv1"))).expect("stream");
        let (layout, frames) = frames_of(&name, &bytes);
        let reference = read_y4m(&encoder_dir().join(format!("{name}.ref-fp32.y4m")));
        let mut decoder = decoder_for(context, &layout);
        for (index, frame) in frames.iter().enumerate() {
            let decoded = decoder.decode(frame).expect("GPU decode");
            assert_matches_reference(
                &name,
                index,
                &decoded,
                &reference.frames[index],
                Reference::Fp32,
                &mut worst,
            );
        }
    }
    assert!(worst >= Reference::Fp32.min_psnr(), "{worst:.2} dB");
}

#[test]
fn dequantised_coefficients_are_bit_identical_to_the_cpu() {
    let context = gpu_or_return!();
    let mut compared = 0usize;

    for name in vector_names() {
        let (layout, frames) = vector_frames(&name);
        let mut decoder = decoder_for(context, &layout);

        for (index, frame) in frames.iter().enumerate() {
            decoder.decode(frame).expect("GPU decode");
            let assembler = assembled(&layout, frame);
            let bands = cpu::dequant::dequantize(&assembler.frame()).expect("CPU dequantisation");

            for (component, level, band, info) in layout.bands() {
                let expected = bands.plane(component, level, band).expect("CPU band");
                let actual = decoder
                    .read_band(component, level, band)
                    .expect("read back")
                    .expect("GPU band");
                assert_eq!(actual.len(), (info.width * info.height) as usize);

                for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        e.to_bits(),
                        "{name} frame {index} {component:?} level {level} {band:?} sample {i}: {a} vs {e}"
                    );
                }
                compared += actual.len();
            }
        }
    }
    assert!(compared > 10_000, "only {compared} coefficients compared");
}

#[test]
fn inverse_transform_matches_the_cpu_level_by_level() {
    let context = gpu_or_return!();
    // The GPU and the CPU evaluate the same expressions, but the GPU compiler
    // is free to fuse a multiply and an add, which moves a result by one unit
    // in the last place. Five levels of lifting stay far below this.
    const TOLERANCE: f32 = 1e-5;
    let mut worst = 0.0f32;

    for name in vector_names() {
        let (layout, frames) = vector_frames(&name);
        let mut decoder = decoder_for(context, &layout);

        for (index, frame) in frames.iter().enumerate() {
            decoder.decode(frame).expect("GPU decode");
            let assembler = assembled(&layout, frame);
            let (_, ll) = cpu::decode_frame_with_ll(&assembler.frame()).expect("CPU decode");

            for component in Component::ALL {
                // Level 4 comes straight from the bitstream; the others are
                // produced by the inverse transform of the level above.
                for level in (layout.finest_level(component)..DECOMPOSITION_LEVELS).rev() {
                    let expected = ll.get(component, level).expect("CPU LL band");
                    let actual = decoder
                        .read_band(component, level, Band::Ll)
                        .expect("read back")
                        .expect("GPU LL band");
                    assert_eq!(actual.len(), expected.len());

                    let diff = actual
                        .iter()
                        .zip(expected)
                        .map(|(a, e)| (a - e).abs())
                        .fold(0.0f32, f32::max);
                    worst = worst.max(diff);
                    assert!(
                        diff <= TOLERANCE,
                        "{name} frame {index} {component:?} level {level}: max difference {diff}"
                    );
                }
            }
        }
    }
    println!("largest difference to the CPU in any LL band: {worst:e}");
}

#[test]
fn gpu_decode_matches_the_reference_decoder() {
    let context = gpu_or_return!();

    for which in Reference::ALL {
        let mut worst = f64::INFINITY;
        for name in vector_names() {
            let reference = read_reference(&name, which);
            let (layout, frames) = vector_frames(&name);
            let mut decoder = decoder_for(context, &layout);

            for (index, frame) in frames.iter().enumerate() {
                let decoded = decoder.decode(frame).expect("GPU decode");
                assert_eq!(decoded.chroma, layout.chroma());
                assert_matches_reference(
                    &name,
                    index,
                    &decoded,
                    &reference.frames[index],
                    which,
                    &mut worst,
                );
            }
        }
        println!(
            "{which:?}: worst PSNR {worst:.2} dB (needs {})",
            which.min_psnr()
        );
        assert!(
            worst >= which.min_psnr(),
            "GPU decode disagrees with the {which:?} reference: {worst:.2} dB"
        );
    }
}

#[test]
fn a_decoder_can_be_reused_and_gives_the_same_frame_every_time() {
    let context = gpu_or_return!();

    // Frames of different content through one decoder, then the first again:
    // nothing may be left over from an earlier frame.
    let (layout, frames) = vector_frames("noise_192x136_420");
    let mut decoder = decoder_for(context, &layout);
    let first = decoder.decode(&frames[0]).unwrap();
    for frame in &frames[1..] {
        decoder.decode(frame).unwrap();
    }
    assert_eq!(decoder.decode(&frames[0]).unwrap(), first);
    assert_eq!(decoder.decode(&frames[0]).unwrap(), first);
}

#[test]
fn the_rows_handed_out_are_those_of_the_decoded_frame() {
    let context = gpu_or_return!();
    let color = ColorParams::default();

    // Rows whose width is not a multiple of four are padded to whole words
    // in the GPU's memory: in 4:4:4 at 131x67, and in the 98-sample chroma
    // rows of a 196x70 picture in 4:2:0.
    let mut cases: Vec<(String, FrameLayout, Vec<u8>)> =
        ["gradient_200x120_420", "noise_131x67_444"]
            .into_iter()
            .map(|name| {
                let (layout, frames) = vector_frames(name);
                (name.to_owned(), layout, frames[0].clone())
            })
            .collect();
    let layout = FrameLayout::new(196, 70, ChromaSubsampling::Yuv420).unwrap();
    let frame = synthetic_frame(&layout, 0x5EED_0196, false);
    cases.push(("synthetic 196x70 4:2:0".to_owned(), layout, frame));

    for (name, layout, bytes) in &cases {
        let mut decoder = decoder_for(context, layout);
        let frame = decoder.decode(bytes).expect("decode");
        let rgb = decoder.decode_rgb(bytes, &color).expect("RGB decode");

        // The frame was decoded already: forget it, or it would be ignored.
        decoder.clear();
        decoder.push_packet(bytes).expect("packet");
        let mut planes: [Vec<u8>; 3] = Default::default();
        let mut next = [0u32; 3];
        decoder
            .decode_assembled_rows(|component, row, samples| {
                let i = component.index();
                assert_eq!(row, next[i], "{name}: rows of plane {i} in order");
                assert_eq!(samples.len(), frame.planes[i].width as usize);
                next[i] += 1;
                planes[i].extend_from_slice(samples);
            })
            .expect("rows");
        // The frame takes its rows from the same place, so the CPU decoder
        // checks them as well: the GPU rounds as it does, within one level.
        let cpu_frame = cpu::decode_frame(&assembled(layout, bytes).frame()).unwrap();
        for (i, plane) in frame.planes.iter().enumerate() {
            assert_eq!(next[i], plane.height, "{name}: rows of plane {i}");
            assert_eq!(planes[i], plane.data, "{name}: plane {i}");
            let expected = cpu_frame.planes[i].to_unorm8();
            assert!(
                max_abs_diff(&expected, &planes[i]) <= 1,
                "{name}: plane {i}"
            );
        }

        decoder.clear();
        decoder.push_packet(bytes).expect("packet");
        let mut pixels = Vec::new();
        let mut next = 0;
        decoder
            .decode_assembled_rgb_rows(&color, |row, line| {
                assert_eq!(row, next, "{name}: RGB rows in order");
                assert_eq!(line.len(), 3 * rgb.width as usize);
                next += 1;
                pixels.extend_from_slice(line);
            })
            .expect("RGB rows");
        assert_eq!(next, rgb.height, "{name}: RGB rows");
        assert_eq!(pixels, rgb.data, "{name}: RGB");
    }
}

/// The pixels of a 2D texture of 4-byte texels, row after row, read back
/// through a buffer whose rows are padded to 256 bytes as copies need.
fn read_texture(context: &GpuContext, texture: &wgpu::Texture) -> Vec<u8> {
    let (width, height) = (texture.width(), texture.height());
    let row = width * 4;
    let padded =
        row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let device = context.device();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("texture readback"),
        size: u64::from(padded * height),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        texture.size(),
    );
    context.queue().submit([encoder.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |result| result.expect("map"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let mapped = slice.get_mapped_range().expect("mapped range");
    mapped
        .chunks(padded as usize)
        .flat_map(|line| line[..row as usize].to_vec())
        .collect()
}

/// RGB pixels as RGBA, alpha 255, or as BGRA with `bgra`.
fn with_alpha(rgb: &[u8], bgra: bool) -> Vec<u8> {
    rgb.chunks_exact(3)
        .flat_map(|p| {
            if bgra {
                [p[2], p[1], p[0], 255]
            } else {
                [p[0], p[1], p[2], 255]
            }
        })
        .collect()
}

#[test]
fn a_picture_drawn_into_a_texture_is_the_rgb_picture() {
    let context = gpu_or_return!();
    let color = ColorParams::default();
    for name in ["gradient_200x120_420", "noise_131x67_444"] {
        let (layout, frames) = vector_frames(name);
        let mut decoder = decoder_for(context, &layout);

        // Into a texture of the decoder's, waiting for the GPU.
        let rgb = decoder.decode_rgb(&frames[0], &color).expect("RGB decode");
        let texture = decoder
            .create_texture(wgpu::TextureFormat::Rgba8Unorm)
            .expect("texture");
        let view = texture.create_view(&Default::default());
        decoder.clear();
        decoder.push_packet(&frames[0]).expect("packet");
        decoder
            .decode_assembled_to_texture(&color, &view)
            .expect("decode into a texture");
        assert_eq!(
            read_texture(context, &texture),
            with_alpha(&rgb.data, false),
            "{name}: RGBA"
        );

        // Into a BGRA texture of the caller's, recorded into its encoder.
        let rgb = decoder.decode_rgb(&frames[1], &color).expect("RGB decode");
        let bgra = context.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("caller's texture"),
            size: texture.size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let mut encoder = context.device().create_command_encoder(&Default::default());
        decoder
            .encode_rgb_to_texture(
                &frames[1],
                &color,
                &bgra.create_view(&Default::default()),
                &mut encoder,
            )
            .expect("recorded decode into a texture");
        context.queue().submit([encoder.finish()]);
        assert_eq!(
            read_texture(context, &bgra),
            with_alpha(&rgb.data, true),
            "{name}: BGRA"
        );
    }
}

#[test]
fn only_a_texture_the_picture_fits_takes_it() {
    let context = gpu_or_return!();
    let color = ColorParams::default();
    let (layout, frames) = vector_frames("gradient_200x120_420");
    let mut decoder = decoder_for(context, &layout);
    let frame = decoder.decode(&frames[0]).expect("decode");
    let make_with = |width, height, format, usage, mips, layers| {
        context.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("unsuitable"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: layers,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let make = |width, height, format, usage| make_with(width, height, format, usage, 1, 1);
    let target = wgpu::TextureUsages::RENDER_ATTACHMENT;
    let rgba = wgpu::TextureFormat::Rgba8Unorm;
    let unsuitable = [
        // Views of these could show a part of them.
        make_with(200, 120, rgba, target, 2, 1),
        make_with(200, 120, rgba, target, 1, 2),
        make(199, 120, wgpu::TextureFormat::Rgba8Unorm, target),
        make(200, 120, wgpu::TextureFormat::R8Unorm, target),
        make(200, 120, wgpu::TextureFormat::Rgba8UnormSrgb, target),
        make(
            200,
            120,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING,
        ),
    ];
    for texture in &unsuitable {
        let view = texture.create_view(&Default::default());
        decoder.clear();
        decoder.push_packet(&frames[0]).expect("packet");
        let outcome = decoder.decode_assembled_to_texture(&color, &view);
        assert!(matches!(outcome, Err(GpuError::Texture(_))), "{outcome:?}");
        // The frame is still there to decode.
        assert_eq!(decoder.decode_assembled().expect("decode"), frame);
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let outcome = decoder.encode_rgb_to_texture(&frames[0], &color, &view, &mut encoder);
        assert!(matches!(outcome, Err(GpuError::Texture(_))), "{outcome:?}");
    }
}

#[test]
fn a_panic_while_rows_are_handed_out_leaves_the_decoder_usable() {
    let context = gpu_or_return!();
    let (layout, frames) = vector_frames("gradient_200x120_420");
    let mut decoder = decoder_for(context, &layout);
    let frame = decoder.decode(&frames[0]).expect("decode");
    let color = ColorParams::default();
    let rgb = decoder.decode_rgb(&frames[0], &color).expect("RGB decode");

    // The read-back buffers are mapped while the rows are handed out. A
    // buffer left mapped would make the next decode fail.
    for round in 0..2 {
        decoder.clear();
        decoder.push_packet(&frames[0]).expect("packet");
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if round == 0 {
                decoder.decode_assembled_rows(|_, _, _| panic!("a row that cannot be taken"))
            } else {
                decoder
                    .decode_assembled_rgb_rows(&color, |_, _| panic!("a row that cannot be taken"))
            }
        }));
        assert!(outcome.is_err(), "the panic goes on to the caller");
        assert_eq!(decoder.decode(&frames[0]).expect("decode"), frame);
        assert_eq!(
            decoder.decode_rgb(&frames[0], &color).expect("RGB decode"),
            rgb
        );
    }
}

#[test]
fn an_empty_frame_decodes_to_mid_grey() {
    let context = gpu_or_return!();
    let layout = FrameLayout::new(160, 96, ChromaSubsampling::Yuv420).unwrap();
    let mut decoder = decoder_for(context, &layout);

    // Only a start of frame header announcing no blocks. Every coefficient is
    // zero, so every sample is exactly 0.5, which is 127.5 in 8 bits: a tie,
    // and ties go to the even neighbour.
    let word0 = 159u32 | (95 << 14) | (1 << 28) | (1 << 31);
    let header = [word0.to_le_bytes(), 0u32.to_le_bytes()].concat();
    let frame = decoder.decode(&header).unwrap();

    assert_eq!(frame.planes[0].data.len(), 160 * 96);
    assert_eq!(frame.planes[1].data.len(), 80 * 48);
    for plane in &frame.planes {
        assert!(plane.data.iter().all(|&v| v == 128), "mid grey expected");
    }

    let assembler = assembled(&layout, &header);
    let cpu_frame = cpu::decode_frame(&assembler.frame()).unwrap();
    assert_eq!(cpu_frame.planes[0].to_unorm8(), frame.planes[0].data);
}

#[test]
fn an_incomplete_frame_is_an_error_until_it_is_allowed_to_be_partial() {
    let context = gpu_or_return!();
    let (layout, frames) = vector_frames("gradient_200x120_420");
    let mut decoder = decoder_for(context, &layout);

    // Cut the frame short: some blocks are missing.
    let cut = &frames[0][..frames[0].len() / 2];
    // The cut lands inside a block, so drop the partial one by parsing block
    // headers until the cut.
    let mut end = 8; // start of frame header
    while end + 8 <= cut.len() {
        let words = usize::from(u16::from_le_bytes([cut[end + 2], cut[end + 3]]) & 0xfff);
        if end + words * 4 > cut.len() {
            break;
        }
        end += words * 4;
    }

    let error = decoder.decode(&cut[..end]).unwrap_err();
    assert!(
        matches!(error, emberwave_core::gpu::GpuError::IncompleteFrame { .. }),
        "{error}"
    );

    // As a partial frame it decodes, with the missing blocks as zeros.
    decoder.push_packet(&cut[..end]).unwrap();
    assert!(!decoder.is_ready(false));
    let frame = decoder.decode_assembled().unwrap();
    assert_eq!(frame.planes[0].data.len(), 200 * 120);

    // It is what the CPU decoder makes of the same blocks, and it is not the
    // complete picture.
    let assembler = assembled(&layout, &cut[..end]);
    let expected = cpu::decode_frame(&assembler.frame()).unwrap();
    for (plane, cpu_plane) in frame.planes.iter().zip(&expected.planes) {
        assert!(max_abs_diff(&cpu_plane.to_unorm8(), &plane.data) <= 1);
    }
    let complete = decoder.decode(&frames[0]).unwrap();
    assert_ne!(frame, complete, "the missing blocks must make a difference");
}

#[test]
fn a_device_limited_to_webgpu_guarantees_decodes_the_same() {
    // The portable subset: default limits and no optional features. If the
    // shaders needed anything beyond it, creating the pipelines or running
    // them would fail here.
    let options = GpuOptions {
        strict_webgpu: true,
        ..GpuOptions::default()
    };
    let Some(context) = create_context(&options) else {
        return;
    };

    let (layout, frames) = vector_frames("noise_131x67_444");
    let reference = read_reference("noise_131x67_444", Reference::Fp32);
    let mut decoder = decoder_for(&context, &layout);
    let mut worst = f64::INFINITY;
    for (index, frame) in frames.iter().enumerate() {
        let decoded = decoder.decode(frame).unwrap();
        assert_matches_reference(
            "noise_131x67_444",
            index,
            &decoded,
            &reference.frames[index],
            Reference::Fp32,
            &mut worst,
        );
    }
    assert!(worst >= Reference::Fp32.min_psnr(), "{worst:.2} dB");
}

/// A small xorshift generator, so synthetic frames are reproducible.
struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
}

/// A valid block with random content: random ballot, tables, bit-planes and
/// signs. Not something an encoder would write, but structurally sound, so
/// the CPU and GPU decoders have to agree on it.
fn synthetic_block(rng: &mut Rng, block_index: u32, quant_code: u8, sequence: u8) -> Vec<u8> {
    let ballot: u16 = match rng.next() % 8 {
        0 => 0,
        1 => 0xFFFF,
        _ => rng.next() as u16,
    };
    let active = ballot.count_ones() as usize;

    let mut controls = Vec::new();
    let mut qscales = Vec::new();
    let mut planes = Vec::new();
    for _ in 0..active {
        let base_planes = (rng.next() % 6) as u8;
        // Some blocks have an all-zero control word, which decodes to nothing.
        let control: u16 = if rng.next().is_multiple_of(6) {
            0
        } else {
            rng.next() as u16
        };
        let scale_code = (rng.next() % 16) as u8;
        controls.extend_from_slice(&control.to_le_bytes());
        qscales.push((scale_code << 4) | base_planes);

        let extra: usize = (0..8).map(|s| usize::from((control >> (2 * s)) & 3)).sum();
        for _ in 0..extra + 8 * usize::from(base_planes) {
            // Plenty of zero bytes keep the coefficients sparse.
            planes.push(if rng.next().is_multiple_of(3) {
                0
            } else {
                rng.next() as u8
            });
        }
    }

    // More sign bytes than any block can use: the decoders only read what the
    // magnitudes call for.
    let signs: Vec<u8> = (0..active * 8).map(|_| rng.next() as u8).collect();

    let mut body = Vec::new();
    body.extend_from_slice(&controls);
    body.extend_from_slice(&qscales);
    body.extend_from_slice(&planes);
    body.extend_from_slice(&signs);
    while (8 + body.len()) % 4 != 0 {
        body.push(0);
    }

    let words = (8 + body.len()) / 4;
    assert!(words < 4096, "block does not fit the 12-bit size field");
    let word0 = u32::from(ballot) | ((words as u32) << 16) | (u32::from(sequence) << 28);
    let word1 = u32::from(quant_code) | (block_index << 8);
    [word0.to_le_bytes().as_slice(), &word1.to_le_bytes(), &body].concat()
}

/// A whole synthetic frame: every block index gets a block, except that about
/// one in ten is left out if `drop_blocks` is set. The quantiser code cycles
/// through all 256 values by block index.
fn synthetic_frame(layout: &FrameLayout, seed: u32, drop_blocks: bool) -> Vec<u8> {
    let mut rng = Rng(seed);
    let mut blocks = Vec::new();
    let mut included = 0u32;
    for block_index in 0..layout.block_count() {
        // The random number is drawn either way so that the content of the
        // blocks does not depend on `drop_blocks`.
        if rng.next().is_multiple_of(10) && drop_blocks {
            continue; // missing block
        }
        let quant_code = (block_index % 256) as u8;
        blocks.extend(synthetic_block(&mut rng, block_index, quant_code, 1));
        included += 1;
    }

    let word0 = (layout.width() - 1) | ((layout.height() - 1) << 14) | (1 << 28) | (1 << 31);
    let chroma_bit = u32::from(layout.chroma() == ChromaSubsampling::Yuv444) << 26;
    let word1 = included | chroma_bit;
    [
        word0.to_le_bytes().as_slice(),
        &word1.to_le_bytes(),
        &blocks,
    ]
    .concat()
}

#[test]
fn synthetic_frames_dequantise_identically_on_gpu_and_cpu() {
    let context = gpu_or_return!();

    // 512x288 has 285 blocks, more than the 256 quantiser codes. The first frame
    // keeps every block, so every code occurs; the others drop some blocks to
    // exercise missing ones.
    let mut codes_seen = [false; 256];
    for (width, height, chroma, seed, drop_blocks) in [
        (512, 288, ChromaSubsampling::Yuv420, 0x1234_5678, false),
        (512, 288, ChromaSubsampling::Yuv420, 0x9E37_79B9, true),
        (130, 66, ChromaSubsampling::Yuv444, 0x0BAD_CAFE, true),
    ] {
        let layout = FrameLayout::new(width, height, chroma).unwrap();
        let frame = synthetic_frame(&layout, seed, drop_blocks);
        let mut decoder = decoder_for(context, &layout);
        decoder.decode(&frame).expect("GPU decode");

        let assembler = assembled(&layout, &frame);
        assert_eq!(
            assembler.received_blocks() == layout.block_count(),
            !drop_blocks,
            "{width}x{height} seed {seed:#x}"
        );
        for block_index in 0..layout.block_count() {
            if let Some(words) = assembler.frame().block(block_index) {
                codes_seen[usize::from(BlockHeader::from_words(words[0], words[1]).quant_code)] =
                    true;
            }
        }

        let bands = cpu::dequant::dequantize(&assembler.frame()).expect("CPU dequantisation");
        let mut nonzero = 0usize;
        for (component, level, band, _) in layout.bands() {
            let expected = bands.plane(component, level, band).unwrap();
            let actual = decoder.read_band(component, level, band).unwrap().unwrap();
            for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    e.to_bits(),
                    "{width}x{height} seed {seed:#x} {component:?} level {level} {band:?} sample {i}"
                );
                nonzero += usize::from(*e != 0.0);
            }
        }
        assert!(
            nonzero > 1000,
            "the synthetic frame is too empty ({nonzero})"
        );
    }

    let missing: Vec<usize> = (0..256).filter(|&code| !codes_seen[code]).collect();
    assert!(
        missing.is_empty(),
        "quantiser codes never decoded: {missing:?}"
    );
}

#[test]
fn a_large_payload_grows_the_buffer_and_still_decodes() {
    let context = gpu_or_return!();
    let layout = FrameLayout::new(512, 288, ChromaSubsampling::Yuv420).unwrap();
    let frame = synthetic_frame(&layout, 0x5EED_0001, true);
    assert!(
        frame.len() > 64 * 1024,
        "needs more than the initial buffer: {}",
        frame.len()
    );

    let mut decoder = decoder_for(context, &layout);
    // An empty frame first, then the large one, then the empty one again: the
    // payload buffer grows for the second and nothing may linger afterwards.
    // Correctness of the large frame itself is covered by the bit-exact
    // comparison of synthetic frames above, which are this large too.
    let word0 = 511u32 | (287 << 14) | (1 << 28) | (1 << 31);
    let empty = [word0.to_le_bytes(), 0u32.to_le_bytes()].concat();
    let first = decoder.decode(&empty).unwrap();
    let large = decoder.decode(&frame).unwrap();
    assert_ne!(first, large);
    assert_eq!(decoder.decode(&empty).unwrap(), first);
    assert_eq!(decoder.decode(&frame).unwrap(), large);
}

/// How a GPU RGB picture compares with the CPU conversion of the same planes.
struct RgbComparison {
    differing: usize,
    largest: u8,
    samples: usize,
}

fn compare_rgb(gpu: &[u8], cpu: &[u8]) -> RgbComparison {
    assert_eq!(gpu.len(), cpu.len());
    RgbComparison {
        differing: gpu.iter().zip(cpu).filter(|(a, b)| a != b).count(),
        largest: gpu
            .iter()
            .zip(cpu)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0),
        samples: gpu.len(),
    }
}

#[test]
fn rgb_conversion_on_the_gpu_matches_the_cpu() {
    let context = gpu_or_return!();

    // Every vector (4:2:0 with odd chroma sizes, 4:4:4 with odd sizes) with
    // every matrix, both ranges and both sitings.
    let matrices = [ColorMatrix::Bt601, ColorMatrix::Bt709, ColorMatrix::Bt2020];
    let ranges = [YcbcrRange::Limited, YcbcrRange::Full];
    let sitings = [ChromaSiting::Center, ChromaSiting::Left];

    let mut worst_fraction = 0.0f64;
    for name in vector_names() {
        let (layout, frames) = vector_frames(&name);
        let mut decoder = decoder_for(context, &layout);

        // The YUV planes the GPU produces are what the conversion starts from.
        let yuv = decoder.decode(&frames[0]).expect("GPU decode");
        let planes = [
            &yuv.planes[0].data,
            &yuv.planes[1].data,
            &yuv.planes[2].data,
        ];

        for matrix in matrices {
            for range in ranges {
                for chroma_siting in sitings {
                    let params = ColorParams {
                        matrix,
                        range,
                        chroma_siting,
                    };
                    let rgb = decoder
                        .decode_rgb(&frames[0], &params)
                        .expect("GPU RGB decode");
                    assert_eq!((rgb.width, rgb.height), (layout.width(), layout.height()));
                    assert_eq!(
                        rgb.data.len(),
                        (layout.width() * layout.height() * 3) as usize
                    );

                    let expected = yuv_to_rgb8(
                        [planes[0], planes[1], planes[2]],
                        layout.width(),
                        layout.height(),
                        layout.chroma(),
                        &params,
                    )
                    .expect("the decoder's planes match its layout");
                    let c = compare_rgb(&rgb.data, &expected);
                    // The same arithmetic in the same order; the GPU may fuse a
                    // multiply and an add, which moves a rounding by one.
                    assert!(
                        c.largest <= 1,
                        "{name} {params:?}: largest difference {}",
                        c.largest
                    );
                    let fraction = c.differing as f64 / c.samples as f64;
                    assert!(
                        fraction < 0.01,
                        "{name} {params:?}: {} of {} samples differ",
                        c.differing,
                        c.samples
                    );
                    worst_fraction = worst_fraction.max(fraction);
                }
            }
        }
    }
    println!("largest share of samples that differ from the CPU conversion: {worst_fraction:.5}");
}

#[test]
fn rgb_output_changes_with_the_conversion_settings() {
    let context = gpu_or_return!();
    let (layout, frames) = vector_frames("gradient_200x120_420");
    let mut decoder = decoder_for(context, &layout);

    // The parameters are read for every call: the same frame converted twice
    // with different settings must differ, and back again must repeat.
    let a = ColorParams::default();
    let b = ColorParams {
        matrix: ColorMatrix::Bt601,
        range: YcbcrRange::Full,
        chroma_siting: ChromaSiting::Left,
    };
    let first = decoder.decode_rgb(&frames[0], &a).unwrap();
    let other = decoder.decode_rgb(&frames[0], &b).unwrap();
    assert_ne!(first, other);
    assert_eq!(decoder.decode_rgb(&frames[0], &a).unwrap(), first);

    // The YUV path still works after the RGB buffers exist.
    let yuv = decoder.decode(&frames[0]).unwrap();
    assert_eq!(yuv.planes[0].data.len(), 200 * 120);
}

#[test]
fn timing_a_frame_reports_where_the_time_goes() {
    // Timestamp queries are optional, so ask for them and accept their absence.
    let options = GpuOptions {
        timestamps: true,
        ..GpuOptions::default()
    };
    let Some(context) = create_context(&options) else {
        return;
    };
    let (layout, frames) = vector_frames("noise_192x136_420");
    let mut decoder = decoder_for(&context, &layout);

    for output in [
        emberwave_core::gpu::TimedOutput::Yuv,
        emberwave_core::gpu::TimedOutput::Rgb(ColorParams::default()),
    ] {
        let timings = decoder.time_frame(&frames[0], output).unwrap();
        assert!(timings.gpu_wall > std::time::Duration::ZERO);
        assert!(timings.prepare > std::time::Duration::ZERO);

        match timings.stages {
            Some(stages) => {
                assert!(context.supports_timestamps());
                // A short stage can be below the timer's resolution and read as
                // zero, but the whole decode cannot.
                assert!(stages.total() > std::time::Duration::ZERO);
                let rgb = matches!(output, emberwave_core::gpu::TimedOutput::Rgb(_));
                assert_eq!(stages.yuv2rgb.is_some(), rgb);
                // The GPU cannot take longer than the wall clock around it.
                assert!(stages.total() <= timings.gpu_wall + timings.prepare);
            }
            // A device that offers timestamps has to deliver them: "no stage
            // times" must not hide a resolve that never landed or a pass whose
            // end came before its beginning.
            None => assert!(
                !context.supports_timestamps(),
                "the device supports timestamp queries but returned no stage times"
            ),
        }
    }
}

/// Decodes `frame` in the way `round` selects, so that a loop goes through
/// all of them: planes, RGB, and both timed variants.
fn decode_one_way(decoder: &mut Decoder, frame: &[u8], round: usize) {
    let color = ColorParams::default();
    match round % 4 {
        0 => drop(decoder.decode(frame).expect("decode")),
        1 => drop(decoder.decode_rgb(frame, &color).expect("RGB decode")),
        2 => drop(
            decoder
                .time_frame(frame, TimedOutput::Yuv)
                .expect("timed decode"),
        ),
        _ => drop(
            decoder
                .time_frame(frame, TimedOutput::Rgb(color))
                .expect("timed RGB decode"),
        ),
    }
}

/// Decodes `count` frames, cycling through `frames` and through the ways of
/// decoding, and returns the live resources before and after the loop.
fn resources_around(
    context: &GpuContext,
    decoder: &mut Decoder,
    frames: &[Vec<u8>],
    count: usize,
) -> (String, String) {
    // First every frame once and every way of decoding once: that creates
    // everything a decoder ever needs, including the payload buffer at the
    // size of the largest frame (it grows, and stays grown).
    for round in 0..frames.len().max(4) {
        decode_one_way(decoder, &frames[round % frames.len()], round);
    }
    assert!(
        context.device().get_internal_counters().hal.buffers.read() > 0,
        "wgpu's counters read zero: are they enabled for the tests?"
    );
    let before = live_resources(context);
    for (round, frame) in frames.iter().cycle().take(count).enumerate() {
        decode_one_way(decoder, frame, round);
    }
    (before, live_resources(context))
}

#[test]
fn decoding_frame_after_frame_allocates_nothing() {
    // A context of its own: the counters belong to the device, and the shared
    // context is busy with the other tests.
    let options = GpuOptions {
        timestamps: true,
        ..GpuOptions::default()
    };
    let Some(context) = create_context(&options) else {
        return;
    };
    let (layout, frames) = vector_frames("noise_192x136_420");
    let mut decoder = decoder_for(&context, &layout);

    let (before, after) = resources_around(&context, &mut decoder, &frames, 48);
    assert_eq!(after, before, "live GPU objects and memory");
}

#[test]
fn a_decoder_makes_its_read_back_buffers_when_a_decode_first_reads_back() {
    // A context of its own: the counters belong to the device.
    let Some(context) = create_context(&GpuOptions::default()) else {
        return;
    };
    let buffers = || context.device().get_internal_counters().hal.buffers.read();
    let (layout, frames) = vector_frames("gradient_200x120_420");
    let mut decoder = decoder_for(&context, &layout);
    let color = ColorParams::default();

    // What a player does: record into its own encoder, or wait without
    // reading back. The 4K decoder of a browser saves 43.5 MiB this way.
    let mut encoder = context.device().create_command_encoder(&Default::default());
    decoder
        .encode_rgb(&frames[0], &color, &mut encoder)
        .expect("recorded decode");
    context.queue().submit([encoder.finish()]);
    decoder
        .time_frame(&frames[1], TimedOutput::Rgb(color))
        .expect("timed decode");
    let player = buffers();
    assert!(player > 0, "wgpu's counters read zero: are they enabled?");

    decoder.decode(&frames[0]).expect("decode");
    assert_eq!(buffers(), player + 1, "the planes' read-back buffer");
    decoder.decode_rgb(&frames[0], &color).expect("RGB decode");
    assert_eq!(buffers(), player + 2, "and the RGB one");
    decoder.decode(&frames[1]).expect("decode");
    decoder.decode_rgb(&frames[1], &color).expect("RGB decode");
    assert_eq!(buffers(), player + 2, "each made once");
}

#[test]
fn a_lost_device_is_an_error_rather_than_a_stale_result() {
    // A context of its own, since destroying the device loses it for good.
    let Some(context) = create_context(&GpuOptions::default()) else {
        return;
    };
    let (layout, frames) = vector_frames("gradient_200x120_420");
    let mut decoder = decoder_for(&context, &layout);
    decoder
        .decode(&frames[0])
        .expect("decode on a working device");
    assert_eq!(context.lost(), None);

    context.device().destroy();
    let color = ColorParams::default();
    let results = [
        decoder.decode(&frames[0]).map(drop),
        decoder.decode_rgb(&frames[0], &color).map(drop),
        decoder.time_frame(&frames[0], TimedOutput::Yuv).map(drop),
        decoder
            .time_frame(&frames[0], TimedOutput::Rgb(color))
            .map(drop),
    ];
    for result in &results {
        assert!(
            matches!(result, Err(GpuError::DeviceLost(_))),
            "{results:?}"
        );
    }
    assert!(context.lost().is_some());

    // The row methods fail alike, without a row. The loss is on record by
    // now, so they fail before any work; a loss found in the wait for the
    // frame is the unit test `no_row_is_handed_out_once_the_device_is_lost`.
    let mut rows = 0;
    decoder.clear();
    decoder.push_packet(&frames[0]).expect("packet");
    let outcome = decoder.decode_assembled_rows(|_, _, _| rows += 1);
    assert!(
        matches!(outcome, Err(GpuError::DeviceLost(_))),
        "{outcome:?}"
    );
    let outcome = decoder.decode_assembled_rgb_rows(&color, |_, _| rows += 1);
    assert!(
        matches!(outcome, Err(GpuError::DeviceLost(_))),
        "{outcome:?}"
    );
    assert_eq!(rows, 0);
}

/// The leak check of milestone M5: 1000 frames of a 4K stream, decoded in
/// every way in turn, leave the GPU resources as they were after the first
/// frames. It needs the streams of `scripts/gen_bench_data.sh` and takes a
/// while, so it only runs on request:
/// `cargo test --release -p emberwave-core --test gpu -- --ignored --nocapture`.
/// `scripts/soak.sh` watches the memory of the process for the same purpose.
#[test]
#[ignore = "decodes 1000 4K frames; needs scripts/gen_bench_data.sh"]
fn a_thousand_4k_frames_leave_no_trace() {
    const FRAMES: usize = 1000;
    let path = testdata_dir().join("generated/scene_4k_420.pwv1");
    let bytes = std::fs::read(&path).unwrap_or_else(|error| {
        panic!("{}: {error}; run scripts/gen_bench_data.sh", path.display())
    });
    let options = GpuOptions {
        timestamps: true,
        ..GpuOptions::default()
    };
    let context = create_context(&options).expect("this test needs a GPU");
    let (layout, frames) = frames_of("scene_4k_420", &bytes);
    let mut decoder = decoder_for(&context, &layout);

    let start = std::time::Instant::now();
    let (before, after) = resources_around(&context, &mut decoder, &frames, FRAMES);
    eprintln!(
        "{FRAMES} 4K frames in {:.1?}\nbefore: {before}\nafter:  {after}",
        start.elapsed()
    );
    assert_eq!(after, before);
}

#[test]
fn encoding_into_a_callers_encoder_gives_the_same_picture() {
    let context = gpu_or_return!();
    let (layout, frames) = vector_frames("gradient_200x120_420");
    let color = ColorParams::default();

    let mut decoder = decoder_for(context, &layout);
    let expected = decoder.decode_rgb(&frames[1], &color).unwrap();

    // The same frame recorded into an encoder of our own, then the RGBA words
    // copied out the way a caller would.
    let mut other = decoder_for(context, &layout);
    assert!(other.rgba_buffer().is_none());
    let device = context.device();
    let mut encoder = device.create_command_encoder(&Default::default());
    other.encode_rgb(&frames[1], &color, &mut encoder).unwrap();
    let rgba = other.rgba_buffer().expect("RGBA buffer after encode_rgb");
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test readback"),
        size: rgba.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(rgba, 0, &readback, 0, rgba.size());
    context.queue().submit([encoder.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |result| result.expect("map"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let words = slice.get_mapped_range().expect("mapped").to_vec();

    let rgb: Vec<u8> = words
        .chunks_exact(4)
        .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
        .collect();
    assert!(words.chunks_exact(4).all(|pixel| pixel[3] == 255));
    assert_eq!(rgb, expected.data);

    // The same decoder, back to the decodes that wait, which upload through a
    // buffer of their own (decision D34) where encode_rgb writes through the
    // queue: each path finds the payload the other left and replaces it.
    for frame in [&frames[0], &frames[1]] {
        let from_both = other.decode_rgb(frame, &color).unwrap();
        assert_eq!(
            from_both.data,
            decoder.decode_rgb(frame, &color).unwrap().data
        );
    }
}
