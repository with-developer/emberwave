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
use emberwave_core::bitstream::header::Header;
use emberwave_core::bitstream::{
    Band, BlockHeader, ChromaSiting, ChromaSubsampling, Component, FrameAssembler, FrameLayout,
    YcbcrRange,
};
use emberwave_core::color::{ColorMatrix, ColorParams, yuv_to_rgb8};
use emberwave_core::cpu;
use emberwave_core::gpu::{Decoder, Frame, GpuContext, GpuError, TEXTURE_FORMATS, TimedOutput};

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

/// The texels of a 2D texture, row after row, read back through a buffer
/// whose rows are padded to 256 bytes as copies need.
fn read_texture(context: &GpuContext, texture: &wgpu::Texture) -> Vec<u8> {
    let (width, height) = (texture.width(), texture.height());
    let texel = texture.format().block_copy_size(None).expect("texel size");
    let row = width * texel;
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
    // 192 pixels make rows of 768 bytes, which a copy takes (decision D42);
    // the others are drawn whatever the texture.
    for name in [
        "noise_192x136_420",
        "gradient_200x120_420",
        "noise_131x67_444",
    ] {
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

        // Into textures of the caller's, recorded into its encoder: RGBA and
        // BGRA, with the RENDER_ATTACHMENT usage that a draw needs, the
        // COPY_DST usage that lets a copy fill them, or both. The RGB decode
        // in between writes RGBA words again.
        let (draw, copy) = (
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            wgpu::TextureUsages::COPY_DST,
        );
        let copies = (layout.width() * 4).is_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        for format in TEXTURE_FORMATS {
            for usage in [draw, draw | copy, copy] {
                let rgb = decoder.decode_rgb(&frames[1], &color).expect("RGB decode");
                let target = context.device().create_texture(&wgpu::TextureDescriptor {
                    label: Some("caller's texture"),
                    size: texture.size(),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::COPY_SRC | usage,
                    view_formats: &[],
                });
                let mut encoder = context.device().create_command_encoder(&Default::default());
                let outcome = decoder.record_rgb_to_texture(
                    &frames[1],
                    &color,
                    &target.create_view(&Default::default()),
                    &mut encoder,
                );
                if usage == copy && !copies {
                    // Neither a copy nor a draw can fill it (decision D44).
                    assert!(matches!(outcome, Err(GpuError::Texture(_))), "{outcome:?}");
                    continue;
                }
                outcome.expect("recorded decode into a texture");
                context.queue().submit([encoder.finish()]);
                let bgra = format == wgpu::TextureFormat::Bgra8Unorm;
                assert_eq!(
                    read_texture(context, &target),
                    with_alpha(&rgb.data, bgra),
                    "{name}: {format:?} with {usage:?}"
                );
            }
        }
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
        let outcome = decoder.record_rgb_to_texture(&frames[0], &color, &view, &mut encoder);
        assert!(matches!(outcome, Err(GpuError::Texture(_))), "{outcome:?}");
    }
}

#[test]
fn planes_drawn_into_textures_are_the_decoded_planes() {
    let context = gpu_or_return!();
    // 4:2:0 with half-size chroma, and 4:4:4 of an odd size, whose rows end
    // inside a word of the packed planes.
    for name in ["gradient_200x120_420", "noise_131x67_444"] {
        let (layout, frames) = vector_frames(name);
        let mut decoder = decoder_for(context, &layout);

        // Into textures of the decoder's, waiting for the GPU.
        let frame = decoder.decode(&frames[0]).expect("decode");
        let textures =
            Component::ALL.map(|component| decoder.create_plane_texture(component).expect("plane"));
        let views = textures
            .each_ref()
            .map(|texture| texture.create_view(&Default::default()));
        decoder.clear();
        decoder.push_packet(&frames[0]).expect("packet");
        decoder
            .decode_assembled_to_plane_textures(views.each_ref())
            .expect("decode into plane textures");
        for (component, texture) in Component::ALL.into_iter().zip(&textures) {
            let plane = &frame.planes[component.index()];
            assert_eq!(
                (texture.width(), texture.height()),
                (plane.width, plane.height),
                "{name}: {component:?} size"
            );
            assert_eq!(
                read_texture(context, texture),
                plane.data,
                "{name}: {component:?}"
            );
        }

        // Into textures of the caller's, recorded into its encoder.
        let frame = decoder.decode(&frames[1]).expect("decode");
        let textures = Component::ALL.map(|component| {
            let (width, height) = layout.plane_size(component);
            context.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("caller's plane"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: emberwave_core::gpu::PLANE_FORMAT,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        });
        let views = textures
            .each_ref()
            .map(|texture| texture.create_view(&Default::default()));
        let mut encoder = context.device().create_command_encoder(&Default::default());
        decoder
            .record_to_plane_textures(&frames[1], views.each_ref(), &mut encoder)
            .expect("recorded decode into plane textures");
        context.queue().submit([encoder.finish()]);
        for (component, texture) in Component::ALL.into_iter().zip(&textures) {
            assert_eq!(
                read_texture(context, texture),
                frame.planes[component.index()].data,
                "{name}: {component:?}, recorded"
            );
        }
    }
}

#[test]
fn decodes_submitted_without_waiting_reach_their_textures() {
    let context = gpu_or_return!();
    let color = ColorParams::default();
    // A picture copied and one drawn into its texture (decision D42).
    for name in ["noise_192x136_420", "gradient_200x120_420"] {
        let (layout, frames) = vector_frames(name);
        let mut decoder = decoder_for(context, &layout);
        let texture = decoder
            .create_texture(wgpu::TextureFormat::Bgra8Unorm)
            .expect("texture");
        let view = texture.create_view(&Default::default());
        let planes =
            Component::ALL.map(|component| decoder.create_plane_texture(component).expect("plane"));
        let plane_views = planes
            .each_ref()
            .map(|texture| texture.create_view(&Default::default()));

        // Frame after frame into the same textures: the copy that reads one
        // back is submitted after the decode, and the next decode after it.
        for frame in &frames {
            let rgb = decoder.decode_rgb(frame, &color).expect("RGB decode");
            let yuv = decoder.decode(frame).expect("decode");
            decoder.clear();
            decoder.push_packet(frame).expect("packet");
            decoder
                .submit_assembled_to_texture(&color, &view)
                .expect("submitted decode");
            assert_eq!(
                read_texture(context, &texture),
                with_alpha(&rgb.data, true),
                "{name}: BGRA"
            );
            decoder.clear();
            decoder.push_packet(frame).expect("packet");
            decoder
                .submit_assembled_to_plane_textures(plane_views.each_ref())
                .expect("submitted decode into planes");
            for (component, texture) in Component::ALL.into_iter().zip(&planes) {
                assert_eq!(
                    read_texture(context, texture),
                    yuv.planes[component.index()].data,
                    "{name}: {component:?}"
                );
            }
        }

        // An unsuitable texture leaves the frame to decode.
        let unsuitable = context.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("too small"),
            size: wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        decoder.clear();
        decoder.push_packet(&frames[0]).expect("packet");
        let outcome = decoder
            .submit_assembled_to_texture(&color, &unsuitable.create_view(&Default::default()));
        assert!(matches!(outcome, Err(GpuError::Texture(_))), "{outcome:?}");
        assert!(decoder.is_ready(false), "{name}: the frame is still there");
    }

    // A view in the texture's sRGB format, which the checks cannot see, fails
    // when the draw's commands are finished: the frame is not used up.
    let (layout, frames) = vector_frames("gradient_200x120_420");
    let mut decoder = decoder_for(context, &layout);
    let texture = context.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("sRGB view"),
        size: wgpu::Extent3d {
            width: layout.width(),
            height: layout.height(),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
    });
    let srgb = texture.create_view(&wgpu::TextureViewDescriptor {
        format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
        ..Default::default()
    });
    decoder.push_packet(&frames[0]).expect("packet");
    let outcome = decoder.submit_assembled_to_texture(&color, &srgb);
    assert!(matches!(outcome, Err(GpuError::Device(_))), "{outcome:?}");
    assert!(decoder.is_ready(false), "the frame is still there");
    decoder
        .submit_assembled_to_texture(&color, &texture.create_view(&Default::default()))
        .expect("a view in the texture's format");
}

/// A texture for `component`'s plane of `layout` with `usage`, as a caller
/// makes one.
fn plane_texture(
    context: &GpuContext,
    layout: &FrameLayout,
    component: Component,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    let (width, height) = layout.plane_size(component);
    context.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("caller's plane"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: emberwave_core::gpu::PLANE_FORMAT,
        usage,
        view_formats: &[],
    })
}

#[test]
fn planes_stored_into_textures_are_the_copied_planes() {
    let context = gpu_or_return!();
    // Metal stores the planes from the inverse transform, the other backends
    // copy them in (decision D52).
    let metal = context.adapter_info().backend == wgpu::Backend::Metal;
    assert_eq!(context.stores_planes(), metal);
    if !metal {
        return;
    }
    // Textures a client samples, which decodes copy into, or store into.
    let copy = wgpu::TextureUsages::COPY_DST
        | wgpu::TextureUsages::COPY_SRC
        | wgpu::TextureUsages::TEXTURE_BINDING;
    let store = copy | wgpu::TextureUsages::STORAGE_BINDING;
    // 4:2:0, whose chroma planes come out a level before luma, and 4:4:4 of
    // an odd size, whose last tiles stick out of the planes.
    for name in ["noise_192x136_420", "noise_131x67_444"] {
        let (layout, frames) = vector_frames(name);
        let mut decoder = decoder_for(context, &layout);
        assert!(decoder.stores_planes(), "{name}");
        let made = decoder.create_plane_texture(Component::Cr).expect("plane");
        assert!(
            made.usage().contains(wgpu::TextureUsages::STORAGE_BINDING),
            "{name}"
        );
        // Stored into all three; copied into all three; copied into all
        // three when one of them takes no stores; and copied into textures
        // that take stores through views for sampling, which do not.
        let sampled = wgpu::TextureViewDescriptor {
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            ..Default::default()
        };
        let plain = wgpu::TextureViewDescriptor::default();
        for (usages, view) in [
            ([store; 3], &plain),
            ([copy; 3], &plain),
            ([store, copy, store], &plain),
            ([store; 3], &sampled),
        ] {
            let textures: Vec<wgpu::Texture> = Component::ALL
                .into_iter()
                .zip(usages)
                .map(|(component, usage)| plane_texture(context, &layout, component, usage))
                .collect();
            let views: Vec<wgpu::TextureView> = textures
                .iter()
                .map(|texture| texture.create_view(view))
                .collect();
            let targets = [&views[0], &views[1], &views[2]];
            for (index, frame) in frames.iter().enumerate() {
                let planes = decoder.decode(frame).expect("decode").planes;
                // Waiting for the GPU, and recorded into the caller's
                // encoder, in turn.
                if index % 2 == 0 {
                    decoder.clear();
                    decoder.push_packet(frame).expect("packet");
                    decoder
                        .decode_assembled_to_plane_textures(targets)
                        .expect("decode into plane textures");
                } else {
                    let mut encoder = context.device().create_command_encoder(&Default::default());
                    decoder
                        .record_to_plane_textures(frame, targets, &mut encoder)
                        .expect("recorded decode into plane textures");
                    context.queue().submit([encoder.finish()]);
                }
                for (component, texture) in Component::ALL.into_iter().zip(&textures) {
                    assert_eq!(
                        read_texture(context, texture),
                        planes[component.index()].data,
                        "{name}: {component:?}, frame {index}, {usages:?}, {:?}",
                        view.usage
                    );
                }
            }
        }
    }
}

#[test]
fn frames_written_in_place_decode_as_copied_ones() {
    // On Metal on Apple silicon the decodes that wait write a frame where the
    // shaders read it; elsewhere, and with `shared_uploads` off, it goes
    // through an upload buffer and copies (decision D53).
    let context = gpu_or_return!();
    let info = context.adapter_info();
    let shares =
        info.backend == wgpu::Backend::Metal && info.device_type == wgpu::DeviceType::IntegratedGpu;
    assert_eq!(context.shares_uploads(), shares);
    let options = {
        let mut options = common::options();
        options.shared_uploads = false;
        options
    };
    let Some(copied) = create_context(&options) else {
        return;
    };
    assert!(!copied.shares_uploads());
    let color = ColorParams::default();
    // Decodes that do not wait in between, which write through the queue.
    for name in [
        "noise_192x136_420",
        "gradient_200x120_420",
        "noise_131x67_444",
    ] {
        let (layout, frames) = vector_frames(name);
        let mut shared = decoder_for(context, &layout);
        let mut through = decoder_for(&copied, &layout);
        for (index, frame) in frames.iter().enumerate() {
            assert_eq!(
                shared.decode(frame).expect("decode"),
                through.decode(frame).expect("decode"),
                "{name}: frame {index}"
            );
            assert_eq!(
                shared.decode_rgb(frame, &color).expect("RGB decode"),
                through.decode_rgb(frame, &color).expect("RGB decode"),
                "{name}: frame {index}, RGB"
            );
            let mut encoder = context.device().create_command_encoder(&Default::default());
            shared
                .record_rgb(frame, &color, &mut encoder)
                .expect("recorded decode");
            context.queue().submit([encoder.finish()]);
        }
    }
}

#[test]
fn frames_written_in_place_wait_for_the_frame_before() {
    // A decode that does not wait, still on the GPU when the next decode,
    // which waits, maps the buffers it reads to write its own frame into
    // them: the first must still decode its own frame (decision D53). The
    // frames are large enough to keep the GPU busy, and the first one grows
    // the payload buffer.
    let context = gpu_or_return!();
    let options = {
        let mut options = common::options();
        options.shared_uploads = false;
        options
    };
    let Some(copied) = create_context(&options) else {
        return;
    };
    let layout = FrameLayout::new(1920, 1080, ChromaSubsampling::Yuv420).unwrap();
    let frames: Vec<Vec<u8>> = (0..4)
        .map(|seed| synthetic_frame(&layout, 0x5EED_0100 + seed, seed % 2 == 1))
        .collect();
    assert!(frames[0].len() > 64 * 1024, "{}", frames[0].len());
    let color = ColorParams::default();
    let mut reference = decoder_for(&copied, &layout);
    let mut decoder = decoder_for(context, &layout);
    let device = context.device();
    for pair in frames.windows(2) {
        let (first, second) = (&pair[0], &pair[1]);
        let mut encoder = device.create_command_encoder(&Default::default());
        decoder
            .record_rgb(first, &color, &mut encoder)
            .expect("recorded decode");
        let rgba = decoder.rgba_buffer().expect("RGBA buffer");
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test readback"),
            size: rgba.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(rgba, 0, &readback, 0, rgba.size());
        context.queue().submit([encoder.finish()]);
        // No wait: the next decode maps the buffers at once.
        let planes = decoder.decode(second).expect("decode");
        assert_eq!(planes, reference.decode(second).expect("decode"));
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
        assert_eq!(
            rgb,
            reference
                .decode_rgb(first, &color)
                .expect("RGB decode")
                .data
        );
    }
}

#[test]
fn without_plane_stores_the_planes_are_copied() {
    let options = {
        let mut options = common::options();
        options.plane_stores = false;
        options
    };
    let Some(context) = create_context(&options) else {
        return;
    };
    assert!(!context.stores_planes());
    let (layout, frames) = vector_frames("gradient_200x120_420");
    let mut decoder = decoder_for(&context, &layout);
    assert!(!decoder.stores_planes());
    let textures =
        Component::ALL.map(|component| decoder.create_plane_texture(component).expect("plane"));
    let views = textures
        .each_ref()
        .map(|texture| texture.create_view(&Default::default()));
    for texture in &textures {
        let usage = texture.usage();
        assert!(!usage.contains(wgpu::TextureUsages::STORAGE_BINDING));
    }
    for frame in &frames {
        let planes = decoder.decode(frame).expect("decode").planes;
        decoder.clear();
        decoder.push_packet(frame).expect("packet");
        decoder
            .decode_assembled_to_plane_textures(views.each_ref())
            .expect("decode into plane textures");
        for (component, texture) in Component::ALL.into_iter().zip(&textures) {
            assert_eq!(
                read_texture(&context, texture),
                planes[component.index()].data,
                "{component:?}"
            );
        }
    }
}

#[test]
fn decoding_into_plane_textures_frame_after_frame_leaves_nothing_behind() {
    // A context of its own: the counters belong to the device. Where the
    // planes are stored, every decode makes the bind groups of its textures,
    // which must go again.
    let Some(context) = create_context(&common::options()) else {
        return;
    };
    let (layout, frames) = vector_frames("noise_192x136_420");
    let mut decoder = decoder_for(&context, &layout);
    let textures =
        Component::ALL.map(|component| decoder.create_plane_texture(component).expect("plane"));
    let views = textures
        .each_ref()
        .map(|texture| texture.create_view(&Default::default()));
    let mut decode = |frame: &[u8]| {
        decoder.clear();
        decoder.push_packet(frame).expect("packet");
        decoder
            .decode_assembled_to_plane_textures(views.each_ref())
            .expect("decode into plane textures");
    };
    decode(&frames[0]);
    let before = live_resources(&context);
    for frame in frames.iter().cycle().take(24) {
        decode(frame);
    }
    assert_eq!(
        live_resources(&context),
        before,
        "live GPU objects and memory"
    );
}

#[test]
fn only_textures_the_planes_fit_take_them() {
    let context = gpu_or_return!();
    let (layout, frames) = vector_frames("gradient_200x120_420");
    let mut decoder = decoder_for(context, &layout);
    let frame = decoder.decode(&frames[0]).expect("decode");
    let make = |(width, height), format, usage| {
        context.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("plane"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let target = wgpu::TextureUsages::COPY_DST;
    let r8 = wgpu::TextureFormat::R8Unorm;
    let (luma, chroma) = ((200, 120), (100, 60));
    let y = make(luma, r8, target);
    let cb = make(chroma, r8, target);
    let cr = make(chroma, r8, target);
    let unsuitable: [[&wgpu::Texture; 3]; 5] = [
        // The planes in the wrong order: the sizes give it away.
        [&cb, &y, &cr],
        [&y, &cb, &cb],
        [
            &y,
            &cb,
            &make(chroma, wgpu::TextureFormat::Rgba8Unorm, target),
        ],
        [
            &y,
            &make(chroma, r8, wgpu::TextureUsages::TEXTURE_BINDING),
            &cr,
        ],
        [&make((200, 60), r8, target), &cb, &cr],
    ];
    for textures in unsuitable {
        let views = textures.map(|texture| texture.create_view(&Default::default()));
        decoder.clear();
        decoder.push_packet(&frames[0]).expect("packet");
        let outcome = decoder.decode_assembled_to_plane_textures(views.each_ref());
        assert!(matches!(outcome, Err(GpuError::Texture(_))), "{outcome:?}");
        // The frame is still there to decode.
        assert_eq!(decoder.decode_assembled().expect("decode"), frame);
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let outcome = decoder.record_to_plane_textures(&frames[0], views.each_ref(), &mut encoder);
        assert!(matches!(outcome, Err(GpuError::Texture(_))), "{outcome:?}");
    }

    // The picture's texture takes no planes, and the planes' no picture.
    let views = [&y, &cb, &cr].map(|texture| texture.create_view(&Default::default()));
    decoder.clear();
    decoder.push_packet(&frames[0]).expect("packet");
    let outcome = decoder.decode_assembled_to_texture(&ColorParams::default(), &views[0]);
    assert!(matches!(outcome, Err(GpuError::Texture(_))), "{outcome:?}");
    let picture = decoder
        .create_texture(wgpu::TextureFormat::Rgba8Unorm)
        .expect("texture")
        .create_view(&Default::default());
    let outcome = decoder.decode_assembled_to_plane_textures([&picture, &views[1], &views[2]]);
    assert!(matches!(outcome, Err(GpuError::Texture(_))), "{outcome:?}");
    decoder
        .decode_assembled_to_plane_textures(views.each_ref())
        .expect("the right textures");
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

/// A 192x128 4:2:0 picture with texture everywhere, its luma raised by
/// `lift`, and its frame from the CPU encoder.
fn textured_frame(lift: u8) -> (FrameLayout, Vec<u8>) {
    let (width, height) = (192usize, 128usize);
    let luma: Vec<u8> = (0..width * height)
        .map(|i| {
            let (x, y) = (i % width, i / width);
            ((x * 37 + y * 101 + (x * y) % 23) % 171) as u8 + 20 + lift
        })
        .collect();
    let chroma = (vec![110; width * height / 4], vec![140; width * height / 4]);
    let mut encoder = emberwave_core::encode::Encoder::new(
        width as u32,
        height as u32,
        ChromaSubsampling::Yuv420,
        emberwave_core::encode::Signalling::default(),
    )
    .unwrap();
    let frame = encoder
        .encode([&luma, &chroma.0, &chroma.1], 20_000)
        .unwrap();
    (encoder.layout().clone(), frame)
}

/// The packets of `frame` but every third that holds blocks below the
/// coarsest level only, from the `phase`th on, as a stream that lost them.
fn lossy_packets<'a>(layout: &FrameLayout, frame: &'a [u8], phase: usize) -> Vec<&'a [u8]> {
    let coarsest = layout
        .bands()
        .find(|&(_, level, _, _)| level < DECOMPOSITION_LEVELS - 1)
        .map_or(layout.block_count(), |(.., band)| band.first_block);
    emberwave_core::encode::packets(frame, 64)
        .into_iter()
        .enumerate()
        .filter(|(n, packet)| {
            let detail = matches!(Header::parse(packet), Ok(Header::Block(block)) if block.block_index >= coarsest);
            !(detail && n % 3 == phase)
        })
        .map(|(_, packet)| packet)
        .collect()
}

#[test]
fn frames_drawn_in_part_conceal_what_they_lack_as_the_cpu_does() {
    // Concealment (decision D66): a frame drawn in part keeps the last
    // frame's coefficients for the blocks of detail it lacks where the
    // picture is still, and has zeros where it changed, on the GPU as
    // cpu::Concealer has them. Whole frames are as without it.
    let context = gpu_or_return!();
    let (layout, still) = textured_frame(0);
    let (_, changed) = textured_frame(40);
    let mut decoder = decoder_for(context, &layout);
    assert!(!decoder.concealment());
    decoder.set_concealment(true);
    let mut concealer = cpu::Concealer::new();
    // Whole, the same picture in part, the same whole, a brighter picture in
    // part, and the brighter in part again, now still against the last,
    // which has the blocks it lacks.
    let sequence = [
        (&still, None),
        (&still, Some(0)),
        (&still, None),
        (&changed, Some(0)),
        (&changed, Some(1)),
    ];
    let mut decoded = Vec::new();
    for (index, &(frame, phase)) in sequence.iter().enumerate() {
        let lossy = phase.is_some();
        let packets = match phase {
            Some(phase) => lossy_packets(&layout, frame, phase),
            None => vec![&frame[..]],
        };
        decoder.clear();
        let mut assembler = FrameAssembler::new(layout.clone());
        for packet in &packets {
            decoder.push_packet(packet).unwrap();
            assembler.push_packet(packet).unwrap();
        }
        assert_eq!(decoder.is_ready(false), !lossy);
        let gpu = decoder.decode_assembled().unwrap();
        let cpu = concealer.decode(&assembler.frame(), lossy).unwrap();
        for (plane, expected) in gpu.planes.iter().zip(&cpu.planes) {
            assert!(
                max_abs_diff(&expected.to_unorm8(), &plane.data) <= 1,
                "frame {index}: {}",
                max_abs_diff(&expected.to_unorm8(), &plane.data)
            );
        }
        let zeros = cpu::decode_frame(&assembler.frame()).unwrap();
        decoded.push((gpu, zeros));
    }
    let whole = &decoded[0].0;
    // The still picture in part is the whole picture.
    assert_eq!(&decoded[1].0, whole);
    assert_ne!(
        decoded[1].0.planes[0].data,
        decoded[1].1.planes[0].to_unorm8()
    );
    // The changed picture in part has zeros where it lacks blocks: as
    // without concealment, to the GPU's rounding.
    for (plane, zeros) in decoded[3].0.planes.iter().zip(&decoded[3].1.planes) {
        assert!(max_abs_diff(&zeros.to_unorm8(), &plane.data) <= 1);
    }
    // Then still against the frame before, it keeps that frame's detail
    // rather than the zeros.
    assert_ne!(
        decoded[4].0.planes[0].data,
        decoded[4].1.planes[0].to_unorm8()
    );

    // Off again, a frame in part has zeros.
    decoder.set_concealment(false);
    decoder.clear();
    let mut assembler = FrameAssembler::new(layout.clone());
    for packet in lossy_packets(&layout, &still, 0) {
        decoder.push_packet(packet).unwrap();
        assembler.push_packet(packet).unwrap();
    }
    let gpu = decoder.decode_assembled().unwrap();
    let zeros = cpu::decode_frame(&assembler.frame()).unwrap();
    for (plane, expected) in gpu.planes.iter().zip(&zeros.planes) {
        assert!(max_abs_diff(&expected.to_unorm8(), &plane.data) <= 1);
    }
}

/// A step of a scene for the concealment's differential test: a texture of
/// sines and noise in every plane, moved, with a patch of luma or of chroma
/// changed, or noise of its own in every frame.
#[derive(Debug, Clone, Copy)]
struct Scene {
    seed: u64,
    shift: i64,
    patch: bool,
    chroma_patch: bool,
    noise: i32,
}

/// A deterministic hash of a sample's place, from -15 to 15.
fn grain(x: i64, y: i64, salt: u64) -> i32 {
    let mut v = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ salt.wrapping_mul(0x1656_67B1_9E37_79F9);
    v ^= v >> 29;
    v = v.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    v ^= v >> 32;
    (v % 31) as i32 - 15
}

fn scene_picture(layout: &FrameLayout, scene: Scene, rng: &mut SceneRng) -> [Vec<u8>; 3] {
    let freq = 1.0 + (scene.seed % 5) as f64 * 0.37;
    std::array::from_fn(|plane| {
        let (w, h) = layout.plane_size(Component::ALL[plane]);
        let (w, h) = (w as i64, h as i64);
        let shift = scene.shift / if plane == 0 { 1 } else { 2 };
        let (amplitude, period) = if plane == 0 {
            (60.0, 17.0)
        } else {
            (30.0, 23.0 + 9.0 * plane as f64)
        };
        let patched = if plane == 0 {
            scene.patch
        } else {
            scene.chroma_patch
        };
        let mut samples = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let xs = x + shift;
                let mut v = 128.0
                    + amplitude * (xs as f64 / (period * freq)).sin()
                    + 0.7 * amplitude * (y as f64 / (13.0 * freq)).cos()
                    + f64::from(grain(xs, y, scene.seed + plane as u64 * 11));
                if patched && x >= w / 3 && x < w / 2 + 1 && y >= h / 4 && y < h / 2 + 1 {
                    v += 60.0;
                }
                let mut v = v as i32;
                if scene.noise > 0 {
                    v += (rng.next() % (2 * scene.noise as u64 + 1)) as i32 - scene.noise;
                }
                samples.push(v.clamp(0, 255) as u8);
            }
        }
        samples
    })
}

/// A small xorshift generator of 64 bits, for the scenes.
struct SceneRng(u64);

impl SceneRng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Blocks kept from the last frame and blocks zeroed, in luma and in
/// chroma, over a run of [`conceal_like_the_cpu`].
#[derive(Debug, Default)]
struct ConcealCounts {
    kept: [usize; 2],
    zeroed: [usize; 2],
}

/// Decodes `steps` frames of an evolving scene of `layout`, each with packets
/// lost in one of several ways, on the GPU through one of its decode paths
/// with concealment now and then turned off and on, and on the CPU with
/// `cpu::Concealer`. After every decode the GPU's bands must be the CPU's to
/// the bit. Returns how many blocks were kept and zeroed.
fn conceal_like_the_cpu(
    context: &GpuContext,
    layout: &FrameLayout,
    steps: usize,
    seed: u64,
) -> ConcealCounts {
    let mut rng = SceneRng(seed | 1);
    let mut encoder = emberwave_core::encode::Encoder::new(
        layout.width(),
        layout.height(),
        layout.chroma(),
        emberwave_core::encode::Signalling::default(),
    )
    .unwrap();
    let coarsest = layout
        .bands()
        .find(|&(_, level, _, _)| level < DECOMPOSITION_LEVELS - 1)
        .map_or(layout.block_count(), |(.., band)| band.first_block);
    let mut decoder = decoder_for(context, layout);
    decoder.set_concealment(true);
    let mut on = true;
    let mut concealer = cpu::Concealer::new();
    let color = ColorParams::default();
    let samples: usize = Component::ALL
        .iter()
        .map(|&c| {
            let (w, h) = layout.plane_size(c);
            (w * h) as usize
        })
        .sum();
    let mut scene = Scene {
        seed: rng.next() % 1000,
        shift: 0,
        patch: false,
        chroma_patch: false,
        noise: 0,
    };
    let mut counts = ConcealCounts::default();
    for step in 0..steps {
        match rng.next() % 10 {
            0..=3 => {}
            4 => scene.noise = if scene.noise == 0 { 2 } else { 0 },
            5 => scene.shift += 1,
            6 => scene.shift += 2,
            7 => scene.patch = !scene.patch,
            8 => scene.chroma_patch = !scene.chroma_patch,
            _ => scene.seed = rng.next() % 1000,
        }
        let planes = scene_picture(layout, scene, &mut rng);
        let budget = (samples as f64 * [0.15, 0.6][step % 2]) as usize;
        let frame = encoder
            .encode([&planes[0], &planes[1], &planes[2]], budget.max(3000))
            .unwrap();
        let packets =
            emberwave_core::encode::packets(&frame, [64, 200, 1403][(rng.next() % 3) as usize]);
        let (loss, burst) = (rng.next() % 6, (rng.next() % packets.len() as u64) as usize);
        let kept: Vec<&[u8]> = packets
            .iter()
            .enumerate()
            .filter(|&(n, packet)| {
                let detail = matches!(Header::parse(packet), Ok(Header::Block(block)) if block.block_index >= coarsest);
                !detail
                    || match loss {
                        0 => true,
                        1 | 2 => rng.unit() >= 0.5,
                        3 | 4 => !(burst..burst + packets.len() / 3).contains(&n),
                        _ => false,
                    }
            })
            .map(|(_, packet)| *packet)
            .collect();
        let mut assembler = FrameAssembler::new(layout.clone());
        decoder.clear();
        for packet in &kept {
            decoder.push_packet(packet).unwrap();
            assembler.push_packet(packet).unwrap();
        }
        let partial = assembler.received_blocks() < assembler.total_blocks();
        if rng.next().is_multiple_of(8) {
            on = !on;
            decoder.set_concealment(on);
        }
        let path = rng.next() % 4;
        let what = format!(
            "{}x{} {:?}, step {step}, loss {loss}, path {path}, concealing {}",
            layout.width(),
            layout.height(),
            layout.chroma(),
            partial && on
        );
        match path {
            0 => {
                decoder.decode_assembled().unwrap();
            }
            1 => {
                decoder.decode_assembled_rgb(&color).unwrap();
            }
            2 => {
                let textures = Component::ALL.map(|c| decoder.create_plane_texture(c).unwrap());
                let views = textures
                    .each_ref()
                    .map(|t| t.create_view(&Default::default()));
                decoder
                    .submit_assembled_to_plane_textures([&views[0], &views[1], &views[2]])
                    .unwrap();
            }
            _ => {
                let texture = decoder
                    .create_texture(wgpu::TextureFormat::Bgra8Unorm)
                    .unwrap();
                decoder
                    .submit_assembled_to_texture(&color, &texture.create_view(&Default::default()))
                    .unwrap();
            }
        }
        let last = concealer.bands().cloned();
        concealer.decode(&assembler.frame(), partial && on).unwrap();
        let bands = concealer.bands().unwrap();
        for (component, level, band, _) in layout.bands() {
            if band == Band::Ll && level < DECOMPOSITION_LEVELS - 1 {
                continue; // the inverse transform's, not the decode's
            }
            let gpu = decoder.read_band(component, level, band).unwrap().unwrap();
            let cpu = bands.plane(component, level, band).unwrap();
            let differ = gpu
                .iter()
                .zip(cpu)
                .filter(|(a, b)| a.to_bits() != b.to_bits() && !(**a == 0.0 && **b == 0.0))
                .count();
            assert_eq!(differ, 0, "{what}: {component:?} level {level} {band:?}");
        }
        // What the concealment did with the blocks the frame lacked.
        if let (Some(last), true) = (last, partial && on) {
            let view = assembler.frame();
            for block in 0..layout.block_count() {
                let location = layout.locate(block).unwrap();
                if view.block(block).is_some() || location.level == DECOMPOSITION_LEVELS - 1 {
                    continue;
                }
                let info = layout
                    .band(location.component, location.level, location.band)
                    .unwrap();
                let (old, new) = (
                    last.plane(location.component, location.level, location.band)
                        .unwrap(),
                    bands
                        .plane(location.component, location.level, location.band)
                        .unwrap(),
                );
                let (mut had, mut has, mut same) = (false, false, true);
                for y in location.block_y * 32..((location.block_y + 1) * 32).min(info.height) {
                    for x in location.block_x * 32..((location.block_x + 1) * 32).min(info.width) {
                        let at = (y * info.width + x) as usize;
                        had |= old[at] != 0.0;
                        has |= new[at] != 0.0;
                        same &= old[at] == new[at];
                    }
                }
                let chroma = usize::from(location.component != Component::Y);
                if had && same {
                    counts.kept[chroma] += 1;
                }
                if had && !has {
                    counts.zeroed[chroma] += 1;
                }
            }
        }
    }
    counts
}

#[test]
fn concealment_matches_the_cpu_through_scenes_losses_and_decode_paths() {
    // After a review's differential test: still, moved, patched and noisy
    // pictures, losses at random, in bursts and of every block of detail,
    // packets of three sizes, four decode paths, concealment turned off and
    // on, in both chroma formats and sizes odd and even. The GPU must leave
    // the CPU's coefficients in every band, and both outcomes must happen,
    // in luma and in chroma.
    let context = gpu_or_return!();
    let mut total = ConcealCounts::default();
    let shapes = [
        (192, 128, ChromaSubsampling::Yuv420),
        (130, 66, ChromaSubsampling::Yuv420),
        (66, 258, ChromaSubsampling::Yuv420),
        (2, 2, ChromaSubsampling::Yuv420),
        (131, 67, ChromaSubsampling::Yuv444),
        (65, 129, ChromaSubsampling::Yuv444),
        (7, 3, ChromaSubsampling::Yuv444),
    ];
    for (index, &(width, height, chroma)) in shapes.iter().enumerate() {
        let layout = FrameLayout::new(width, height, chroma).unwrap();
        let counts = conceal_like_the_cpu(context, &layout, 16, 0x1234_5678 + index as u64 * 7);
        for i in 0..2 {
            total.kept[i] += counts.kept[i];
            total.zeroed[i] += counts.zeroed[i];
        }
    }
    println!(
        "blocks kept {:?} and zeroed {:?} (luma, chroma)",
        total.kept, total.zeroed
    );
    assert!(
        total
            .kept
            .iter()
            .chain(&total.zeroed)
            .all(|&count| count > 0),
        "{total:?}"
    );
}

#[test]
fn a_partial_frame_needs_the_share_of_blocks_its_decoder_asks() {
    let context = gpu_or_return!();
    // A frame of emberwave's encoder, which sends every block of the
    // coarsest level (decision D58).
    let name = "gradient_200x120_420";
    let bytes = std::fs::read(encoder_dir().join(format!("{name}.pwv1"))).expect("stream");
    let (layout, frames) = frames_of(name, &bytes);
    let mut decoder = decoder_for(context, &layout);
    let coarsest = layout
        .bands()
        .find(|&(_, level, _, _)| level < DECOMPOSITION_LEVELS - 1)
        .map_or(layout.block_count(), |(_, _, _, band)| band.first_block);

    // Packets of a block each, mostly; every other one past the coarsest
    // level is lost.
    let mut assembler = FrameAssembler::new(layout.clone());
    for (number, packet) in emberwave_core::encode::packets(&frames[0], 64)
        .into_iter()
        .enumerate()
    {
        // The start of frame header may go alone, and is kept.
        let mut at = 0;
        let first_block = loop {
            match Header::parse(&packet[at..]) {
                Ok(Header::StartOfFrame(_)) => at += 8,
                Ok(Header::Block(block)) => break Some(block.block_index),
                Ok(_) | Err(_) => break None,
            }
        };
        if first_block.is_some_and(|block| block >= coarsest) && number % 2 == 1 {
            continue;
        }
        decoder.push_packet(packet).unwrap();
        assembler.push_packet(packet).unwrap();
    }
    let share = assembler.received_blocks() as f32 / assembler.total_blocks() as f32;
    assert!((0.4..0.9).contains(&share), "{share}");

    assert_eq!(decoder.partial_share(), 0.9);
    assert!(!decoder.is_ready(true), "90 % by default");
    decoder.set_partial_share(share - 0.05).unwrap();
    assert!(decoder.is_ready(true));
    assert!(!decoder.is_ready(false));
    // A share above what arrived; the edge itself, where exactly the share
    // is not enough, is the assembler's test.
    decoder.set_partial_share(share + 0.01).unwrap();
    assert!(!decoder.is_ready(true));
    decoder.set_partial_share(1.0).unwrap();
    assert!(!decoder.is_ready(true), "1 takes whole frames only");
    // Outside 0 to 1, or NaN: refused, and the share stays.
    for refused in [-1.0, 1.01, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            decoder.set_partial_share(refused),
            Err(GpuError::InvalidArgument(_))
        ));
        assert_eq!(decoder.partial_share(), 1.0);
    }

    // What arrived decodes, as it always could.
    decoder.set_partial_share(0.0).unwrap();
    assert!(decoder.is_ready(true));
    let frame = decoder.decode_assembled().unwrap();
    assert_eq!(frame.planes[0].data.len(), 200 * 120);
}

#[test]
fn a_device_limited_to_webgpu_guarantees_decodes_the_same() {
    // The portable subset: default limits and no optional features. If the
    // shaders needed anything beyond it, creating the pipelines or running
    // them would fail here.
    let options = {
        let mut options = common::options();
        options.strict_webgpu = true;
        options
    };
    let Some(context) = create_context(&options) else {
        return;
    };
    // Not even the stores into plane textures, which Metal's own format
    // features allow (decision D52), nor frames written where the shaders
    // read them, which mappable storage buffers allow (decision D53).
    assert!(!context.stores_planes());
    assert!(!context.shares_uploads());
    let features = context.device().features();
    assert!(!features.contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES));
    assert!(!features.contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS));

    let (layout, frames) = vector_frames("noise_131x67_444");
    let reference = read_reference("noise_131x67_444", Reference::Fp32);
    let mut decoder = decoder_for(&context, &layout);
    assert!(!decoder.stores_planes());
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
                    let params = ColorParams::new(matrix, range, chroma_siting);
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
    let b = ColorParams::new(ColorMatrix::Bt601, YcbcrRange::Full, ChromaSiting::Left);
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
    let options = {
        let mut options = common::options();
        options.timestamps = true;
        options
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
    let options = {
        let mut options = common::options();
        options.timestamps = true;
        options
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
    let Some(context) = create_context(&common::options()) else {
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
        .record_rgb(&frames[0], &color, &mut encoder)
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
    let Some(context) = create_context(&common::options()) else {
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
    let options = {
        let mut options = common::options();
        options.timestamps = true;
        options
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
    other.record_rgb(&frames[1], &color, &mut encoder).unwrap();
    let rgba = other.rgba_buffer().expect("RGBA buffer after record_rgb");
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
    // buffer of their own (decision D34) where record_rgb writes through the
    // queue: each path finds the payload the other left and replaces it.
    for frame in [&frames[0], &frames[1]] {
        let from_both = other.decode_rgb(frame, &color).unwrap();
        assert_eq!(
            from_both.data,
            decoder.decode_rgb(frame, &color).unwrap().data
        );
    }
}
