//! The encoder of the C API, called the way C calls it, through raw pointers,
//! and checked against the Rust encoder it wraps and the C decoder.
//!
//! The tests that need a GPU skip without one, unless `EMBERWAVE_REQUIRE_GPU`
//! is set, like those of `emberwave-core`.

use std::ffi::CStr;
use std::path::PathBuf;
use std::ptr;

use emberwave_core::bitstream::{
    ChromaSiting, ChromaSubsampling, ColorPrimaries, Component, FrameLayout, Header, Pwv1File,
    TransferFunction, YcbcrRange, YcbcrTransform,
};
use emberwave_core::color::{ColorMatrix, ColorParams, RgbOrder, rgb_to_ycbcr8};
use emberwave_core::encode::{self, Signalling};
use emberwave_core::gpu::{Encoder, GpuContext, GpuOptions};
use emberwave_ffi::*;

fn last_error() -> String {
    // SAFETY: emberwave_last_error always returns a C string.
    unsafe { CStr::from_ptr(emberwave_last_error()) }
        .to_string_lossy()
        .into_owned()
}

/// A context on the default GPU, or `None` without one, unless the
/// environment insists on one.
fn context() -> Option<*mut EmberwaveContext> {
    let mut context = ptr::null_mut();
    // SAFETY: NULL options and a pointer to store the context.
    let status = unsafe { emberwave_context_create(ptr::null(), &mut context) };
    if status == EmberwaveStatus::Ok {
        return Some(context);
    }
    if std::env::var_os("EMBERWAVE_REQUIRE_GPU").is_some_and(|value| !value.is_empty()) {
        panic!("no GPU: {}", last_error());
    }
    eprintln!("skipping: no GPU ({})", last_error());
    None
}

/// An encoder made through the C API, which must succeed.
fn encoder_for(
    context: *mut EmberwaveContext,
    layout: &FrameLayout,
    signalling: Option<&EmberwaveSignalling>,
) -> *mut EmberwaveEncoder {
    let chroma = match layout.chroma() {
        ChromaSubsampling::Yuv420 => EMBERWAVE_CHROMA_420,
        ChromaSubsampling::Yuv444 => EMBERWAVE_CHROMA_444,
    };
    let mut encoder = ptr::null_mut();
    // SAFETY: a live context, a signalling or NULL, and a pointer to store
    // the encoder.
    let status = unsafe {
        emberwave_encoder_create(
            context,
            layout.width(),
            layout.height(),
            chroma,
            signalling.map_or(ptr::null(), |s| s as *const _),
            &mut encoder,
        )
    };
    assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    encoder
}

/// A picture of `layout` with gradients, edges and some noise, different for
/// each `seed`.
fn picture(layout: &FrameLayout, seed: u32) -> [Vec<u8>; 3] {
    let mut state = seed.wrapping_mul(2_654_435_761) | 1;
    std::array::from_fn(|plane| {
        let (width, height) = layout.plane_size(Component::ALL[plane]);
        (0..width * height)
            .map(|i| {
                let (x, y) = (i % width, i / width);
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let edge = if (x + seed) % 37 < 18 { 40 } else { 0 };
                let value = 40 + x * 120 / width + y * 60 / height + edge + (state >> 29);
                (value + 30 * plane as u32).min(255) as u8
            })
            .collect()
    })
}

/// The planes of `picture` as the C API takes them, with `padding` bytes
/// after each row; the planes are kept in `storage`.
fn source_planes(
    layout: &FrameLayout,
    picture: &[Vec<u8>; 3],
    padding: usize,
    storage: &mut Vec<Vec<u8>>,
) -> [EmberwaveSourcePlane; 3] {
    storage.clear();
    for (plane, samples) in picture.iter().enumerate() {
        let width = layout.plane_size(Component::ALL[plane]).0 as usize;
        storage.push(
            samples
                .chunks_exact(width)
                .flat_map(|row| {
                    row.iter()
                        .copied()
                        .chain(std::iter::repeat_n(0xEE, padding))
                })
                .collect(),
        );
    }
    std::array::from_fn(|plane| {
        let width = layout.plane_size(Component::ALL[plane]).0 as usize;
        EmberwaveSourcePlane {
            data: storage[plane].as_ptr(),
            stride: width + padding,
            size: storage[plane].len(),
        }
    })
}

/// Encodes through the C API: the status, and the frame or, after a failure,
/// what the size out-parameter holds.
fn encode_c(
    encoder: *mut EmberwaveEncoder,
    planes: [EmberwaveSourcePlane; 3],
    budget: usize,
) -> (EmberwaveStatus, Vec<u8>, usize) {
    let mut frame = vec![0u8; budget];
    let mut size = usize::MAX;
    // SAFETY: a live encoder, planes of its sizes and a frame of `budget`
    // bytes.
    let status = unsafe {
        emberwave_encoder_encode(
            encoder,
            planes[0],
            planes[1],
            planes[2],
            frame.as_mut_ptr(),
            budget,
            &mut size,
        )
    };
    if status == EmberwaveStatus::Ok {
        frame.truncate(size);
    } else {
        frame.clear();
    }
    (status, frame, size)
}

/// The packets of a frame as the C API splits it.
fn packets_c(frame: &[u8], max_bytes: usize) -> Vec<EmberwavePacket> {
    let mut count = 0;
    // SAFETY: a frame of `frame.len()` bytes, and no packets, only a count.
    let status = unsafe {
        emberwave_frame_packets(
            frame.as_ptr(),
            frame.len(),
            max_bytes,
            ptr::null_mut(),
            0,
            &mut count,
        )
    };
    assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    let mut packets = vec![EmberwavePacket::default(); count];
    // SAFETY: room for `count` packets.
    let status = unsafe {
        emberwave_frame_packets(
            frame.as_ptr(),
            frame.len(),
            max_bytes,
            packets.as_mut_ptr(),
            packets.len(),
            &mut count,
        )
    };
    assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    assert_eq!(count, packets.len());
    packets
}

const SHAPES: [(u32, u32, ChromaSubsampling); 2] = [
    (200, 120, ChromaSubsampling::Yuv420),
    (131, 67, ChromaSubsampling::Yuv444),
];

#[test]
fn frames_through_the_c_api_equal_the_rust_encoder() {
    let Some(context) = context() else {
        return;
    };
    let rust_context = GpuContext::new(&GpuOptions::default()).expect("a GPU, as the C API's");
    for (width, height, chroma) in SHAPES {
        let layout = FrameLayout::new(width, height, chroma).unwrap();
        let tight = encoder_for(context, &layout, None);
        let mut info = EmberwaveFrameInfo::default();
        // SAFETY: a live encoder and an info to fill in.
        let status = unsafe { emberwave_encoder_info(tight, &mut info) };
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
        let (chroma_width, chroma_height) = layout.plane_size(Component::Cb);
        assert_eq!(
            (
                info.width,
                info.height,
                info.chroma_width,
                info.chroma_height
            ),
            (width, height, chroma_width, chroma_height)
        );
        assert_eq!(
            info.chroma,
            if chroma == ChromaSubsampling::Yuv420 {
                EMBERWAVE_CHROMA_420
            } else {
                EMBERWAVE_CHROMA_444
            }
        );
        let padded = encoder_for(context, &layout, None);
        let mut rust =
            Encoder::new(&rust_context, width, height, chroma, Signalling::default()).unwrap();
        let mut storage = Vec::new();
        for seed in 0..3 {
            let source = picture(&layout, seed);
            let expected = rust
                .encode([&source[0], &source[1], &source[2]], 9_000)
                .unwrap();
            let planes = source_planes(&layout, &source, 0, &mut storage);
            let (status, frame, _) = encode_c(tight, planes, 9_000);
            assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
            assert!(
                frame == expected,
                "{width}x{height} {chroma:?} frame {seed}"
            );
            let planes = source_planes(&layout, &source, 5 + seed as usize, &mut storage);
            let (status, frame, _) = encode_c(padded, planes, 9_000);
            assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
            assert!(
                frame == expected,
                "{width}x{height} {chroma:?} padded frame {seed}"
            );
        }
        // SAFETY: both live, destroyed once.
        unsafe {
            emberwave_encoder_destroy(tight);
            emberwave_encoder_destroy(padded);
        }
    }
    // SAFETY: live, destroyed once.
    unsafe { emberwave_context_destroy(context) };
}

#[test]
fn a_budget_too_small_gives_the_size_of_the_smallest_frame() {
    let Some(context) = context() else {
        return;
    };
    let layout = FrameLayout::new(200, 120, ChromaSubsampling::Yuv420).unwrap();
    let encoder = encoder_for(context, &layout, None);
    let source = picture(&layout, 7);
    let mut storage = Vec::new();
    let planes = source_planes(&layout, &source, 0, &mut storage);
    let (status, _, minimum) = encode_c(encoder, planes, 7);
    assert_eq!(status, EmberwaveStatus::BufferTooSmall);
    assert!(last_error().contains("smallest frame"), "{}", last_error());
    // The header and the blocks of the coarsest level, which every frame
    // carries, each as small as it gets: 12 bytes at least.
    assert!(minimum >= 8 + 12 * 12, "{minimum}");
    let (status, frame, size) = encode_c(encoder, planes, minimum);
    assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    assert!(size <= minimum);
    // The refused budget did not use up a sequence number.
    match Header::parse(&frame).unwrap() {
        Header::StartOfFrame(header) => assert_eq!(header.sequence, 1),
        other => panic!("{other:?}"),
    }
    // SAFETY: live, destroyed once each.
    unsafe {
        emberwave_encoder_destroy(encoder);
        emberwave_context_destroy(context);
    }
}

#[test]
fn every_start_of_frame_header_says_what_the_signalling_does() {
    let Some(context) = context() else {
        return;
    };
    let layout = FrameLayout::new(200, 120, ChromaSubsampling::Yuv420).unwrap();
    // The fields in the header's terms: matrix, range, siting, primaries,
    // transfer function.
    type Fields = (
        YcbcrTransform,
        YcbcrRange,
        ChromaSiting,
        ColorPrimaries,
        TransferFunction,
    );
    let default_fields: Fields = (
        YcbcrTransform::Bt709,
        YcbcrRange::Limited,
        ChromaSiting::Center,
        ColorPrimaries::Srgb,
        TransferFunction::Srgb,
    );
    let defaults = EmberwaveSignalling {
        matrix: EMBERWAVE_MATRIX_BT709,
        range: EMBERWAVE_RANGE_LIMITED,
        chroma_siting: EMBERWAVE_SITING_CENTER,
        primaries: EMBERWAVE_PRIMARIES_SRGB,
        transfer: EMBERWAVE_TRANSFER_SRGB,
        ..Default::default()
    };
    // NULL, the defaults spelt out, and each field away from its default on
    // its own, so that one field cannot stand in for another.
    let cases: [(Option<EmberwaveSignalling>, Fields); 7] = [
        (None, default_fields),
        (Some(defaults), default_fields),
        (
            Some(EmberwaveSignalling {
                matrix: EMBERWAVE_MATRIX_BT2020,
                ..defaults
            }),
            (
                YcbcrTransform::Bt2020,
                default_fields.1,
                default_fields.2,
                default_fields.3,
                default_fields.4,
            ),
        ),
        (
            Some(EmberwaveSignalling {
                range: EMBERWAVE_RANGE_FULL,
                ..defaults
            }),
            (
                default_fields.0,
                YcbcrRange::Full,
                default_fields.2,
                default_fields.3,
                default_fields.4,
            ),
        ),
        (
            Some(EmberwaveSignalling {
                chroma_siting: EMBERWAVE_SITING_LEFT,
                ..defaults
            }),
            (
                default_fields.0,
                default_fields.1,
                ChromaSiting::Left,
                default_fields.3,
                default_fields.4,
            ),
        ),
        (
            Some(EmberwaveSignalling {
                primaries: EMBERWAVE_PRIMARIES_BT2020,
                ..defaults
            }),
            (
                default_fields.0,
                default_fields.1,
                default_fields.2,
                ColorPrimaries::Bt2020,
                default_fields.4,
            ),
        ),
        (
            Some(EmberwaveSignalling {
                transfer: EMBERWAVE_TRANSFER_PQ,
                ..defaults
            }),
            (
                default_fields.0,
                default_fields.1,
                default_fields.2,
                default_fields.3,
                TransferFunction::Pq,
            ),
        ),
    ];
    let source = picture(&layout, 1);
    let mut storage = Vec::new();
    for (index, (signalling, fields)) in cases.iter().enumerate() {
        let encoder = encoder_for(context, &layout, signalling.as_ref());
        let planes = source_planes(&layout, &source, 0, &mut storage);
        let (status, frame, _) = encode_c(encoder, planes, 9_000);
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
        let Header::StartOfFrame(header) = Header::parse(&frame).unwrap() else {
            panic!("a start of frame header");
        };
        assert_eq!(
            (
                header.ycbcr_transform,
                header.ycbcr_range,
                header.chroma_siting,
                header.color_primaries,
                header.transfer_function,
            ),
            *fields,
            "case {index}"
        );
        // SAFETY: live, destroyed once.
        unsafe { emberwave_encoder_destroy(encoder) };
    }
    // SAFETY: live, destroyed once.
    unsafe { emberwave_context_destroy(context) };
}

#[test]
fn bad_encoder_arguments_are_errors_with_a_message() {
    let Some(context) = context() else {
        return;
    };
    let good = EmberwaveSignalling {
        matrix: EMBERWAVE_MATRIX_BT709,
        range: EMBERWAVE_RANGE_LIMITED,
        chroma_siting: EMBERWAVE_SITING_CENTER,
        primaries: EMBERWAVE_PRIMARIES_SRGB,
        transfer: EMBERWAVE_TRANSFER_SRGB,
        ..Default::default()
    };
    let signalling_cases = [
        (
            EmberwaveSignalling {
                matrix: EMBERWAVE_MATRIX_BT601,
                ..good
            },
            "BT.601",
        ),
        (
            EmberwaveSignalling {
                transfer: 2,
                ..good
            },
            "transfer",
        ),
        (
            EmberwaveSignalling {
                struct_size: 4,
                ..good
            },
            "signalling.struct_size is 4",
        ),
    ];
    let create = |width, height, chroma, signalling: *const EmberwaveSignalling| {
        // Not NULL, so that the call has to set it.
        let mut encoder = ptr::dangling_mut::<EmberwaveEncoder>();
        // SAFETY: a live context and a pointer to store the encoder.
        let status = unsafe {
            emberwave_encoder_create(context, width, height, chroma, signalling, &mut encoder)
        };
        assert!(encoder.is_null(), "set to NULL on failure");
        (status, last_error())
    };
    for (signalling, mention) in &signalling_cases {
        let (status, message) = create(64, 32, EMBERWAVE_CHROMA_420, signalling);
        assert_eq!(status, EmberwaveStatus::InvalidArgument);
        assert!(message.contains(mention), "{message}");
    }
    for (width, height, chroma) in [(0, 32, 0), (63, 32, 0), (64, 32, 9), (20_000, 32, 1)] {
        let (status, message) = create(width, height, chroma, ptr::null());
        assert_eq!(status, EmberwaveStatus::InvalidArgument, "{message}");
    }

    let layout = FrameLayout::new(64, 32, ChromaSubsampling::Yuv420).unwrap();
    let encoder = encoder_for(context, &layout, Some(&good));
    let source = picture(&layout, 3);
    let mut storage = Vec::new();
    let planes = source_planes(&layout, &source, 0, &mut storage);
    let mut frame = vec![0u8; 10_000];
    let mut size = 0;
    let call = |encoder, planes: [EmberwaveSourcePlane; 3], frame: *mut u8, size: *mut usize| {
        // SAFETY: what the arguments point to lives, or is NULL on purpose.
        let status = unsafe {
            emberwave_encoder_encode(
                encoder, planes[0], planes[1], planes[2], frame, 10_000, size,
            )
        };
        (status, last_error())
    };
    let mut no_data = planes;
    no_data[1].data = ptr::null();
    let mut narrow = planes;
    narrow[2].stride = 31;
    // Rows so far apart that the plane would span more than memory: refused
    // before anything is read.
    let mut huge = planes;
    huge[0].stride = usize::MAX / 2;
    let cases = [
        (
            ptr::null_mut(),
            planes,
            frame.as_mut_ptr(),
            &mut size as *mut usize,
            "encoder is NULL",
        ),
        (
            encoder,
            no_data,
            frame.as_mut_ptr(),
            &mut size,
            "cb.data is NULL",
        ),
        (
            encoder,
            narrow,
            frame.as_mut_ptr(),
            &mut size,
            "cr.stride is 31",
        ),
        (
            encoder,
            huge,
            frame.as_mut_ptr(),
            &mut size,
            "y spans more memory than a pointer can address",
        ),
        (encoder, planes, ptr::null_mut(), &mut size, "frame is NULL"),
        (
            encoder,
            planes,
            frame.as_mut_ptr(),
            ptr::null_mut(),
            "size is NULL",
        ),
    ];
    for (encoder, planes, frame, size, mention) in cases {
        let (status, message) = call(encoder, planes, frame, size);
        assert_eq!(status, EmberwaveStatus::InvalidArgument, "{mention}");
        assert!(message.contains(mention), "{message}");
    }
    // The failures left the encoder as it was.
    let (status, frame, _) = encode_c(encoder, planes, 10_000);
    assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    match Header::parse(&frame).unwrap() {
        Header::StartOfFrame(header) => assert_eq!(header.sequence, 1),
        other => panic!("{other:?}"),
    }
    // SAFETY: live, destroyed once each; NULL does nothing.
    unsafe {
        emberwave_encoder_destroy(encoder);
        emberwave_encoder_destroy(ptr::null_mut());
        emberwave_context_destroy(context);
    }
}

#[test]
fn frames_split_into_the_packets_of_the_rust_function() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/encoder/scene_200x120_420.pwv1");
    let bytes = std::fs::read(&path).expect("the committed vector");
    let file = Pwv1File::parse(&bytes).unwrap();
    for frame in file.frames() {
        let frame = frame.unwrap();
        for max_bytes in [1, 100, 1000, usize::MAX] {
            let expected: Vec<EmberwavePacket> = encode::packets(frame, max_bytes)
                .iter()
                .map(|packet| EmberwavePacket {
                    offset: packet.as_ptr() as usize - frame.as_ptr() as usize,
                    size: packet.len(),
                })
                .collect();
            assert_eq!(packets_c(frame, max_bytes), expected, "{max_bytes} bytes");
        }
        // Too little room: an error, the count, and nothing written.
        let needed = packets_c(frame, 100).len();
        let mut packets = vec![EmberwavePacket::default(); needed - 1];
        let mut count = 0;
        // SAFETY: room for `needed - 1` packets.
        let status = unsafe {
            emberwave_frame_packets(
                frame.as_ptr(),
                frame.len(),
                100,
                packets.as_mut_ptr(),
                packets.len(),
                &mut count,
            )
        };
        assert_eq!(status, EmberwaveStatus::BufferTooSmall);
        assert_eq!(count, needed);
        assert!(packets.iter().all(|p| *p == EmberwavePacket::default()));
        // Room to spare: the packets, and the rest untouched.
        let unused = EmberwavePacket {
            offset: usize::MAX,
            size: usize::MAX,
        };
        let mut packets = vec![unused; needed + 3];
        // SAFETY: room for `needed + 3` packets.
        let status = unsafe {
            emberwave_frame_packets(
                frame.as_ptr(),
                frame.len(),
                100,
                packets.as_mut_ptr(),
                packets.len(),
                &mut count,
            )
        };
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
        assert_eq!(count, needed);
        assert_eq!(packets[..needed], packets_c(frame, 100)[..]);
        assert!(packets[needed..].iter().all(|p| *p == unused));
    }
    let mut count = 0;
    // SAFETY: a byte that is not read, and a count.
    let status =
        unsafe { emberwave_frame_packets([0u8].as_ptr(), 1, 0, ptr::null_mut(), 0, &mut count) };
    assert_eq!(status, EmberwaveStatus::InvalidArgument);
    assert!(last_error().contains("max_packet_bytes"));
}

#[test]
fn the_c_decoder_shows_the_c_encoders_pictures() {
    let Some(context) = context() else {
        return;
    };
    for (width, height, chroma) in SHAPES {
        let layout = FrameLayout::new(width, height, chroma).unwrap();
        let encoder = encoder_for(context, &layout, None);
        let mut decoder = ptr::null_mut();
        // SAFETY: a live context and a pointer to store the decoder.
        let status = unsafe {
            emberwave_decoder_create(
                context,
                width,
                height,
                if chroma == ChromaSubsampling::Yuv420 {
                    EMBERWAVE_CHROMA_420
                } else {
                    EMBERWAVE_CHROMA_444
                },
                &mut decoder,
            )
        };
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
        let mut storage = Vec::new();
        for seed in 0..2 {
            let source = picture(&layout, seed);
            let planes = source_planes(&layout, &source, 3, &mut storage);
            let (status, frame, _) = encode_c(encoder, planes, 60_000);
            assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
            // Packets the size of a network datagram, sent last first.
            for packet in packets_c(&frame, 1_200).iter().rev() {
                let bytes = &frame[packet.offset..packet.offset + packet.size];
                // SAFETY: a live decoder and `bytes.len()` bytes.
                let status =
                    unsafe { emberwave_decoder_push_packet(decoder, bytes.as_ptr(), bytes.len()) };
                assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
            }
            // SAFETY: a live decoder.
            assert!(unsafe { emberwave_decoder_is_ready(decoder, 0) });
            let mut decoded: Vec<Vec<u8>> = (0..3)
                .map(|plane| {
                    let (w, h) = layout.plane_size(Component::ALL[plane]);
                    vec![0; (w * h) as usize]
                })
                .collect();
            let targets: Vec<EmberwavePlane> = decoded
                .iter_mut()
                .enumerate()
                .map(|(plane, bytes)| EmberwavePlane {
                    data: bytes.as_mut_ptr(),
                    stride: layout.plane_size(Component::ALL[plane]).0 as usize,
                    size: bytes.len(),
                })
                .collect();
            // SAFETY: a live decoder and planes of its sizes.
            let status = unsafe {
                emberwave_decoder_decode_yuv(decoder, targets[0], targets[1], targets[2])
            };
            assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
            for (plane, (expected, actual)) in source.iter().zip(&decoded).enumerate() {
                let squared: f64 = expected
                    .iter()
                    .zip(actual)
                    .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                    .sum();
                let mse = squared / expected.len() as f64;
                let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-12)).log10();
                assert!(
                    psnr >= 40.0,
                    "{width}x{height} {chroma:?} plane {plane}: {psnr:.2} dB"
                );
            }
        }
        // SAFETY: live, destroyed once each.
        unsafe {
            emberwave_decoder_destroy(decoder);
            emberwave_encoder_destroy(encoder);
        }
    }
    // SAFETY: live, destroyed once.
    unsafe { emberwave_context_destroy(context) };
}

/// A decoder made through the C API for pictures of `layout`, which must
/// succeed.
fn decoder_for(context: *mut EmberwaveContext, layout: &FrameLayout) -> *mut EmberwaveDecoder {
    let chroma = match layout.chroma() {
        ChromaSubsampling::Yuv420 => EMBERWAVE_CHROMA_420,
        ChromaSubsampling::Yuv444 => EMBERWAVE_CHROMA_444,
    };
    let mut decoder = ptr::null_mut();
    // SAFETY: a live context and a pointer to store the decoder.
    let status = unsafe {
        emberwave_decoder_create(
            context,
            layout.width(),
            layout.height(),
            chroma,
            &mut decoder,
        )
    };
    assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    decoder
}

/// Three textures of `decoder`'s planes, Y, Cb and Cr.
fn plane_textures(decoder: *mut EmberwaveDecoder) -> [*mut EmberwaveTexture; 3] {
    [EMBERWAVE_PLANE_Y, EMBERWAVE_PLANE_CB, EMBERWAVE_PLANE_CR].map(|plane| {
        let mut texture = ptr::null_mut();
        // SAFETY: a live decoder and a pointer to store the texture.
        let status = unsafe { emberwave_texture_create_plane(decoder, plane, &mut texture) };
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
        texture
    })
}

/// Encodes the planes in `textures` through the C API: the status, and the
/// frame or, after a failure, what the size out-parameter holds.
fn encode_textures_c(
    encoder: *mut EmberwaveEncoder,
    [y, cb, cr]: [*const EmberwaveTexture; 3],
    budget: usize,
) -> (EmberwaveStatus, Vec<u8>, usize) {
    let mut frame = vec![0u8; budget];
    let mut size = usize::MAX;
    // SAFETY: a live encoder, textures that are NULL or live, and a frame of
    // `budget` bytes.
    let status = unsafe {
        emberwave_encoder_encode_textures(encoder, y, cb, cr, frame.as_mut_ptr(), budget, &mut size)
    };
    if status == EmberwaveStatus::Ok {
        frame.truncate(size);
    } else {
        frame.clear();
    }
    (status, frame, size)
}

#[test]
fn planes_a_decode_left_in_textures_encode_as_the_same_planes_in_memory() {
    let Some(context) = context() else {
        return;
    };
    for (width, height, chroma) in SHAPES {
        let layout = FrameLayout::new(width, height, chroma).unwrap();
        let maker = encoder_for(context, &layout, None);
        let decoder = decoder_for(context, &layout);
        let textures = plane_textures(decoder);
        let mut storage = Vec::new();
        let source = picture(&layout, 4);
        let (status, frame, _) = encode_c(
            maker,
            source_planes(&layout, &source, 0, &mut storage),
            9_000,
        );
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());

        // The decoded planes, in the textures and in CPU memory.
        let mut decoded: Vec<Vec<u8>> = (0..3)
            .map(|plane| {
                let (w, h) = layout.plane_size(Component::ALL[plane]);
                vec![0; (w * h) as usize]
            })
            .collect();
        // SAFETY: a live decoder, its textures, a frame of the length passed
        // and planes of its sizes.
        unsafe {
            for into_textures in [true, false] {
                emberwave_decoder_clear(decoder);
                let status = emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len());
                assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
                let status = if into_textures {
                    emberwave_decoder_decode_plane_textures(
                        decoder,
                        textures[0],
                        textures[1],
                        textures[2],
                    )
                } else {
                    let [y, cb, cr] = [0, 1, 2].map(|plane| EmberwavePlane {
                        data: decoded[plane].as_mut_ptr(),
                        stride: layout.plane_size(Component::ALL[plane]).0 as usize,
                        size: decoded[plane].len(),
                    });
                    emberwave_decoder_decode_yuv(decoder, y, cb, cr)
                };
                assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
            }
        }

        // Fresh encoders, so that both frames take the first sequence
        // number.
        let from_textures = encoder_for(context, &layout, None);
        let from_memory = encoder_for(context, &layout, None);
        let (status, frame, _) =
            encode_textures_c(from_textures, textures.map(|t| t.cast_const()), 6_000);
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
        let in_memory = [decoded[0].clone(), decoded[1].clone(), decoded[2].clone()];
        let (status, expected, _) = encode_c(
            from_memory,
            source_planes(&layout, &in_memory, 3, &mut storage),
            6_000,
        );
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
        assert!(frame == expected, "{width}x{height} {chroma:?}");

        // A budget too small, as from CPU memory.
        let (status, _, size) =
            encode_textures_c(from_textures, textures.map(|t| t.cast_const()), 13);
        assert_eq!(status, EmberwaveStatus::BufferTooSmall);
        assert!(size > 13, "the smallest frame's size, not {size}");

        // SAFETY: live, destroyed once each.
        unsafe {
            for texture in textures {
                emberwave_texture_destroy(texture);
            }
            emberwave_decoder_destroy(decoder);
            for encoder in [maker, from_textures, from_memory] {
                emberwave_encoder_destroy(encoder);
            }
        }
    }
    // SAFETY: live, destroyed once.
    unsafe { emberwave_context_destroy(context) };
}

#[test]
fn textures_an_encode_cannot_read_are_refused() {
    let Some(context) = context() else {
        return;
    };
    let layout = FrameLayout::new(200, 120, ChromaSubsampling::Yuv420).unwrap();
    let encoder = encoder_for(context, &layout, None);
    let decoder = decoder_for(context, &layout);
    let [y, cb, cr] = plane_textures(decoder);
    let mut picture = ptr::null_mut();
    // SAFETY: a live decoder and a pointer to store the texture.
    let status =
        unsafe { emberwave_texture_create(decoder, EMBERWAVE_TEXTURE_RGBA8, &mut picture) };
    assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    let other_size = FrameLayout::new(208, 120, ChromaSubsampling::Yuv420).unwrap();
    let other_decoder = decoder_for(context, &other_size);
    let [other_y, _, _] = plane_textures(other_decoder);
    let other_context = context_or_panic();
    let elsewhere = decoder_for(other_context, &layout);
    let [elsewhere_y, _, _] = plane_textures(elsewhere);

    let refused = |textures: [*mut EmberwaveTexture; 3], says: &str| {
        let (status, frame, size) =
            encode_textures_c(encoder, textures.map(|t| t.cast_const()), 9_000);
        assert_eq!(status, EmberwaveStatus::InvalidArgument, "{says}");
        assert!(frame.is_empty() && size == 0, "{says}");
        let error = last_error();
        assert!(error.contains(says), "{says}: {error}");
    };
    let none = ptr::null_mut();
    refused([none, cb, cr], "y is NULL");
    refused([y, none, cr], "cb is NULL");
    refused([y, cb, none], "cr is NULL");
    refused([cb, y, cr], "y is not a texture for the Y plane");
    refused([y, cr, cb], "cb is not a texture for the Cb plane");
    refused(
        [picture, cb, cr],
        "cb or cr is not NULL, though y holds the picture in RGB",
    );
    refused(
        [picture, cb, none],
        "cb or cr is not NULL, though y holds the picture in RGB",
    );
    refused([elsewhere_y, cb, cr], "another context");
    // Of another picture size: the encoder's own check.
    refused([other_y, cb, cr], "texture");

    // Semaphores and images of the caller's need a Vulkan context.
    let mut frame = vec![0u8; 100];
    let mut size = 0;
    // SAFETY: a live encoder and textures, a frame of the length passed and
    // pointers to store results in.
    unsafe {
        let status = emberwave_encoder_encode_textures_vulkan(
            encoder,
            y,
            cb,
            cr,
            ptr::null(),
            frame.as_mut_ptr(),
            frame.len(),
            &mut size,
        );
        assert_eq!(status, EmberwaveStatus::InvalidArgument);
        assert!(last_error().contains("not on the caller's Vulkan device"));
        let image = EmberwaveVulkanImage {
            image: 1,
            format: 9,
            usage: 4,
            layout: 5,
            release: None,
            user: ptr::null_mut(),
            ..Default::default()
        };
        let mut texture = ptr::null_mut();
        let status =
            emberwave_texture_from_vulkan_source(encoder, EMBERWAVE_PLANE_Y, &image, &mut texture);
        assert_eq!(status, EmberwaveStatus::InvalidArgument);
        assert!(texture.is_null());
        let status = emberwave_texture_from_metal_source(encoder, 7, ptr::null_mut(), &mut texture);
        assert_eq!(status, EmberwaveStatus::InvalidArgument);
        assert!(last_error().contains("EMBERWAVE_PLANE_CBCR"));
        // Cb and Cr interleaved is for encoders only.
        let status = emberwave_texture_create_plane(decoder, EMBERWAVE_PLANE_CBCR, &mut texture);
        assert_eq!(status, EmberwaveStatus::InvalidArgument);
    }

    // SAFETY: live, destroyed once each.
    unsafe {
        for texture in [y, cb, cr, picture, other_y, elsewhere_y] {
            emberwave_texture_destroy(texture);
        }
        emberwave_decoder_destroy(decoder);
        emberwave_decoder_destroy(other_decoder);
        emberwave_decoder_destroy(elsewhere);
        emberwave_encoder_destroy(encoder);
        emberwave_context_destroy(other_context);
        emberwave_context_destroy(context);
    }
}

/// A picture of `width` x `height` pixels in RGB, four bytes each in
/// `order`, rows `stride` bytes apart with 0xEE between them: gradients,
/// edges and noise.
fn rgb_picture(width: u32, height: u32, stride: usize, order: RgbOrder) -> Vec<u8> {
    let (width, height) = (width as usize, height as usize);
    let mut state = 0x9E37_79B9_u32;
    let mut bytes = vec![0xEE; stride * (height - 1) + 4 * width];
    for y in 0..height {
        for x in 0..width {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let [r, g, b] = if (x / 9 + y / 7) % 2 == 0 {
                [(x * 255 / width) as u8, (y * 255 / height) as u8, 200]
            } else {
                [state as u8, (state >> 8) as u8, (state >> 16) as u8]
            };
            let pixel = match order {
                RgbOrder::Rgba => [r, g, b, (state >> 24) as u8],
                RgbOrder::Bgra => [b, g, r, (state >> 24) as u8],
                other => panic!("{other:?}"),
            };
            bytes[y * stride + 4 * x..][..4].copy_from_slice(&pixel);
        }
    }
    bytes
}

/// Encodes a picture in RGB in CPU memory through the C API: the status, and
/// the frame or, after a failure, what the size out-parameter holds.
fn encode_rgb_c(
    encoder: *mut EmberwaveEncoder,
    rgb: EmberwaveSourcePlane,
    format: u32,
    budget: usize,
) -> (EmberwaveStatus, Vec<u8>, usize) {
    let mut frame = vec![0u8; budget];
    let mut size = usize::MAX;
    // SAFETY: a live encoder, a picture of its size or NULL, and a frame of
    // `budget` bytes.
    let status = unsafe {
        emberwave_encoder_encode_rgb(encoder, rgb, format, frame.as_mut_ptr(), budget, &mut size)
    };
    if status == EmberwaveStatus::Ok {
        frame.truncate(size);
    } else {
        frame.clear();
    }
    (status, frame, size)
}

#[test]
fn pictures_in_rgb_encode_as_the_planes_they_convert_to() {
    // RGB in CPU memory in either order, with bytes between the rows,
    // encodes to the frame of the planes that rgb_to_ycbcr8 makes of it with
    // the encoder's signalling (decision D65).
    let Some(context) = context() else {
        return;
    };
    let other = EmberwaveSignalling {
        matrix: EMBERWAVE_MATRIX_BT2020,
        range: EMBERWAVE_RANGE_FULL,
        chroma_siting: EMBERWAVE_SITING_LEFT,
        primaries: EMBERWAVE_PRIMARIES_BT2020,
        transfer: EMBERWAVE_TRANSFER_PQ,
        ..Default::default()
    };
    let other_params = ColorParams::new(ColorMatrix::Bt2020, YcbcrRange::Full, ChromaSiting::Left);
    for (width, height, chroma) in SHAPES {
        let layout = FrameLayout::new(width, height, chroma).unwrap();
        for (signalling, params) in [(None, ColorParams::default()), (Some(&other), other_params)] {
            let row = 4 * width as usize;
            let rgba = rgb_picture(width, height, row, RgbOrder::Rgba);
            let planes =
                rgb_to_ycbcr8(&rgba, row, RgbOrder::Rgba, width, height, chroma, &params).unwrap();
            let from_planes = encoder_for(context, &layout, signalling);
            let mut storage = Vec::new();
            let (status, expected, _) = encode_c(
                from_planes,
                source_planes(&layout, &planes, 0, &mut storage),
                9_000,
            );
            assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
            for (format, order) in [
                (EMBERWAVE_TEXTURE_RGBA8, RgbOrder::Rgba),
                (EMBERWAVE_TEXTURE_BGRA8, RgbOrder::Bgra),
            ] {
                let from_rgb = encoder_for(context, &layout, signalling);
                let stride = row + 8;
                let bytes = rgb_picture(width, height, stride, order);
                let rgb = EmberwaveSourcePlane {
                    data: bytes.as_ptr(),
                    stride,
                    size: bytes.len(),
                };
                let (status, frame, _) = encode_rgb_c(from_rgb, rgb, format, 9_000);
                assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
                assert!(
                    frame == expected,
                    "{width}x{height} {chroma:?} {order:?} {signalling:?}"
                );
                // SAFETY: live, destroyed once.
                unsafe { emberwave_encoder_destroy(from_rgb) };
            }
            // SAFETY: live, destroyed once.
            unsafe { emberwave_encoder_destroy(from_planes) };
        }
    }
    // SAFETY: live, destroyed once.
    unsafe { emberwave_context_destroy(context) };
}

#[test]
fn a_picture_a_decode_left_in_a_texture_encodes_as_the_same_picture_in_memory() {
    // A decode's picture texture is an RGB picture like any other for an
    // encoder: it encodes as the bytes of the same decode in CPU memory do.
    let Some(context) = context() else {
        return;
    };
    for (width, height, chroma) in SHAPES {
        let layout = FrameLayout::new(width, height, chroma).unwrap();
        let maker = encoder_for(context, &layout, None);
        let mut storage = Vec::new();
        let (status, frame, _) = encode_c(
            maker,
            source_planes(&layout, &picture(&layout, 6), 0, &mut storage),
            9_000,
        );
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
        let decoder = decoder_for(context, &layout);
        let (w, h) = (width as usize, height as usize);
        let mut rgb = vec![0u8; 3 * w * h];
        for format in [EMBERWAVE_TEXTURE_RGBA8, EMBERWAVE_TEXTURE_BGRA8] {
            let mut texture = ptr::null_mut();
            // SAFETY: a live decoder, its texture, a frame of the length
            // passed, and a picture of its size.
            unsafe {
                let status = emberwave_texture_create(decoder, format, &mut texture);
                assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
                for into_texture in [true, false] {
                    emberwave_decoder_clear(decoder);
                    let status =
                        emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len());
                    assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
                    let status = if into_texture {
                        emberwave_decoder_decode_texture(decoder, ptr::null(), texture)
                    } else {
                        let plane = EmberwavePlane {
                            data: rgb.as_mut_ptr(),
                            stride: 3 * w,
                            size: rgb.len(),
                        };
                        emberwave_decoder_decode_rgb(decoder, ptr::null(), plane)
                    };
                    assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
                }
            }
            let rgba: Vec<u8> = rgb
                .chunks_exact(3)
                .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 255])
                .collect();
            let (from_texture, from_memory) = (
                encoder_for(context, &layout, None),
                encoder_for(context, &layout, None),
            );
            let (status, frame, _) = encode_textures_c(
                from_texture,
                [texture, ptr::null_mut(), ptr::null_mut()].map(|t| t.cast_const()),
                9_000,
            );
            assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
            let picture = EmberwaveSourcePlane {
                data: rgba.as_ptr(),
                stride: 4 * w,
                size: rgba.len(),
            };
            let (status, expected, _) =
                encode_rgb_c(from_memory, picture, EMBERWAVE_TEXTURE_RGBA8, 9_000);
            assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
            assert!(
                frame == expected,
                "{width}x{height} {chroma:?} format {format}"
            );
            // SAFETY: live, destroyed once each.
            unsafe {
                emberwave_texture_destroy(texture);
                emberwave_encoder_destroy(from_texture);
                emberwave_encoder_destroy(from_memory);
            }
        }
        // SAFETY: live, destroyed once each.
        unsafe {
            emberwave_decoder_destroy(decoder);
            emberwave_encoder_destroy(maker);
        }
    }
    // SAFETY: live, destroyed once.
    unsafe { emberwave_context_destroy(context) };
}

#[test]
fn rgb_pictures_an_encode_cannot_read_are_refused() {
    let Some(context) = context() else {
        return;
    };
    let layout = FrameLayout::new(64, 32, ChromaSubsampling::Yuv420).unwrap();
    let encoder = encoder_for(context, &layout, None);
    let bytes = rgb_picture(64, 32, 256, RgbOrder::Bgra);
    let refused = |rgb: EmberwaveSourcePlane, format: u32, says: &str| {
        let (status, frame, size) = encode_rgb_c(encoder, rgb, format, 9_000);
        assert_eq!(status, EmberwaveStatus::InvalidArgument, "{says}");
        assert!(frame.is_empty() && size == 0, "{says}");
        let error = last_error();
        assert!(error.contains(says), "{says}: {error}");
    };
    let good = EmberwaveSourcePlane {
        data: bytes.as_ptr(),
        stride: 256,
        size: bytes.len(),
    };
    refused(
        good,
        7,
        "format is 7, not EMBERWAVE_TEXTURE_RGBA8 or EMBERWAVE_TEXTURE_BGRA8",
    );
    refused(
        EmberwaveSourcePlane {
            data: ptr::null(),
            stride: 256,
            size: 0,
        },
        EMBERWAVE_TEXTURE_BGRA8,
        "rgb.data is NULL",
    );
    refused(
        EmberwaveSourcePlane {
            data: bytes.as_ptr(),
            stride: 255,
            size: bytes.len(),
        },
        EMBERWAVE_TEXTURE_BGRA8,
        "rgb.stride is 255, less than the 256 bytes of a row",
    );
    // The arguments are checked before anything is encoded: the next frame
    // is the first.
    let (status, frame, _) = encode_rgb_c(encoder, good, EMBERWAVE_TEXTURE_BGRA8, 9_000);
    assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    let Header::StartOfFrame(header) = Header::parse(&frame).unwrap() else {
        panic!("a start of frame header");
    };
    assert_eq!(header.sequence, 1);
    // SAFETY: live, destroyed once each.
    unsafe {
        emberwave_encoder_destroy(encoder);
        emberwave_context_destroy(context);
    }
}

/// A second context, where the first was made.
fn context_or_panic() -> *mut EmberwaveContext {
    context().expect("a second context where there is a first")
}

#[test]
fn source_planes_shorter_than_their_rows_are_refused() {
    let Some(context) = context() else {
        return;
    };
    let layout = FrameLayout::new(200, 120, ChromaSubsampling::Yuv420).unwrap();
    let encoder = encoder_for(context, &layout, None);
    let source = picture(&layout, 7);
    let mut storage = Vec::new();
    let mut planes = source_planes(&layout, &source, 0, &mut storage);
    planes[1].size -= 1;
    let (status, _, _) = encode_c(encoder, planes, 20_000);
    assert_eq!(status, EmberwaveStatus::InvalidArgument);
    assert!(last_error().contains("cb.size"), "{}", last_error());
    planes[1].size += 1;
    let (status, _, _) = encode_c(encoder, planes, 20_000);
    assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    // SAFETY: live, destroyed once each.
    unsafe {
        emberwave_encoder_destroy(encoder);
        emberwave_context_destroy(context);
    }
}
