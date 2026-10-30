//! The C API, called the way C calls it, through raw pointers, and checked
//! against the Rust decoder it wraps.
//!
//! The tests that need a GPU skip without one, unless `EMBERWAVE_REQUIRE_GPU`
//! is set, like those of `emberwave-core`.

use std::ffi::CStr;
use std::path::PathBuf;
use std::ptr;

use emberwave_core::bitstream::consts::DECOMPOSITION_LEVELS;
use emberwave_core::bitstream::{FrameLayout, Header, Pwv1File};
use emberwave_core::color::ColorParams;
use emberwave_core::gpu::{Decoder, GpuContext, GpuOptions};
use emberwave_ffi::*;

fn testdata(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn frames(bytes: &[u8]) -> Vec<&[u8]> {
    Pwv1File::parse(bytes)
        .unwrap()
        .frames()
        .collect::<Result<_, _>>()
        .unwrap()
}

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
    assert!(context.is_null());
    if std::env::var_os("EMBERWAVE_REQUIRE_GPU").is_some_and(|value| !value.is_empty()) {
        panic!("no GPU: {}", last_error());
    }
    eprintln!("skipping: no GPU ({})", last_error());
    None
}

/// A decoder for the stream in `bytes`, made through the C API.
fn decoder_for(context: *mut EmberwaveContext, bytes: &[u8]) -> *mut EmberwaveDecoder {
    let mut stream = EmberwaveStreamInfo::default();
    let mut decoder = ptr::null_mut();
    // SAFETY: valid buffers, a live context and pointers to store results.
    unsafe {
        assert_eq!(
            emberwave_probe_pwv1(bytes.as_ptr(), bytes.len(), &mut stream),
            EmberwaveStatus::Ok
        );
        let status = emberwave_decoder_create(
            context,
            stream.width,
            stream.height,
            stream.chroma,
            &mut decoder,
        );
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    }
    decoder
}

/// Memory for a plane of `width` x `height` bytes at a stride of `width +
/// padding`, filled with a marker that decoding must leave alone in the
/// padding.
struct Target {
    bytes: Vec<u8>,
    width: usize,
    height: usize,
    stride: usize,
}

const MARKER: u8 = 0xA5;

impl Target {
    fn new(width: usize, height: usize, padding: usize) -> Self {
        let stride = width + padding;
        Self {
            bytes: vec![MARKER; stride * height],
            width,
            height,
            stride,
        }
    }

    fn plane(&mut self) -> EmberwavePlane {
        EmberwavePlane {
            data: self.bytes.as_mut_ptr(),
            stride: self.stride,
        }
    }

    /// The rows without the padding, which must still hold the marker.
    fn rows(&self) -> Vec<u8> {
        let mut rows = Vec::with_capacity(self.width * self.height);
        for row in self.bytes.chunks(self.stride) {
            rows.extend_from_slice(&row[..self.width]);
            assert!(
                row[self.width..].iter().all(|&b| b == MARKER),
                "padding written"
            );
        }
        rows
    }
}

#[test]
fn the_version_is_the_crate_version() {
    // SAFETY: a static C string.
    let version = unsafe { CStr::from_ptr(emberwave_version()) };
    assert_eq!(version.to_str(), Ok(env!("CARGO_PKG_VERSION")));
}

#[test]
fn a_pwv1_header_and_a_first_packet_describe_the_stream() {
    let bytes = testdata("noise_192x136_420.pwv1");
    let mut info = EmberwaveStreamInfo::default();
    // SAFETY: valid buffers.
    let status = unsafe { emberwave_probe_pwv1(bytes.as_ptr(), bytes.len(), &mut info) };
    assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    assert_eq!(
        (info.width, info.height, info.chroma),
        (192, 136, EMBERWAVE_CHROMA_420)
    );
    assert_eq!((info.frame_rate_num, info.frame_rate_den), (30, 1));
    // The samples are in video range, which the file header says.
    assert_eq!(
        (info.matrix, info.range, info.chroma_siting),
        (
            EMBERWAVE_MATRIX_BT709,
            EMBERWAVE_RANGE_LIMITED,
            EMBERWAVE_SITING_CENTER
        )
    );

    // Every frame starts with a start of frame header with the same picture
    // size and chroma format. Its colour signalling is the encoder's, and
    // upstream's Metal encoder cannot set any: its frames say BT.709, full
    // range and centred chroma, whatever the content (testdata/README.md).
    let frame = frames(&bytes)[0];
    let mut from_packet = EmberwaveStreamInfo::default();
    // SAFETY: valid buffers.
    let status = unsafe { emberwave_probe_packet(frame.as_ptr(), frame.len(), &mut from_packet) };
    assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    assert_eq!(
        (from_packet.width, from_packet.height, from_packet.chroma),
        (info.width, info.height, info.chroma)
    );
    assert_eq!(
        (
            from_packet.matrix,
            from_packet.range,
            from_packet.chroma_siting
        ),
        (
            EMBERWAVE_MATRIX_BT709,
            EMBERWAVE_RANGE_FULL,
            EMBERWAVE_SITING_CENTER
        )
    );
    assert_eq!(
        (from_packet.frame_rate_num, from_packet.frame_rate_den),
        (0, 0)
    );

    // The block after the 8 byte start of frame header is no start of frame.
    // SAFETY: valid buffers.
    let status =
        unsafe { emberwave_probe_packet(frame[8..].as_ptr(), frame.len() - 8, &mut from_packet) };
    assert_eq!(status, EmberwaveStatus::Bitstream);
}

#[test]
fn bad_arguments_are_errors_with_a_message_not_crashes() {
    let bytes = testdata("noise_192x136_420.pwv1");
    let mut info = EmberwaveStreamInfo::default();
    // SAFETY: every pointer is valid where its size says so, or NULL.
    unsafe {
        assert_eq!(
            emberwave_probe_pwv1(c"not a file".as_ptr().cast(), 10, &mut info),
            EmberwaveStatus::Bitstream
        );
        assert!(!last_error().is_empty());
        assert_eq!(
            emberwave_probe_pwv1(ptr::null(), 24, &mut info),
            EmberwaveStatus::InvalidArgument
        );
        assert_eq!(
            emberwave_probe_pwv1(ptr::null(), 0, &mut info),
            EmberwaveStatus::Bitstream
        );
        assert_eq!(
            emberwave_probe_pwv1(bytes.as_ptr(), bytes.len(), ptr::null_mut()),
            EmberwaveStatus::InvalidArgument
        );
        assert_eq!(
            emberwave_context_create(ptr::null(), ptr::null_mut()),
            EmberwaveStatus::InvalidArgument
        );

        let mut context = ptr::null_mut();
        for options in [
            EmberwaveContextOptions {
                backends: 1 << 7,
                flags: 0,
            },
            EmberwaveContextOptions {
                backends: EMBERWAVE_BACKEND_ANY,
                flags: 1 << 5,
            },
        ] {
            assert_eq!(
                emberwave_context_create(&options, &mut context),
                EmberwaveStatus::InvalidArgument
            );
            assert!(context.is_null());
        }

        let mut decoder = ptr::null_mut();
        assert_eq!(
            emberwave_decoder_create(ptr::null(), 64, 64, EMBERWAVE_CHROMA_420, &mut decoder),
            EmberwaveStatus::InvalidArgument
        );
        assert!(decoder.is_null());
        assert_eq!(
            emberwave_decoder_push_packet(ptr::null_mut(), bytes.as_ptr(), 8),
            EmberwaveStatus::InvalidArgument
        );
        assert!(!emberwave_decoder_is_ready(ptr::null(), 1));
        assert_eq!(
            emberwave_decoder_set_partial_share(ptr::null_mut(), 0.5),
            EmberwaveStatus::InvalidArgument
        );
        assert_eq!(*emberwave_context_adapter(ptr::null()), 0);
        emberwave_decoder_clear(ptr::null_mut());
        emberwave_decoder_destroy(ptr::null_mut());
        emberwave_context_destroy(ptr::null_mut());
    }
}

#[test]
fn the_last_error_belongs_to_its_thread() {
    // SAFETY: one valid byte, which is not a PWV1 file.
    let status = unsafe {
        emberwave_probe_pwv1(c"x".as_ptr().cast(), 1, &mut EmberwaveStreamInfo::default())
    };
    assert_eq!(status, EmberwaveStatus::Bitstream);
    let here = last_error();
    let there = std::thread::spawn(last_error).join().unwrap();
    assert!(!here.is_empty());
    assert_eq!(there, "");
}

#[test]
fn planes_through_the_c_api_equal_the_rust_decoder() {
    let Some(context) = context() else {
        return;
    };
    for name in ["noise_192x136_420.pwv1", "noise_131x67_444.pwv1"] {
        let bytes = testdata(name);
        let decoder = decoder_for(context, &bytes);
        let mut info = EmberwaveFrameInfo::default();
        // SAFETY: a live decoder.
        unsafe {
            assert_eq!(
                emberwave_decoder_info(decoder, &mut info),
                EmberwaveStatus::Ok
            )
        };

        let rust_context = GpuContext::new(&GpuOptions::default()).unwrap();
        let header = *Pwv1File::parse(&bytes).unwrap().header();
        let mut rust = Decoder::new(
            &rust_context,
            header.stream.width,
            header.stream.height,
            header.stream.chroma,
        )
        .unwrap();

        for frame in frames(&bytes) {
            let (width, height) = (info.width as usize, info.height as usize);
            let (chroma_width, chroma_height) =
                (info.chroma_width as usize, info.chroma_height as usize);
            // Strides wider than the planes, with padding to leave alone.
            let mut targets = [
                Target::new(width, height, 13),
                Target::new(chroma_width, chroma_height, 3),
                Target::new(chroma_width, chroma_height, 1),
            ];
            // SAFETY: a live decoder, and targets as large as their strides
            // say.
            unsafe {
                assert_eq!(
                    emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
                    EmberwaveStatus::Ok
                );
                assert!(emberwave_decoder_is_ready(decoder, 0));
                let [y, cb, cr] = &mut targets;
                let status =
                    emberwave_decoder_decode_yuv(decoder, y.plane(), cb.plane(), cr.plane());
                assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
                assert!(!emberwave_decoder_is_ready(decoder, 0));
            }

            let expected = rust.decode(frame).unwrap();
            for (target, plane) in targets.iter().zip(&expected.planes) {
                assert_eq!(target.rows(), plane.data, "{name}");
            }
        }
        // SAFETY: a live decoder, destroyed once.
        unsafe { emberwave_decoder_destroy(decoder) };
    }
    // SAFETY: a live context, destroyed once.
    unsafe { emberwave_context_destroy(context) };
}

#[test]
fn rgb_through_the_c_api_follows_the_stream_or_the_caller() {
    let Some(context) = context() else {
        return;
    };
    let bytes = testdata("gradient_200x120_420.pwv1");
    let decoder = decoder_for(context, &bytes);
    let frame = frames(&bytes)[0];
    let header = *Pwv1File::parse(&bytes).unwrap().header();
    let rust_context = GpuContext::new(&GpuOptions::default()).unwrap();
    let mut rust = Decoder::new(&rust_context, 200, 120, header.stream.chroma).unwrap();

    let decode = |color: *const EmberwaveColor| {
        let mut target = Target::new(3 * 200, 120, 7);
        // SAFETY: a live decoder and a target as large as its stride says.
        unsafe {
            assert_eq!(
                emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
                EmberwaveStatus::Ok
            );
            let status = emberwave_decoder_decode_rgb(decoder, color, target.plane());
            assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
        }
        target.rows()
    };

    // NULL: what the frame's start of frame header signals.
    let Ok(Header::StartOfFrame(start_of_frame)) = Header::parse(frame) else {
        panic!("a frame starts with a start of frame header");
    };
    let signalled = rust
        .decode_rgb(frame, &ColorParams::from_stream(&start_of_frame))
        .unwrap();
    assert_eq!(decode(ptr::null()), signalled.data);

    // What the file header says, passed on as a caller with a PWV1 file does.
    let mut info = EmberwaveStreamInfo::default();
    // SAFETY: valid buffers.
    unsafe { emberwave_probe_pwv1(bytes.as_ptr(), bytes.len(), &mut info) };
    let from_file = EmberwaveColor {
        matrix: info.matrix,
        range: info.range,
        chroma_siting: info.chroma_siting,
    };
    let expected = rust
        .decode_rgb(frame, &ColorParams::from_stream(&header.stream))
        .unwrap();
    assert_eq!(decode(&from_file), expected.data);

    // The caller's choice, here full range BT.601.
    let chosen = EmberwaveColor {
        matrix: EMBERWAVE_MATRIX_BT601,
        range: EMBERWAVE_RANGE_FULL,
        chroma_siting: EMBERWAVE_SITING_LEFT,
    };
    let expected = rust
        .decode_rgb(
            frame,
            &ColorParams {
                matrix: emberwave_core::color::ColorMatrix::Bt601,
                range: emberwave_core::bitstream::YcbcrRange::Full,
                chroma_siting: emberwave_core::bitstream::ChromaSiting::Left,
            },
        )
        .unwrap();
    let rgb = decode(&chosen);
    assert_eq!(rgb, expected.data);
    assert_ne!(rgb, signalled.data, "the choice must make a difference");

    // SAFETY: a live decoder, a bad colour, then destroyed once.
    unsafe {
        let bad = EmberwaveColor {
            matrix: 9,
            ..chosen
        };
        let mut target = Target::new(3 * 200, 120, 0);
        assert_eq!(
            emberwave_decoder_decode_rgb(decoder, &bad, target.plane()),
            EmberwaveStatus::InvalidArgument
        );
        emberwave_decoder_destroy(decoder);
        emberwave_context_destroy(context);
    }
}

#[test]
fn a_bad_target_is_refused_before_the_frame_is_used() {
    let Some(context) = context() else {
        return;
    };
    let bytes = testdata("noise_192x136_420.pwv1");
    let decoder = decoder_for(context, &bytes);
    let frame = frames(&bytes)[0];
    let mut y = Target::new(192, 136, 0);
    let mut cb = Target::new(96, 68, 0);
    let mut cr = Target::new(96, 68, 0);
    // SAFETY: a live decoder, and targets as large as their strides say,
    // except where the test means them to be wrong.
    unsafe {
        assert_eq!(
            emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
            EmberwaveStatus::Ok
        );
        let short = EmberwavePlane {
            stride: 95,
            ..cb.plane()
        };
        assert_eq!(
            emberwave_decoder_decode_yuv(decoder, y.plane(), short, cr.plane()),
            EmberwaveStatus::InvalidArgument
        );
        assert!(last_error().contains("cb.stride"), "{}", last_error());
        let missing = EmberwavePlane {
            data: ptr::null_mut(),
            stride: 96,
        };
        assert_eq!(
            emberwave_decoder_decode_yuv(decoder, y.plane(), cb.plane(), missing),
            EmberwaveStatus::InvalidArgument
        );

        // The frame is still there to decode.
        assert!(emberwave_decoder_is_ready(decoder, 0));
        assert_eq!(
            emberwave_decoder_decode_yuv(decoder, y.plane(), cb.plane(), cr.plane()),
            EmberwaveStatus::Ok
        );
        emberwave_decoder_destroy(decoder);
        emberwave_context_destroy(context);
    }
}

#[test]
fn a_picture_size_the_format_cannot_have_is_an_invalid_argument() {
    let Some(context) = context() else {
        return;
    };
    // Zero, too large, and odd for 4:2:0.
    for (width, height, chroma) in [
        (0, 0, EMBERWAVE_CHROMA_420),
        (16385, 64, EMBERWAVE_CHROMA_444),
        (191, 100, EMBERWAVE_CHROMA_420),
    ] {
        let mut decoder = ptr::null_mut();
        // SAFETY: a live context and a pointer to store the decoder.
        let status =
            unsafe { emberwave_decoder_create(context, width, height, chroma, &mut decoder) };
        assert_eq!(status, EmberwaveStatus::InvalidArgument, "{width}x{height}");
        assert!(decoder.is_null());
        assert!(
            last_error().contains("invalid dimensions"),
            "{}",
            last_error()
        );
    }
    // SAFETY: a live context, destroyed once.
    unsafe { emberwave_context_destroy(context) };
}

#[test]
fn packets_of_another_stream_are_refused_and_clear_starts_over() {
    let Some(context) = context() else {
        return;
    };
    let bytes = testdata("noise_192x136_420.pwv1");
    let decoder = decoder_for(context, &bytes);
    let other = testdata("gradient_200x120_420.pwv1");
    let other_frame = frames(&other)[0];
    let frame = frames(&bytes)[0];
    // SAFETY: a live decoder and valid buffers.
    unsafe {
        assert_eq!(
            emberwave_decoder_push_packet(decoder, other_frame.as_ptr(), other_frame.len()),
            EmberwaveStatus::Bitstream
        );
        assert!(!last_error().is_empty());
        assert!(!emberwave_decoder_is_ready(decoder, 1));

        assert_eq!(
            emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
            EmberwaveStatus::Ok
        );
        assert!(emberwave_decoder_is_ready(decoder, 0));
        emberwave_decoder_clear(decoder);
        assert!(!emberwave_decoder_is_ready(decoder, 1));
        emberwave_decoder_destroy(decoder);
        emberwave_context_destroy(context);
    }
}

#[test]
fn a_partial_frame_needs_the_share_the_caller_sets() {
    let Some(context) = context() else {
        return;
    };
    // A frame of emberwave's encoder, which sends every block of the
    // coarsest level, without every other packet past that level.
    let bytes = testdata("encoder/gradient_200x120_420.pwv1");
    let decoder = decoder_for(context, &bytes);
    let frame = frames(&bytes)[0];
    let stream = Pwv1File::parse(&bytes).unwrap().header().stream;
    let layout = FrameLayout::new(stream.width, stream.height, stream.chroma).unwrap();
    let coarsest = layout
        .bands()
        .find(|&(_, level, _, _)| level < DECOMPOSITION_LEVELS - 1)
        .map_or(layout.block_count(), |(_, _, _, band)| band.first_block);
    // SAFETY: a live decoder and valid buffers.
    unsafe {
        for (number, packet) in emberwave_core::encode::packets(frame, 64)
            .into_iter()
            .enumerate()
        {
            let mut at = 0;
            let first_block = loop {
                match Header::parse(&packet[at..]) {
                    Ok(Header::StartOfFrame(_)) => at += 8,
                    Ok(Header::Block(block)) => break Some(block.block_index),
                    Err(_) => break None,
                }
            };
            if first_block.is_some_and(|block| block >= coarsest) && number % 2 == 1 {
                continue;
            }
            assert_eq!(
                emberwave_decoder_push_packet(decoder, packet.as_ptr(), packet.len()),
                EmberwaveStatus::Ok
            );
        }
        assert!(!emberwave_decoder_is_ready(decoder, 1), "90 % by default");
        for share in [-0.1, 1.5, f32::NAN] {
            assert_eq!(
                emberwave_decoder_set_partial_share(decoder, share),
                EmberwaveStatus::InvalidArgument
            );
            assert!(!last_error().is_empty());
        }
        assert!(!emberwave_decoder_is_ready(decoder, 1), "nothing changed");
        assert_eq!(
            emberwave_decoder_set_partial_share(decoder, 0.25),
            EmberwaveStatus::Ok
        );
        assert!(emberwave_decoder_is_ready(decoder, 1));
        assert!(!emberwave_decoder_is_ready(decoder, 0));
        // Any value but 0 allows a partial frame, as C's truth does.
        assert!(emberwave_decoder_is_ready(decoder, 2));
        assert!(emberwave_decoder_is_ready(decoder, u32::MAX));
        assert_eq!(
            emberwave_decoder_set_partial_share(decoder, 1.0),
            EmberwaveStatus::Ok
        );
        assert!(!emberwave_decoder_is_ready(decoder, 1));
        emberwave_decoder_destroy(decoder);
        emberwave_context_destroy(context);
    }
}

#[test]
fn a_decoder_outlives_its_context() {
    let Some(context) = context() else {
        return;
    };
    let bytes = testdata("solid_160x96_420.pwv1");
    let decoder = decoder_for(context, &bytes);
    // SAFETY: the context is destroyed once; the decoder stays live.
    unsafe { emberwave_context_destroy(context) };
    let frame = frames(&bytes)[0];
    let mut y = Target::new(160, 96, 0);
    let mut cb = Target::new(80, 48, 0);
    let mut cr = Target::new(80, 48, 0);
    // SAFETY: a live decoder and targets as large as their strides say.
    unsafe {
        assert_eq!(
            emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
            EmberwaveStatus::Ok
        );
        assert_eq!(
            emberwave_decoder_decode_yuv(decoder, y.plane(), cb.plane(), cr.plane()),
            EmberwaveStatus::Ok,
            "{}",
            last_error()
        );
        emberwave_decoder_destroy(decoder);
    }
}

/// Whether `context` runs on Metal, as its adapter's name says.
fn on_metal(context: *const EmberwaveContext) -> bool {
    // SAFETY: a live context, whose adapter name is a C string it keeps.
    unsafe { CStr::from_ptr(emberwave_context_adapter(context)) }
        .to_string_lossy()
        .ends_with("(Metal)")
}

#[test]
fn a_picture_reaches_a_texture_and_its_metal_handle() {
    let Some(context) = context() else {
        return;
    };
    let bytes = testdata("gradient_200x120_420.pwv1");
    let frame = frames(&bytes)[0];
    let decoder = decoder_for(context, &bytes);
    let mut texture = ptr::null_mut();
    // SAFETY: a live context, decoder and texture, and pointers to store
    // results in.
    unsafe {
        assert_eq!(
            emberwave_texture_create(decoder, EMBERWAVE_TEXTURE_RGBA8, &mut texture),
            EmberwaveStatus::Ok,
            "{}",
            last_error()
        );
        assert_eq!(
            emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
            EmberwaveStatus::Ok
        );
        assert_eq!(
            emberwave_decoder_decode_texture(decoder, ptr::null(), texture),
            EmberwaveStatus::Ok,
            "{}",
            last_error()
        );

        // Metal handles exactly when the context runs on Metal.
        let metal = on_metal(context);
        assert_eq!(!emberwave_texture_metal(texture).is_null(), metal);
        assert_eq!(!emberwave_context_metal_device(context).is_null(), metal);
        assert!(emberwave_texture_metal(ptr::null()).is_null());
        assert!(emberwave_context_metal_device(ptr::null()).is_null());

        emberwave_texture_destroy(texture);
        emberwave_texture_destroy(ptr::null_mut());
        emberwave_decoder_destroy(decoder);
        emberwave_context_destroy(context);
    }
}

#[test]
fn a_texture_that_does_not_fit_is_refused_before_the_frame_is_used() {
    let Some(context) = context() else {
        return;
    };
    let Some(other_context) = crate::context() else {
        return;
    };
    let gradient = testdata("gradient_200x120_420.pwv1");
    let solid = testdata("solid_160x96_420.pwv1");
    let frame = frames(&gradient)[0];
    let decoder = decoder_for(context, &gradient);
    let small = decoder_for(context, &solid);
    let elsewhere = decoder_for(other_context, &gradient);
    let texture = |decoder| {
        let mut texture = ptr::null_mut();
        // SAFETY: a live decoder and a pointer to store the texture in.
        let status =
            unsafe { emberwave_texture_create(decoder, EMBERWAVE_TEXTURE_RGBA8, &mut texture) };
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
        texture
    };
    let (fits, too_small, other_device) = (texture(decoder), texture(small), texture(elsewhere));
    // SAFETY: live decoders and textures, and NULL where the test means it.
    unsafe {
        assert_eq!(
            emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
            EmberwaveStatus::Ok
        );
        for (texture, complaint) in [
            (too_small, "160x96"),
            (other_device, "another context"),
            (ptr::null_mut(), "texture is NULL"),
        ] {
            assert_eq!(
                emberwave_decoder_decode_texture(decoder, ptr::null(), texture),
                EmberwaveStatus::InvalidArgument
            );
            assert!(last_error().contains(complaint), "{}", last_error());
        }
        assert_eq!(
            emberwave_decoder_decode_texture(ptr::null_mut(), ptr::null(), fits),
            EmberwaveStatus::InvalidArgument
        );

        // The frame is still there to decode.
        assert!(emberwave_decoder_is_ready(decoder, 0));
        assert_eq!(
            emberwave_decoder_decode_texture(decoder, ptr::null(), fits),
            EmberwaveStatus::Ok,
            "{}",
            last_error()
        );
        for texture in [fits, too_small, other_device] {
            emberwave_texture_destroy(texture);
        }
        for decoder in [decoder, small, elsewhere] {
            emberwave_decoder_destroy(decoder);
        }
        emberwave_context_destroy(context);
        emberwave_context_destroy(other_context);
    }
}

/// Textures for the Y, Cb and Cr planes of `decoder`'s pictures.
fn plane_textures(decoder: *const EmberwaveDecoder) -> [*mut EmberwaveTexture; 3] {
    [EMBERWAVE_PLANE_Y, EMBERWAVE_PLANE_CB, EMBERWAVE_PLANE_CR].map(|plane| {
        let mut texture = ptr::null_mut();
        // SAFETY: a live decoder and a pointer to store the texture in.
        let status = unsafe { emberwave_texture_create_plane(decoder, plane, &mut texture) };
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
        texture
    })
}

#[test]
fn planes_reach_textures_and_their_metal_handles() {
    let Some(context) = context() else {
        return;
    };
    let bytes = testdata("noise_131x67_444.pwv1");
    let frame = frames(&bytes)[0];
    let decoder = decoder_for(context, &bytes);
    let [y, cb, cr] = plane_textures(decoder);
    // SAFETY: a live context, decoder and textures, and pointers to store
    // results in.
    unsafe {
        assert_eq!(
            emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
            EmberwaveStatus::Ok
        );
        assert_eq!(
            emberwave_decoder_decode_plane_textures(decoder, y, cb, cr),
            EmberwaveStatus::Ok,
            "{}",
            last_error()
        );
        let metal = on_metal(context);
        for texture in [y, cb, cr] {
            assert_eq!(!emberwave_texture_metal(texture).is_null(), metal);
            emberwave_texture_destroy(texture);
        }

        let mut texture = ptr::null_mut();
        assert_eq!(
            emberwave_texture_create_plane(decoder, 3, &mut texture),
            EmberwaveStatus::InvalidArgument
        );
        assert!(texture.is_null());
        assert!(last_error().contains("plane is 3"), "{}", last_error());
        emberwave_decoder_destroy(decoder);
        emberwave_context_destroy(context);
    }
}

#[test]
fn plane_textures_that_do_not_fit_are_refused_before_the_frame_is_used() {
    let Some(context) = context() else {
        return;
    };
    let Some(other_context) = crate::context() else {
        return;
    };
    // 4:4:4, where the planes are all of a size and only the textures' planes
    // tell them apart.
    let noise = testdata("noise_131x67_444.pwv1");
    let solid = testdata("solid_160x96_420.pwv1");
    let frame = frames(&noise)[0];
    let decoder = decoder_for(context, &noise);
    let small = decoder_for(context, &solid);
    let elsewhere = decoder_for(other_context, &noise);
    let [y, cb, cr] = plane_textures(decoder);
    let small_planes = plane_textures(small);
    let other_planes = plane_textures(elsewhere);
    let (small_y, other_y) = (small_planes[0], other_planes[0]);
    let mut picture = ptr::null_mut();
    // SAFETY: live decoders and textures, and NULL where the test means it.
    unsafe {
        assert_eq!(
            emberwave_texture_create(decoder, EMBERWAVE_TEXTURE_RGBA8, &mut picture),
            EmberwaveStatus::Ok
        );
        assert_eq!(
            emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
            EmberwaveStatus::Ok
        );
        for ([y, cb, cr], complaint) in [
            ([cb, y, cr], "y is not a texture for the Y plane"),
            ([y, cr, cb], "cb is not a texture for the Cb plane"),
            ([y, cb, cb], "cr is not a texture for the Cr plane"),
            ([picture, cb, cr], "y is not a texture for the Y plane"),
            ([small_y, cb, cr], "160x96"),
            ([other_y, cb, cr], "another context"),
            ([y, cb, ptr::null_mut()], "cr is NULL"),
        ] {
            assert_eq!(
                emberwave_decoder_decode_plane_textures(decoder, y, cb, cr),
                EmberwaveStatus::InvalidArgument
            );
            assert!(last_error().contains(complaint), "{}", last_error());
        }
        assert_eq!(
            emberwave_decoder_decode_plane_textures(ptr::null_mut(), y, cb, cr),
            EmberwaveStatus::InvalidArgument
        );
        // A plane's texture is not the picture's.
        assert_eq!(
            emberwave_decoder_decode_texture(decoder, ptr::null(), y),
            EmberwaveStatus::InvalidArgument
        );
        assert!(last_error().contains("Y plane"), "{}", last_error());

        // The frame is still there to decode.
        assert!(emberwave_decoder_is_ready(decoder, 0));
        assert_eq!(
            emberwave_decoder_decode_plane_textures(decoder, y, cb, cr),
            EmberwaveStatus::Ok,
            "{}",
            last_error()
        );
        for texture in [[y, cb, cr], small_planes, other_planes]
            .into_iter()
            .flatten()
            .chain([picture])
        {
            emberwave_texture_destroy(texture);
        }
        for decoder in [decoder, small, elsewhere] {
            emberwave_decoder_destroy(decoder);
        }
        emberwave_context_destroy(context);
        emberwave_context_destroy(other_context);
    }
}

#[test]
fn a_decode_without_a_wait_needs_the_callers_own_queue() {
    // Only a context on the caller's queue orders the caller's reads after a
    // decode that does not wait (decision D43); emberwave_context_create
    // makes a queue of its own.
    let Some(context) = context() else {
        return;
    };
    let bytes = testdata("gradient_200x120_420.pwv1");
    let frame = frames(&bytes)[0];
    let decoder = decoder_for(context, &bytes);
    let [y, cb, cr] = plane_textures(decoder);
    let mut picture = ptr::null_mut();
    // SAFETY: a live context, decoder and textures, and pointers to store
    // results in.
    unsafe {
        assert_eq!(
            emberwave_texture_create(decoder, EMBERWAVE_TEXTURE_RGBA8, &mut picture),
            EmberwaveStatus::Ok
        );
        assert_eq!(
            emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
            EmberwaveStatus::Ok
        );
        assert_eq!(
            emberwave_decoder_submit_texture(decoder, ptr::null(), picture),
            EmberwaveStatus::InvalidArgument
        );
        assert!(
            last_error().contains("caller's Metal queue"),
            "{}",
            last_error()
        );
        assert_eq!(
            emberwave_decoder_submit_plane_textures(decoder, y, cb, cr),
            EmberwaveStatus::InvalidArgument
        );
        assert!(emberwave_decoder_is_ready(decoder, 0));
        assert_eq!(
            emberwave_decoder_decode_texture(decoder, ptr::null(), picture),
            EmberwaveStatus::Ok
        );
        for texture in [picture, y, cb, cr] {
            emberwave_texture_destroy(texture);
        }
        emberwave_decoder_destroy(decoder);
        emberwave_context_destroy(context);
    }
}

#[test]
fn a_metal_context_needs_a_device_and_a_queue() {
    let mut context = ptr::null_mut();
    // SAFETY: NULL where the test means it, and a pointer to store the
    // context in.
    unsafe {
        assert_eq!(
            emberwave_context_create_metal(ptr::null_mut(), ptr::null_mut(), &mut context),
            EmberwaveStatus::InvalidArgument
        );
        assert!(last_error().contains("device is NULL"), "{}", last_error());
        assert!(context.is_null());
    }
    // Elsewhere there is no Metal: the pointers are not even looked at.
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        let mut marker = 0u8;
        let anything = std::ptr::from_mut(&mut marker).cast();
        // SAFETY: never dereferenced off Apple systems.
        let status = unsafe { emberwave_context_create_metal(anything, anything, &mut context) };
        assert_eq!(status, EmberwaveStatus::NoGpu);
        assert!(context.is_null());
    }
}

#[test]
fn vulkan_contexts_need_their_handles_and_vulkan() {
    let mut context = ptr::null_mut();
    let mut requirements = ptr::null_mut();
    let mut count = 7;
    // SAFETY: NULL where the test means it, and pointers to store results in.
    unsafe {
        assert_eq!(
            emberwave_context_create_vulkan(ptr::null(), &mut context),
            EmberwaveStatus::InvalidArgument
        );
        assert!(context.is_null());
        assert_eq!(
            emberwave_vulkan_requirements_create(ptr::null(), ptr::null_mut(), &mut requirements),
            EmberwaveStatus::InvalidArgument
        );
        assert!(requirements.is_null());
        assert_eq!(
            emberwave_vulkan_instance_extensions(None, ptr::null_mut(), ptr::null_mut()),
            EmberwaveStatus::InvalidArgument
        );
        assert!(emberwave_vulkan_requirements_extensions(ptr::null(), &mut count).is_null());
        assert_eq!(count, 0);
        assert!(emberwave_vulkan_requirements_features(ptr::null()).is_null());
        emberwave_vulkan_requirements_destroy(ptr::null_mut());
    }
    // Where wgpu does not run on Vulkan, the handles are not even looked at.
    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        all(windows, target_pointer_width = "64")
    )))]
    {
        unsafe extern "C" fn nothing(
            _instance: *mut std::ffi::c_void,
            _name: *const std::ffi::c_char,
        ) -> Option<unsafe extern "C" fn()> {
            None
        }
        let mut marker = 0u8;
        let anything = std::ptr::from_mut(&mut marker).cast::<std::ffi::c_void>();
        let instance = EmberwaveVulkanInstance {
            get_instance_proc_addr: Some(nothing),
            instance: anything,
            api_version: 0,
            extensions: ptr::null(),
            extension_count: 0,
        };
        let device = EmberwaveVulkanDevice {
            instance,
            physical_device: anything,
            device: anything,
            extensions: ptr::null(),
            extension_count: 0,
            queue_family_index: 0,
            queue_index: 0,
            lock_queue: None,
            unlock_queue: None,
            user: ptr::null_mut(),
        };
        // SAFETY: never dereferenced where there is no Vulkan.
        unsafe {
            assert_eq!(
                emberwave_vulkan_requirements_create(&instance, anything, &mut requirements),
                EmberwaveStatus::NoGpu
            );
            assert!(requirements.is_null());
            assert_eq!(
                emberwave_context_create_vulkan(&device, &mut context),
                EmberwaveStatus::NoGpu
            );
            assert!(context.is_null());
            assert_eq!(
                emberwave_vulkan_instance_extensions(Some(nothing), ptr::null_mut(), &mut count),
                EmberwaveStatus::NoGpu
            );
        }
    }
}

#[test]
fn direct3d_calls_need_a_context_of_their_own() {
    // No adapter has the LUID 0, and off Windows there is no Direct3D.
    let mut made = ptr::null_mut();
    // SAFETY: a pointer to store the context in.
    let status = unsafe { emberwave_context_create_d3d12(0, 0, &mut made) };
    assert_eq!(status, EmberwaveStatus::NoGpu, "{}", last_error());
    assert!(made.is_null());

    // A context of emberwave_context_create takes no shared handle nor
    // fence, whatever its backend: its calls do not take turns.
    let Some(context) = context() else {
        return;
    };
    let bytes = testdata("gradient_200x120_420.pwv1");
    let decoder = decoder_for(context, &bytes);
    let mut marker = 0u8;
    let handle = std::ptr::from_mut(&mut marker).cast::<std::ffi::c_void>();
    let mut fence = ptr::null_mut();
    let mut texture = ptr::null_mut();
    // SAFETY: a live context and decoder, a handle that is never used, and
    // pointers to store results in.
    unsafe {
        assert_eq!(
            emberwave_d3d12_fence_open(context, handle, &mut fence),
            EmberwaveStatus::InvalidArgument
        );
        assert!(last_error().contains("emberwave_context_create_d3d12"));
        assert!(fence.is_null());
        assert_eq!(
            emberwave_texture_from_d3d12(decoder, handle, &mut texture),
            EmberwaveStatus::InvalidArgument
        );
        assert!(last_error().contains("emberwave_context_create_d3d12"));
        assert!(texture.is_null());
        assert_eq!(
            emberwave_decoder_submit_texture_d3d12(decoder, ptr::null(), ptr::null(), ptr::null()),
            EmberwaveStatus::InvalidArgument
        );
        assert!(last_error().contains("emberwave_context_create_d3d12"));
        emberwave_d3d12_fence_destroy(ptr::null_mut());
        emberwave_decoder_destroy(decoder);
        emberwave_context_destroy(context);
    }
}

#[test]
fn metal_textures_of_the_callers_own_need_metal() {
    let Some(context) = context() else {
        return;
    };
    let bytes = testdata("gradient_200x120_420.pwv1");
    let decoder = decoder_for(context, &bytes);
    let mut texture = ptr::null_mut();
    // SAFETY: a live context and decoder, NULL where the test means it, and a
    // pointer to store the texture in.
    unsafe {
        // On Apple systems a NULL texture is refused; elsewhere there is no
        // Metal, and the pointer is not even looked at.
        let expected = if cfg!(any(target_os = "macos", target_os = "ios")) {
            EmberwaveStatus::InvalidArgument
        } else {
            EmberwaveStatus::NoGpu
        };
        assert_eq!(
            emberwave_texture_from_metal(decoder, ptr::null_mut(), &mut texture),
            expected
        );
        assert!(texture.is_null());
        assert_eq!(
            emberwave_texture_from_metal_plane(decoder, 3, ptr::null_mut(), &mut texture),
            EmberwaveStatus::InvalidArgument
        );
        assert!(last_error().contains("plane is 3"), "{}", last_error());
        assert!(texture.is_null());
        emberwave_decoder_destroy(decoder);
        emberwave_context_destroy(context);
    }
}
