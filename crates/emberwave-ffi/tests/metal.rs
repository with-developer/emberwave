//! The C API's textures read through Metal itself, as an app on macOS or iOS
//! reads them: the `id<MTLTexture>` and `id<MTLDevice>` it hands out, a copy
//! on a command queue of the app's own, and the pixels against the RGB the C
//! API gives in CPU memory.

#![cfg(target_os = "macos")]

use std::ffi::CStr;
use std::path::PathBuf;
use std::ptr;

use emberwave_core::bitstream::Pwv1File;
use emberwave_ffi::*;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLDevice, MTLOrigin, MTLPixelFormat, MTLResource, MTLResourceOptions, MTLSize, MTLTexture,
};

fn testdata(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn last_error() -> String {
    // SAFETY: emberwave_last_error always returns a C string.
    unsafe { CStr::from_ptr(emberwave_last_error()) }
        .to_string_lossy()
        .into_owned()
}

/// The pixels of `texture`, copied into a buffer the CPU can read by a blit
/// on a queue made here, as an app's own would be.
fn read(
    device: &ProtocolObject<dyn MTLDevice>,
    texture: &ProtocolObject<dyn MTLTexture>,
) -> Vec<u8> {
    let (width, height) = (texture.width(), texture.height());
    let row = width * 4;
    let buffer = device
        .newBufferWithLength_options(row * height, MTLResourceOptions::StorageModeShared)
        .expect("a buffer");
    let queue = device.newCommandQueue().expect("a command queue");
    let commands = queue.commandBuffer().expect("a command buffer");
    let blit = commands.blitCommandEncoder().expect("a blit encoder");
    let origin = MTLOrigin { x: 0, y: 0, z: 0 };
    let size = MTLSize {
        width,
        height,
        depth: 1,
    };
    // SAFETY: the texture is on `device`, the region lies inside it, and the
    // buffer holds `height` rows of `row` bytes.
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
            texture, 0, 0, origin, size, &buffer, 0, row, row * height,
        );
    }
    blit.endEncoding();
    commands.commit();
    commands.waitUntilCompleted();
    // SAFETY: shared storage, which the CPU may read once the copy completed.
    unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u8>(), row * height) }
        .to_vec()
}

#[test]
fn an_app_reads_the_picture_from_the_metal_texture() {
    let mut context = ptr::null_mut();
    // SAFETY: NULL options and a pointer to store the context in.
    if unsafe { emberwave_context_create(ptr::null(), &mut context) } != EmberwaveStatus::Ok {
        assert!(
            std::env::var_os("EMBERWAVE_REQUIRE_GPU").is_none_or(|value| value.is_empty()),
            "no GPU: {}",
            last_error()
        );
        return;
    }
    let bytes = testdata("noise_131x67_444.pwv1");
    let frames: Vec<&[u8]> = Pwv1File::parse(&bytes)
        .unwrap()
        .frames()
        .collect::<Result<_, _>>()
        .unwrap();
    // SAFETY: live objects, buffers of the sizes passed, and pointers to store
    // results in; the Metal pointers are borrowed while their objects live.
    unsafe {
        let device = emberwave_context_metal_device(context);
        if device.is_null() {
            // Another backend, which EMBERWAVE_BACKEND can choose on a Mac.
            emberwave_context_destroy(context);
            return;
        }
        let device: &ProtocolObject<dyn MTLDevice> = &*device.cast();
        let mut stream = EmberwaveStreamInfo::default();
        assert_eq!(
            emberwave_probe_pwv1(bytes.as_ptr(), bytes.len(), &mut stream),
            EmberwaveStatus::Ok
        );
        let mut decoder = ptr::null_mut();
        assert_eq!(
            emberwave_decoder_create(
                context,
                stream.width,
                stream.height,
                stream.chroma,
                &mut decoder
            ),
            EmberwaveStatus::Ok
        );
        let (width, height) = (stream.width as usize, stream.height as usize);

        for (frame, format, pixel_format) in [
            (
                frames[0],
                EMBERWAVE_TEXTURE_RGBA8,
                MTLPixelFormat::RGBA8Unorm,
            ),
            (
                frames[1],
                EMBERWAVE_TEXTURE_BGRA8,
                MTLPixelFormat::BGRA8Unorm,
            ),
        ] {
            let mut texture = ptr::null_mut();
            assert_eq!(
                emberwave_texture_create(decoder, format, &mut texture),
                EmberwaveStatus::Ok,
                "{}",
                last_error()
            );
            assert_eq!(
                emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
                EmberwaveStatus::Ok
            );
            let mut rgb = vec![0u8; width * height * 3];
            let plane = EmberwavePlane {
                data: rgb.as_mut_ptr(),
                stride: width * 3,
            };
            assert_eq!(
                emberwave_decoder_decode_rgb(decoder, ptr::null(), plane),
                EmberwaveStatus::Ok
            );
            assert_eq!(
                emberwave_decoder_decode_texture(decoder, ptr::null(), texture),
                EmberwaveStatus::Ok,
                "{}",
                last_error()
            );

            let metal: &ProtocolObject<dyn MTLTexture> = &*emberwave_texture_metal(texture).cast();
            assert_eq!((metal.width(), metal.height()), (width, height));
            assert_eq!(metal.pixelFormat(), pixel_format);
            let on: Retained<ProtocolObject<dyn MTLDevice>> = metal.device();
            assert!(
                ptr::eq(&*on, device),
                "the texture is on the context's device"
            );

            let expected: Vec<u8> = rgb
                .chunks_exact(3)
                .flat_map(|p| match format {
                    EMBERWAVE_TEXTURE_RGBA8 => [p[0], p[1], p[2], 255],
                    _ => [p[2], p[1], p[0], 255],
                })
                .collect();
            assert_eq!(read(device, metal), expected, "format {format}");
            emberwave_texture_destroy(texture);
        }
        emberwave_decoder_destroy(decoder);
        emberwave_context_destroy(context);
    }
}
