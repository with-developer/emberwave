//! The C API's textures read through Metal itself, as an app on macOS or iOS
//! reads them: the `id<MTLTexture>` and `id<MTLDevice>` it hands out, a copy
//! on a command queue of the app's own, and the pixels against the RGB, or
//! the planes, the C API gives in CPU memory.

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
    MTLCreateSystemDefaultDevice, MTLDevice, MTLHazardTrackingMode, MTLOrigin, MTLPixelFormat,
    MTLResource, MTLResourceOptions, MTLSize, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
    MTLTextureUsage,
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
    let queue = device.newCommandQueue().expect("a command queue");
    blit(&queue, texture).contents()
}

/// A copy of a texture into a buffer the CPU can read, committed and maybe
/// not done yet.
struct Copy {
    commands: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    size: usize,
}

impl Copy {
    /// The pixels, once the copy is done.
    fn contents(self) -> Vec<u8> {
        self.commands.waitUntilCompleted();
        // SAFETY: shared storage, which the CPU may read once the copy
        // completed.
        unsafe {
            std::slice::from_raw_parts(self.buffer.contents().as_ptr().cast::<u8>(), self.size)
        }
        .to_vec()
    }
}

/// Copies the pixels of `texture` into a buffer the CPU can read with a blit
/// committed to `queue`, a queue of the texture's device, without waiting.
fn blit(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    texture: &ProtocolObject<dyn MTLTexture>,
) -> Copy {
    let (width, height) = (texture.width(), texture.height());
    let texel = if texture.pixelFormat() == MTLPixelFormat::R8Unorm {
        1
    } else {
        4
    };
    let row = width * texel;
    let buffer = texture
        .device()
        .newBufferWithLength_options(row * height, MTLResourceOptions::StorageModeShared)
        .expect("a buffer");
    let commands = queue.commandBuffer().expect("a command buffer");
    let blit = commands.blitCommandEncoder().expect("a blit encoder");
    let origin = MTLOrigin { x: 0, y: 0, z: 0 };
    let size = MTLSize {
        width,
        height,
        depth: 1,
    };
    // SAFETY: the texture is on the queue's device, the region lies inside
    // it, and the buffer holds `height` rows of `row` bytes.
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
            texture, 0, 0, origin, size, &buffer, 0, row, row * height,
        );
    }
    blit.endEncoding();
    commands.commit();
    Copy {
        commands,
        buffer,
        size: row * height,
    }
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

#[test]
fn an_app_reads_the_planes_from_the_metal_textures() {
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
    // SAFETY: a live context.
    let device = unsafe { emberwave_context_metal_device(context) };
    if device.is_null() {
        // Another backend, which EMBERWAVE_BACKEND can choose on a Mac.
        // SAFETY: a live context.
        unsafe { emberwave_context_destroy(context) };
        return;
    }
    // SAFETY: an id<MTLDevice>, which the context keeps.
    let device: &ProtocolObject<dyn MTLDevice> = unsafe { &*device.cast() };
    // 4:2:0, where the chroma textures are half the size of the luma's, and
    // 4:4:4, where only their contents tell the planes apart.
    for name in ["gradient_200x120_420.pwv1", "noise_131x67_444.pwv1"] {
        read_planes(context, device, &testdata(name));
    }
    // SAFETY: a live context.
    unsafe { emberwave_context_destroy(context) };
}

/// Decodes the first frame of the PWV1 file `bytes` into plane textures and
/// checks them, read through Metal, against the planes in CPU memory.
fn read_planes(
    context: *mut EmberwaveContext,
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
) {
    let frame = Pwv1File::parse(bytes)
        .unwrap()
        .frames()
        .next()
        .unwrap()
        .unwrap();
    // SAFETY: live objects, buffers of the sizes passed, and pointers to store
    // results in; the Metal pointers are borrowed while their objects live.
    unsafe {
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
        let mut info = EmberwaveFrameInfo::default();
        assert_eq!(
            emberwave_decoder_info(decoder, &mut info),
            EmberwaveStatus::Ok
        );
        let sizes = [
            (info.width as usize, info.height as usize),
            (info.chroma_width as usize, info.chroma_height as usize),
            (info.chroma_width as usize, info.chroma_height as usize),
        ];

        // The planes in CPU memory, then the same frame into textures.
        let mut planes = sizes.map(|(width, height)| vec![0u8; width * height]);
        let [y, cb, cr] = planes.each_mut().map(|plane| plane.as_mut_ptr());
        let target = |data, (width, _): (usize, usize)| EmberwavePlane {
            data,
            stride: width,
        };
        assert_eq!(
            emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
            EmberwaveStatus::Ok
        );
        assert_eq!(
            emberwave_decoder_decode_yuv(
                decoder,
                target(y, sizes[0]),
                target(cb, sizes[1]),
                target(cr, sizes[2])
            ),
            EmberwaveStatus::Ok
        );
        let textures = [EMBERWAVE_PLANE_Y, EMBERWAVE_PLANE_CB, EMBERWAVE_PLANE_CR].map(|plane| {
            let mut texture = ptr::null_mut();
            assert_eq!(
                emberwave_texture_create_plane(decoder, plane, &mut texture),
                EmberwaveStatus::Ok,
                "{}",
                last_error()
            );
            texture
        });
        assert_eq!(
            emberwave_decoder_decode_plane_textures(decoder, textures[0], textures[1], textures[2]),
            EmberwaveStatus::Ok,
            "{}",
            last_error()
        );

        for ((texture, plane), size) in textures.into_iter().zip(&planes).zip(sizes) {
            let metal: &ProtocolObject<dyn MTLTexture> = &*emberwave_texture_metal(texture).cast();
            assert_eq!((metal.width(), metal.height()), size);
            assert_eq!(metal.pixelFormat(), MTLPixelFormat::R8Unorm);
            // The decodes store the planes into it from their shaders, rather
            // than copy them in (decision D52).
            assert!(
                metal.usage().contains(MTLTextureUsage::ShaderWrite),
                "{:?}",
                metal.usage()
            );
            let on: Retained<ProtocolObject<dyn MTLDevice>> = metal.device();
            assert!(
                ptr::eq(&*on, device),
                "the texture is on the context's device"
            );
            assert_eq!(&read(device, metal), plane);
            emberwave_texture_destroy(texture);
        }
        emberwave_decoder_destroy(decoder);
    }
}

#[test]
fn decodes_on_the_app_queue_need_no_wait() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        assert!(
            std::env::var_os("EMBERWAVE_REQUIRE_GPU").is_none_or(|value| value.is_empty()),
            "no Metal device"
        );
        return;
    };
    let queue = device.newCommandQueue().expect("a command queue");
    let mut context = ptr::null_mut();
    // SAFETY: a Metal device and its queue, and a pointer to store the
    // context in.
    let status = unsafe {
        emberwave_context_create_metal(
            Retained::as_ptr(&device).cast_mut().cast(),
            Retained::as_ptr(&queue).cast_mut().cast(),
            &mut context,
        )
    };
    assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    // SAFETY: a live context.
    let on = unsafe { emberwave_context_metal_device(context) };
    assert!(
        ptr::eq(on.cast_const().cast(), Retained::as_ptr(&device)),
        "the context is on the app's device"
    );
    // 192 pixels wide, the picture is copied into its texture; 200, drawn
    // (decision D42).
    for name in ["noise_192x136_420.pwv1", "gradient_200x120_420.pwv1"] {
        decode_on_the_app_queue(context, &queue, &testdata(name));
    }
    // SAFETY: a live context.
    unsafe { emberwave_context_destroy(context) };
}

/// Decodes every frame of the PWV1 file `bytes` into the same textures
/// without waiting, each followed by copies out of them committed to the
/// context's queue, and waits only at the end: each copy finds its frame.
fn decode_on_the_app_queue(
    context: *mut EmberwaveContext,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    bytes: &[u8],
) {
    let frames: Vec<&[u8]> = Pwv1File::parse(bytes)
        .unwrap()
        .frames()
        .collect::<Result<_, _>>()
        .unwrap();
    assert!(frames.len() > 1, "frames that differ");
    // SAFETY: live objects, buffers of the sizes passed, and pointers to store
    // results in; the Metal pointers are borrowed while their objects live.
    unsafe {
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
        let mut info = EmberwaveFrameInfo::default();
        assert_eq!(
            emberwave_decoder_info(decoder, &mut info),
            EmberwaveStatus::Ok
        );
        let (width, height) = (info.width as usize, info.height as usize);
        let sizes = [
            (width, height),
            (info.chroma_width as usize, info.chroma_height as usize),
            (info.chroma_width as usize, info.chroma_height as usize),
        ];

        // What the decodes that wait give: BGRA and the planes.
        let expected: Vec<(Vec<u8>, [Vec<u8>; 3])> = frames
            .iter()
            .map(|frame| {
                let mut rgb = vec![0u8; width * height * 3];
                // The same frames again: forget their order.
                emberwave_decoder_clear(decoder);
                assert_eq!(
                    emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
                    EmberwaveStatus::Ok
                );
                let target = EmberwavePlane {
                    data: rgb.as_mut_ptr(),
                    stride: width * 3,
                };
                assert_eq!(
                    emberwave_decoder_decode_rgb(decoder, ptr::null(), target),
                    EmberwaveStatus::Ok
                );
                let mut planes = sizes.map(|(w, h)| vec![0u8; w * h]);
                let [y, cb, cr] = planes.each_mut().map(|plane| plane.as_mut_ptr());
                let plane = |data, (w, _): (usize, usize)| EmberwavePlane { data, stride: w };
                // The same frames again: forget their order.
                emberwave_decoder_clear(decoder);
                assert_eq!(
                    emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
                    EmberwaveStatus::Ok
                );
                assert_eq!(
                    emberwave_decoder_decode_yuv(
                        decoder,
                        plane(y, sizes[0]),
                        plane(cb, sizes[1]),
                        plane(cr, sizes[2])
                    ),
                    EmberwaveStatus::Ok
                );
                let bgra = rgb
                    .chunks_exact(3)
                    .flat_map(|p| [p[2], p[1], p[0], 255])
                    .collect();
                (bgra, planes)
            })
            .collect();

        let mut texture = ptr::null_mut();
        assert_eq!(
            emberwave_texture_create(decoder, EMBERWAVE_TEXTURE_BGRA8, &mut texture),
            EmberwaveStatus::Ok
        );
        let planes = [EMBERWAVE_PLANE_Y, EMBERWAVE_PLANE_CB, EMBERWAVE_PLANE_CR].map(|plane| {
            let mut texture = ptr::null_mut();
            assert_eq!(
                emberwave_texture_create_plane(decoder, plane, &mut texture),
                EmberwaveStatus::Ok
            );
            texture
        });
        let metal = |texture| -> &ProtocolObject<dyn MTLTexture> {
            &*emberwave_texture_metal(texture).cast()
        };
        // The queue orders the work by the textures only while Metal tracks
        // them, as wgpu-hal 30 has it. A wgpu that stops would fail here.
        for texture in [texture].into_iter().chain(planes) {
            assert_eq!(
                metal(texture).hazardTrackingMode(),
                MTLHazardTrackingMode::Tracked
            );
        }

        let mut copies = Vec::new();
        for frame in &frames {
            emberwave_decoder_clear(decoder);
            assert_eq!(
                emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
                EmberwaveStatus::Ok
            );
            assert_eq!(
                emberwave_decoder_submit_texture(decoder, ptr::null(), texture),
                EmberwaveStatus::Ok,
                "{}",
                last_error()
            );
            let picture = blit(queue, metal(texture));
            emberwave_decoder_clear(decoder);
            assert_eq!(
                emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
                EmberwaveStatus::Ok
            );
            assert_eq!(
                emberwave_decoder_submit_plane_textures(decoder, planes[0], planes[1], planes[2]),
                EmberwaveStatus::Ok,
                "{}",
                last_error()
            );
            let plane_copies = planes.map(|plane| blit(queue, metal(plane)));
            copies.push((picture, plane_copies));
        }
        for (index, ((picture, plane_copies), (bgra, yuv))) in
            copies.into_iter().zip(&expected).enumerate()
        {
            assert!(picture.contents() == *bgra, "frame {index}: the picture");
            for (component, (copy, plane)) in plane_copies.into_iter().zip(yuv).enumerate() {
                assert!(
                    copy.contents() == *plane,
                    "frame {index}: plane {component}"
                );
            }
        }

        for texture in [texture].into_iter().chain(planes) {
            emberwave_texture_destroy(texture);
        }
        emberwave_decoder_destroy(decoder);
    }
}

#[test]
fn calls_on_one_app_queue_take_turns_at_its_command_buffers() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        assert!(
            std::env::var_os("EMBERWAVE_REQUIRE_GPU").is_none_or(|value| value.is_empty()),
            "no Metal device"
        );
        return;
    };
    // Room for an encode's 18 command buffers, the most a call makes, but
    // not for two calls at once (#126, decision D60).
    let queue = device
        .newCommandQueueWithMaxCommandBufferCount(20)
        .expect("a command queue");
    if !holds_to_its_room(&queue, 20) {
        return;
    }
    let contexts: Vec<Shared<EmberwaveContext>> =
        (0..2).map(|_| metal_context(&device, &queue)).collect();
    let bytes = testdata("noise_192x136_420.pwv1");
    let file = Pwv1File::parse(&bytes).unwrap();
    let frame = file.frames().next().unwrap().unwrap().to_vec();

    // On each context a thread that encodes and one that decodes, each into
    // its own objects, all at once. One that waits for good waits for
    // command buffers that the others hold uncommitted, and they for its.
    let threads: Vec<_> = contexts
        .iter()
        .flat_map(|context| [(Shared(context.0), true), (Shared(context.0), false)])
        .map(|(context, encodes)| {
            let frame = frame.clone();
            std::thread::spawn(move || {
                let context = context;
                if encodes {
                    encode_on(context.0);
                } else {
                    decode_on(context.0, &frame);
                }
            })
        })
        .collect();
    finish_within(
        threads,
        "calls on one queue waited for each other's command buffers (#126)",
    );
    for context in contexts {
        // SAFETY: a live context, which no thread uses any more.
        unsafe { emberwave_context_destroy(context.0) };
    }
}

#[test]
fn contexts_hold_no_command_buffer_of_an_app_queue_between_calls() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        assert!(
            std::env::var_os("EMBERWAVE_REQUIRE_GPU").is_none_or(|value| value.is_empty()),
            "no Metal device"
        );
        return;
    };
    let queue = device
        .newCommandQueueWithMaxCommandBufferCount(20)
        .expect("a command queue");
    if !holds_to_its_room(&queue, 20) {
        return;
    }
    // wgpu writes through the queue in making a device, a decoder and an
    // encoder, and holds the writes in a command buffer until a submission:
    // each context made, and each that made a decoder or an encoder, held
    // one of the queue's 20 until its first call (decision D60). So did the
    // making of more contexts than that wait for good, and so did an
    // encode, which needs 18, beside a few such contexts.
    let device_at: Shared<std::ffi::c_void> = Shared(Retained::as_ptr(&device).cast_mut().cast());
    let queue_at: Shared<std::ffi::c_void> = Shared(Retained::as_ptr(&queue).cast_mut().cast());
    let thread = std::thread::spawn(move || {
        let (device_at, queue_at) = (device_at, queue_at);
        let contexts: Vec<*mut EmberwaveContext> = (0..24)
            .map(|_| {
                let mut context = ptr::null_mut();
                // SAFETY: a Metal device and its queue, which the test keeps,
                // and a pointer to store the context in.
                let status = unsafe {
                    emberwave_context_create_metal(
                        device_at.0.cast(),
                        queue_at.0.cast(),
                        &mut context,
                    )
                };
                assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
                context
            })
            .collect();
        let decoders: Vec<*mut EmberwaveDecoder> = contexts[..4]
            .iter()
            .map(|&context| {
                let mut decoder = ptr::null_mut();
                // SAFETY: a live context, and a pointer to store the decoder in.
                let status = unsafe {
                    emberwave_decoder_create(context, 192, 136, EMBERWAVE_CHROMA_420, &mut decoder)
                };
                assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
                decoder
            })
            .collect();
        let encoders: Vec<*mut EmberwaveEncoder> = contexts[5..9]
            .iter()
            .map(|&context| {
                let mut encoder = ptr::null_mut();
                // SAFETY: a live context, and a pointer to store the encoder in.
                let status = unsafe {
                    emberwave_encoder_create(
                        context,
                        192,
                        136,
                        EMBERWAVE_CHROMA_420,
                        ptr::null(),
                        &mut encoder,
                    )
                };
                assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
                encoder
            })
            .collect();
        encode_on(contexts[4]);
        // SAFETY: live objects, which nothing else uses.
        unsafe {
            for decoder in decoders {
                emberwave_decoder_destroy(decoder);
            }
            for encoder in encoders {
                emberwave_encoder_destroy(encoder);
            }
            for context in contexts {
                emberwave_context_destroy(context);
            }
        }
    });
    finish_within(
        vec![thread],
        "contexts held command buffers of the queue between calls",
    );
}

/// A pointer to hand to another thread: the C API lets several threads use
/// one context, and Metal's devices and command queues are thread-safe.
struct Shared<T>(*mut T);

// SAFETY: as above.
unsafe impl<T> Send for Shared<T> {}

/// A context on `queue` of `device`.
fn metal_context(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>,
) -> Shared<EmberwaveContext> {
    let mut context = ptr::null_mut();
    // SAFETY: a Metal device and its queue, and a pointer to store the
    // context in.
    let status = unsafe {
        emberwave_context_create_metal(
            Retained::as_ptr(device).cast_mut().cast(),
            Retained::as_ptr(queue).cast_mut().cast(),
            &mut context,
        )
    };
    assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    Shared(context)
}

/// Whether `queue`, made with room for `room` command buffers, makes the one
/// past them wait, as Metal does on a Mac. The GPU of a virtual machine may
/// not, and then no test of the room tells anything: it says so and passes.
fn holds_to_its_room(queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>, room: usize) -> bool {
    let held: Vec<_> = (0..room)
        .map(|_| queue.commandBuffer().expect("a command buffer"))
        .collect();
    let at = Shared(Retained::as_ptr(queue).cast_mut());
    let (made, waited) = std::sync::mpsc::channel();
    let past = std::thread::spawn(move || {
        let at = at;
        // SAFETY: the queue, which the caller keeps until this thread ends.
        let queue = unsafe { &*at.0 };
        let commands = queue.commandBuffer().expect("a command buffer");
        let _ = made.send(());
        commands.commit();
    });
    let holds = waited
        .recv_timeout(std::time::Duration::from_millis(500))
        .is_err();
    for commands in held {
        commands.commit();
    }
    past.join().unwrap();
    if !holds {
        eprintln!(
            "this Metal queue does not make a command buffer past its room wait: not tested here"
        );
    }
    holds
}

/// Waits for `threads` to end, at most 60 s, and fails with `stuck` if one
/// has not, or with what a thread panicked with.
fn finish_within(threads: Vec<std::thread::JoinHandle<()>>, stuck: &str) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while threads.iter().any(|thread| !thread.is_finished()) {
        assert!(std::time::Instant::now() < deadline, "{stuck}");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    for thread in threads {
        if let Err(panic) = thread.join() {
            std::panic::resume_unwind(panic);
        }
    }
}

/// Encodes a 192x136 picture 20 times with an encoder of `context`.
fn encode_on(context: *mut EmberwaveContext) {
    let (width, height) = (192, 136);
    let y: Vec<u8> = (0..width * height).map(|i| (i * 7 % 251) as u8).collect();
    let chroma = vec![128u8; width * height / 4];
    let mut frame = vec![0u8; 100_000];
    // SAFETY: a live context, planes of the sizes and strides passed, a
    // frame buffer of `budget` bytes, and pointers to store results in.
    unsafe {
        let mut encoder = ptr::null_mut();
        assert_eq!(
            emberwave_encoder_create(
                context,
                width as u32,
                height as u32,
                EMBERWAVE_CHROMA_420,
                ptr::null(),
                &mut encoder
            ),
            EmberwaveStatus::Ok,
            "{}",
            last_error()
        );
        let plane = |data: &[u8], stride| EmberwaveSourcePlane {
            data: data.as_ptr(),
            stride,
        };
        for _ in 0..20 {
            let mut size = 0;
            assert_eq!(
                emberwave_encoder_encode(
                    encoder,
                    plane(&y, width),
                    plane(&chroma, width / 2),
                    plane(&chroma, width / 2),
                    frame.as_mut_ptr(),
                    frame.len(),
                    &mut size
                ),
                EmberwaveStatus::Ok,
                "{}",
                last_error()
            );
        }
        emberwave_encoder_destroy(encoder);
    }
}

/// Decodes `frame`, 192x136 4:2:0, 20 times into a texture with a decoder of
/// `context`: by turns a decode that waits and one that does not.
fn decode_on(context: *mut EmberwaveContext, frame: &[u8]) {
    // SAFETY: a live context, a frame of the length passed, and pointers to
    // store results in.
    unsafe {
        let mut decoder = ptr::null_mut();
        assert_eq!(
            emberwave_decoder_create(context, 192, 136, EMBERWAVE_CHROMA_420, &mut decoder),
            EmberwaveStatus::Ok,
            "{}",
            last_error()
        );
        let mut texture = ptr::null_mut();
        assert_eq!(
            emberwave_texture_create(decoder, EMBERWAVE_TEXTURE_BGRA8, &mut texture),
            EmberwaveStatus::Ok
        );
        for round in 0..20 {
            emberwave_decoder_clear(decoder);
            assert_eq!(
                emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
                EmberwaveStatus::Ok
            );
            let status = if round % 2 == 0 {
                emberwave_decoder_decode_texture(decoder, ptr::null(), texture)
            } else {
                emberwave_decoder_submit_texture(decoder, ptr::null(), texture)
            };
            assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
        }
        emberwave_texture_destroy(texture);
        emberwave_decoder_destroy(decoder);
    }
}

/// A texture of the app's own, made as a `CVMetalTextureCache` makes one:
/// shared storage, `usage`, and tracked unless `untracked`, as a heap's are.
fn app_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    format: MTLPixelFormat,
    (width, height): (usize, usize),
    usage: MTLTextureUsage,
    untracked: bool,
) -> Retained<ProtocolObject<dyn MTLTexture>> {
    // SAFETY: a 2D texture of a pixel format and a size Metal takes.
    let descriptor = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            format, width, height, false,
        )
    };
    descriptor.setUsage(usage);
    descriptor.setStorageMode(MTLStorageMode::Shared);
    if untracked {
        descriptor.setHazardTrackingMode(MTLHazardTrackingMode::Untracked);
    }
    device
        .newTextureWithDescriptor(&descriptor)
        .expect("a texture")
}

/// The pointer the C API takes for `texture`.
fn raw(texture: &Retained<ProtocolObject<dyn MTLTexture>>) -> *mut std::ffi::c_void {
    Retained::as_ptr(texture).cast_mut().cast()
}

#[test]
fn the_apps_own_textures_take_the_picture_and_the_planes() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        assert!(
            std::env::var_os("EMBERWAVE_REQUIRE_GPU").is_none_or(|value| value.is_empty()),
            "no Metal device"
        );
        return;
    };
    let queue = device.newCommandQueue().expect("a command queue");
    let mut context = ptr::null_mut();
    // SAFETY: a Metal device and its queue, and a pointer to store the
    // context in.
    let status = unsafe {
        emberwave_context_create_metal(
            Retained::as_ptr(&device).cast_mut().cast(),
            Retained::as_ptr(&queue).cast_mut().cast(),
            &mut context,
        )
    };
    assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    // 192 pixels wide, a copy fills the picture's texture; 200, a draw
    // (decision D42).
    for name in ["noise_192x136_420.pwv1", "gradient_200x120_420.pwv1"] {
        into_app_textures(context, &device, &queue, &testdata(name));
    }
    // SAFETY: a live context.
    unsafe { emberwave_context_destroy(context) };
}

/// Decodes the frames of `bytes` into textures of the app's own, waiting and
/// not, and checks what the C API refuses to take.
fn into_app_textures(
    context: *mut EmberwaveContext,
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    bytes: &[u8],
) {
    let frames: Vec<&[u8]> = Pwv1File::parse(bytes)
        .unwrap()
        .frames()
        .collect::<Result<_, _>>()
        .unwrap();
    // SAFETY: live objects, buffers of the sizes passed, and pointers to store
    // results in; the Metal textures outlive the C API's textures of them.
    unsafe {
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
        let mut info = EmberwaveFrameInfo::default();
        assert_eq!(
            emberwave_decoder_info(decoder, &mut info),
            EmberwaveStatus::Ok
        );
        let (width, height) = (info.width as usize, info.height as usize);
        let sizes = [
            (width, height),
            (info.chroma_width as usize, info.chroma_height as usize),
            (info.chroma_width as usize, info.chroma_height as usize),
        ];
        let push = |frame: &[u8]| {
            emberwave_decoder_clear(decoder);
            assert_eq!(
                emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
                EmberwaveStatus::Ok
            );
        };
        // What the decodes to CPU memory give: BGRA and the planes.
        let expected: Vec<(Vec<u8>, [Vec<u8>; 3])> = frames[..2]
            .iter()
            .map(|frame| {
                let mut rgb = vec![0u8; width * height * 3];
                push(frame);
                let target = EmberwavePlane {
                    data: rgb.as_mut_ptr(),
                    stride: width * 3,
                };
                assert_eq!(
                    emberwave_decoder_decode_rgb(decoder, ptr::null(), target),
                    EmberwaveStatus::Ok
                );
                let mut planes = sizes.map(|(w, h)| vec![0u8; w * h]);
                let [y, cb, cr] = planes.each_mut().map(|plane| plane.as_mut_ptr());
                let plane = |data, (w, _): (usize, usize)| EmberwavePlane { data, stride: w };
                push(frame);
                assert_eq!(
                    emberwave_decoder_decode_yuv(
                        decoder,
                        plane(y, sizes[0]),
                        plane(cb, sizes[1]),
                        plane(cr, sizes[2])
                    ),
                    EmberwaveStatus::Ok
                );
                let bgra = rgb
                    .chunks_exact(3)
                    .flat_map(|p| [p[2], p[1], p[0], 255])
                    .collect();
                (bgra, planes)
            })
            .collect();

        // The picture. A texture of a CVMetalTextureCache can be read but not
        // drawn into, unless asked for: a copy fills it if it can, and the
        // draw that fills it otherwise needs it to be a render target.
        let copies = (width * 4).is_multiple_of(256);
        let usage = if copies {
            MTLTextureUsage::ShaderRead
        } else {
            MTLTextureUsage::ShaderRead | MTLTextureUsage::RenderTarget
        };
        let picture = app_texture(device, MTLPixelFormat::BGRA8Unorm, sizes[0], usage, false);
        let mut texture = ptr::null_mut();
        assert_eq!(
            emberwave_texture_from_metal(decoder, raw(&picture), &mut texture),
            EmberwaveStatus::Ok,
            "{}",
            last_error()
        );
        push(frames[0]);
        assert_eq!(
            emberwave_decoder_decode_texture(decoder, ptr::null(), texture),
            EmberwaveStatus::Ok,
            "{}",
            last_error()
        );
        assert!(read(device, &picture) == expected[0].0, "the picture");
        // Without a wait, read by the app's next command buffer.
        push(frames[1]);
        assert_eq!(
            emberwave_decoder_submit_texture(decoder, ptr::null(), texture),
            EmberwaveStatus::Ok,
            "{}",
            last_error()
        );
        assert!(
            blit(queue, &picture).contents() == expected[1].0,
            "the picture, submitted"
        );

        // The planes.
        let planes = [EMBERWAVE_PLANE_Y, EMBERWAVE_PLANE_CB, EMBERWAVE_PLANE_CR]
            .into_iter()
            .zip(sizes)
            .map(|(plane, size)| {
                let metal = app_texture(
                    device,
                    MTLPixelFormat::R8Unorm,
                    size,
                    MTLTextureUsage::ShaderRead,
                    false,
                );
                let mut texture = ptr::null_mut();
                assert_eq!(
                    emberwave_texture_from_metal_plane(decoder, plane, raw(&metal), &mut texture),
                    EmberwaveStatus::Ok,
                    "{}",
                    last_error()
                );
                (metal, texture)
            })
            .collect::<Vec<_>>();
        push(frames[0]);
        assert_eq!(
            emberwave_decoder_decode_plane_textures(decoder, planes[0].1, planes[1].1, planes[2].1),
            EmberwaveStatus::Ok,
            "{}",
            last_error()
        );
        for ((metal, _), plane) in planes.iter().zip(&expected[0].1) {
            assert!(read(device, metal) == *plane, "a plane");
        }
        // Textures the app's shaders may write as well: the decodes store the
        // planes into them (decision D52), waiting and not.
        let writable = [EMBERWAVE_PLANE_Y, EMBERWAVE_PLANE_CB, EMBERWAVE_PLANE_CR]
            .into_iter()
            .zip(sizes)
            .map(|(plane, size)| {
                let usage = MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite;
                let metal = app_texture(device, MTLPixelFormat::R8Unorm, size, usage, false);
                let mut texture = ptr::null_mut();
                assert_eq!(
                    emberwave_texture_from_metal_plane(decoder, plane, raw(&metal), &mut texture),
                    EmberwaveStatus::Ok,
                    "{}",
                    last_error()
                );
                (metal, texture)
            })
            .collect::<Vec<_>>();
        push(frames[0]);
        assert_eq!(
            emberwave_decoder_decode_plane_textures(
                decoder,
                writable[0].1,
                writable[1].1,
                writable[2].1
            ),
            EmberwaveStatus::Ok,
            "{}",
            last_error()
        );
        for ((metal, _), plane) in writable.iter().zip(&expected[0].1) {
            assert!(read(device, metal) == *plane, "a plane stored into");
        }
        push(frames[1]);
        assert_eq!(
            emberwave_decoder_submit_plane_textures(
                decoder,
                writable[0].1,
                writable[1].1,
                writable[2].1
            ),
            EmberwaveStatus::Ok,
            "{}",
            last_error()
        );
        for ((metal, _), plane) in writable.iter().zip(&expected[1].1) {
            assert!(
                blit(queue, metal).contents() == *plane,
                "a plane stored into, submitted"
            );
        }
        let shared = app_texture(
            device,
            MTLPixelFormat::R8Unorm,
            sizes[1],
            MTLTextureUsage::ShaderRead,
            false,
        );
        let twice = [EMBERWAVE_PLANE_CB, EMBERWAVE_PLANE_CR].map(|plane| {
            let mut texture = ptr::null_mut();
            assert_eq!(
                emberwave_texture_from_metal_plane(decoder, plane, raw(&shared), &mut texture),
                EmberwaveStatus::Ok,
                "{}",
                last_error()
            );
            texture
        });
        push(frames[0]);
        assert_eq!(
            emberwave_decoder_decode_plane_textures(decoder, planes[0].1, twice[0], twice[1]),
            EmberwaveStatus::InvalidArgument
        );
        assert!(
            last_error().contains("same Metal texture"),
            "{}",
            last_error()
        );
        assert_eq!(
            emberwave_decoder_submit_plane_textures(decoder, planes[0].1, twice[0], twice[1]),
            EmberwaveStatus::InvalidArgument
        );
        assert!(
            last_error().contains("same Metal texture"),
            "{}",
            last_error()
        );
        assert!(
            emberwave_decoder_is_ready(decoder, 0),
            "the frame is still there"
        );
        for texture in twice {
            emberwave_texture_destroy(texture);
        }

        // A texture Metal does not track: a waiting decode takes it, a
        // decode that does not wait cannot be ordered.
        let untracked = app_texture(device, MTLPixelFormat::BGRA8Unorm, sizes[0], usage, true);
        let mut loose = ptr::null_mut();
        assert_eq!(
            emberwave_texture_from_metal(decoder, raw(&untracked), &mut loose),
            EmberwaveStatus::Ok,
            "{}",
            last_error()
        );
        push(frames[0]);
        assert_eq!(
            emberwave_decoder_decode_texture(decoder, ptr::null(), loose),
            EmberwaveStatus::Ok
        );
        assert!(read(device, &untracked) == expected[0].0, "untracked");
        assert_eq!(
            emberwave_decoder_submit_texture(decoder, ptr::null(), loose),
            EmberwaveStatus::InvalidArgument
        );
        assert!(last_error().contains("track"), "{}", last_error());

        // What is refused.
        let ink = MTLTextureUsage::ShaderRead | MTLTextureUsage::RenderTarget;
        let wrong_size = app_texture(device, MTLPixelFormat::BGRA8Unorm, (64, 64), ink, false);
        let wrong_format = app_texture(device, MTLPixelFormat::RGBA16Float, sizes[0], ink, false);
        let mut refused = ptr::null_mut();
        for (metal, complaint) in [
            (raw(&wrong_size), "rather than the picture's"),
            (raw(&wrong_format), "pixel format"),
            (ptr::null_mut(), "metal is NULL"),
        ] {
            assert_eq!(
                emberwave_texture_from_metal(decoder, metal, &mut refused),
                EmberwaveStatus::InvalidArgument
            );
            assert!(last_error().contains(complaint), "{}", last_error());
            assert!(refused.is_null());
        }
        assert_eq!(
            emberwave_texture_from_metal_plane(
                decoder,
                EMBERWAVE_PLANE_Y,
                raw(&picture),
                &mut refused
            ),
            EmberwaveStatus::InvalidArgument
        );
        assert!(
            last_error().contains("MTLPixelFormatR8Unorm"),
            "{}",
            last_error()
        );
        if !copies {
            // Neither a copy nor a draw can fill a texture that is only read.
            let read_only = app_texture(
                device,
                MTLPixelFormat::BGRA8Unorm,
                sizes[0],
                MTLTextureUsage::ShaderRead,
                false,
            );
            assert_eq!(
                emberwave_texture_from_metal(decoder, raw(&read_only), &mut refused),
                EmberwaveStatus::InvalidArgument
            );
            assert!(
                last_error().contains("neither a copy nor a draw"),
                "{}",
                last_error()
            );
        }

        for texture in [texture, loose]
            .into_iter()
            .chain(planes.iter().map(|(_, texture)| *texture))
            .chain(writable.iter().map(|(_, texture)| *texture))
        {
            emberwave_texture_destroy(texture);
        }
        emberwave_decoder_destroy(decoder);
    }
}
