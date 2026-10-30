//! The C API's Direct3D 12 interop, used as a Windows host or client uses
//! it: a Direct3D 11 device on the same adapter shares textures and a fence
//! through NT handles, and orders its work against emberwave's with the
//! fence (decision D63). WARP, Direct3D's software adapter, stands in for a
//! GPU here, as in CI.

#![cfg(windows)]

use std::ffi::CStr;
use std::path::PathBuf;
use std::ptr;

use emberwave_core::bitstream::Pwv1File;
use emberwave_ffi::*;
use windows::Win32::Foundation::{CloseHandle, GENERIC_ALL, HANDLE, HMODULE};
use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL_11_1};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BIND_FLAG, D3D11_BIND_RENDER_TARGET, D3D11_BIND_SHADER_RESOURCE, D3D11_CPU_ACCESS_READ,
    D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_FENCE_FLAG_SHARED, D3D11_MAP_READ,
    D3D11_MAPPED_SUBRESOURCE, D3D11_RESOURCE_MISC_SHARED, D3D11_RESOURCE_MISC_SHARED_NTHANDLE,
    D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT, D3D11_USAGE_STAGING,
    D3D11CreateDevice, ID3D11Device, ID3D11Device5, ID3D11DeviceContext4, ID3D11Fence,
    ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, DXGI_FORMAT_R8_UNORM,
    DXGI_FORMAT_R8G8_UNORM, DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, DXGI_SHARED_RESOURCE_READ, DXGI_SHARED_RESOURCE_WRITE, IDXGIAdapter,
    IDXGIFactory4, IDXGIResource1,
};
use windows::core::{Interface, PCWSTR};

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

/// A Direct3D 11 device of a host's or a client's on WARP, an emberwave
/// context on the same adapter, and a fence the two share.
struct Host {
    device: ID3D11Device,
    immediate: ID3D11DeviceContext4,
    fence: ID3D11Fence,
    /// The fence, opened by the context.
    shared_fence: *mut EmberwaveD3D12Fence,
    context: *mut EmberwaveContext,
    /// The fence's last value.
    value: u64,
}

impl Host {
    /// The host. WARP is part of Windows, so the tests never skip, and a
    /// pass means they ran.
    fn new() -> Self {
        Self::make().expect("a Direct3D 11.4 device on WARP, with a shared fence")
    }

    fn make() -> Option<Self> {
        // SAFETY: DXGI and Direct3D 11 called as documented; every out
        // parameter is a local that outlives its call.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().ok()?;
            let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
            let (mut device, mut immediate) = (None, None);
            D3D11CreateDevice(
                &adapter,
                D3D_DRIVER_TYPE_UNKNOWN,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                Some(&[D3D_FEATURE_LEVEL_11_1]),
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut immediate),
            )
            .ok()?;
            let device = device?;
            let immediate: ID3D11DeviceContext4 = immediate?.cast().ok()?;
            let luid = adapter.GetDesc().ok()?.AdapterLuid;

            let mut context = ptr::null_mut();
            let status = emberwave_context_create_d3d12(luid.LowPart, luid.HighPart, &mut context);
            assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());

            let mut fence: Option<ID3D11Fence> = None;
            device
                .cast::<ID3D11Device5>()
                .ok()?
                .CreateFence(0, D3D11_FENCE_FLAG_SHARED, &mut fence)
                .ok()?;
            let fence = fence?;
            let handle = fence
                .CreateSharedHandle(None, GENERIC_ALL.0, PCWSTR::null())
                .expect("a handle of the fence");
            let mut shared_fence = ptr::null_mut();
            let status = emberwave_d3d12_fence_open(context, handle.0, &mut shared_fence);
            assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
            // The context keeps a fence of its own.
            CloseHandle(handle).expect("closed");
            Some(Self {
                device,
                immediate,
                fence,
                shared_fence,
                context,
                value: 0,
            })
        }
    }

    /// A shared texture of `format` and `size`, with `bind` flags, and its
    /// NT handle.
    fn texture(
        &self,
        format: DXGI_FORMAT,
        (width, height): (usize, usize),
        bind: D3D11_BIND_FLAG,
    ) -> (ID3D11Texture2D, HANDLE) {
        let description = D3D11_TEXTURE2D_DESC {
            Width: width as u32,
            Height: height as u32,
            MipLevels: 1,
            ArraySize: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: bind.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: (D3D11_RESOURCE_MISC_SHARED_NTHANDLE.0 | D3D11_RESOURCE_MISC_SHARED.0)
                as u32,
        };
        // SAFETY: a valid description, and a local for the texture.
        unsafe {
            let mut texture = None;
            self.device
                .CreateTexture2D(&description, None, Some(&mut texture))
                .unwrap_or_else(|error| panic!("a shared texture of format {}: {error}", format.0));
            let texture: ID3D11Texture2D = texture.expect("a texture");
            let handle = texture
                .cast::<IDXGIResource1>()
                .expect("a DXGI resource")
                .CreateSharedHandle(
                    None,
                    DXGI_SHARED_RESOURCE_READ.0 | DXGI_SHARED_RESOURCE_WRITE.0,
                    PCWSTR::null(),
                )
                .expect("a handle of the texture");
            (texture, handle)
        }
    }

    /// Writes `samples`, rows of `row` bytes, into `texture` after the
    /// fence's last value, and signals the next.
    fn write(&mut self, texture: &ID3D11Texture2D, samples: &[u8], row: usize) -> u64 {
        // SAFETY: a texture of the device as large as the rows, and the
        // device's own fence.
        unsafe {
            self.immediate
                .Wait(&self.fence, self.value)
                .expect("a wait");
            self.immediate.UpdateSubresource(
                texture,
                0,
                None,
                samples.as_ptr().cast(),
                row as u32,
                0,
            );
            self.signal()
        }
    }

    /// Signals the fence's next value after the work so far, and returns it.
    fn signal(&mut self) -> u64 {
        self.signal_to(self.value + 1);
        self.value
    }

    /// Signals the fence to `value` after the work so far.
    fn signal_to(&mut self, value: u64) {
        self.value = value;
        // SAFETY: the device's own fence.
        unsafe {
            self.immediate.Signal(&self.fence, value).expect("a signal");
            self.immediate.Flush();
        }
    }

    /// The value the fence has reached.
    fn completed(&self) -> u64 {
        // SAFETY: the device's own fence.
        unsafe { self.fence.GetCompletedValue() }
    }

    /// The texels of `texture`, `texel` bytes each, read once the fence has
    /// reached `after`, or with no wait.
    fn read(&self, texture: &ID3D11Texture2D, texel: usize, after: Option<u64>) -> Vec<u8> {
        // SAFETY: a staging copy of the texture's description, mapped for
        // reading once the copy is done, which Map waits for.
        unsafe {
            let mut description = D3D11_TEXTURE2D_DESC::default();
            texture.GetDesc(&mut description);
            description.Usage = D3D11_USAGE_STAGING;
            description.BindFlags = 0;
            description.CPUAccessFlags = D3D11_CPU_ACCESS_READ.0 as u32;
            description.MiscFlags = 0;
            let mut staging = None;
            self.device
                .CreateTexture2D(&description, None, Some(&mut staging))
                .expect("a staging texture");
            let staging: ID3D11Texture2D = staging.expect("a texture");
            if let Some(value) = after {
                self.immediate.Wait(&self.fence, value).expect("a wait");
            }
            self.immediate.CopyResource(&staging, texture);
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            self.immediate
                .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .expect("mapped");
            let (width, height) = (description.Width as usize, description.Height as usize);
            let mut texels = Vec::with_capacity(width * height * texel);
            for y in 0..height {
                let row = std::slice::from_raw_parts(
                    mapped.pData.cast::<u8>().add(y * mapped.RowPitch as usize),
                    width * texel,
                );
                texels.extend_from_slice(row);
            }
            self.immediate.Unmap(&staging, 0);
            texels
        }
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        // SAFETY: live objects of the C API, destroyed once.
        unsafe {
            emberwave_d3d12_fence_destroy(self.shared_fence);
            emberwave_context_destroy(self.context);
        }
    }
}

/// Wraps the texture of `handle` with `wrap`, then closes the handle.
fn wrapped(
    handle: HANDLE,
    wrap: impl FnOnce(*mut std::ffi::c_void, &mut *mut EmberwaveTexture) -> EmberwaveStatus,
) -> Result<*mut EmberwaveTexture, (EmberwaveStatus, String)> {
    let mut texture = ptr::null_mut();
    let status = wrap(handle.0, &mut texture);
    let error = last_error();
    // SAFETY: a handle of this process, which the texture does not need.
    unsafe { CloseHandle(handle) }.expect("closed");
    if status == EmberwaveStatus::Ok {
        Ok(texture)
    } else {
        assert!(texture.is_null());
        Err((status, error))
    }
}

/// The frames of the PWV1 file `name`, and its picture size.
fn stream(name: &str) -> (Vec<Vec<u8>>, (usize, usize)) {
    let bytes = testdata(name);
    let file = Pwv1File::parse(&bytes).unwrap();
    let frames = file.frames().map(|frame| frame.unwrap().to_vec()).collect();
    let header = file.header();
    (
        frames,
        (header.stream.width as usize, header.stream.height as usize),
    )
}

/// A decoder of `context` for 4:2:0 pictures of `size`.
fn decoder_of(
    context: *mut EmberwaveContext,
    (width, height): (usize, usize),
) -> *mut EmberwaveDecoder {
    let mut decoder = ptr::null_mut();
    // SAFETY: a live context, and a pointer to store the decoder in.
    let status = unsafe {
        emberwave_decoder_create(
            context,
            width as u32,
            height as u32,
            EMBERWAVE_CHROMA_420,
            &mut decoder,
        )
    };
    assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    decoder
}

/// Pushes `frame` into `decoder`, after clearing what it had.
fn push(decoder: *mut EmberwaveDecoder, frame: &[u8]) {
    // SAFETY: a live decoder and a frame of the length passed.
    unsafe {
        emberwave_decoder_clear(decoder);
        let status = emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len());
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    }
}

/// The picture of `frame` as the decode that waits gives it, in BGRA, and
/// its Y, Cb and Cr planes.
fn expected(
    decoder: *mut EmberwaveDecoder,
    frame: &[u8],
    (width, height): (usize, usize),
) -> (Vec<u8>, [Vec<u8>; 3]) {
    let mut rgb = vec![0u8; width * height * 3];
    let sizes = [
        (width, height),
        (width / 2, height / 2),
        (width / 2, height / 2),
    ];
    let mut planes = sizes.map(|(w, h)| vec![0u8; w * h]);
    // SAFETY: a live decoder, and buffers of the sizes and strides passed.
    unsafe {
        push(decoder, frame);
        let target = EmberwavePlane {
            data: rgb.as_mut_ptr(),
            stride: width * 3,
        };
        let status = emberwave_decoder_decode_rgb(decoder, ptr::null(), target);
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
        push(decoder, frame);
        let [y, cb, cr] = [0, 1, 2].map(|i| EmberwavePlane {
            data: planes[i].as_mut_ptr(),
            stride: sizes[i].0,
        });
        let status = emberwave_decoder_decode_yuv(decoder, y, cb, cr);
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    }
    let bgra = rgb
        .chunks_exact(3)
        .flat_map(|p| [p[2], p[1], p[0], 255])
        .collect();
    (bgra, planes)
}

#[test]
fn a_client_reads_pictures_and_planes_after_the_fence() {
    let mut host = Host::new();
    // 192 pixels wide, a copy fills the picture's texture; 200, a draw.
    for name in ["noise_192x136_420.pwv1", "gradient_200x120_420.pwv1"] {
        let (frames, size) = stream(name);
        let decoder = decoder_of(host.context, size);
        let chroma = (size.0 / 2, size.1 / 2);
        let readable = D3D11_BIND_FLAG(D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0);
        let (picture, handle) = host.texture(DXGI_FORMAT_B8G8R8A8_UNORM, size, readable);
        // SAFETY: a live decoder, a handle of this process, and a pointer to
        // store the texture in.
        let picture_texture = wrapped(handle, |handle, texture| unsafe {
            emberwave_texture_from_d3d12(decoder, handle, texture)
        })
        .unwrap();
        let planes = [size, chroma, chroma].map(|plane_size| {
            host.texture(DXGI_FORMAT_R8_UNORM, plane_size, D3D11_BIND_SHADER_RESOURCE)
        });
        let names = [EMBERWAVE_PLANE_Y, EMBERWAVE_PLANE_CB, EMBERWAVE_PLANE_CR];
        let plane_textures: Vec<*mut EmberwaveTexture> = planes
            .iter()
            .zip(names)
            .map(|((_, handle), plane)| {
                // SAFETY: as above.
                wrapped(*handle, |handle, texture| unsafe {
                    emberwave_texture_from_d3d12_plane(decoder, plane, handle, texture)
                })
                .unwrap()
            })
            .collect();

        for frame in &frames[..2] {
            let (bgra, yuv) = expected(decoder, frame, size);
            // The decode waits for the client's reads of the last picture,
            // and the client's reads wait for the decode.
            push(decoder, frame);
            let sync = EmberwaveD3D12Sync {
                wait_fence: host.shared_fence,
                wait_value: host.value,
                signal_fence: host.shared_fence,
                signal_value: host.value + 1,
            };
            host.value += 1;
            // SAFETY: a live decoder and texture, and live fences.
            let status = unsafe {
                emberwave_decoder_submit_texture_d3d12(decoder, ptr::null(), picture_texture, &sync)
            };
            assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
            assert!(
                host.read(&picture, 4, Some(host.value)) == bgra,
                "{name}: the picture after the fence"
            );

            push(decoder, frame);
            let sync = EmberwaveD3D12Sync {
                wait_fence: host.shared_fence,
                wait_value: host.value,
                signal_fence: host.shared_fence,
                signal_value: host.value + 1,
            };
            host.value += 1;
            // SAFETY: a live decoder and textures, and live fences.
            let status = unsafe {
                emberwave_decoder_submit_plane_textures_d3d12(
                    decoder,
                    plane_textures[0],
                    plane_textures[1],
                    plane_textures[2],
                    &sync,
                )
            };
            assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
            for ((texture, _), expected) in planes.iter().zip(&yuv) {
                assert!(
                    host.read(texture, 1, Some(host.value)) == *expected,
                    "{name}: a plane after the fence"
                );
            }
        }

        // A decode waits for its fence: until the client signals it, the
        // texture keeps the last picture and the decode signals nothing.
        let (last, _) = expected(decoder, &frames[1], size);
        let (next, _) = expected(decoder, &frames[2], size);
        push(decoder, &frames[2]);
        let held = host.value + 1;
        let sync = EmberwaveD3D12Sync {
            wait_fence: host.shared_fence,
            wait_value: held,
            signal_fence: host.shared_fence,
            signal_value: held + 1,
        };
        // SAFETY: a live decoder and texture, and live fences.
        let status = unsafe {
            emberwave_decoder_submit_texture_d3d12(decoder, ptr::null(), picture_texture, &sync)
        };
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert!(
            host.completed() < held + 1,
            "{name}: the decode signalled before its wait was met"
        );
        assert!(
            host.read(&picture, 4, None) == last,
            "{name}: the decode wrote the texture before its wait was met"
        );
        host.signal_to(held);
        host.value = held + 1;
        assert!(
            host.read(&picture, 4, Some(held + 1)) == next,
            "{name}: the picture once the wait was met"
        );
        if size.0 % 64 != 0 {
            // A picture a decode draws needs a texture it can draw into.
            let (_texture, handle) =
                host.texture(DXGI_FORMAT_B8G8R8A8_UNORM, size, D3D11_BIND_SHADER_RESOURCE);
            // SAFETY: a live decoder, a handle of this process, and a pointer
            // to store the texture in.
            let (status, error) = wrapped(handle, |handle, texture| unsafe {
                emberwave_texture_from_d3d12(decoder, handle, texture)
            })
            .unwrap_err();
            assert_eq!(status, EmberwaveStatus::InvalidArgument);
            assert!(error.contains("ALLOW_RENDER_TARGET"), "{error}");
        }

        // The decodes that wait leave textures in other states, and refuse
        // the shared ones.
        push(decoder, &frames[0]);
        // SAFETY: a live decoder and texture.
        let status =
            unsafe { emberwave_decoder_decode_texture(decoder, ptr::null(), picture_texture) };
        assert_eq!(status, EmberwaveStatus::InvalidArgument);
        assert!(last_error().contains("_d3d12"), "{}", last_error());
        push(decoder, &frames[0]);
        // SAFETY: a live decoder and textures.
        let status = unsafe {
            emberwave_decoder_decode_plane_textures(
                decoder,
                plane_textures[0],
                plane_textures[1],
                plane_textures[2],
            )
        };
        assert_eq!(status, EmberwaveStatus::InvalidArgument);
        assert!(last_error().contains("_d3d12"), "{}", last_error());

        // SAFETY: live objects, destroyed once each.
        unsafe {
            emberwave_texture_destroy(picture_texture);
            for texture in plane_textures {
                emberwave_texture_destroy(texture);
            }
            emberwave_decoder_destroy(decoder);
        }
    }
}

/// An encode's arguments, for a thread of its own.
struct Held {
    encoder: *mut EmberwaveEncoder,
    textures: [*mut EmberwaveTexture; 3],
    sync: EmberwaveD3D12Sync,
}

// SAFETY: the C API's objects may move between threads, and the test's
// thread does not use them while the encode's does.
unsafe impl Send for Held {}

#[test]
fn a_hosts_shared_planes_are_encoded_after_the_fence() {
    let mut host = Host::new();
    let (size, chroma) = ((192usize, 136usize), (96usize, 68usize));
    let mut encoders = [ptr::null_mut(); 3];
    for encoder in &mut encoders {
        // SAFETY: a live context, and a pointer to store the encoder in.
        let status = unsafe {
            emberwave_encoder_create(
                host.context,
                size.0 as u32,
                size.1 as u32,
                EMBERWAVE_CHROMA_420,
                ptr::null(),
                encoder,
            )
        };
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    }
    let [memory, as_planes, as_nv12] = encoders;
    let planes = [size, chroma, chroma].map(|plane_size| {
        host.texture(DXGI_FORMAT_R8_UNORM, plane_size, D3D11_BIND_SHADER_RESOURCE)
    });
    let interleaved = host.texture(DXGI_FORMAT_R8G8_UNORM, chroma, D3D11_BIND_SHADER_RESOURCE);
    let names = [EMBERWAVE_PLANE_Y, EMBERWAVE_PLANE_CB, EMBERWAVE_PLANE_CR];
    let sources: Vec<*mut EmberwaveTexture> = planes
        .iter()
        .zip(names)
        .map(|((_, handle), plane)| {
            // SAFETY: a live encoder, a handle of this process, and a pointer
            // to store the texture in.
            wrapped(*handle, |handle, texture| unsafe {
                emberwave_texture_from_d3d12_source(as_planes, plane, handle, texture)
            })
            .unwrap()
        })
        .collect();
    // SAFETY: as above.
    let cbcr = wrapped(interleaved.1, |handle, texture| unsafe {
        emberwave_texture_from_d3d12_source(as_nv12, EMBERWAVE_PLANE_CBCR, handle, texture)
    })
    .unwrap();

    let encode = |encoder, [y, cb, cr]: [*mut EmberwaveTexture; 3], sync: &EmberwaveD3D12Sync| {
        let mut frame = vec![0u8; 9_000];
        let mut size = 0;
        // SAFETY: a live encoder, textures that are NULL or live, live
        // fences, and a frame of the length passed.
        let status = unsafe {
            emberwave_encoder_encode_textures_d3d12(
                encoder,
                y,
                cb,
                cr,
                sync,
                frame.as_mut_ptr(),
                frame.len(),
                &mut size,
            )
        };
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
        frame.truncate(size);
        frame
    };
    for seed in 0..3u32 {
        let picture: [Vec<u8>; 3] = std::array::from_fn(|plane| {
            let (w, h) = if plane == 0 { size } else { chroma };
            (0..w * h)
                .map(|i| {
                    ((i % w) * 160 / w + (i / w) * 60 / h + 20 * plane + 9 * seed as usize) as u8
                })
                .collect()
        });
        let both: Vec<u8> = picture[1]
            .iter()
            .zip(&picture[2])
            .flat_map(|(&b, &r)| [b, r])
            .collect();
        // The host writes the planes, and its fence tells the encode.
        for ((texture, _), samples) in planes.iter().zip(&picture) {
            let row = samples.len()
                / if samples.len() == size.0 * size.1 {
                    size.1
                } else {
                    chroma.1
                };
            host.write(texture, samples, row);
        }
        host.write(&interleaved.0, &both, chroma.0 * 2);
        let written = host.value;

        let mut frame = vec![0u8; 9_000];
        let mut length = 0;
        let [y, cb, cr] = [0, 1, 2].map(|i| EmberwaveSourcePlane {
            data: picture[i].as_ptr(),
            stride: if i == 0 { size.0 } else { chroma.0 },
        });
        // SAFETY: a live encoder, planes of its sizes, and a frame of the
        // length passed.
        let status = unsafe {
            emberwave_encoder_encode(
                memory,
                y,
                cb,
                cr,
                frame.as_mut_ptr(),
                frame.len(),
                &mut length,
            )
        };
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
        frame.truncate(length);

        if seed == 2 {
            // An encode waits for its fence: until the host signals it, the
            // call does not return.
            let held = host.value + 1;
            let call = Held {
                encoder: as_planes,
                textures: [sources[0], sources[1], sources[2]],
                sync: EmberwaveD3D12Sync {
                    wait_fence: host.shared_fence,
                    wait_value: held,
                    signal_fence: host.shared_fence,
                    signal_value: held + 1,
                },
            };
            let thread = std::thread::spawn(move || {
                let call = call;
                encode(call.encoder, call.textures, &call.sync)
            });
            std::thread::sleep(std::time::Duration::from_millis(300));
            assert!(
                !thread.is_finished(),
                "the encode returned before its wait was met"
            );
            host.signal_to(held);
            host.value = held + 1;
            let planar = thread.join().expect("the encode's thread");
            assert!(planar == frame, "the planes, once the wait was met");
            continue;
        }

        let sync = EmberwaveD3D12Sync {
            wait_fence: host.shared_fence,
            wait_value: written,
            signal_fence: host.shared_fence,
            signal_value: written + 1,
        };
        host.value = written + 1;
        let planar = encode(as_planes, [sources[0], sources[1], sources[2]], &sync);
        assert!(planar == frame, "the planes, picture {seed}");
        let sync = EmberwaveD3D12Sync {
            wait_fence: host.shared_fence,
            wait_value: host.value,
            signal_fence: host.shared_fence,
            signal_value: host.value + 1,
        };
        host.value += 1;
        let nv12 = encode(as_nv12, [sources[0], cbcr, ptr::null_mut()], &sync);
        assert!(nv12 == frame, "NV12's layout, picture {seed}");
    }

    // What an encode cannot take.
    let (wrong, handle) = host.texture(DXGI_FORMAT_R8G8_UNORM, size, D3D11_BIND_SHADER_RESOURCE);
    // SAFETY: as above.
    let (status, error) = wrapped(handle, |handle, texture| unsafe {
        emberwave_texture_from_d3d12_source(memory, EMBERWAVE_PLANE_Y, handle, texture)
    })
    .unwrap_err();
    assert_eq!(status, EmberwaveStatus::InvalidArgument);
    assert!(error.contains("DXGI_FORMAT_R8_UNORM"), "{error}");
    drop(wrong);
    let (wrong, handle) = host.texture(DXGI_FORMAT_R8_UNORM, chroma, D3D11_BIND_SHADER_RESOURCE);
    // SAFETY: as above.
    let (status, error) = wrapped(handle, |handle, texture| unsafe {
        emberwave_texture_from_d3d12_source(memory, EMBERWAVE_PLANE_Y, handle, texture)
    })
    .unwrap_err();
    assert_eq!(status, EmberwaveStatus::InvalidArgument);
    assert!(error.contains("rather than the plane's 192x136"), "{error}");
    drop(wrong);
    // The encodes that take no fences refuse the shared textures.
    let mut frame = vec![0u8; 9_000];
    let mut length = 0;
    // SAFETY: a live encoder, live textures, and a frame of the length
    // passed.
    let status = unsafe {
        emberwave_encoder_encode_textures(
            as_planes,
            sources[0],
            sources[1],
            sources[2],
            frame.as_mut_ptr(),
            frame.len(),
            &mut length,
        )
    };
    assert_eq!(status, EmberwaveStatus::InvalidArgument);
    assert!(last_error().contains("_d3d12"), "{}", last_error());

    // SAFETY: live objects, destroyed once each.
    unsafe {
        for texture in sources.into_iter().chain([cbcr]) {
            emberwave_texture_destroy(texture);
        }
        for encoder in encoders {
            emberwave_encoder_destroy(encoder);
        }
    }
}

#[test]
fn a_hosts_shared_bgra_picture_is_encoded_as_rgb() {
    // A picture in DXGI_FORMAT_B8G8R8A8_UNORM, as a desktop duplication
    // gives it, encodes after the host's fence as the same bytes in CPU
    // memory do (decision D65).
    let mut host = Host::new();
    let size = (192usize, 136usize);
    let mut encoders = [ptr::null_mut(); 2];
    for encoder in &mut encoders {
        // SAFETY: a live context, and a pointer to store the encoder in.
        let status = unsafe {
            emberwave_encoder_create(
                host.context,
                size.0 as u32,
                size.1 as u32,
                EMBERWAVE_CHROMA_420,
                ptr::null(),
                encoder,
            )
        };
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    }
    let [memory, from_texture] = encoders;
    let (texture, handle) =
        host.texture(DXGI_FORMAT_B8G8R8A8_UNORM, size, D3D11_BIND_SHADER_RESOURCE);
    // SAFETY: a live encoder, a handle of this process, and a pointer to
    // store the texture in.
    let rgb = wrapped(handle, |handle, texture| unsafe {
        emberwave_texture_from_d3d12_source(from_texture, EMBERWAVE_PLANE_RGB, handle, texture)
    })
    .unwrap();
    for seed in 0..2usize {
        let bgra: Vec<u8> = (0..size.0 * size.1)
            .flat_map(|i| {
                let (x, y) = (i % size.0, i / size.0);
                [
                    (40 + y * 150 / size.1 + 7 * seed) as u8,
                    (20 + x * 200 / size.0) as u8,
                    (230 - x * 100 / size.0) as u8,
                    (x * 13 + y) as u8,
                ]
            })
            .collect();
        let written = host.write(&texture, &bgra, 4 * size.0);

        let mut expected = vec![0u8; 9_000];
        let mut length = 0;
        let picture = EmberwaveSourcePlane {
            data: bgra.as_ptr(),
            stride: 4 * size.0,
        };
        // SAFETY: a live encoder, a picture of its size, and a frame of the
        // length passed.
        let status = unsafe {
            emberwave_encoder_encode_rgb(
                memory,
                picture,
                EMBERWAVE_TEXTURE_BGRA8,
                expected.as_mut_ptr(),
                expected.len(),
                &mut length,
            )
        };
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
        expected.truncate(length);

        let sync = EmberwaveD3D12Sync {
            wait_fence: host.shared_fence,
            wait_value: written,
            signal_fence: host.shared_fence,
            signal_value: written + 1,
        };
        host.value = written + 1;
        let mut frame = vec![0u8; 9_000];
        // SAFETY: a live encoder, a live texture, live fences, and a frame
        // of the length passed.
        let status = unsafe {
            emberwave_encoder_encode_textures_d3d12(
                from_texture,
                rgb,
                ptr::null(),
                ptr::null(),
                &sync,
                frame.as_mut_ptr(),
                frame.len(),
                &mut length,
            )
        };
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
        frame.truncate(length);
        assert!(frame == expected, "picture {seed}");
    }

    // An sRGB texture, whose reads would undo the transfer function, is
    // refused.
    let (srgb, handle) = host.texture(
        DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
        size,
        D3D11_BIND_SHADER_RESOURCE,
    );
    // SAFETY: as above.
    let (status, error) = wrapped(handle, |handle, texture| unsafe {
        emberwave_texture_from_d3d12_source(memory, EMBERWAVE_PLANE_RGB, handle, texture)
    })
    .unwrap_err();
    assert_eq!(status, EmberwaveStatus::InvalidArgument);
    assert!(
        error.contains("not DXGI_FORMAT_B8G8R8A8_UNORM or DXGI_FORMAT_R8G8B8A8_UNORM"),
        "{error}"
    );
    drop(srgb);
    // SAFETY: live objects, destroyed once each.
    unsafe {
        emberwave_texture_destroy(rgb);
        for encoder in encoders {
            emberwave_encoder_destroy(encoder);
        }
    }
}

#[test]
fn contexts_need_a_direct3d_12_adapter_and_fences_of_their_own() {
    let mut context = ptr::null_mut();
    // SAFETY: a pointer to store the context in.
    let status = unsafe { emberwave_context_create_d3d12(0, 0, &mut context) };
    assert_eq!(status, EmberwaveStatus::NoGpu, "{}", last_error());
    assert!(context.is_null());
    let host = Host::new();
    let mut fence = ptr::null_mut();
    // SAFETY: a live context, and a pointer to store the fence in.
    let status = unsafe { emberwave_d3d12_fence_open(host.context, ptr::null_mut(), &mut fence) };
    assert_eq!(status, EmberwaveStatus::InvalidArgument);
    assert!(fence.is_null());
    // A handle of a texture is no fence's.
    let (_texture, handle) =
        host.texture(DXGI_FORMAT_R8_UNORM, (64, 64), D3D11_BIND_SHADER_RESOURCE);
    // SAFETY: as above, and a handle of this process.
    let status = unsafe { emberwave_d3d12_fence_open(host.context, handle.0, &mut fence) };
    // SAFETY: a handle of this process.
    unsafe { CloseHandle(handle) }.expect("closed");
    assert_eq!(status, EmberwaveStatus::InvalidArgument, "{}", last_error());
    assert!(fence.is_null());
}
