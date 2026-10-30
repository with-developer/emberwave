# emberwave C API

`emberwave-ffi` is the C interface of [emberwave](https://github.com/with-developer/emberwave): it decodes and encodes PyroWave video on the GPU, through Vulkan, Metal or Direct3D 12, for programs in C, C++ or any language that can call C functions: a streaming client that decodes, and a host that encodes. One header, `include/emberwave.h`, and one library.

## Getting it

Each [release](https://github.com/with-developer/emberwave/releases) has an archive `emberwave-ffi-VERSION-TARGET` for x86_64 Linux, Apple Silicon macOS and x86_64 Windows:

| Path | Content |
|---|---|
| `include/emberwave.h` | The header |
| `lib/libemberwave_ffi.so`, `lib/libemberwave_ffi.dylib`, or `lib/emberwave_ffi.dll` with its import library `lib/emberwave_ffi.dll.lib` | The shared library |
| `lib/libemberwave_ffi.a` or `lib/emberwave_ffi.lib` | The static library |
| `lib/native-static-libs.txt` | The system libraries a program linked to the static library needs as well |

For apps on iPhones, iPads and Macs, `emberwave-ffi-VERSION-xcframework.zip` holds the static library as an XCFramework (see [iPhone, iPad and Mac apps](#iphone-ipad-and-mac-apps)).

To build it yourself, with Rust 1.87 or newer:

```sh
cargo build --release -p emberwave-ffi        # target/release: the shared library
cargo rustc --release -p emberwave-ffi --lib --crate-type staticlib \
    -- --print native-static-libs              # the static library, and what it needs
```

## Linking

The commands below assume an archive unpacked into `emberwave/` and a program `app.c`.

The shared library names itself `@rpath/libemberwave_ffi.dylib` on macOS and `libemberwave_ffi.so` on Linux, so a program looks for it on its run path. These commands set the run path to a `lib` directory next to the program and put the library there:

```sh
# macOS
cc app.c -I emberwave/include -L emberwave/lib -lemberwave_ffi -Wl,-rpath,@executable_path/lib
# Linux
cc app.c -I emberwave/include -L emberwave/lib -lemberwave_ffi -Wl,-rpath,'$ORIGIN/lib'
# both
mkdir -p lib && cp emberwave/lib/libemberwave_ffi.* lib/
```

On Windows, link the import library and keep the DLL next to the program (from a Visual Studio developer prompt):

```bat
cl app.c /I emberwave\include emberwave\lib\emberwave_ffi.dll.lib
copy emberwave\lib\emberwave_ffi.dll .
```

To link the static library instead, name its file, as `-lemberwave_ffi` would pick the shared one, and add the system libraries it needs:

```sh
cc app.c -I emberwave/include emberwave/lib/libemberwave_ffi.a $(cat emberwave/lib/native-static-libs.txt)
```

On Windows that is `emberwave\lib\emberwave_ffi.lib` with the libraries named in `emberwave\lib\native-static-libs.txt`.

## iPhone, iPad and Mac apps

`Emberwave.xcframework`, in each release's `emberwave-ffi-VERSION-xcframework.zip`, has the static library for iPhones and iPads (iOS 12 or later), for the iOS Simulator on Macs with Apple silicon (iOS 14 or later) and for Macs with Apple silicon (macOS 11 or later), each with the header and a module map. The `native-static-libs-*.txt` files next to it name the system libraries of each; they are the same for all three. To build it yourself on a Mac with Xcode, from the root of the repository:

```sh
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
scripts/build_xcframework.sh        # target/xcframework/Emberwave.xcframework
```

In Xcode, add `Emberwave.xcframework` to the app target's *Frameworks, Libraries, and Embedded Content* as *Do Not Embed*, since its libraries are static, and add Metal, QuartzCore, CoreGraphics, Foundation, CoreFoundation and `libiconv.tbd` there too. Swift imports the API as the module `Emberwave`; C and Objective-C include `<Emberwave/emberwave.h>`.

```swift
import Emberwave

var context: OpaquePointer?
guard emberwave_context_create(nil, &context) == EMBERWAVE_STATUS_OK else {
    fatalError(String(cString: emberwave_last_error()))
}
defer { emberwave_context_destroy(context) }
print("decoding on", String(cString: emberwave_context_adapter(context)))
```

A Swift package names the XCFramework as a binary target. From a release, it takes `Emberwave.xcframework.zip`, the XCFramework alone at the top of an archive, by its URL and checksum. The checksum is the archive's SHA-256, the line of `Emberwave.xcframework.zip` in the release's `SHA256SUMS`, which `swift package compute-checksum` also prints. The target that uses the library links the system libraries:

```swift
// In the app's Package.swift, with the version of the release:
.binaryTarget(
    name: "Emberwave",
    url: "https://github.com/with-developer/emberwave/releases/download/v0.15.0/Emberwave.xcframework.zip",
    checksum: "..."   // the release's SHA256SUMS line for Emberwave.xcframework.zip
),
.target(
    name: "Player",
    dependencies: ["Emberwave"],
    linkerSettings: [
        .linkedFramework("Metal"), .linkedFramework("QuartzCore"),
        .linkedFramework("CoreGraphics"), .linkedFramework("Foundation"),
        .linkedLibrary("iconv"), .linkedLibrary("objc"),
    ]
),
```

[`examples/swift`](https://github.com/with-developer/emberwave/tree/main/crates/emberwave-ffi/examples/swift) is such a package: `decode.c` in Swift, with the XCFramework of a local build. From the root of the repository:

```sh
scripts/build_xcframework.sh macos-arm64     # the Mac's library alone
swift run --package-path crates/emberwave-ffi/examples/swift \
    emberwave-decode testdata/noise_192x136_420.pwv1 out.y4m
```

`scripts/check_xcframework.sh` builds the C examples and the Swift example against every library of the XCFramework. It also builds the Swift example from `Emberwave.xcframework.zip` as a binary target, as an app's package takes a release's archive. CI runs them on a Mac and in an iOS Simulator, and links them for devices. They have not run on an iPhone or an iPad yet, where an app has to be signed by its developer. The Swift example is for macOS 12, the oldest that the Swift Build of Xcode 27 takes; the library runs on macOS 11.

## Use

```c
#include "emberwave.h"

EmberwaveContext *context;   /* one per GPU; compiles the shaders */
EmberwaveDecoder *decoder;   /* one per stream: picture size and chroma format */

emberwave_context_create(NULL, &context);
emberwave_decoder_create(context, 1920, 1080, EMBERWAVE_CHROMA_420, &decoder);

/* For every packet that arrives: */
emberwave_decoder_push_packet(decoder, packet, packet_size);
if (emberwave_decoder_is_ready(decoder, false)) {
    emberwave_decoder_decode_yuv(decoder, y, cb, cr);   /* or _decode_rgb */
}

emberwave_decoder_destroy(decoder);
emberwave_context_destroy(context);
```

A host encodes each picture within a byte budget and sends the frame in packets:

```c
EmberwaveEncoder *encoder;   /* one per stream, on a context as above */
emberwave_encoder_create(context, 1920, 1080, EMBERWAVE_CHROMA_420, NULL, &encoder);

/* For every picture: planes with their strides, a buffer of `budget` bytes. */
EmberwaveSourcePlane y = { y_data, y_stride }, cb = { cb_data, cb_stride }, cr = { cr_data, cr_stride };
size_t size, count;
emberwave_encoder_encode(encoder, y, cb, cr, frame, budget, &size);
/* or a picture in BGRA: emberwave_encoder_encode_rgb(encoder, bgra, EMBERWAVE_TEXTURE_BGRA8, …) */
emberwave_frame_packets(frame, size, 1200, NULL, 0, &count);           /* how many */
emberwave_frame_packets(frame, size, 1200, packets, capacity, &count); /* where */
/* send frame + packets[i].offset, packets[i].size bytes, for each i */

emberwave_encoder_destroy(encoder);
```

[`examples/c/decode.c`](https://github.com/with-developer/emberwave/blob/main/crates/emberwave-ffi/examples/c/decode.c) in the repository is a complete program: it decodes a PWV1 file to a Y4M file. [`examples/c/encode.c`](https://github.com/with-developer/emberwave/blob/main/crates/emberwave-ffi/examples/c/encode.c) encodes the frames of a Y4M file, splits them into packets, decodes the packets again and prints the size and PSNR of each frame. Its [`CMakeLists.txt`](https://github.com/with-developer/emberwave/blob/main/crates/emberwave-ffi/examples/c/CMakeLists.txt) links it to the shared library of a cargo build, from the root of the repository:

```sh
cargo build -p emberwave-ffi
cmake -S crates/emberwave-ffi/examples/c -B target/c-example -DEMBERWAVE_LIB_DIR=target/debug
cmake --build target/c-example
target/c-example/emberwave_decode testdata/noise_192x136_420.pwv1 out.y4m
target/c-example/emberwave_encode testdata/encoder/scene_200x120_420.source.y4m 20000
```

`scripts/check_c_example.sh` decodes every test vector that way and compares the output with upstream's decoder, and encodes two of the encoder's sources; CI runs it on Linux, macOS and Windows.

## Pictures on the GPU: Metal

A client that draws with Metal can have the picture put into a texture on the GPU instead of copied to CPU memory: on an Apple M3 Max, a 4K frame takes 1.3 to 1.4 ms that way, where `emberwave_decoder_decode_rgb` takes 2.6 to 2.7 ms, and a 1080p frame 0.56 ms against 0.81 ms. The texture is emberwave's, the Metal objects behind it the client's to use:

```c
EmberwaveTexture *textures[3];   /* one for each frame the display may still show */
for (int i = 0; i < 3; i++) {
    emberwave_texture_create(decoder, EMBERWAVE_TEXTURE_BGRA8, &textures[i]);
}
id<MTLDevice> device = (__bridge id<MTLDevice>)emberwave_context_metal_device(context);

/* For every frame that is ready, into a texture whose readers have completed: */
emberwave_decoder_decode_texture(decoder, NULL, textures[next]);   /* waits for the GPU */
id<MTLTexture> picture = (__bridge id<MTLTexture>)emberwave_texture_metal(textures[next]);
/* sample `picture` on a queue of `device`, or copy it onto a BGRA drawable */
```

In Swift, `Unmanaged<MTLTexture>.fromOpaque(pointer).takeUnretainedValue()` takes the pointer.

- **Formats.** `EMBERWAVE_TEXTURE_RGBA8` or `EMBERWAVE_TEXTURE_BGRA8`, unsigned normalised 8 bits per channel: `MTLPixelFormatRGBA8Unorm` or `MTLPixelFormatBGRA8Unorm`. A copy onto a display's drawable, which is BGRA, takes a BGRA texture. The storage is private, so only the GPU reads it.
- **Device.** The textures are on the device of `emberwave_context_metal_device`. Make the queue and the resources that read them on that device. On iPhones, iPads and Macs with Apple silicon, the libraries' systems, that is the system's only GPU, `MTLCreateSystemDefaultDevice()`.
- **Ordering.** `emberwave_decoder_decode_texture` returns once the GPU has drawn the picture, so work the client commits afterwards on any queue of the device sees it. On a context of `emberwave_context_create`, nothing orders the next decode after the client's work that still reads a texture. The client keeps one texture for each command buffer that may still sample one and one to decode into, three with the display's synchronisation and two frames in flight, and decodes only into one whose readers have completed. On a context of the client's own queue one texture does (below). The header's `EmberwaveTexture` sets out all the cases.
- **Pointers.** The Metal pointers are borrowed, `__bridge` without a transfer. A texture keeps its `id<MTLTexture>` until it is destroyed, and a context's device stays while the context or anything made on it lives. They are `NULL` off Metal, as for a Vulkan context on a Mac.
- **Colours.** As for `emberwave_decoder_decode_rgb`: `NULL` follows the frame, and upstream's Metal encoder signals BT.709, full range and centred chroma whatever the pictures are.

A client that converts the colours itself, for high dynamic range, for its display, or for a stream that signals the wrong colours, takes the Y, Cb and Cr planes instead: three one-channel textures of the planes' sizes, chroma at half size in 4:2:0. They hold the samples of the stream, as `emberwave_decoder_decode_yuv` writes them, and are the fastest way to the GPU: on an Apple M3 Max a 4K frame takes 0.91 to 0.93 ms, and a 1080p frame 0.41 ms. In 4:2:0 they also take 1.5 bytes a pixel, where RGBA takes 4.

```c
EmberwaveTexture *planes[3][3];   /* Y, Cb, Cr, for each frame the display may still show */
for (int i = 0; i < 3; i++) {
    emberwave_texture_create_plane(decoder, EMBERWAVE_PLANE_Y, &planes[i][0]);
    emberwave_texture_create_plane(decoder, EMBERWAVE_PLANE_CB, &planes[i][1]);
    emberwave_texture_create_plane(decoder, EMBERWAVE_PLANE_CR, &planes[i][2]);
}

/* For every frame that is ready, into a set whose readers have completed: */
emberwave_decoder_decode_plane_textures(decoder, planes[next][0], planes[next][1], planes[next][2]);
id<MTLTexture> luma = (__bridge id<MTLTexture>)emberwave_texture_metal(planes[next][0]);
/* ... and Cb and Cr: MTLPixelFormatR8Unorm, sampled by the client's own conversion */
```

A texture is for the plane it was made for, and a decode refuses textures passed in another order. The client converts with the matrix and range the stream signals (`emberwave_probe_packet`) or that it knows better. In 4:2:0 a chroma sample sits vertically halfway between two rows of luma, and horizontally halfway between two luma samples with `EMBERWAVE_SITING_CENTER`, on the left one with `EMBERWAVE_SITING_LEFT`. Device, ordering and pointers are as for the RGB texture.

### On the client's own queue, without waiting

The decodes above wait for the GPU, because emberwave works on a queue of its own and nothing else would order the client's reads after its writes. A context made on the client's own device and command queue needs neither the wait nor the textures in turn. Metal orders the command buffers of one queue that use the same texture by when they were committed, so the client's draw committed after a decode reads the new picture, and the next decode waits for that draw (decision D43):

```objc
id<MTLDevice> device = MTLCreateSystemDefaultDevice();
id<MTLCommandQueue> queue = [device newCommandQueue];
EmberwaveContext *context;
emberwave_context_create_metal((__bridge void *)device, (__bridge void *)queue, &context);
/* ... a decoder on it, and one texture for every frame ... */

/* For every frame that is ready: */
emberwave_decoder_submit_texture(decoder, NULL, texture);   /* returns once the work is on the queue */
id<MTLCommandBuffer> commands = [queue commandBuffer];       /* the same queue, committed after */
/* ... encode the draw that samples the texture ... */
[commands commit];
```

`emberwave_decoder_submit_plane_textures` does the same for the planes. On an Apple M3 Max a 4K frame takes the decoding thread 0.26 ms instead of 1.3 ms, and the client's draw completes as soon as after the waiting decode, at 1080p 0.1 ms sooner. The rules of a shared queue:
- Use the textures directly in the command buffers that read them, not through a heap or through an argument buffer without `useResource`, as Metal does not track those.
- Commit those command buffers after the call has returned.
- A call makes all its command buffers before it commits them: on an Apple M3 Max, 9 for a decode to a texture or to RGB, 6 to planes or to plane textures and 18 for an encode, one for each pass and transition of wgpu's. The decoders and encoders of every context made on the queue take turns at this, and hold none between calls (decision D60). A full queue makes a call wait until one of its command buffers has completed and that buffer's completed handlers have returned; a queue without room for 20 besides those the client holds uncommitted can wait for good. The default of 64 is plenty.
- Do not call emberwave inside a completion handler of a command buffer of the same queue. A call that waits for the GPU never returns there: the wait ends in a handler of emberwave's that Metal runs after that one. The others can wait there too, for another thread's call or for room on the queue, and hold up the handlers behind them.

On a context of `emberwave_context_create` the submit functions refuse to work, as nothing there orders the reads.

### Into the client's own textures

`emberwave_texture_from_metal` wraps a texture of the client's own, so that any decode above fills it (decision D44). The usual one belongs to a `CVPixelBuffer`, to hand to an `AVSampleBufferDisplayLayer`:

```objc
/* Once: a cache on the context's device, and BGRA pixel buffers of the picture's size,
   from a pool made with kCVPixelBufferMetalCompatibilityKey. */
CVMetalTextureCacheRef cache;
CVMetalTextureCacheCreate(NULL, NULL, device, NULL, &cache);

/* For every frame that is ready, into a free pixel buffer: */
CVMetalTextureRef image;
CVMetalTextureCacheCreateTextureFromImage(NULL, cache, pixelBuffer, NULL, MTLPixelFormatBGRA8Unorm,
                                          width, height, 0, &image);
EmberwaveTexture *texture;
emberwave_texture_from_metal(decoder, (__bridge void *)CVMetalTextureGetTexture(image), &texture);
emberwave_decoder_decode_texture(decoder, NULL, texture);   /* returns once the GPU is done */
emberwave_texture_destroy(texture);
CFRelease(image);
/* ... the pixel buffer holds the picture: enqueue it on the display layer */
```

For a pixel buffer that goes to a display layer, the decode that waits is the simple way. The display layer is not on the queue, so the submit gains nothing there unless the client queues GPU work of its own behind the decode. With `emberwave_decoder_submit_texture` the call returns before the GPU has written the pixel buffer. The queue orders only command buffers that use the texture: one that does not use it may run beside the decode and finish first. Commit one that uses it, such as the client's own pass over the texture or a copy of one pixel from it. In its completed handler, release the `CVMetalTextureRef` and enqueue the pixel buffer:

```objc
emberwave_decoder_submit_texture(decoder, NULL, texture);
id<MTLTexture> metal = CVMetalTextureGetTexture(image);
id<MTLCommandBuffer> commands = [queue commandBuffer];   /* the same queue, after the call */
id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
[blit copyFromTexture:metal sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
           sourceSize:MTLSizeMake(1, 1, 1) toBuffer:pixel destinationOffset:0
       destinationBytesPerRow:4 destinationBytesPerImage:4];   /* uses the texture: after the decode */
[blit endEncoding];
[commands addCompletedHandler:^(id<MTLCommandBuffer> done) {
    CFRelease(image);
    /* the pixel buffer holds the picture: enqueue it on the display layer */
}];
[commands commit];
emberwave_texture_destroy(texture);   /* the work on the GPU keeps what it uses */
```

- **The texture** is on the device of the decoder's context, 2D, of one level, layer and sample, of the picture's size, and BGRA8 or RGBA8, or R8 of a plane's size for `emberwave_texture_from_metal_plane`. A texture of a `CVPixelBuffer` is a texture of its own, not a view.
- **Copy or draw.** A copy fills the texture when its width is a multiple of 64 pixels, as at 720p, 1080p, 1440p and 4K, and it is not framebuffer-only. A texture of a `CVMetalTextureCache` is not framebuffer-only, so a copy fills it whatever usage `kCVMetalTextureUsage` asks for; on a Mac it is also a render target by default. Other textures take a draw, which needs `MTLTextureUsageRenderTarget`, as a drawable has it. A texture that neither can fill is refused, as is one of the wrong device, format or size.
- **Lifetime.** emberwave keeps a reference to the `MTLTexture` until the texture is destroyed. Keep the `CVMetalTextureRef`, and hand the pixel buffer to anything else, only once the GPU has written the picture: once a waiting decode has returned, or once a command buffer that uses the texture, committed after a submit, has completed.
- **Planes.** `emberwave_texture_from_metal_plane(decoder, plane, metal, &texture)` wraps an R8 texture for one plane. A Metal texture wrapped for two planes is refused by the decode. When all three textures of a decode have `MTLTextureUsageShaderWrite`, as those of `emberwave_texture_create_plane` do, the decode stores the planes into them from its shaders; otherwise it copies them in, which takes a little longer.
- **Hazards.** A submit takes only textures whose hazards Metal tracks, as it does for those of a `CVMetalTextureCache`. A heap's textures are usually untracked; the waiting decodes take those too.

On Vulkan, see [A context on the client's Vulkan device](#a-context-on-the-clients-vulkan-device), and on Direct3D, [Textures shared with Direct3D](#textures-shared-with-direct3d). A host encodes pictures from textures as well: see [Pictures to encode on the GPU](#pictures-to-encode-on-the-gpu).

[`examples/ios`](https://github.com/with-developer/emberwave/tree/main/crates/emberwave-ffi/examples/ios) is an iPhone and iPad app that does all of this in Swift: it times decoding into textures, decoding to planes and encoding on the device's GPU, and draws the last picture from its texture. Its README says how to run it on a phone.

## A context on the client's Vulkan device

On Linux, Android and 64-bit Windows, where wgpu runs on Vulkan, a context can run on the client's own `VkDevice` and submit to one of its queues (decision D61). The client makes the instance with the extensions of `emberwave_vulkan_instance_extensions` among its own, as wgpu uses only those enabled, and the device with what emberwave needs, besides what it needs itself:

```c
uint32_t wanted = 0;
emberwave_vulkan_instance_extensions((EmberwaveVkGetInstanceProcAddr)vkGetInstanceProcAddr, NULL, &wanted);
/* ... room for `wanted` names, then the same call fills them in; vkCreateInstance with them,
   and an apiVersion at least the physical device's, up to VK_API_VERSION_1_3 ... */

EmberwaveVulkanInstance instance = {
    .get_instance_proc_addr = (EmberwaveVkGetInstanceProcAddr)vkGetInstanceProcAddr,
    .instance = vk_instance,
    .api_version = VK_API_VERSION_1_3,            /* the instance's apiVersion */
    .extensions = instance_extensions,            /* the instance's extensions */
    .extension_count = instance_extension_count,
};
EmberwaveVulkanRequirements *needs = NULL;
emberwave_vulkan_requirements_create(&instance, physical_device, &needs);
uint32_t count = 0;
const char *const *names = emberwave_vulkan_requirements_extensions(needs, &count);
const VkPhysicalDeviceFeatures2 *features = emberwave_vulkan_requirements_features(needs);
/* ... vkCreateDevice with these extensions and features among the client's ... */
emberwave_vulkan_requirements_destroy(needs);

EmberwaveVulkanDevice device = {
    .instance = instance,
    .physical_device = physical_device,
    .device = vk_device,
    .extensions = device_extensions,              /* all the device's extensions */
    .extension_count = device_extension_count,
    .queue_family_index = family,
    .queue_index = 0,
    .lock_queue = lock_queue,                     /* NULL if nothing else uses the queue meanwhile */
    .unlock_queue = unlock_queue,
    .user = &queue_mutex,
};
EmberwaveContext *context = NULL;
emberwave_context_create_vulkan(&device, &context);
```

- **Features.** `emberwave_vulkan_requirements_features` gives a `VkPhysicalDeviceFeatures2` whose `features`, and the structures chained to it, have `VK_TRUE` for each feature emberwave needs: those wgpu enables on a device of its own, and then counts on. Set those members in the client's own structures, or chain copies of the structures it lacks. Where the client chains `VkPhysicalDeviceVulkan11Features`, `VkPhysicalDeviceVulkan12Features` or `VkPhysicalDeviceVulkan13Features`, set the same features there: Vulkan allows only one of the two kinds. A missing extension makes `emberwave_context_create_vulkan` fail with `EMBERWAVE_STATUS_INVALID_ARGUMENT`. A missing feature is undefined behaviour, as Vulkan cannot tell which were enabled.
- **The queue.** It belongs to a family with graphics and compute, which wgpu records work for. Vulkan requires the uses of a queue to be externally synchronised. emberwave calls `lock_queue` and `unlock_queue` with `user` around each of its submissions, and around each destroy of a context, decoder or encoder, from any thread. The destroy that lets go of the last hold on the queue waits for the queue to be idle while it holds the lock, which holds up the client's own submissions until then.
  - Do not call emberwave while holding that lock.
  - Give every context on one queue the same callbacks and `user`, and keep them valid until everything made on those contexts is destroyed. A context with other callbacks is refused. The decoders and encoders of those contexts take turns at the queue, as on Metal (decision D60).
  - Without callbacks, nothing else may use the queue during emberwave's calls. A second queue of the family, where the device has one, needs neither.
- **The device.** It stays the client's: destroy it after the context and everything made on it.

### Pictures in the client's Vulkan images

On such a context the decodes into textures put the picture, or its planes, into Vulkan images that the client reads on the GPU:
- `emberwave_texture_vulkan` gives the `VkImage` of a texture of `emberwave_texture_create` or `emberwave_texture_create_plane`.
- `emberwave_texture_from_vulkan` and `emberwave_texture_from_vulkan_plane` wrap an image of the client's own, described by an `EmberwaveVulkanImage`: its `VkImage`, `VkFormat` (`VK_FORMAT_B8G8R8A8_UNORM` or `VK_FORMAT_R8G8B8A8_UNORM` for the picture, `VK_FORMAT_R8_UNORM` for a plane), usage, and current layout.

The images are 2D, of one level, layer and sample, with optimal tiling, exclusive to the context's queue family, and have `VK_IMAGE_USAGE_SAMPLED_BIT`. A decode copies the picture in with `VK_IMAGE_USAGE_TRANSFER_DST_BIT` when its width is a multiple of 64 pixels, and draws it in with `VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT` otherwise. Planes are copied.

Between emberwave's calls an image is in `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`, from the first decode into it on. A decode leaves it so, and the client's barriers start from that layout and return the image to it before the next call that uses it. An image wrapped in `VK_IMAGE_LAYOUT_UNDEFINED` may lose its contents to the first decode.

The decodes that wait, `emberwave_decoder_decode_texture` and `emberwave_decoder_decode_plane_textures`, return once the GPU has written the images. The ones that do not wait take semaphores instead:

```c
EmberwaveVulkanSync sync = {
    .wait_semaphore = (uint64_t)reads_done,       /* signalled when the client's reads of the image end */
    .wait_value = frame,
    .signal_semaphore = (uint64_t)picture_ready,  /* the client's reads wait for it */
    .signal_value = frame + 1,
    .flags = EMBERWAVE_SYNC_WAIT_TIMELINE | EMBERWAVE_SYNC_SIGNAL_TIMELINE,  /* 0 for binary semaphores */
};
emberwave_decoder_submit_texture_vulkan(decoder, NULL, texture, &sync);
/* ... the client's command buffer that samples the image waits for picture_ready at frame + 1 ... */
```

`emberwave_decoder_submit_plane_textures_vulkan` does the same for the planes. Either semaphore may be 0, and `sync` `NULL`. The queue orders the work it carries, but not memory: work of the client's on the same queue after the call needs a barrier from all commands, or the signal semaphore.
- A binary semaphore to wait for has its signal submitted already, and one to signal is waited for before it is signalled again.
- A wait that is never met holds up every later submission to the queue, the client's too: the decode waits at every stage.
- Keep the semaphores until the work that signals them has completed. When the call fails, the semaphores may or may not have been waited for and signalled.

A decode that does not wait may still run after `emberwave_texture_destroy` returns. So the client destroys a wrapped image only once `release` of its `EmberwaveVulkanImage` has been called, which emberwave does when it and the GPU are done with the image, from any thread. Without `release`, the client waits for the work of every decode into the image itself.

`tests/vulkan.rs` makes an instance, a device and images as a client does, and runs on Linux CI's lavapipe.

## Textures shared with Direct3D

On Windows, a host or a client that draws with Direct3D 12 or Direct3D 11 shares textures and fences with emberwave through NT handles (decision D63). wgpu cannot run on the caller's `ID3D12Device`, so emberwave opens them on a device of its own, on the same adapter:

```c
/* Once: a context on the adapter of the client's device, and its fence. */
LUID luid = adapter_desc.AdapterLuid;   /* from IDXGIAdapter::GetDesc, or ID3D12Device::GetAdapterLuid */
emberwave_context_create_d3d12(luid.LowPart, luid.HighPart, &context);

ID3D11Fence *fence;   /* or an ID3D12Fence made with D3D12_FENCE_FLAG_SHARED */
device5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence));
HANDLE handle;
fence->CreateSharedHandle(NULL, GENERIC_ALL, NULL, &handle);
EmberwaveD3D12Fence *shared;
emberwave_d3d12_fence_open(context, handle, &shared);
CloseHandle(handle);

/* A texture of the picture's size, made with D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED
   (or a Direct3D 12 resource on a heap with D3D12_HEAP_FLAG_SHARED), and its handle: */
EmberwaveTexture *texture;
emberwave_texture_from_d3d12(decoder, texture_handle, &texture);

/* For every frame that is ready: the decode waits for the client's reads of the last picture,
   and signals when the texture holds the new one. */
EmberwaveD3D12Sync sync = {
    .wait_fence = shared, .wait_value = reads_done,
    .signal_fence = shared, .signal_value = reads_done + 1,
};
emberwave_decoder_submit_texture_d3d12(decoder, NULL, texture, &sync);
context4->Wait(fence, reads_done + 1);   /* the client's work that reads the picture */
```

- **Context.** `emberwave_context_create_d3d12` takes the adapter's LUID, its `LowPart` and `HighPart`. Shared handles open only on the adapter that made them. The context runs on Direct3D 12, and its decoders and encoders take turns at its queue, so that the fences of one call go with that call's submission alone. A context of `emberwave_context_create`, which takes no turns, opens no shared handle or fence, even where it runs on Direct3D 12.
- **Fences.** `emberwave_d3d12_fence_open` opens an `ID3D12Fence` made with `D3D12_FENCE_FLAG_SHARED`, or an `ID3D11Fence` made with `D3D11_FENCE_FLAG_SHARED`. An `EmberwaveD3D12Sync` names a fence and a value to wait for before the call's GPU work, all of it, and a fence and a value to signal after it. Either may be `NULL`, as may `sync`. The client destroys its fences after emberwave's. Fences order the work, not keyed mutexes.
  - A wait that is never met holds up every later call on the context. The destroy that lets go of the last hold on the context's queue waits for it to be idle, and then never returns.
  - A budget too small for an encode shows only once the GPU has run: the wait has been met and the signal made by then, so a retry takes a new signal value.
- **Textures.** `emberwave_texture_from_d3d12` opens a texture for the picture, `_plane` for a plane, and `_source` for a plane of a picture to encode. emberwave reads everything it checks from the resource itself:
  - 2D, of one level, layer and sample, and of the size of the picture or the plane;
  - `DXGI_FORMAT_B8G8R8A8_UNORM` or `DXGI_FORMAT_R8G8B8A8_UNORM` for the picture, `DXGI_FORMAT_R8_UNORM` for a plane, and `DXGI_FORMAT_R8G8_UNORM` for Cb and Cr interleaved (`EMBERWAVE_PLANE_CBCR`);
  - no `D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE`;
  - for a picture that is not a multiple of 64 pixels wide, which a decode draws rather than copies, `D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET` (`D3D11_BIND_RENDER_TARGET`).

  emberwave does not close the handles: the client may close each one once the call that took it returns.
- **Calls.** Only the `_d3d12` calls take such textures:
  - `emberwave_decoder_submit_texture_d3d12` and `emberwave_decoder_submit_plane_textures_d3d12` decode without waiting;
  - `emberwave_encoder_encode_textures_d3d12` encodes and waits, as every encode does.

  Each leaves the client's textures in `D3D12_RESOURCE_STATE_COMMON`, the state a texture is in when it changes devices, before it signals. The other calls refuse such textures, since they would leave them in other states.
- **Lifetime.** A texture keeps a reference to the client's resource until it is destroyed and the GPU is done with it, so the client may release its own as it likes.

`tests/d3d12.rs` shares textures and a fence of a Direct3D 11 device on WARP, as a host or a client does, and runs in Windows CI.

## Pictures to encode on the GPU

A host whose pictures are on the GPU already, as a screen capture's are, encodes them from textures. That spares the read-back to CPU memory and the upload. The encoder reads one texture for each plane: Y, Cb and Cr, each with one 8-bit channel. It can also take Y and a second texture with Cb and Cr interleaved in two channels, as the second plane of NV12 holds them (decision D62). The frame has the bytes that `emberwave_encoder_encode` makes of the same picture in CPU memory.

On Metal, `emberwave_texture_from_metal_source` wraps a texture of the host's own, such as the planes of an NV12 pixel buffer from ScreenCaptureKit:

```objc
/* Once: a cache on the context's device. The capture delivers NV12 pixel buffers of the
   encoder's size: '420v' for an encoder signalling video range, '420f' for full range. */
CVMetalTextureCacheRef cache;
CVMetalTextureCacheCreate(NULL, NULL, device, NULL, &cache);

/* For every picture the capture delivers: */
CVMetalTextureRef luma, chroma;
CVMetalTextureCacheCreateTextureFromImage(NULL, cache, pixelBuffer, NULL, MTLPixelFormatR8Unorm,
                                          width, height, 0, &luma);
CVMetalTextureCacheCreateTextureFromImage(NULL, cache, pixelBuffer, NULL, MTLPixelFormatRG8Unorm,
                                          width / 2, height / 2, 1, &chroma);
EmberwaveTexture *y, *cbcr;
emberwave_texture_from_metal_source(encoder, EMBERWAVE_PLANE_Y,
                                    (__bridge void *)CVMetalTextureGetTexture(luma), &y);
emberwave_texture_from_metal_source(encoder, EMBERWAVE_PLANE_CBCR,
                                    (__bridge void *)CVMetalTextureGetTexture(chroma), &cbcr);
emberwave_encoder_encode_textures(encoder, y, cbcr, NULL, frame, budget, &size);   /* waits for the GPU */
emberwave_texture_destroy(y);
emberwave_texture_destroy(cbcr);
CFRelease(luma);
CFRelease(chroma);
/* the pixel buffer may go back to the capture */
```

With three textures of the planes, `emberwave_encoder_encode_textures(encoder, y, cb, cr, …)` takes them in that order.

- **Textures.** They are on the device of the encoder's context, 2D, of one level, layer and sample, of the plane's size as `emberwave_encoder_info` gives it, and readable by shaders.
  - On Metal: `MTLPixelFormatR8Unorm` for a plane and `MTLPixelFormatRG8Unorm` for Cb and Cr, with `MTLTextureUsageShaderRead`, which textures of a `CVMetalTextureCache` have.
  - On Vulkan: `VK_FORMAT_R8_UNORM` and `VK_FORMAT_R8G8_UNORM`, with `VK_IMAGE_USAGE_SAMPLED_BIT`.
  - Textures of planes that a decoder of the same context put a picture into can be encoded again.
- **Waiting.** Every encode waits for the GPU. When it returns, the GPU has read the textures, and the host may write them again, give the pixel buffer back to its pool or destroy the texture. Wrapping a texture for one encode and destroying it afterwards is cheap, and so is keeping it for many.
- **Ordering.** The encode reads the textures after the host's work that writes them only where something orders the two:
  - On a context of the host's own Metal queue, Metal orders the command buffers committed to it before the call, for textures whose hazards it tracks.
  - On a context of the host's Vulkan device, work submitted to the context's queue before the call orders itself with a barrier that makes its writes visible to compute shaders. Work on other queues needs semaphores (below).
  - Otherwise, on a context of emberwave's own queue or for untracked textures, only work that has completed is ordered. A captured pixel buffer is complete once the capture delivers it.

On Vulkan, `emberwave_texture_from_vulkan_source` wraps an image of the host's, and `emberwave_encoder_encode_textures_vulkan` takes the semaphores of an `EmberwaveVulkanSync`, as the decodes that do not wait do:

```c
EmberwaveVulkanImage image = {
    .image = (uint64_t)luma_image,
    .format = VK_FORMAT_R8_UNORM,
    .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,   /* whenever an encode reads it */
};
EmberwaveTexture *y, *cbcr;   /* once, for images the host keeps */
emberwave_texture_from_vulkan_source(encoder, EMBERWAVE_PLANE_Y, &image, &y);
/* ... and the VK_FORMAT_R8G8_UNORM image of Cb and Cr for EMBERWAVE_PLANE_CBCR */

/* For every picture: the host's copy of the capture into the images signals `copied`. */
EmberwaveVulkanSync sync = { .wait_semaphore = (uint64_t)copied };
emberwave_encoder_encode_textures_vulkan(encoder, y, cbcr, NULL, &sync, frame, budget, &size);
```

- **Layout.** An image is in `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` whenever an encode reads it. The host's barriers start from that layout and return the image to it before the next encode, as for the images of decodes. An image described in any other layout is refused, `VK_IMAGE_LAYOUT_UNDEFINED` included, whose contents a barrier may discard.
- **Semaphores.** The encode's GPU work waits for `wait_semaphore` at every stage before it reads the images. It signals `signal_semaphore` once it has read them, for the host's work on other queues that writes them again. As the call waits for the GPU, it returns only once its wait is met. A binary semaphore needs its signal submitted already. A timeline value can be reached by work submitted to the context's queue before the call, by work on another queue, or by `vkSignalSemaphore` on another thread. Work submitted to the context's queue after the call began runs only after the encode, so it never meets the wait, and the call never returns then.
- **Budgets.** A budget too small for the smallest frame shows only once the GPU has encoded the picture. The semaphores have been waited for and signalled by then, so a retry with the smallest frame's size takes other semaphores, or new timeline values.
- **Images.** They have optimal tiling and are exclusive to the context's queue family. emberwave never destroys an image: `release`, if given, is called once the texture is destroyed, as nothing reads the image after its encodes.

`tests/metal.rs` and `tests/vulkan.rs` encode the host's own textures and images both ways, against the same pictures encoded from CPU memory.

### Pictures in RGB

Most captures hold their pictures in RGB, four bytes a pixel, blue first: ScreenCaptureKit's BGRA pixel buffers, Windows' desktop duplication, a renderer's target. A texture of the whole picture, wrapped with `EMBERWAVE_PLANE_RGB`, goes in as `y`, with `cb` and `cr` `NULL`, and the encoder converts it to Y'CbCr as it reads it, by the matrix, range and chroma siting of its signalling (decision D65):

```objc
CVMetalTextureRef bgra;   /* a '32BGRA' pixel buffer of the encoder's size */
CVMetalTextureCacheCreateTextureFromImage(NULL, cache, pixelBuffer, NULL, MTLPixelFormatBGRA8Unorm,
                                          width, height, 0, &bgra);
EmberwaveTexture *picture;
emberwave_texture_from_metal_source(encoder, EMBERWAVE_PLANE_RGB,
                                    (__bridge void *)CVMetalTextureGetTexture(bgra), &picture);
emberwave_encoder_encode_textures(encoder, picture, NULL, NULL, frame, budget, &size);
emberwave_texture_destroy(picture);
CFRelease(bgra);
```

- **Formats.** `MTLPixelFormatBGRA8Unorm` or `MTLPixelFormatRGBA8Unorm` on Metal, `VK_FORMAT_B8G8R8A8_UNORM` or `VK_FORMAT_R8G8B8A8_UNORM` on Vulkan, and `DXGI_FORMAT_B8G8R8A8_UNORM` or `DXGI_FORMAT_R8G8B8A8_UNORM` on Direct3D, through `emberwave_texture_from_d3d12_source` and `emberwave_encoder_encode_textures_d3d12`. The fourth channel is not read. The sRGB formats are refused, as their reads would undo the transfer function that the picture's bytes carry.
- **The conversion** is in integers and makes the same bytes on every GPU, so a picture encodes into the frame of the planes it converts to; in Rust, `color::rgb_to_ycbcr8` makes them. A 4:2:0 chroma sample is the average of the pixels it covers: across two columns where chroma is centred, across three weighted 1, 2 and 1 where it is on the left.
- **Time.** On an Apple M3 Max a 4K frame encodes from a BGRA texture in 1.82 to 1.84 ms, against 1.74 to 1.75 ms from textures of its planes.
- **A decode's picture.** A texture that a decode put a picture into encodes the same way, for a transcode on the GPU.
- **In CPU memory**, `emberwave_encoder_encode_rgb(encoder, rgb, EMBERWAVE_TEXTURE_BGRA8, frame, budget, &size)` takes the picture's rows, `rgb.stride` bytes apart, and converts them on the GPU. It copies the picture into memory the encoder keeps, and on the GPU into a texture: a 4K frame takes 2.5 to 2.8 ms.

## Rules

- **Errors.** Functions that can fail return an `EmberwaveStatus`. After a failure, `emberwave_last_error()` describes it, for the thread that made the call, until that thread's next failure.
- **Objects.** `*_create` makes a context, a decoder or an encoder, `*_destroy` frees it, and destroying `NULL` does nothing. A decoder or an encoder may outlive its context. Contexts are created one at a time: calls from several threads wait for each other (decision D31).
- **Threads.** A context may be used by several threads at once, for instance to create decoders and encoders. A decoder or an encoder may move between threads, but only one thread may use it at a time. A context keeps a few hundred bytes for every thread that has called it or its decoders and encoders, until they are all destroyed, as wgpu keeps its error scopes by thread: call from a pool of threads, not from new ones without end.
- **Memory.** The decoder writes into memory the caller provides: planes with a stride of at least their width (`emberwave_decoder_info` gives the sizes), or RGB with a stride of at least three times the width. It checks the arguments before it decodes, so a bad one does not use up the frame.
- **Waiting.** `emberwave_decoder_decode_yuv` and `emberwave_decoder_decode_rgb` wait for the GPU and copy the picture to the CPU. On an Apple M3 Max a 4K frame takes 1.2 ms to planes and 2.6 to 2.7 ms to RGB that way, against 0.9 and 1.2 ms when the picture stays on the GPU. A call that fails may have written some rows. `emberwave_decoder_decode_texture` and `emberwave_decoder_decode_plane_textures` wait as well, and leave the picture on the GPU, in textures for Metal (see [Pictures on the GPU: Metal](#pictures-on-the-gpu-metal)). On a context of the client's own Metal queue, no call that waits may be made inside a completion handler of a command buffer of that queue: the wait never ends there. That includes the destroy functions, which wait when they let go of the last hold on the queue.
- **Panics.** A panic inside emberwave is caught and reported as `EMBERWAVE_STATUS_INTERNAL`; destroy the object it happened in.
- **Packets.** `emberwave_decoder_push_packet` takes one packet or several back to back, in any order, and ignores a block that arrived already. Frames are told apart by the bitstream's sequence number, which has three bits: a packet of a frame one to three frames newer discards what was received of the older one and starts its own; one four to seven frames newer, or one to four frames older, is ignored; one five to seven frames older passes for a newer frame, and one eight frames off for the same frame. So a transport keeps frame numbers of its own: it drops packets of frames older than the one being received, and before it gives the first packet of a newer frame it decodes the current one if it is ready, then calls `emberwave_decoder_clear`, not after.
- **Partial frames.** `emberwave_decoder_is_ready` with `allow_partial_frame` accepts a frame that lost packets once every block of the coarsest level and more than 90 percent of the blocks its start of frame header announced have arrived, for a picture despite the loss; `emberwave_decoder_set_partial_share` sets another share, lower to draw frames that lost more, with more of their detail blank, or 1 for whole frames only (decision D59). `emberwave_decoder_set_concealment` fills the blocks of detail such a frame lacks with the last frame's where the picture is still, so that a still picture keeps its detail (decision D66). Until that header arrives the count is of all the picture's blocks. emberwave's encoder sends every block of the coarsest level for this; upstream's leaves out the empty ones, so a frame of it that lost any packet may never be ready in part (decision D58).
- **Encoding.** `emberwave_encoder_encode` reads planes with a stride of at least their width (`emberwave_encoder_info` gives the sizes), `emberwave_encoder_encode_rgb` a picture in RGB with a stride of at least four bytes a pixel, and `emberwave_encoder_encode_textures` reads either from textures on the GPU (see [Pictures to encode on the GPU](#pictures-to-encode-on-the-gpu)), as `_vulkan` and `_d3d12` do with fences or semaphores. Both write the frame, at most `budget` bytes, into a buffer of that many bytes the caller provides, and wait for the GPU. A budget below the smallest frame is `EMBERWAVE_STATUS_INVALID_ARGUMENT` with the smallest frame's size in `size`, and encodes nothing. The frame's sequence number, a three-bit counter from 1 that wraps around, advances only with frames that were encoded. `emberwave_frame_packets` splits a frame at its blocks; a block larger than the packet size gets a packet of its own.
- **Signalling.** `EmberwaveSignalling` sets what every frame says about its colours: matrix (BT.709 or BT.2020; the format cannot signal BT.601), range, chroma siting, primaries and transfer function (sRGB or PQ). `NULL` means BT.709, video range, centred chroma and sRGB. Its `reserved` words are zero.
- **Colours.** With a `NULL` colour, `emberwave_decoder_decode_rgb` converts the way the frame's start of frame header says. Upstream's Metal encoder writes BT.709, full range and centred chroma there whatever the pictures are, so pass what you know, such as the colours of a PWV1 file's header from `emberwave_probe_pwv1`.

The header documents every function; `ARCHITECTURE.md` in the repository explains the design (decisions D27 and D33).
