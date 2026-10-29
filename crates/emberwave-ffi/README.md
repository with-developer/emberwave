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

A Swift package names the XCFramework as a binary target. [`examples/swift`](https://github.com/with-developer/emberwave/tree/main/crates/emberwave-ffi/examples/swift) is one: `decode.c` in Swift, with the system libraries in its linker settings. From the root of the repository:

```sh
scripts/build_xcframework.sh macos-arm64     # the Mac's library alone
swift run --package-path crates/emberwave-ffi/examples/swift \
    emberwave-decode testdata/noise_192x136_420.pwv1 out.y4m
```

`scripts/check_xcframework.sh` builds the C examples and the Swift example against every library of the XCFramework. CI runs them on a Mac and in an iOS Simulator, and links them for devices. They have not run on an iPhone or an iPad yet, where an app has to be signed by its developer.

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
- A call makes all its command buffers before it commits them: on an Apple M3 Max, 9 for a decode to a texture or to RGB, 7 to planes, 6 to plane textures and 18 for an encode, one for each pass and transition of wgpu's. A queue with room for fewer than a call needs waits for good; the default of 64 holds a few calls at a time, and the decoders and encoders of a context take turns (#126).
- Do not call a function that waits for the GPU inside a completion handler of a command buffer of the same queue: the wait ends in a handler of emberwave's that Metal runs after that one. `emberwave_decoder_submit_texture` returns without waiting, and may.

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

Vulkan and Direct3D 12 handles are not part of the interface yet.

[`examples/ios`](https://github.com/with-developer/emberwave/tree/main/crates/emberwave-ffi/examples/ios) is an iPhone and iPad app that does all of this in Swift: it times decoding into textures, decoding to planes and encoding on the device's GPU, and draws the last picture from its texture. Its README says how to run it on a phone.

## Rules

- **Errors.** Functions that can fail return an `EmberwaveStatus`. After a failure, `emberwave_last_error()` describes it, for the thread that made the call, until that thread's next failure.
- **Objects.** `*_create` makes a context, a decoder or an encoder, `*_destroy` frees it, and destroying `NULL` does nothing. A decoder or an encoder may outlive its context. Contexts are created one at a time: calls from several threads wait for each other (decision D31).
- **Threads.** A context may be used by several threads at once, for instance to create decoders and encoders. A decoder or an encoder may move between threads, but only one thread may use it at a time.
- **Memory.** The decoder writes into memory the caller provides: planes with a stride of at least their width (`emberwave_decoder_info` gives the sizes), or RGB with a stride of at least three times the width. It checks the arguments before it decodes, so a bad one does not use up the frame.
- **Waiting.** `emberwave_decoder_decode_yuv` and `emberwave_decoder_decode_rgb` wait for the GPU and copy the picture to the CPU. On an Apple M3 Max a 4K frame takes 1.2 ms to planes and 2.6 to 2.7 ms to RGB that way, against 0.9 and 1.2 ms when the picture stays on the GPU. A call that fails may have written some rows. `emberwave_decoder_decode_texture` and `emberwave_decoder_decode_plane_textures` wait as well, and leave the picture on the GPU, in textures for Metal (see [Pictures on the GPU: Metal](#pictures-on-the-gpu-metal)).
- **Panics.** A panic inside emberwave is caught and reported as `EMBERWAVE_STATUS_INTERNAL`; destroy the object it happened in.
- **Packets.** `emberwave_decoder_push_packet` takes one packet or several back to back, in any order, and ignores a block that arrived already. Frames are told apart by the bitstream's sequence number, which has three bits: a packet of a frame one to three frames newer discards what was received of the older one and starts its own; one four to seven frames newer, or one to four frames older, is ignored; one five to seven frames older passes for a newer frame, and one eight frames off for the same frame. So a transport keeps frame numbers of its own: it drops packets of frames older than the one being received, and before it gives the first packet of a newer frame it decodes the current one if it is ready, then calls `emberwave_decoder_clear`, not after.
- **Partial frames.** `emberwave_decoder_is_ready` with `allow_partial_frame` accepts a frame that lost packets once every block of the coarsest level and more than 90 percent of the blocks its start of frame header announced have arrived, for a picture despite the loss; `emberwave_decoder_set_partial_share` sets another share, lower to draw frames that lost more, with more of their detail blank, or 1 for whole frames only (decision D59). Until that header arrives the count is of all the picture's blocks. emberwave's encoder sends every block of the coarsest level for this; upstream's leaves out the empty ones, so a frame of it that lost any packet may never be ready in part (decision D58).
- **Encoding.** `emberwave_encoder_encode` reads planes with a stride of at least their width (`emberwave_encoder_info` gives the sizes) and writes the frame, at most `budget` bytes, into a buffer of that many bytes the caller provides; it waits for the GPU. A budget below the smallest frame is `EMBERWAVE_STATUS_INVALID_ARGUMENT` with the smallest frame's size in `size`, and encodes nothing. The frame's sequence number, a three-bit counter from 1 that wraps around, advances only with frames that were encoded. `emberwave_frame_packets` splits a frame at its blocks; a block larger than the packet size gets a packet of its own.
- **Signalling.** `EmberwaveSignalling` sets what every frame says about its colours: matrix (BT.709 or BT.2020; the format cannot signal BT.601), range, chroma siting, primaries and transfer function (sRGB or PQ). `NULL` means BT.709, video range, centred chroma and sRGB. Its `reserved` words are zero.
- **Colours.** With a `NULL` colour, `emberwave_decoder_decode_rgb` converts the way the frame's start of frame header says. Upstream's Metal encoder writes BT.709, full range and centred chroma there whatever the pictures are, so pass what you know, such as the colours of a PWV1 file's header from `emberwave_probe_pwv1`.

The header documents every function; `ARCHITECTURE.md` in the repository explains the design (decisions D27 and D33).
