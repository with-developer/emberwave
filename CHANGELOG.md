# Changelog

All notable changes to emberwave. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- A context on the caller's own Vulkan device, on Linux, Android and 64-bit Windows: `emberwave_vulkan_instance_extensions` lists the instance extensions wgpu uses, `emberwave_vulkan_requirements_create` the device extensions and features emberwave needs, `emberwave_context_create_vulkan` makes the context on the device and one of its queues, and optional lock callbacks keep the caller's own uses of the queue apart from emberwave's. Decodes into textures put the picture into Vulkan images: `emberwave_texture_vulkan` gives the image of a texture, `emberwave_texture_from_vulkan` and `emberwave_texture_from_vulkan_plane` wrap the caller's own, and `emberwave_decoder_submit_texture_vulkan` and `emberwave_decoder_submit_plane_textures_vulkan` decode without waiting, with semaphores to wait for and signal, binary or timeline, and a wrapped image's `release` callback says when the caller may destroy it. Between calls the images are in `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` (decision D61, [#115](https://github.com/with-developer/emberwave/issues/115)). In the Rust API, `CallerQueueLock` and `QueueTurns::with_caller_lock` hold such a lock around a context's submissions, `SubmissionHooks` add to a decode's submission (`Decoder::submit_assembled_to_texture_with` and `_to_plane_textures_with`), and a decode into textures leaves them ready for sampling.
- Encoding pictures that are in the caller's textures, through the C API, as a screen capture holds them: `emberwave_texture_from_metal_source` and `emberwave_texture_from_vulkan_source` wrap a Metal texture or a Vulkan image that holds one plane of an encoder's pictures. The plane can also be the new `EMBERWAVE_PLANE_CBCR`, Cb and Cr interleaved as in NV12's second plane. `emberwave_encoder_encode_textures` encodes three planes, or Y and Cb and Cr interleaved, into the bytes of the same picture in CPU memory. `emberwave_encoder_encode_textures_vulkan` adds semaphores to wait for and signal. Textures of planes that a decode filled encode too (decision D62, [#133](https://github.com/with-developer/emberwave/issues/133)). In the Rust API, `SubmissionHooks` also run around an encode's submission (`Encoder::encode_textures_with`), and `Encoder::device` and `Encoder::context_id` tell where an encoder runs.

## [0.13.0] - 2026-10-30

Streams decoded as they arrive, and a client's own Metal queue shared safely. The browser player takes a stream packet by packet, every frame carries the blocks that a decode in part needs, and a client chooses how much of a frame it draws in part. Calls on a context of a client's own Metal queue take turns at its command buffers, where threads that shared the queue could wait for good. Also the encoder's planes written in place on Apple silicon, upstream's encoder fixes tracked, and a command that measures any encoder's stream against its pictures.

### Added

- `GpuContext::with_queue_turns` and `QueueTurns`: the decoders and encoders of contexts made on one queue of the caller's, with `wgpu::Adapter::create_device_from_hal`, take turns at its command buffers, as the C API's on a client's Metal queue now do. `QueueTurns::take` gives a turn to the caller's own submissions (decision D60).
- `emberwave-cli quality SOURCE.y4m FILE.pwv1 [--frames N]` measures a stream from any encoder against the pictures it was encoded from: per frame and over all frames, its size, the PSNR of each plane and the coefficient error that the rate control minimises, weighted per band as it weighs it. `emberwave_core::encode::coefficient_error` returns that error for a frame (decision D57).
- A client chooses the share of a frame's blocks that a frame decoded in part needs, besides every block of the coarsest level: `gpu::Decoder::set_partial_share`, the browser player's `set_partial_share` and the C API's `emberwave_decoder_set_partial_share`, from 0 to 1, 0.9 by default as before. A lower share draws a frame that lost a burst of packets, with its finer detail blank, where it was skipped; the demo page's packet mode offers 90, 70 and 50 % (decision D59, [#123](https://github.com/with-developer/emberwave/issues/123)).
- The browser player decodes a stream packet by packet, as a client of a live stream receives it: `Player.open_stream` for a stream of a known size without a file, `push_packet`, `is_ready`, `draw_assembled`, `drop_packets` and `set_colors`, and `packet_ends`, `frames` and `frame` to cut a loaded file's frames into packets. The rules a transport follows, those of the three-bit sequence number among them, are in its documentation and in the C API's. The demo page's "Play as packets" (`?packets`, `?loss=0.05`) plays its stream that way and shows what a lost packet costs (decision D55, [#109](https://github.com/with-developer/emberwave/issues/109), [#120](https://github.com/with-developer/emberwave/issues/120)).

### Changed

- On Apple silicon the GPU encoder writes a picture's planes into the buffer its shaders read, rather than into an upload buffer copied at the start of every frame: a 4K picture encodes 0.02 to 0.04 ms faster frame after frame on an Apple M3 Max, and the upload buffer's 12 MB go (decision D54, [#106](https://github.com/with-developer/emberwave/issues/106)).
- The encoder sends every block of the coarsest level, a block with nothing left as an empty 8x8 block of 12 bytes, so that a frame that lost packets can be decoded in part: before, a frame whose coarsest chroma detail was zero could not be, since a block never sent counts as lost (decision D58). Frames grow by 12 bytes per such block, 72 bytes in a 1080p frame of the benchmark scene.
- The smallest frame the encoder makes is the header and those blocks at their smallest, 584 bytes for the 1080p scene: a smaller budget is `EncodeError::BudgetTooSmall`, where 8 bytes always fit before. The encoder vector `noise_64x64_444_empty` is now `noise_64x64_444_smallest`.
- `EncodeError` has a new variant, `Frame`, for a frame that `coefficient_error` cannot dequantise: a `match` on it that names every variant needs an arm for it.
- `emberwave_decoder_is_ready` takes `uint32_t allow_partial_frame` rather than `bool`, and any value but 0 allows a partial frame, so that no value a caller passes is an invalid one. C and C++ callers that pass `true` or `false` build as before; Swift callers pass 1 or 0 (decision D60, [#126](https://github.com/with-developer/emberwave/issues/126)).
- The tools that build upstream for the test vectors, the reference decodes and the comparisons follow upstream at 7fe0e26, which fixed two faults of upstream's encoder's rate control; its decoder makes every committed reference decode byte for byte, so the vectors stay. emberwave's encoder never had the faults. Against upstream's fixed encoder at equal bytes, emberwave's frames have 6.5 to 13 % less of the contrast-weighted error both encoders minimise, and luma PSNR from 0.8 dB lower to 2.3 dB higher; the README says so rather than calling it the better encoder (decision D56).

### Fixed

- Calls on contexts made with `emberwave_context_create_metal` could wait for good:
  - when threads shared the queue: a call makes 6 to 18 command buffers before it commits them, not two, so each thread held some and waited for more;
  - beside idle contexts on the queue: wgpu holds the writes of making a context, a decoder or an encoder in a command buffer until a submission, so each such context held one;
  - and when more contexts were made on a queue than it has room for.

  The decoders and encoders of every context made on one queue of the caller's now take turns at making and submitting command buffers, and none holds its turn while it waits for the GPU. Making a context, a decoder or an encoder there submits wgpu's writes at once, and so does a call that fails. Contexts on a client's queue are made one at a time. Contexts on emberwave's own queues are unchanged (decision D60, [#126](https://github.com/with-developer/emberwave/issues/126)).
- The C API's documentation gives those counts and the room a queue needs. It says, also at each call that waits and at the destroy functions, that nothing of emberwave may be called inside a completion handler of a command buffer of the queue, where a call that waits never returns.
- `emberwave_last_error`, and a failing call that records its error, no longer abort the process when called from a thread-local destructor that runs after emberwave's; the error is then empty. The documentation says the string lives no longer than its thread, and that a context keeps a few hundred bytes for every thread that calls it, until it and its objects are destroyed.

## [0.12.0] - 2026-10-29

Fewer stages per frame on Apple's GPUs: the planes stored straight into Metal textures, and on Apple silicon the frames written where the shaders read them. Also HEVC timed beside emberwave on the same Mac and iPhone, timings at a stream's pace in the benchmark app, in the browser benchmark and for upstream's decoder, a browser player that opens with a stream it makes itself, and a README in English and in Korean that shows the picture quality on a frame of a game.

### Changed

- On Metal, a decode into Y, Cb and Cr textures has the inverse transform's last levels store the planes into them, as upstream does, rather than copy them in (decision D52, [#86](https://github.com/with-developer/emberwave/issues/86)).
  - On an Apple M3 Max a 4K frame takes 0.08 to 0.12 ms less frame after frame, 0.91 to 0.93 ms through the C API instead of 1.03 to 1.05 ms, and a 1080p frame about 0.02 ms less. At 60 frames a second the gain is within the noise of the runs.
  - The decode stores into textures with the `STORAGE_BINDING` usage, which `create_plane_texture` and `emberwave_texture_create_plane` now give theirs; the C API gives it to the caller's Metal plane textures with `MTLTextureUsageShaderWrite`. Other textures, views that refuse the stores, and the other backends keep the copies of decision D41.
  - A context on Metal asks for `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`. `GpuOptions::plane_stores` turns the stores off, and `GpuContext::stores_planes` and `Decoder::stores_planes` tell whether they happen.
- On Apple silicon, a decode that waits for its frame writes the frame's block table and payload where its shaders read them, rather than into an upload buffer that the GPU copies from (decision D53, [#86](https://github.com/with-developer/emberwave/issues/86)).
  - On an Apple M3 Max at 60 frames a second, a 4K frame takes 0.14 ms less in the median of ten rounds, and a 1080p frame 0.06 ms less; frame after frame about 2 % less.
  - A context on Metal with an integrated GPU asks for `MAPPABLE_PRIMARY_BUFFERS`. `GpuOptions::shared_uploads` turns it off, and `GpuContext::shares_uploads` tells whether it happens.
- `GpuOptions` has the two new fields, `plane_stores` and `shared_uploads`: a struct literal of it needs them, or `..Default::default()`.
- `LICENSE` is the MIT license alone, so that GitHub reports it as MIT. PyroWave's license, and what the license does not cover, are in `THIRD_PARTY_NOTICES.md`, which the release packages carry next to it (decision D51, [#101](https://github.com/with-developer/emberwave/issues/101)).
- The README is in English, `README.md`, and in Korean, `README.ko.md`, which the command line tool's package carries too. `scripts/bench_report.py write` takes both and writes the benchmark summary into each (decision D50, [#99](https://github.com/with-developer/emberwave/issues/99)).

### Added

- The benchmark app also times HEVC through VideoToolbox, the decoder Moonlight uses, on the same device and pictures: encoding, and decoding into NV12 and BGRA textures, at Moonlight's rates and at emberwave's bytes per frame (decision D45, [#80](https://github.com/with-developer/emberwave/issues/80)). It times its decodes into textures at 60 frames a second as well as frame after frame, and `scripts/bench_app_on_mac.sh` runs it on a Mac. `BENCHMARKS.md` has the M3 Max's and the iPhone 17's results.
- `refgen bench --fps` times upstream's decoder at a stream's pace, and the browser benchmark page times every size at 60 frames a second after frame after frame ([#95](https://github.com/with-developer/emberwave/issues/95)). Decision D48 splits up where the time goes at that pace.
- The browser player opens with a 960x540 stream that it draws and encodes itself with the CPU encoder, before any file is given (decision D46, [#90](https://github.com/with-developer/emberwave/issues/90)).
- The README's front page: a banner and charts of decode and encode times, drawn by `scripts/readme_figures.py` from the numbers of `BENCHMARKS.md` (decision D47, [#87](https://github.com/with-developer/emberwave/issues/87)), and a frame of Crimson Desert within four budgets, made by `scripts/quality_figure.py`, with a key frame of Apple's HEVC encoder beside it in `BENCHMARKS.md` (decision D49, [#97](https://github.com/with-developer/emberwave/issues/97)).

### Fixed

- On a phone, the browser player's picture comes before its controls, so that the demo plays without scrolling ([#93](https://github.com/with-developer/emberwave/issues/93)).
- The benchmark app, started from a Mac with `-run`, waits until it has been in front for three seconds, and reports and fails a run that lost the front ([#80](https://github.com/with-developer/emberwave/issues/80)).

## [0.11.0] - 2026-10-24

Decoding into the Metal client's own textures, such as the texture of a `CVPixelBuffer` for an `AVSampleBufferDisplayLayer`, a C API whose documentation sets out how decodes and the client's reads are ordered, and an iPhone and iPad benchmark app that times and checks every way into textures.

### Added

- Decoding into the Metal client's own textures (decision D44, [#63](https://github.com/with-developer/emberwave/issues/63)).
  - `emberwave_texture_from_metal` and `emberwave_texture_from_metal_plane` wrap an `id<MTLTexture>` of the client's, such as the BGRA texture of a `CVPixelBuffer` for an `AVSampleBufferDisplayLayer`. Every decode into textures then fills it.
  - A copy fills a texture that is not framebuffer-only, as a `CVMetalTextureCache`'s is not, at a width of a multiple of 64 pixels. A draw fills a render target.
  - The decodes that do not wait take only textures whose hazards Metal tracks.
  - In Rust, a picture's target needs `RENDER_ATTACHMENT` only where a draw fills it.
  - The plane decodes refuse one Metal texture wrapped for two planes.
- The C API's documentation sets out in one place, under `EmberwaveTexture`, how the decodes and the caller's reads of a texture are ordered on each kind of context, and how many textures to keep in turn.

### Changed

- The iPhone and iPad benchmark app also times decoding into Y, Cb and Cr textures and on the app's own queue, and checks both through Metal: the planes against those decoded to memory, and a decode on the app's queue against the RGB decode, read by the app's next command buffer without a wait. CI runs the checks in an iOS Simulator.

## [0.10.0] - 2026-10-24

More for Metal clients: the Y, Cb and Cr planes in textures, the fastest way to the GPU; the RGB picture into textures with a copy rather than a draw; and decoding on the client's own queue, which needs no wait and one texture.

### Added

- The Y, Cb and Cr planes in textures, for Metal clients that convert the colours themselves (decision D41, [#64](https://github.com/with-developer/emberwave/issues/64)).
  - `emberwave_texture_create_plane` makes a one-channel texture (`MTLPixelFormatR8Unorm`) of one plane's size, for that plane.
  - `emberwave_decoder_decode_plane_textures` copies the planes of the frame into three of them and waits. They hold the stream's samples, byte for byte.
  - The fastest way to the GPU: on an Apple M3 Max a 4K frame takes 1.0 to 1.1 ms, against 1.3 to 1.4 ms into an RGB texture, and a 1080p frame 0.45 ms against 0.56 ms. In 4:2:0 they take 1.5 bytes a pixel, where RGBA takes 4.
  - In Rust, `Decoder::encode_to_plane_textures`, `decode_assembled_to_plane_textures` and `create_plane_texture`, and `PLANE_FORMAT`.
- Decoding on the Metal client's own queue, without waiting (decision D43, [#62](https://github.com/with-developer/emberwave/issues/62), [#63](https://github.com/with-developer/emberwave/issues/63)).
  - `emberwave_context_create_metal` makes a context on the client's `id<MTLDevice>` and `id<MTLCommandQueue>`.
  - On it, `emberwave_decoder_submit_texture` and `emberwave_decoder_submit_plane_textures` return once the work is on the queue. Metal runs the client's command buffers committed afterwards after the decode, and the next decode after them, so one texture takes every frame.
  - On an Apple M3 Max a 4K frame takes the decoding thread 0.26 ms instead of 1.3 ms, and the client's draw completes as early as after the waiting decode, at 1080p 0.1 ms earlier.
  - In Rust, `Decoder::submit_assembled_to_texture` and `submit_assembled_to_plane_textures`, and `GpuContext::device_descriptor` and `from_device` for a context on a device made elsewhere.

### Changed

- A decoder's packed planes have rows of a multiple of 256 bytes, so that copies take them into textures: 7% more memory for them at 1080p, and as much more to read back to the CPU.
- The RGB picture goes into a texture with a copy, not a draw, when the texture takes copies (`COPY_DST`) and the picture's width is a multiple of 64 pixels, as 720p, 1080p, 1440p and 4K are (decision D42, [#70](https://github.com/with-developer/emberwave/issues/70)). The colour conversion writes blue first for a BGRA texture. On an Apple M3 Max a 4K frame into a texture through the C API takes 1.3 to 1.4 ms instead of 1.5 ms, and a 1080p frame 0.56 ms instead of 0.71 ms. The textures of `emberwave_texture_create` and `Decoder::create_texture` take copies. The browser player still draws its canvas, as wgpu offers no copies into one.

## [0.9.0] - 2026-10-24

Pictures straight into Metal textures for iPhone, iPad and Mac apps, without a trip through CPU memory, and an iPhone and iPad app that times emberwave on the device's own GPU.

### Added

- An iPhone and iPad benchmark app, `crates/emberwave-ffi/examples/ios` ([#66](https://github.com/with-developer/emberwave/issues/66)). It times decoding into Metal textures, decoding to planes in memory and encoding on the device's GPU through the C API, with the pictures and budgets of the browser benchmark. It also draws the last picture from its texture. CI runs it in an iOS Simulator, where it also reads a texture back through Metal.
- Pictures on the GPU for Metal clients (decision D40, [#27](https://github.com/with-developer/emberwave/issues/27)).
  - `emberwave_texture_create` makes a texture of the picture's size, RGBA or BGRA, on the decoder's device.
  - `emberwave_decoder_decode_texture` draws the RGB picture into it and waits.
  - `emberwave_texture_metal` and `emberwave_context_metal_device` give the `id<MTLTexture>` and `id<MTLDevice>` behind them.
  - On an Apple M3 Max a 4K frame takes 1.5 ms that way, against 2.7 ms to RGB in CPU memory.
  - In Rust, `Decoder::encode_rgb_to_texture`, `decode_assembled_to_texture` and `create_texture` draw into a wgpu texture, and `GpuContext::id` tells contexts apart.

### Changed

- The browser player draws its canvas through `Decoder::encode_rgb_to_texture`, the same pass as the C API's textures.

## [0.8.0] - 2026-10-24

Faster native clients: a decoded picture reaches the CPU in a third of the time, through the C API and the command line tool, and pictures in textures encode faster. Also fixes from a review of the browser pages, which now hold a file once, survive a lost GPU and refuse files that would ask for gigabytes.

### Added

- `Decoder::decode_assembled_rows` and `Decoder::decode_assembled_rgb_rows` hand the decoded picture to a closure row by row, straight from the memory the GPU wrote, so that a caller can put it where it wants it with one copy.

### Changed

- Encoding from textures is faster (decision D38, [#40](https://github.com/with-developer/emberwave/issues/40)). The gather pass takes the texels four at a time with `textureGather`, where it loaded them one by one. On an Apple M3 Max a 4K frame from textures encodes in 1.71 ms instead of 1.80 ms.
- Reading a decoded picture back to the CPU is much faster (decision D37, [#56](https://github.com/with-developer/emberwave/issues/56)): it comes straight out of the mapped buffer, and RGBA turns into RGB in a loop the compiler vectorises, where a copy per pixel took 4.5 ms at 4K. On an Apple M3 Max a 4K frame through the C API takes 2.7 ms to RGB instead of 7.6 ms, and 1.2 ms to planes instead of 1.6 ms; at 1080p 0.8 and 0.5 ms instead of 2.2 and 1.1 ms. `emberwave-cli decode` gains as much. A failed `emberwave_decoder_decode_yuv` or `_decode_rgb` may now have written some rows.

### Fixed

- The browser pages, from a review of them (decision D39):
  - The benchmark page gave times for frames the GPU never decoded when the browser reported an error; it now fails that size with the error.
  - The player held a file about twice in memory, and a file with millions of empty frames could panic the module, after which nothing worked until the page was reloaded. It now keeps one copy, refuses frames too short to decode, plays the frames before one that is cut off rather than nothing, and loads files of up to 1 GiB and pictures of up to 8192x4320 pixels' worth.
  - After a lost GPU the next file gets a new device; after a panic the page asks to be reloaded.
  - The benchmark frees its GPU device and wake lock however a run ends, marks a run during which the page was hidden, and no longer blocks for seconds on a coarse timer.
  - Errors from the browser show their message without the JavaScript stack.
- The decoder makes its read-back buffers only when a decode first reads back: a player, in a browser or natively, saves 43.5 MiB at 4K.

## [0.7.0] - 2026-10-24

Apps on iPhones, iPads and Macs: the C library as an XCFramework that Swift imports, checked in an iOS Simulator, where contexts can now be created. Also a benchmark page for browsers, phones included, a faster CPU side of decoding, and fixes from a review of the command line tool.

### Added

- A benchmark page for browsers, [`bench.html`](https://with-developer.github.io/emberwave/bench.html) next to the player: it encodes test streams at 1080p, 1440p and 4K in the page and times their decoding to RGB in GPU memory frame by frame, on any device with WebGPU, a phone included, and gives a report to copy.
- The C library for apps on iPhones, iPads and Macs, as `Emberwave.xcframework` (decision D35, [#52](https://github.com/with-developer/emberwave/issues/52)). Each release has it as `emberwave-ffi-VERSION-xcframework.zip`. It holds static libraries for devices (iOS 12 or later), for the iOS Simulator on Apple silicon and for Macs with Apple silicon, with the header and a module map, so that Swift can `import Emberwave`. `scripts/build_xcframework.sh` builds it, and `crates/emberwave-ffi/examples/swift` is the decoding example in Swift. CI runs the C and Swift examples, linked with it, on a Mac and in an iOS Simulator.

### Changed

- Decoding is faster on the CPU's side (decision D34, [#45](https://github.com/with-developer/emberwave/issues/45)): the decodes that wait for their frame write it into an upload buffer the decoder keeps rather than through `Queue::write_buffer`, which takes new memory for every frame. On an Apple M3 Max a 4K frame decodes to planes in about 0.89 ms instead of 1.06 ms, the CPU's part falling from 0.23 to 0.08 ms. `encode_rgb` still writes through the queue.

### Fixed

- `emberwave-cli decode` reported success when writing a PNG failed, on a full disk or past a file size limit, and left an empty or cut file: the PNG writer was dropped without being finished. It now fails, naming the file.
- Output piped into a program that stops reading, such as `emberwave-cli info FILE | head`, made the command panic; it now ends quietly.
- Errors of `emberwave-cli` name the file they are about. `compare` reads both files frame by frame rather than whole, `bench-encode` refuses a picture size the format cannot have before it reads the video, the Y4M reader reads no header or frame line longer than 64 KiB, `compare --min-psnr` refuses NaN, which every PSNR passed, and `decode` of a PWV1 file without frames is an error, as it is for `bench` and `compare`.
- `FrameAssembler` ignores a block of an older frame even when its size is wrong, as upstream does, rather than rejecting the packet.
- No context could be created where the adapter falls short of a WebGPU default limit that emberwave does not need. Metal gives the iOS Simulator, and Apple GPUs before the A11, 15 inter-stage variables, one fewer than the default. A context now asks for no limit beyond the adapter's (decision D36, [#54](https://github.com/with-developer/emberwave/issues/54)).

## [0.6.0] - 2026-10-24

Encoding for hosts: the encoder in the C API, with planes at any row stride and packets for the network, and in Rust from pictures that are on the GPU already, such as a screen capture's.

### Added

- An encoder in the C API ([#42](https://github.com/with-developer/emberwave/issues/42)): `emberwave_encoder_create`, `_encode`, `_info` and `_destroy` encode planes with row strides into a frame in the caller's buffer, with the colour signalling of `EmberwaveSignalling`, and `emberwave_frame_packets` splits a frame into packets for a decoder (decision D33). `examples/c/encode.c` shows them in use.
- `gpu::Encoder::encode_strided` encodes planes whose rows are a stride apart, and a stride below the width is the new `EncodeError::PlaneStride`.
- Encoding pictures that are on the GPU already ([#33](https://github.com/with-developer/emberwave/issues/33)): `gpu::Encoder::encode_textures` takes a `TexturePicture`, three `R8Unorm` textures or NV12's layout of an `R8Unorm` and an `Rg8Unorm` texture, and makes the bytes `encode` makes of the same picture in CPU memory, without the CPU writing the planes (decision D32). An unsuitable texture is the new `GpuError::Texture`. `emberwave-cli bench-encode --gpu --textures` times it.

### Changed

- `GpuError` has a new variant, `Texture`, and `EncodeError` one, `PlaneStride`; a `match` over every variant needs arms for them.
- A context asks for the adapter's largest 2D textures, as it asks for its largest buffers, so that the textures of pictures to encode can be as large as the pictures; one limited to WebGPU's guarantees keeps 8192.

## [0.5.0] - 2026-10-24

The GPU encoder overtakes upstream's: on an Apple M3 Max it encodes a 4K frame in about 1.9 ms, against 3.4 ms for upstream's encoder at the same precision, and `bench-encode` measures both. Making a decoder or an encoder on a lost device, and making contexts on several threads with some Linux Vulkan loaders, no longer crash.

### Added

- Encoder benchmarks ([#29](https://github.com/with-developer/emberwave/issues/29)): `emberwave-cli bench-encode` times an encoder on the frames of a Y4M video, in the way `bench` times the decoder, and `Encoder::encode_timed` returns `EncodeTimings` with the GPU time of each stage. `refgen bench-encode` times upstream's encoder the same way, at both of its precisions, and `scripts/bench.sh` puts them in `BENCHMARKS.md`, with the pictures from `scripts/gen_bench_data.sh --sources`.

### Changed

- The GPU encoder is faster (decision D30): on an Apple M3 Max a 4K frame encodes in about 1.9 ms instead of about 3.5 ms, against 3.4 ms for upstream's encoder at the same precision. The planes go through an upload buffer the encoder keeps from frame to frame, 12 MB of GPU memory at 4K that the old path allocated anew for every frame, and the quantisation adds each subblock's errors in registers. The CPU encoder adds its errors in the same grouping; its streams are the same as before.

### Fixed

- `Decoder::time_frame` sometimes reported the previous frame's time for its last stages, or none at all, on Metal: the timestamps were resolved in the frame's command buffer, before the GPU had written them. They are now resolved once the frame is done, in a submission of their own, which `gpu_wall` does not include.
- `gpu::Decoder::new` and `gpu::Encoder::new` panicked inside wgpu when the context's device had been lost, a destroyed device for one ([#35](https://github.com/with-developer/emberwave/issues/35)). They now return an error, `GpuError::DeviceLost` as decoding and encoding do, and `emberwave_decoder_create` in the C API returns `EMBERWAVE_STATUS_DEVICE_LOST` rather than `EMBERWAVE_STATUS_INTERNAL`.
- On Linux with a Vulkan loader from 1.3.284 to 1.4.344, such as Debian 13's or Ubuntu 26.04's, creating contexts on several threads at once could crash the process inside the loader ([#36](https://github.com/with-developer/emberwave/issues/36)). `GpuContext::new` and `emberwave_context_create` now create one context at a time (decision D31).

## [0.4.0] - 2026-10-24

emberwave encodes: an encoder on the CPU and on the GPU whose streams upstream's decoder reads, at upstream's PSNR or above for the same bytes. On an Apple M3 Max the GPU encodes a 4K frame in about 4 ms, the CPU in about 240 ms.

### Added

- An encoder on the CPU ([#14](https://github.com/with-developer/emberwave/issues/14)): `emberwave_core::encode::Encoder` turns 8-bit pictures into frames within a byte budget, and `emberwave-cli encode` turns a Y4M video into a PWV1 file. Upstream's decoder reads its streams; at the same bytes its PSNR is upstream's encoder's or higher. It makes the same bytes on every system.
- Writers for the formats the parsers read: `BlockHeader::to_words`, `SequenceHeader::start_of_frame_words` and `stream_words`, and `Pwv1Header::to_bytes`.
- Test vectors of the encoder in `testdata/encoder/`, with upstream's decode of them, and `scripts/gen_encoder_vectors.sh` to make them. `refgen encode` can write its pictures (`--source`) and encode a Y4M file (`--input`).
- An encoder on the GPU ([#14](https://github.com/with-developer/emberwave/issues/14)): `gpu::Encoder` runs the CPU encoder's steps as compute shaders, the forward transform, quantisation, rate control and packing, with one upload and one read-back per frame. Its frames are the CPU encoder's, or differ in some blocks where the GPU rounds a coefficient differently, at the same PSNR; on an Apple M3 Max the committed vectors come out byte for byte, and a 4K frame takes about 4 ms. `emberwave-cli encode --gpu` uses it, with `--backend` and `--software` as for decoding.

### Changed

- `emberwave-cli`'s Y4M reader keeps the range, the chroma siting and the frame rate of the header, reads `C420paldv` as chroma on the left, and reads a file a frame at a time where that is enough.
- `GpuError` has a new variant, `Encode`, for the GPU encoder's refusals; a `match` over every variant needs an arm for it.
- `emberwave-cli encode` ends with the mean, median and 99th percentile of the time per frame instead of the mean alone.

## [0.3.0] - 2026-10-24

emberwave outside Rust: a player in the browser, published at https://with-developer.github.io/emberwave/, and a C API with prebuilt libraries for Linux, macOS and Windows.

### Added

- A browser player ([#15](https://github.com/with-developer/emberwave/issues/15)): `crates/emberwave-web` and the page in `web/`, published to GitHub Pages, decode `.pwv1` files with WebGPU in the page and draw them on a canvas.
- `Decoder::encode_rgb` records a decode into the caller's command encoder without waiting, and `Decoder::rgba_buffer` gives the picture; `GpuContext::from_adapter` creates a context on an adapter the caller chose. Together they let a player keep the picture on the GPU, and they are what the browser player uses.
- A C API ([#16](https://github.com/with-developer/emberwave/issues/16)): `crates/emberwave-ffi` with the header `include/emberwave.h`, made by cbindgen. Contexts and decoders, packets pushed as they arrive, and the decoded planes or RGB written into the caller's memory; status codes with a message per thread, and panics caught at the boundary. Each release ships the header with a shared and a static library for Linux, macOS and Windows, and a C example decodes every test vector in CI on all three.
- `Decoder::clear`, `Decoder::start_of_frame` and `Decoder::lost`, which the C API needed.

### Changed

- `GpuContext::new`, which blocks, is not part of the WebAssembly build; use `GpuContext::new_async` or `GpuContext::from_adapter` there.
- A `Decoder` destroys its GPU buffers when it is dropped. Native wgpu freed them then anyway; in a browser they waited for the garbage collector. A clone of the buffer from `Decoder::rgba_buffer` does not keep it alive.

## [0.2.0] - 2026-10-24

Faster decoding: on an Apple M3 Max a 4K frame decodes to planes in 1.11 ms instead of 4.28 ms (1.37 instead of 4.53 ms with the conversion to RGB), 1.4 times the time of upstream's own Metal decoder at FP32 ([#11](https://github.com/with-developer/emberwave/issues/11), [#12](https://github.com/with-developer/emberwave/issues/12)). The decoded pictures are the same up to rounding ties: in the test vectors 5 of 464,790 samples now differ by one level, as the compiler fuses a multiply and an add differently in the rewritten transform.

### Changed

- The inverse transform and the dequantiser keep the values of an invocation in registers, written out with constant indices: the transform is 4.5 times faster.
- The dequantiser finds the band of a block by binary search. With the change above it is 2.6 times faster.
- The last level of the inverse transform writes the 8-bit planes itself. The separate packing pass and the float planes (47.5 MiB at 4K) are gone.
- The frame assembler validates blocks without building their full structure, 2.8 times faster.
- Workgroup memory is no longer zeroed before every workgroup, which the shaders never needed: the inverse transform is another 1.7 times faster. On WARP, Direct3D's software rasteriser, the first frame no longer takes a minute and a half.
- Software rendering on lavapipe is about a third faster too (4K: 46 ms instead of 69 ms).

### Removed

- `StageTimings::pack`: there is no packing pass any more, and the time of the 8-bit output is part of `idwt`.

## [0.1.0] - 2026-10-23

The first release: a decoder for PyroWave bitstream v1 that runs the same WGSL on Vulkan, Metal and DX12 through wgpu, and a command line tool.

### Added

- `emberwave_core::bitstream`: the PyroWave v1 bitstream, with headers, block layout, packet assembly following upstream's sequence rules, and the PWV1 file format. Malformed input gives errors, never panics (property tests).
- `emberwave_core::cpu`: a reference decoder in plain Rust, used as the test oracle.
- `emberwave_core::gpu`: the wgpu/WGSL decoder. Dequantisation, the five-level inverse CDF 9/7 transform and 8-bit packing on the GPU, with an optional conversion to RGB (BT.601, BT.709 or BT.2020; full or limited range; centred or left chroma siting) and per-pass GPU timings.
- `emberwave_core::color`: the YCbCr to RGB conversion on the CPU, which the shader is tested against.
- Errors instead of panics: wrong-sized planes, device limits, GPU validation, out-of-memory and internal errors come back as values, and a lost GPU device is reported as `GpuError::DeviceLost` instead of returning stale results.
- `emberwave-cli` with `decode` (PNG or Y4M), `bench` (text or JSON), `info` and `compare` (PSNR per plane).
- Scripts to generate test vectors with upstream's encoder, to check a 4K decode against upstream's decoder, to benchmark (Metal, upstream's decoder, lavapipe) and to watch memory over 1000 frames.

### Verified

- Matches upstream's decoder on streams from upstream's encoder: at least 64 dB PSNR per plane against its default precision and 86 dB against its FP32 mode; a 4K stream reaches 65.6 dB and 96.3 dB. The largest difference is one 8-bit step.
- The same WGSL runs in CI on Vulkan (lavapipe), Metal and DX12 (WARP), and validates with no optional GPU capability.
- 1000 decoded 4K frames leave the GPU resources and the memory of the process where they were.

### Performance

On an Apple M3 Max with Metal, a 1080p frame decodes to RGB in GPU memory in 1.5 ms and a 4K frame in 4.5 ms ([`BENCHMARKS.md`](BENCHMARKS.md)). Upstream's own Metal decoder is still several times faster on the same machine ([#11](https://github.com/with-developer/emberwave/issues/11)).

### Known limitations

- The macOS binary is not signed. After downloading, run `xattr -d com.apple.quarantine emberwave-cli`.
- The Linux binary needs the Vulkan loader (`libvulkan1`) and a Vulkan driver, and a glibc at least as new as that of GitHub's `ubuntu-latest` runner, where it is built.
- Output is 8-bit, in 4:2:0 or 4:4:4. There is no encoder and no WebGPU build yet.
