# Changelog

All notable changes to emberwave. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

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
