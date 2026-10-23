# Changelog

All notable changes to emberwave. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- A browser player ([#15](https://github.com/with-developer/emberwave/issues/15)): `crates/emberwave-web` and the page in `web/`, published to GitHub Pages, decode `.pwv1` files with WebGPU in the page and draw them on a canvas.
- `Decoder::encode_rgb` records a decode into the caller's command encoder without waiting, and `Decoder::rgba_buffer` gives the picture; `GpuContext::from_adapter` creates a context on an adapter the caller chose. Together they let a player keep the picture on the GPU, and they are what the browser player uses.

### Changed

- `GpuContext::new`, which blocks, is not part of the WebAssembly build; use `GpuContext::new_async` or `GpuContext::from_adapter` there.

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
