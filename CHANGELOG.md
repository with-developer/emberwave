# Changelog

All notable changes to emberwave. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

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
