# emberwave

[![CI](https://github.com/with-developer/emberwave/actions/workflows/ci.yml/badge.svg)](https://github.com/with-developer/emberwave/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)

A portable ([wgpu](https://wgpu.rs)) port of the [PyroWave](https://github.com/Themaister/pyrowave) low-latency intra-frame wavelet video codec.

PyroWave is an intra-only codec built for very fast GPU encode and decode (upstream reports under ~0.1 ms at 1080p and ~0.2 ms at 4K) for local-network game streaming. Its reference implementation is written as Vulkan compute shaders. **emberwave reimplements the decoder in Rust + WGSL on top of wgpu**, so a single code base can target Vulkan, Metal, DX12 and WebGPU.

> **Status: pre-alpha.** The wgpu/WGSL decoder matches the original decoder on real encoder output (at least 64 dB PSNR against its default precision and 86 dB against its FP32 mode; CI runs the tests on Linux with the lavapipe Vulkan driver, on macOS with Metal and on Windows with DX12 on WARP). It converts to RGB on the GPU and comes with a command line tool that decodes files to PNG or Y4M. Benchmarks and the first release come next. Follow progress in the roadmap below.

## Goals

- Decode bitstreams produced by the original PyroWave encoder, straight from a file, and match the reference decoder's output. PyroWave decoding is floating-point and not bit-exact by design, so the acceptance bar is PSNR ≥ 60 dB per plane against the reference decoder.
- Rely only on features wgpu guarantees on every backend: no Vulkan-only extensions and no subgroup operations.
- Measure per-frame decode time at 1080p / 1440p / 4K on macOS (Metal) and publish the numbers together with their environment.

Out of scope: network transport, RTP and FEC.

## Roadmap

Each milestone is tracked in its own issue.

- [x] **M0** Bootstrap: workspace, CI, documentation skeleton, license ([#1](https://github.com/with-developer/emberwave/issues/1))
- [x] **M1** Bitstream format and parser ([#2](https://github.com/with-developer/emberwave/issues/2))
- [x] **M2** GPU decode pipeline: dequantisation and inverse DWT ([#3](https://github.com/with-developer/emberwave/issues/3))
- [x] **M3** Colour conversion and CLI ([#4](https://github.com/with-developer/emberwave/issues/4))
- [ ] **M4** Benchmarks and documentation ([#5](https://github.com/with-developer/emberwave/issues/5))
- [ ] **M5** Stabilisation and the v0.1.0 release ([#6](https://github.com/with-developer/emberwave/issues/6))

Later, time permitting: an encoder port with round-trip tests, a WebGPU demo (wasm), and a C FFI for Moonlight-style clients.

## Repository layout

| Path | Purpose |
|---|---|
| `crates/emberwave-core` | Bitstream parser and wgpu/WGSL decode pipeline |
| `crates/emberwave-cli` | `emberwave-cli` command line tool |
| `crates/emberwave-ffi` | C FFI (placeholder) |
| `testdata/` | Test vectors and how to regenerate them |
| `scripts/` | Test data and benchmark scripts |
| `ARCHITECTURE.md` | Design notes and the decision log |

## Building

Requires a recent stable Rust toolchain.

```sh
cargo build --workspace
cargo test --workspace
```

## Usage

`emberwave-cli` decodes PyroWave files in the PWV1 format (`.pwv1`). The examples use the vectors in `testdata/`, which come from the upstream encoder, and assume `emberwave-cli` is on the `PATH`; build it with `cargo build --release -p emberwave-cli` and find it in `target/release/`.

```sh
# What is in a file
emberwave-cli info testdata/gradient_200x120_420.pwv1

# Decode to one PNG per frame, or to a Y4M video of the YUV planes
emberwave-cli decode testdata/gradient_200x120_420.pwv1 --out-dir out
emberwave-cli decode testdata/gradient_200x120_420.pwv1 --out-dir out --format y4m

# Time the decoder; --output rgb includes the colour conversion, --json prints JSON
emberwave-cli bench testdata/gradient_200x120_420.pwv1 --output rgb

# Compare two Y4M files plane by plane; fails if a plane is below 60 dB PSNR
emberwave-cli compare out/gradient_200x120_420.y4m testdata/gradient_200x120_420.ref.y4m --min-psnr 60
```

The decoder uses whatever GPU the machine offers: Vulkan, Metal or DX12, or a software driver such as Mesa's lavapipe or Windows' WARP. `--backend vulkan|metal|dx12` and `--software` narrow the choice, as does the `EMBERWAVE_BACKEND` environment variable, and the adapter in use is printed. The RGB conversion follows what the file signals (matrix, range, chroma siting); `--matrix bt601|bt709|bt2020` and `--range full|limited` override it. [`ARCHITECTURE.md`](ARCHITECTURE.md#command-line-tool) lists every option and explains how the timings are measured.

## Relationship to upstream

emberwave follows PyroWave's algorithm structure and its [bitstream definition](https://github.com/Themaister/pyrowave/blob/79a3232a2199281f33f7a2e2659fc4fac62b022a/bitstream/bitstream.md), but contains no code copied from the upstream repository. Test vectors and reference output come from upstream's own encoder and decoder; see [`testdata/README.md`](testdata/README.md).

Upstream also ships an Apple-only Metal port. emberwave aims at one portable code base that additionally covers Vulkan, DX12 and the browser.

## License

MIT. See [`LICENSE`](LICENSE), which also carries the PyroWave copyright notice.

## 한국어 요약

emberwave는 Themaister의 PyroWave(Vulkan 컴퓨트 셰이더 기반 GPU 웨이블릿 코덱)의 **디코더를 Rust + wgpu + WGSL로 재구현**해서 Vulkan / Metal / DX12 / WebGPU에서 같은 코드로 동작하게 만드는 프로젝트입니다. 현재는 프리알파 단계로, 비트스트림 파서, wgpu/WGSL 디코더(역양자화와 역 DWT), GPU에서의 YUV→RGB 변환, 그리고 PNG/Y4M으로 디코딩하는 CLI(`emberwave-cli`)가 동작하며 원본 디코더와 PSNR 64 dB 이상(FP32 모드 기준 86 dB 이상)으로 일치합니다. 성능 측정(M4)과 첫 릴리스(M5)가 남아 있습니다. 1차 목표는 원본 인코더가 만든 비트스트림을 원본 디코더와 PSNR 60 dB 이상으로 일치하게 디코딩하고, macOS(Metal)에서 1080p / 1440p / 4K 프레임당 디코딩 시간을 측정해 환경과 함께 공개하는 것입니다.
