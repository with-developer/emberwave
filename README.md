# emberwave

[![CI](https://github.com/with-developer/emberwave/actions/workflows/ci.yml/badge.svg)](https://github.com/with-developer/emberwave/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)

A portable ([wgpu](https://wgpu.rs)) port of the [PyroWave](https://github.com/Themaister/pyrowave) low-latency intra-frame wavelet video codec.

PyroWave is an intra-only codec built for very fast GPU encode and decode (upstream reports under ~0.1 ms at 1080p and ~0.2 ms at 4K) for local-network game streaming. Its reference implementation is written as Vulkan compute shaders. **emberwave reimplements the decoder in Rust + WGSL on top of wgpu**, so a single code base can target Vulkan, Metal, DX12 and WebGPU.

> **Status: pre-alpha.** The bitstream parser and a CPU reference decoder exist and match the original decoder on real encoder output; the GPU decoder is next. Follow progress in the roadmap below.

## Goals

- Decode bitstreams produced by the original PyroWave encoder, straight from a file, and match the reference decoder's output. PyroWave decoding is floating-point and not bit-exact by design, so the acceptance bar is PSNR ≥ 60 dB per plane against the reference decoder.
- Rely only on features wgpu guarantees on every backend: no Vulkan-only extensions and no subgroup operations.
- Measure per-frame decode time at 1080p / 1440p / 4K on macOS (Metal) and publish the numbers together with their environment.

Out of scope: network transport, RTP and FEC.

## Roadmap

Each milestone is tracked in its own issue.

- [x] **M0** Bootstrap: workspace, CI, documentation skeleton, license ([#1](https://github.com/with-developer/emberwave/issues/1))
- [x] **M1** Bitstream format and parser ([#2](https://github.com/with-developer/emberwave/issues/2))
- [ ] **M2** GPU decode pipeline: dequantisation and inverse DWT ([#3](https://github.com/with-developer/emberwave/issues/3))
- [ ] **M3** Colour conversion and CLI ([#4](https://github.com/with-developer/emberwave/issues/4))
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

## Relationship to upstream

emberwave follows PyroWave's algorithm structure and its [bitstream definition](https://github.com/Themaister/pyrowave/blob/79a3232a2199281f33f7a2e2659fc4fac62b022a/bitstream/bitstream.md), but contains no code copied from the upstream repository. Test vectors and reference output come from upstream's own encoder and decoder; see [`testdata/README.md`](testdata/README.md).

Upstream also ships an Apple-only Metal port. emberwave aims at one portable code base that additionally covers Vulkan, DX12 and the browser.

## License

MIT. See [`LICENSE`](LICENSE), which also carries the PyroWave copyright notice.

## 한국어 요약

emberwave는 Themaister의 PyroWave(Vulkan 컴퓨트 셰이더 기반 GPU 웨이블릿 코덱)의 **디코더를 Rust + wgpu + WGSL로 재구현**해서 Vulkan / Metal / DX12 / WebGPU에서 같은 코드로 동작하게 만드는 프로젝트입니다. 현재는 프리알파 단계로 워크스페이스와 CI 뼈대만 있습니다. 1차 목표는 원본 인코더가 만든 비트스트림을 원본 디코더와 PSNR 60 dB 이상으로 일치하게 디코딩하고, macOS(Metal)에서 1080p / 1440p / 4K 프레임당 디코딩 시간을 측정해 환경과 함께 공개하는 것입니다.
