# emberwave

[![CI](https://github.com/with-developer/emberwave/actions/workflows/ci.yml/badge.svg)](https://github.com/with-developer/emberwave/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)

A portable ([wgpu](https://wgpu.rs)) port of the [PyroWave](https://github.com/Themaister/pyrowave) low-latency intra-frame wavelet video codec.

**[Play a stream in your browser](https://with-developer.github.io/emberwave/)**, decoded by WebGPU in the page (a recent Chrome or Edge has it).

PyroWave is an intra-only codec built for very fast GPU encode and decode (upstream reports under ~0.1 ms at 1080p and ~0.2 ms at 4K) for local-network game streaming. Its reference implementation is written as Vulkan compute shaders. **emberwave reimplements the decoder in Rust + WGSL on top of wgpu**, so a single code base can target Vulkan, Metal, DX12 and WebGPU.

> **Status: pre-alpha.** The wgpu/WGSL decoder matches the original decoder on real encoder output, up to 4K (at least 64 dB PSNR against its default precision and 86 dB against its FP32 mode). CI runs the tests on Linux with the lavapipe Vulkan driver, on macOS with Metal and on Windows with DX12 on WARP. Decoding works, converts to RGB on the GPU and comes with a command line tool, a browser player and a C API. An encoder makes streams that upstream's decoder reads, at upstream's PSNR or above for the same bytes, on the CPU or on the GPU, which encodes a 4K frame in about 1.9 ms on the same machine, faster than upstream's own encoder (3.2 to 3.4 ms at the same precision), also from pictures that are in GPU memory already. A 4K frame decodes in about 0.9 ms on an Apple M3 Max (1.2 ms including the conversion to RGB), 1.0 to 1.25 times the time of upstream's own Metal decoder at FP32, depending on the run (see [Performance](#performance)). Releases and their changes are in [`CHANGELOG.md`](CHANGELOG.md).

## Quick start

You need [Rust](https://rustup.rs) 1.87 or newer and a GPU with Vulkan, Metal or DX12. Without a GPU, a software driver works too: Mesa's lavapipe on Linux (`sudo apt install mesa-vulkan-drivers libvulkan1` on Debian and Ubuntu), WARP on Windows, which is built in.

```sh
git clone https://github.com/with-developer/emberwave.git
cd emberwave
cargo build --release -p emberwave-cli
target/release/emberwave-cli decode testdata/gradient_200x120_420.pwv1 --out-dir out
```

The first build takes a few minutes. The last command prints the adapter it uses and the time of each frame, and leaves `out/gradient_200x120_420_000000.png` to `000002.png`: a colour gradient. The `.pwv1` files in `testdata/` are small streams made by upstream's encoder.

Without Rust, take `emberwave-cli` for Linux (x86_64), macOS (Apple Silicon) or Windows (x86_64) from the [releases](https://github.com/with-developer/emberwave/releases) and decode a `.pwv1` file from `testdata/` with it. The macOS binary is not signed, so run `xattr -d com.apple.quarantine emberwave-cli` once after downloading it.

## Usage

`emberwave-cli` decodes PyroWave files in the PWV1 format (`.pwv1`). The examples assume `emberwave-cli` is on the `PATH` (`cargo install --path crates/emberwave-cli` puts it there, or use `target/release/emberwave-cli`).

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

# Encode a Y4M video (8-bit 4:2:0 or 4:4:4), each frame within 6000 bytes;
# --gpu encodes on the GPU
emberwave-cli encode testdata/encoder/scene_200x120_420.source.y4m --out scene.pwv1 --bytes-per-frame 6000 --gpu

# Time the encoder, on the GPU here
emberwave-cli bench-encode testdata/encoder/scene_200x120_420.source.y4m --bytes-per-frame 6000 --gpu
```

The decoder, and the encoder with `--gpu`, use whatever GPU the machine offers: Vulkan, Metal or DX12, or a software driver such as Mesa's lavapipe or Windows' WARP. `--backend vulkan|metal|dx12` and `--software` narrow the choice, as does the `EMBERWAVE_BACKEND` environment variable, and the adapter in use is printed. The RGB conversion follows what the file signals (matrix, range, chroma siting); `--matrix bt601|bt709|bt2020` and `--range full|limited` override it. [`ARCHITECTURE.md`](ARCHITECTURE.md#command-line-tool) lists every option and explains how the timings are measured.

### From C

The C API decodes packets as a client receives them, into the caller's memory, and encodes pictures into frames and packets as a host sends them: a header, `crates/emberwave-ffi/include/emberwave.h`, and a shared or static library, which each release also ships prebuilt. [`crates/emberwave-ffi/README.md`](crates/emberwave-ffi/README.md) explains how to build and link it, and [`examples/c/decode.c`](crates/emberwave-ffi/examples/c/decode.c) and [`examples/c/encode.c`](crates/emberwave-ffi/examples/c/encode.c) are complete programs.

For apps on iPhones, iPads and Macs, each release also has the static library as an XCFramework, `Emberwave.xcframework`, which Swift imports as `Emberwave`; Metal clients can have the picture drawn into a texture on the GPU, or its Y, Cb and Cr planes copied into three, and read the `id<MTLTexture>` ([Pictures on the GPU](crates/emberwave-ffi/README.md#pictures-on-the-gpu-metal)). [`examples/swift`](crates/emberwave-ffi/examples/swift) decodes a file with it from Swift, [`examples/ios`](crates/emberwave-ffi/examples/ios) is a benchmark app for iPhones and iPads, and CI runs the examples and the app with it in an iOS Simulator ([iPhone, iPad and Mac apps](crates/emberwave-ffi/README.md#iphone-ipad-and-mac-apps)).

## Performance

<!-- bench-summary:begin (generated by scripts/bench.sh; edits are overwritten) -->
Decode time per frame on an Apple M3 Max (Metal), RGB output kept in GPU memory, measured on 2026-10-26, in milliseconds:

| Stream | mean | p99 |
|---|---|---|
| 1080p | 0.53 | 0.79 |
| 1440p | 0.69 | 0.97 |
| 4K | 1.22 | 1.62 |

Upstream's own Metal decoder is faster on the same machine: 0.80 ms for a 4K frame at FP32 against emberwave's 0.90 ms (planes only). [`BENCHMARKS.md`](BENCHMARKS.md) has the method and all the numbers.

Encoding a 4K frame within 1.2 MB takes 1.90 ms on the GPU, from the planes in CPU memory to the frame in CPU memory, and 3.33 ms with upstream's encoder at FP32.
<!-- bench-summary:end -->

In Safari on an iPhone 17 the [benchmark page](https://with-developer.github.io/emberwave/bench.html) decodes a 1080p frame in 2.4 ms and a 4K frame in 8.1 ms, to RGB in GPU memory ([`BENCHMARKS.md`](BENCHMARKS.md#in-the-browser-phones-included)). Natively, through the XCFramework, the [benchmark app](crates/emberwave-ffi/examples/ios) decodes them into a Metal texture in 1.6 and 6.2 to 6.3 ms, and into Y, Cb and Cr textures in 1.3 and 4.8 ms ([`BENCHMARKS.md`](BENCHMARKS.md#on-the-iphone-natively)).

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
- [x] **M4** Benchmarks and documentation ([#5](https://github.com/with-developer/emberwave/issues/5))
- [x] **M5** Stabilisation and the v0.1.0 release ([#6](https://github.com/with-developer/emberwave/issues/6))

After the release: a [browser player](https://with-developer.github.io/emberwave/) (WebGPU, [#15](https://github.com/with-developer/emberwave/issues/15)), with a [benchmark page](https://with-developer.github.io/emberwave/bench.html) that times the decoding on any device with WebGPU, phones included, and a C API for clients such as Moonlight's ([#16](https://github.com/with-developer/emberwave/issues/16)); an encoder ([#14](https://github.com/with-developer/emberwave/issues/14)), on the CPU and on the GPU, which also encodes pictures that are in GPU memory already, such as a screen capture's ([#33](https://github.com/with-developer/emberwave/issues/33)), and the encoder in the C API, for hosts ([#42](https://github.com/with-developer/emberwave/issues/42)); and the C library as an XCFramework for iPhone, iPad and Mac apps ([#52](https://github.com/with-developer/emberwave/issues/52)).

## Repository layout

| Path | Purpose |
|---|---|
| `crates/emberwave-core` | Bitstream parser, CPU reference decoder, colour conversion, the wgpu/WGSL decoder, and the encoder on the CPU and the GPU |
| `crates/emberwave-cli` | `emberwave-cli` command line tool |
| `crates/emberwave-ffi` | The C API: header, library and C examples |
| `crates/emberwave-web`, `web/` | The browser player and its benchmark: WebAssembly module and pages |
| `testdata/` | Test vectors and how to regenerate them |
| `scripts/` | Test data, 4K acceptance check and benchmark scripts |
| `ARCHITECTURE.md` | Bitstream format, GPU pipeline and the decision log |
| `BENCHMARKS.md` | How decode times are measured, and the results |
| `CONTRIBUTING.md` | How to build, test and change the code |
| `CHANGELOG.md` | What each release changed |

## Development

```sh
cargo build --workspace
cargo test --workspace
```

The GPU tests skip when there is no adapter, unless `EMBERWAVE_REQUIRE_GPU` is set. [`CONTRIBUTING.md`](CONTRIBUTING.md) explains the test vectors, the conventions and how changes get in.

## Relationship to upstream

emberwave follows PyroWave's algorithm structure and its [bitstream definition](https://github.com/Themaister/pyrowave/blob/79a3232a2199281f33f7a2e2659fc4fac62b022a/bitstream/bitstream.md), but contains no code copied from the upstream repository. Test vectors and reference output come from upstream's own encoder and decoder; see [`testdata/README.md`](testdata/README.md).

Upstream also ships an Apple-only Metal port. emberwave aims at one portable code base that additionally covers Vulkan, DX12 and the browser.

## License

MIT. See [`LICENSE`](LICENSE), which also carries the PyroWave copyright notice.

## 한국어 요약

emberwave는 Themaister의 PyroWave(LAN 게임 스트리밍용 GPU 웨이블릿 코덱)의 **디코더를 Rust + wgpu + WGSL로 재구현**해서 Vulkan / Metal / DX12 / WebGPU에서 같은 코드로 돌아가게 만드는 프로젝트입니다. 원본 디코더는 Vulkan 전용이라 macOS, iOS, 브라우저용 클라이언트가 없는데, 그 빈자리를 채우는 것이 목표입니다.

- **현재 상태(프리알파)**: 비트스트림 파서, GPU 디코더(역양자화, 역 DWT, YUV→RGB), CLI(`emberwave-cli`)가 동작합니다. 원본 인코더가 만든 스트림을 원본 디코더와 PSNR 64 dB 이상(원본 FP32 모드 기준 86 dB 이상)으로 일치하게 디코딩하고, 4K 스트림도 같은 기준을 통과합니다. CI는 Linux(lavapipe), macOS(Metal), Windows(DX12, WARP)에서 테스트를 돌립니다.
- **빠른 시작**: Rust 1.87 이상과 GPU(없으면 Linux는 Mesa lavapipe, Windows는 내장 WARP)가 있으면 위 Quick start의 네 줄로 PNG를 얻습니다. 처음 빌드는 몇 분 걸립니다.
- **성능**: Apple M3 Max(Metal)에서 4K 한 프레임을 약 0.9 ms(RGB 변환 포함 1.2 ms)에 디코딩하며, 같은 기계에서 원본 Metal 디코더(FP32)의 1.0~1.25배입니다(측정 회차마다 다름). 디코딩한 그림을 CPU로 읽어 오면 4K RGB가 약 2.2 ms입니다(전에는 7.5 ms). 측정 환경과 전체 수치는 위 Performance 표와 [`BENCHMARKS.md`](BENCHMARKS.md)에 있습니다.
- **릴리스**: v0.1.0이 첫 릴리스, v0.2.0이 속도 개선, v0.3.0이 브라우저 플레이어와 C API, v0.4.0이 인코더(CPU와 GPU), v0.5.0이 GPU 인코더 가속(4K 약 1.9 ms로 업스트림보다 빠름)과 기기 손실·Vulkan 로더 관련 수정, v0.6.0이 C API 인코더와 GPU 텍스처 입력 인코딩, v0.7.0이 iPhone·iPad·Mac 앱용 XCFramework, 휴대폰도 되는 브라우저 벤치마크, 디코딩 CPU 시간 단축, v0.8.0이 디코딩한 그림의 CPU 읽기 3배 가속(C API 4K RGB 7.6→2.7 ms), 텍스처 입력 인코딩 가속, 브라우저 페이지 리뷰 수정, v0.9.0이 Metal 클라이언트용 GPU 텍스처 출력(C API, M3 Max 4K 1.5 ms)과 iPhone·iPad 벤치마크 앱, v0.10.0이 Y·Cb·Cr 평면 텍스처(4K 1.0 ms), 복사로 채우는 RGB 텍스처(4K 1.5→1.3 ms), 앱의 Metal 큐에서 대기 없는 디코딩, v0.11.0이 앱 자신의 Metal 텍스처(CVPixelBuffer 등)로 디코딩과 모든 텍스처 경로를 재는 벤치마크 앱 릴리스입니다. Linux(x86_64), macOS(Apple Silicon), Windows(x86_64)용 `emberwave-cli`와 C 라이브러리(`emberwave-ffi`), iOS·macOS 앱용 `Emberwave.xcframework`를 [releases](https://github.com/with-developer/emberwave/releases)에서 받을 수 있습니다. 4K 1000프레임을 반복 디코딩해도 GPU 자원과 프로세스 메모리가 늘지 않는 것을 확인했습니다.
- **브라우저 플레이어**: [with-developer.github.io/emberwave](https://with-developer.github.io/emberwave/)에서 `.pwv1` 파일을 열면 WebGPU로 페이지 안에서 디코딩해 재생합니다. 파일은 어디에도 올라가지 않습니다. [벤치마크 페이지](https://with-developer.github.io/emberwave/bench.html)는 그 기기(휴대폰 포함)에서 1080p·1440p·4K 디코딩 시간을 재고 복사할 수 있는 결과를 보여 줍니다. iPhone 17(Safari)에서는 1080p 한 프레임에 2.4 ms, 4K에 8.1 ms입니다.
- **C API**: `crates/emberwave-ffi`가 C에서 쓰는 헤더(`emberwave.h`)와 공유·정적 라이브러리를 제공합니다. 네트워크로 받은 패킷을 넣고 YUV 평면이나 RGB를 호출자 메모리로 받습니다. 인코더도 있어서(#42), 호스트는 stride가 있는 평면을 바이트 예산 안의 프레임으로 인코딩하고 패킷으로 나눕니다. 릴리스마다 3개 OS용으로 빌드해 올리고, C 예제가 CI에서 모든 테스트 벡터를 디코딩하고 인코더 소스 둘을 인코딩해 다시 디코딩합니다. iPhone·iPad·Mac 앱용으로는 정적 라이브러리를 `Emberwave.xcframework`로 묶어 릴리스에 함께 올립니다(#52). Swift에서는 `import Emberwave`로 쓰고, Swift 예제(`examples/swift`)와 iPhone·iPad용 벤치마크 앱(`examples/ios`, Xcode 프로젝트)이 있습니다. CI가 C·Swift 예제를 iOS 시뮬레이터와 Mac에서, 앱을 iOS 시뮬레이터에서 돌려 검증합니다. iPhone 17 실기기에서도 앱의 Metal 검사 3개가 통과했고, 1080p 한 프레임을 Metal 텍스처로 1.6 ms, Y·Cb·Cr 텍스처로 1.3 ms, 4K는 6.2~6.3 ms와 4.8 ms에 디코딩합니다. 같은 폰의 Safari보다 22~31% 빠릅니다.
- **인코더**: `emberwave-cli encode`가 Y4M을 프레임당 바이트 예산 안에서 PWV1로 인코딩합니다. 업스트림 디코더가 그대로 읽고, 같은 바이트에서 PSNR이 업스트림 인코더와 같거나 높습니다(장면 영상에서 +0.6~+3.4 dB). `--gpu`를 주면 GPU에서 인코딩하며, Apple M3 Max에서 4K 한 프레임이 업로드와 읽기를 포함해 약 1.9 ms로 CPU보다 100배 이상 빠르고, 같은 정밀도의 원본 인코더(3.2~3.4 ms)보다 빠릅니다. GPU 메모리의 텍스처에 있는 그림도 바로 인코딩합니다(`Encoder::encode_textures`, 4K 약 1.8 ms). GPU 인코더의 프레임은 CPU 인코더와 같거나 일부 블록만 다르며(자연 영상에서 보낸 블록의 약 0.5%) PSNR은 같고, Apple M3 Max에서 커밋된 벡터는 모두 바이트 단위로 같습니다.
- **GPU에서 그림 넘기기(#27, #64)**: Metal 클라이언트는 디코딩한 그림을 CPU로 복사하지 않고 텍스처로 받아 `id<MTLTexture>`로 바로 그릴 수 있습니다(M3 Max 4K: 1.3~1.4 ms, CPU로 RGB 복사 시 2.6 ms). 색 변환을 직접 하는 클라이언트는 Y·Cb·Cr 평면을 R8 텍스처 세 장으로 받습니다. 가장 빠른 경로입니다(4K 1.0~1.1 ms, 1080p 0.45 ms).
- **앱의 큐에서 대기 없이(#62, #63)**: 앱의 `MTLDevice`·`MTLCommandQueue`로 컨텍스트를 만들면 `emberwave_decoder_submit_texture`가 GPU를 기다리지 않고 돌아옵니다. Metal이 같은 큐의 순서로 앱의 그리기를 디코딩 뒤에 둡니다. 4K에서 디코딩 스레드가 1.3 ms 대신 0.26 ms만 묶이고, 텍스처 하나로 충분합니다.
- **앱의 텍스처로(#63)**: `emberwave_texture_from_metal`로 앱의 `MTLTexture`(예: `AVSampleBufferDisplayLayer`에 넘길 `CVPixelBuffer`의 텍스처)에 바로 디코딩합니다.
- **남은 일**: Vulkan·D3D12 핸들.
