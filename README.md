<p align="center">
  <img src="docs/images/banner.svg" alt="emberwave: PyroWave's wavelet video codec, rewritten for wgpu" width="100%">
</p>

<p align="center">
  <a href="https://github.com/with-developer/emberwave/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/with-developer/emberwave/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/with-developer/emberwave/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/with-developer/emberwave?color=e8302a"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <a href="https://with-developer.github.io/emberwave/"><img alt="Try it in the browser" src="https://img.shields.io/badge/try_it-WebGPU_player-ff7a1a"></a>
</p>

<p align="center">
  <b><a href="https://with-developer.github.io/emberwave/">Play a stream in your browser</a></b> ·
  <a href="https://with-developer.github.io/emberwave/bench.html">Benchmark your device</a> ·
  <a href="BENCHMARKS.md">Benchmarks</a> ·
  <a href="crates/emberwave-ffi/README.md">C API</a> ·
  <a href="crates/emberwave-ffi/examples/ios">iPhone app</a>
</p>

[PyroWave](https://github.com/Themaister/pyrowave) is a video codec for game streaming on a local network: a GPU encodes and decodes each frame in a millisecond or two, at the cost of bandwidth. Valve has added it to Steam Remote Play, in the Steam client's beta ([GamingOnLinux](https://www.gamingonlinux.com/2026/10/steam-client-beta-for-october-1st-brings-many-remote-play-pyrowave-improvements/)). Upstream's implementation is written as Vulkan compute shaders, with a separate port to Metal. **emberwave is PyroWave rewritten in Rust and WGSL on [wgpu](https://wgpu.rs)**: one code base, and the same bitstream, on Metal, Vulkan, DX12 and WebGPU, from a native app to a browser tab and an iPhone.

- **Fast.** On an Apple M3 Max, frames one after the other, a 4K frame decodes in 0.90 ms to its planes and 1.22 ms to RGB, and encodes in 1.90 ms on the GPU, from CPU memory to CPU memory. Upstream's own Metal port takes 0.80 ms to decode it at the same FP32 precision and 0.56 ms at its default FP16, and 3.2 to 3.3 ms to encode it ([`BENCHMARKS.md`](BENCHMARKS.md), 2026-10-26).
- **Every frame stands alone.** No frame refers to another, so a lost packet costs one frame, with no wait for a keyframe. Each frame keeps to a fixed byte budget.
- **Compatible with upstream, and a better encoder.** It decodes the streams of upstream's encoder as upstream's decoder does, at 64 dB PSNR or more, and upstream's decoder reads emberwave's streams. At the same bytes per frame, emberwave's encoder beats upstream's luma PSNR by up to 3.4 dB on the test scenes, and matches it within 0.01 dB on noise.
- **Native, in the browser and on the iPhone.** A command line tool, a [browser player](https://with-developer.github.io/emberwave/) that decodes with WebGPU in the page and opens with a demo stream it encodes itself, a C API with prebuilt libraries for Linux and Windows (x86_64) and macOS (Apple silicon), and an XCFramework for iPhone, iPad and Mac apps. Tested on Apple GPUs, and in CI on software Vulkan and DX12.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/decode-dark.svg">
  <img alt="Decode time per frame, to RGB in GPU memory. Apple M3 Max: 0.53 ms at 1080p and 1.22 ms at 4K natively through Metal, 0.70 to 1.00 and 1.69 to 1.88 ms in Chrome through WebGPU. iPhone 17: 1.62 to 1.63 and 6.21 to 6.29 ms in a native app, 2.36 and 8.05 ms in Safari." src="docs/images/decode-light.svg" width="100%">
</picture>

These times are for frames one after the other. A stream's frames arrive 60 a second, and between them the GPU idles and slows down, so each decode takes longer ([Performance](#performance)).

> **Status: 0.x.** The bitstream is upstream's, and the Rust and C APIs may still change between minor versions. CI tests every change on Linux with the lavapipe Vulkan driver, on macOS with Metal and on Windows with DX12 on WARP. Releases and their changes are in [`CHANGELOG.md`](CHANGELOG.md).

## Picture quality

<p align="center">
  <img alt="A 4K frame of the game Crimson Desert, and a 320 by 180 region of it, a pine before a stone wall, enlarged twice: from the source, and decoded from emberwave's frames within four budgets, with the bit rate at 60 frames a second and the luma PSNR of the whole frame: 322 Mbit/s, 25.8 dB; 499 Mbit/s, 27.1 dB; 816 Mbit/s, 29.9 dB; 1200 Mbit/s, 32.9 dB." src="docs/images/quality.png" width="100%">
</p>

PyroWave buys its speed with bandwidth, and how much it takes depends on the picture. This 4K frame of a game is a hard one, much of it grass, needles and stone. emberwave's encoder made the frames. Each decoded panel gives the bit rate at 60 frames a second and the luma PSNR over the whole frame, and shows the region that loses the most, enlarged twice. At 322 Mbit/s the needles run together and the joints of the wall blur; at 816 Mbit/s they are mostly back, and at 1.2 Gbit/s the region is close to the source. At 322 Mbit/s the first picture of the benchmarks' synthetic 4K scene keeps 42.7 dB. On this frame, a key frame of Apple's hardware HEVC encoder, outside its real-time mode, has 1.9 to 3.5 dB more luma PSNR within the same budget (Apple M3 Max, macOS 27.0.1, 2026-10-27), and an HEVC stream needs key frames only at its start and after a loss. PyroWave is meant for local networks with bandwidth to spare ([`BENCHMARKS.md`](BENCHMARKS.md#picture-quality)).

<sub>The frame is from Crimson Desert, © Pearl Abyss. This comparison is unofficial content: it contains copyrighted material and IP from Pearl Abyss, and it is not made or endorsed by Pearl Abyss. The picture is not covered by this repository's license.</sub>

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/pipeline-dark.svg">
  <img alt="A 4K frame through emberwave on an Apple M3 Max. Encoding on the GPU: wavelet transform 0.40 ms, quantisation 0.59 ms, rate control 0.13 ms, packing 0.18 ms, a frame of at most 1.2 MB. The packets cross the network. Decoding: assembly on the CPU 0.10 ms, dequantisation 0.19 ms, inverse transform 0.40 ms, YUV to RGB 0.24 ms." src="docs/images/pipeline-light.svg" width="100%">
</picture>

A frame is a wavelet transform of the picture, its coefficients quantised in blocks and sent as bit-planes, most significant first. To fit a frame into its budget, the encoder drops the least significant planes where that costs the picture least, weighted by the eye's sensitivity to each band. Every stage of the figure but the decoder's assembly of the packets is a compute shader in WGSL; that assembly runs on the CPU, as in upstream's decoder. Around them, the CPU copies the planes in, the frame out and the picture's header. [`ARCHITECTURE.md`](ARCHITECTURE.md) describes the bitstream, the passes and the decisions behind them.

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

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/encode-dark.svg">
  <img alt="Encoding a 4K frame on an Apple M3 Max: emberwave on the GPU 1.90 ms, upstream PyroWave 3.17 ms at FP16 and 3.33 ms at FP32, Apple's HEVC encoder at 80 Mbit/s 12.76 to 13.00 ms." src="docs/images/encode-light.svg" width="100%">
</picture>

In Safari on an iPhone 17 the [benchmark page](https://with-developer.github.io/emberwave/bench.html) decodes a 1080p frame in 2.4 ms and a 4K frame in 8.1 ms, to RGB in GPU memory ([`BENCHMARKS.md`](BENCHMARKS.md#in-the-browser-phones-included)). Natively, through the XCFramework, the [benchmark app](crates/emberwave-ffi/examples/ios) decodes them into a Metal texture in 1.6 and 6.2 to 6.3 ms, and into Y, Cb and Cr textures in 1.3 and 4.8 ms ([`BENCHMARKS.md`](BENCHMARKS.md#on-the-iphone-natively)).

The same app times the hardware HEVC decoder that Moonlight uses, at Moonlight's default bit rates. Frame after frame, emberwave decodes into textures 2.2 to 3.9 times as quickly on the M3 Max, 1.0 against 2.6 to 3.4 ms at 4K, and 1.4 to 1.7 times as quickly on the iPhone 17, 4.5 against HEVC's 6.2 to 6.3 ms, in runs a day after those above. When frames come 60 a second, the GPU slows down while it idles between them, and the media engine does not. The two then take about as long on the Mac, and on the phone HEVC is 1.5 to 2.7 times as quick ([`BENCHMARKS.md`](BENCHMARKS.md#against-hevc-on-the-same-device)).

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
| `docs/images/` | The README's banner and charts, drawn by `scripts/readme_figures.py` from the numbers in `BENCHMARKS.md`, and its picture of the quality, made by `scripts/quality_figure.py` |
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

MIT. See [`LICENSE`](LICENSE), which also carries the PyroWave copyright notice. The frame of Crimson Desert in `docs/images/quality.png` is © Pearl Abyss and not covered by the license; the source archives of releases leave it out.

## 한국어 요약

emberwave는 Themaister의 PyroWave(LAN 게임 스트리밍용 GPU 웨이블릿 코덱, Valve가 Steam Remote Play 베타에 넣은 코덱)를 **Rust + wgpu + WGSL로 재구현**해서 Metal / Vulkan / DX12 / WebGPU에서 같은 코드로 돌아가게 만드는 프로젝트입니다. 원본은 Vulkan 컴퓨트 셰이더와 Apple 전용 Metal 포트로 따로 구현되어 있고 브라우저에서는 돌지 않습니다. emberwave는 한 코드로 네이티브 앱, 브라우저 탭, iPhone까지 같은 비트스트림을 인코딩·디코딩합니다.

- **현재 상태(0.x)**: 비트스트림은 업스트림 것이고, Rust·C API는 마이너 버전 사이에 바뀔 수 있습니다. 비트스트림 파서, GPU 디코더(역양자화, 역 DWT, YUV→RGB), CLI(`emberwave-cli`)가 동작합니다. 원본 인코더가 만든 스트림을 원본 디코더와 PSNR 64 dB 이상(원본 FP32 모드 기준 86 dB 이상)으로 일치하게 디코딩하고, 4K 스트림도 같은 기준을 통과합니다. CI는 Linux(lavapipe), macOS(Metal), Windows(DX12, WARP)에서 테스트를 돌립니다.
- **빠른 시작**: Rust 1.87 이상과 GPU(없으면 Linux는 Mesa lavapipe, Windows는 내장 WARP)가 있으면 위 Quick start의 네 줄로 PNG를 얻습니다. 처음 빌드는 몇 분 걸립니다.
- **성능**: Apple M3 Max(Metal)에서 4K 한 프레임을 약 0.9 ms(RGB 변환 포함 1.2 ms)에 디코딩하며, 같은 기계에서 원본 Metal 디코더(FP32)의 1.0~1.25배입니다(측정 회차마다 다름). 디코딩한 그림을 CPU로 읽어 오면 4K RGB가 약 2.2 ms입니다(전에는 7.5 ms). 측정 환경과 전체 수치는 위 Performance 표와 [`BENCHMARKS.md`](BENCHMARKS.md)에 있습니다.
- **화질**: 필요한 대역폭은 그림에 따라 다릅니다. 풀·솔잎·돌벽이 많아 압축하기 어려운 붉은사막 4K 프레임을 emberwave 인코더로 초당 60프레임 기준 322 Mbit/s에 맞추면 솔잎이 뭉개지고(프레임 전체의 luma PSNR 25.8 dB), 816 Mbit/s에서는 대부분 살아나며(29.9 dB), 1.2 Gbit/s에서는 원본에 가깝습니다(32.9 dB). 벤치마크의 합성 4K 장면 첫 프레임은 322 Mbit/s에서도 42.7 dB입니다. 같은 프레임을 Apple 하드웨어 HEVC 인코더가 실시간 모드가 아닌 키프레임으로 같은 예산 안에 압축하면 luma PSNR이 1.9~3.5 dB 높습니다(Apple M3 Max, macOS 27.0.1, 2026-10-27). PyroWave는 대역폭이 넉넉한 로컬 네트워크용입니다([`BENCHMARKS.md`](BENCHMARKS.md#picture-quality)).
- **릴리스**: v0.1.0이 첫 릴리스, v0.2.0이 속도 개선, v0.3.0이 브라우저 플레이어와 C API, v0.4.0이 인코더(CPU와 GPU), v0.5.0이 GPU 인코더 가속(4K 약 1.9 ms로 업스트림보다 빠름)과 기기 손실·Vulkan 로더 관련 수정, v0.6.0이 C API 인코더와 GPU 텍스처 입력 인코딩, v0.7.0이 iPhone·iPad·Mac 앱용 XCFramework, 휴대폰도 되는 브라우저 벤치마크, 디코딩 CPU 시간 단축, v0.8.0이 디코딩한 그림의 CPU 읽기 3배 가속(C API 4K RGB 7.6→2.7 ms), 텍스처 입력 인코딩 가속, 브라우저 페이지 리뷰 수정, v0.9.0이 Metal 클라이언트용 GPU 텍스처 출력(C API, M3 Max 4K 1.5 ms)과 iPhone·iPad 벤치마크 앱, v0.10.0이 Y·Cb·Cr 평면 텍스처(4K 1.0 ms), 복사로 채우는 RGB 텍스처(4K 1.5→1.3 ms), 앱의 Metal 큐에서 대기 없는 디코딩, v0.11.0이 앱 자신의 Metal 텍스처(CVPixelBuffer 등)로 디코딩과 모든 텍스처 경로를 재는 벤치마크 앱 릴리스입니다. Linux(x86_64), macOS(Apple Silicon), Windows(x86_64)용 `emberwave-cli`와 C 라이브러리(`emberwave-ffi`), iOS·macOS 앱용 `Emberwave.xcframework`를 [releases](https://github.com/with-developer/emberwave/releases)에서 받을 수 있습니다. 4K 1000프레임을 반복 디코딩해도 GPU 자원과 프로세스 메모리가 늘지 않는 것을 확인했습니다.
- **브라우저 플레이어**: [with-developer.github.io/emberwave](https://with-developer.github.io/emberwave/)에서 `.pwv1` 파일을 열면 WebGPU로 페이지 안에서 디코딩해 재생합니다. 파일 없이 열면 페이지가 직접 그리고 인코딩한 960x540 데모 스트림을 바로 재생합니다. 파일은 어디에도 올라가지 않습니다. [벤치마크 페이지](https://with-developer.github.io/emberwave/bench.html)는 그 기기(휴대폰 포함)에서 1080p·1440p·4K 디코딩 시간을 재고 복사할 수 있는 결과를 보여 줍니다. iPhone 17(Safari)에서는 1080p 한 프레임에 2.4 ms, 4K에 8.1 ms입니다.
- **C API**: `crates/emberwave-ffi`가 C에서 쓰는 헤더(`emberwave.h`)와 공유·정적 라이브러리를 제공합니다. 네트워크로 받은 패킷을 넣고 YUV 평면이나 RGB를 호출자 메모리로 받습니다. 인코더도 있어서(#42), 호스트는 stride가 있는 평면을 바이트 예산 안의 프레임으로 인코딩하고 패킷으로 나눕니다. 릴리스마다 3개 OS용으로 빌드해 올리고, C 예제가 CI에서 모든 테스트 벡터를 디코딩하고 인코더 소스 둘을 인코딩해 다시 디코딩합니다. iPhone·iPad·Mac 앱용으로는 정적 라이브러리를 `Emberwave.xcframework`로 묶어 릴리스에 함께 올립니다(#52). Swift에서는 `import Emberwave`로 쓰고, Swift 예제(`examples/swift`)와 iPhone·iPad용 벤치마크 앱(`examples/ios`, Xcode 프로젝트)이 있습니다. CI가 C·Swift 예제를 iOS 시뮬레이터와 Mac에서, 앱을 iOS 시뮬레이터에서 돌려 검증합니다. iPhone 17 실기기에서도 앱의 Metal 검사 3개가 통과했고, 1080p 한 프레임을 Metal 텍스처로 1.6 ms, Y·Cb·Cr 텍스처로 1.3 ms, 4K는 6.2~6.3 ms와 4.8 ms에 디코딩합니다. 같은 폰의 Safari보다 22~31% 빠릅니다. 앱은 같은 기기의 하드웨어 HEVC(VideoToolbox, Moonlight가 쓰는 경로)도 함께 잽니다. M3 Max에서 프레임을 연달아 디코딩하면 emberwave가 HEVC보다 2.2~3.9배 빠릅니다(4K 1.0 ms 대 2.6~3.4 ms). 하지만 초당 60프레임 간격으로 받으면 프레임 사이에 쉬는 GPU는 느려지고 미디어 엔진은 그렇지 않아서, 둘이 비슷해집니다. iPhone 17에서는 연달아 디코딩하면 emberwave가 1.4~1.7배 빠르지만(하루 뒤 측정에서 4K 4.5 ms 대 HEVC 6.2~6.3 ms), 초당 60프레임 간격에서는 HEVC가 1.5~2.7배 빠릅니다.
- **인코더**: `emberwave-cli encode`가 Y4M을 프레임당 바이트 예산 안에서 PWV1로 인코딩합니다. 업스트림 디코더가 그대로 읽고, 같은 바이트에서 PSNR이 업스트림 인코더와 같거나 높습니다(장면 영상에서 +0.6~+3.4 dB). `--gpu`를 주면 GPU에서 인코딩하며, Apple M3 Max에서 4K 한 프레임이 업로드와 읽기를 포함해 약 1.9 ms로 CPU보다 100배 이상 빠르고, 같은 정밀도의 원본 인코더(3.2~3.4 ms)보다 빠릅니다. GPU 메모리의 텍스처에 있는 그림도 바로 인코딩합니다(`Encoder::encode_textures`, 4K 약 1.8 ms). GPU 인코더의 프레임은 CPU 인코더와 같거나 일부 블록만 다르며(자연 영상에서 보낸 블록의 약 0.5%) PSNR은 같고, Apple M3 Max에서 커밋된 벡터는 모두 바이트 단위로 같습니다.
- **GPU에서 그림 넘기기(#27, #64)**: Metal 클라이언트는 디코딩한 그림을 CPU로 복사하지 않고 텍스처로 받아 `id<MTLTexture>`로 바로 그릴 수 있습니다(M3 Max 4K: 1.3~1.4 ms, CPU로 RGB 복사 시 2.6 ms). 색 변환을 직접 하는 클라이언트는 Y·Cb·Cr 평면을 R8 텍스처 세 장으로 받습니다. 가장 빠른 경로입니다(4K 1.0~1.1 ms, 1080p 0.45 ms).
- **앱의 큐에서 대기 없이(#62, #63)**: 앱의 `MTLDevice`·`MTLCommandQueue`로 컨텍스트를 만들면 `emberwave_decoder_submit_texture`가 GPU를 기다리지 않고 돌아옵니다. Metal이 같은 큐의 순서로 앱의 그리기를 디코딩 뒤에 둡니다. 4K에서 디코딩 스레드가 1.3 ms 대신 0.26 ms만 묶이고, 텍스처 하나로 충분합니다.
- **앱의 텍스처로(#63)**: `emberwave_texture_from_metal`로 앱의 `MTLTexture`(예: `AVSampleBufferDisplayLayer`에 넘길 `CVPixelBuffer`의 텍스처)에 바로 디코딩합니다.
- **남은 일**: Vulkan·D3D12 핸들.
