<p align="center"><b>English</b> | <a href="README.ko.md">한국어</a></p>

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
- **Every frame stands alone.** No frame refers to another, so a lost packet costs one frame at most, with no wait for a keyframe: emberwave's frames can still be shown in part, unless the packet held the coarsest level or the frame's header. Each frame keeps to a fixed byte budget.
- **Compatible with upstream, both ways.** It decodes the streams of upstream's encoder as upstream's decoder does, at 64 dB PSNR or more, and upstream's decoder reads emberwave's streams. At the same bytes per frame, emberwave's frames have 6.5 to 13 % less of the contrast-weighted error that both encoders minimise than those of upstream's encoder, with the fixes upstream made to its rate control in October 2026, on every test picture but noise, where the two are within 1 %. In luma PSNR, emberwave's ranges from 0.8 dB below upstream's, on a gradient, to 2.3 dB above it, on a test scene, and is the same on noise.
- **Native, in the browser and on the iPhone.** A command line tool, a [browser player](https://with-developer.github.io/emberwave/) that decodes with WebGPU in the page, opens with a demo stream it encodes itself and can play a stream packet by packet, shuffled and some of them lost, as a network delivers it, a C API with prebuilt libraries for Linux and Windows (x86_64) and macOS (Apple silicon), and an XCFramework for iPhone, iPad and Mac apps. Tested on Apple GPUs, and in CI on software Vulkan and DX12.

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

# Measure a stream, from any encoder, against the pictures it was encoded
# from: PSNR, and the error that the encoder's rate control minimises
emberwave-cli quality testdata/encoder/scene_200x120_420.source.y4m scene.pwv1

# Time the encoder, on the GPU here
emberwave-cli bench-encode testdata/encoder/scene_200x120_420.source.y4m --bytes-per-frame 6000 --gpu
```

The decoder, and the encoder with `--gpu`, use whatever GPU the machine offers: Vulkan, Metal or DX12, or a software driver such as Mesa's lavapipe or Windows' WARP. `--backend vulkan|metal|dx12` and `--software` narrow the choice, as does the `EMBERWAVE_BACKEND` environment variable, and the adapter in use is printed. The RGB conversion follows what the file signals (matrix, range, chroma siting); `--matrix bt601|bt709|bt2020` and `--range full|limited` override it. [`ARCHITECTURE.md`](ARCHITECTURE.md#command-line-tool) lists every option and explains how the timings are measured.

### From C

The C API decodes packets as a client receives them, into the caller's memory, and encodes pictures into frames and packets as a host sends them: a header, `crates/emberwave-ffi/include/emberwave.h`, and a shared or static library, which each release also ships prebuilt. [`crates/emberwave-ffi/README.md`](crates/emberwave-ffi/README.md) explains how to build and link it, and [`examples/c/decode.c`](crates/emberwave-ffi/examples/c/decode.c) and [`examples/c/encode.c`](crates/emberwave-ffi/examples/c/encode.c) are complete programs.

For apps on iPhones, iPads and Macs, each release also has the static library as an XCFramework, `Emberwave.xcframework`, which Swift imports as `Emberwave`; Metal clients can have the picture drawn into a texture on the GPU, or its Y, Cb and Cr planes put into three, and read the `id<MTLTexture>` ([Pictures on the GPU](crates/emberwave-ffi/README.md#pictures-on-the-gpu-metal)). [`examples/swift`](crates/emberwave-ffi/examples/swift) decodes a file with it from Swift, [`examples/ios`](crates/emberwave-ffi/examples/ios) is a benchmark app for iPhones and iPads, and CI runs the examples and the app with it in an iOS Simulator ([iPhone, iPad and Mac apps](crates/emberwave-ffi/README.md#iphone-ipad-and-mac-apps)).

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
- Rely only on features wgpu guarantees on every backend: no Vulkan-only extensions and no subgroup operations. Where Metal offers more, an optional path may use it with the portable one as its fallback (decisions D52 and D53).
- Measure per-frame decode time at 1080p / 1440p / 4K on macOS (Metal) and publish the numbers together with their environment.

Out of scope: network transport, RTP and FEC.

## Roadmap

Done: the milestones to the first release, M0 to M5 ([#1](https://github.com/with-developer/emberwave/issues/1) to [#6](https://github.com/with-developer/emberwave/issues/6)); since then the encoder, on the CPU and on the GPU ([#14](https://github.com/with-developer/emberwave/issues/14)), the browser player and its benchmark page ([#15](https://github.com/with-developer/emberwave/issues/15)), the C API for clients and hosts ([#16](https://github.com/with-developer/emberwave/issues/16), [#42](https://github.com/with-developer/emberwave/issues/42)), pictures to and from GPU memory, Metal textures included ([#27](https://github.com/with-developer/emberwave/issues/27), [#33](https://github.com/with-developer/emberwave/issues/33)), and the XCFramework for iPhone, iPad and Mac apps ([#52](https://github.com/with-developer/emberwave/issues/52)). 0.12 took two copies out of a frame's work on Apple's GPUs: the planes are stored straight into textures on Metal, and on Apple silicon the frames are written where the shaders read them ([#86](https://github.com/with-developer/emberwave/issues/86)). [`CHANGELOG.md`](CHANGELOG.md) has each release.

Next, in this order:

1. **0.13, measured in real use.** Decoding inside a client's display loop, timed to the screen, on a Mac and an iPhone ([#81](https://github.com/with-developer/emberwave/issues/81)); NVENC's HEVC and AV1 streams decoded beside emberwave's, and the quality of moving game footage against HEVC with predicted frames at the same bit rate ([#82](https://github.com/with-developer/emberwave/issues/82)); a Swift package for the XCFramework; and the CPU's preparation of a frame, the last item of [#86](https://github.com/with-developer/emberwave/issues/86).
2. **0.14, quality and platforms.** A better encoder within the same bitstream, and Vulkan and Direct3D 12 handles in the C API's GPU interop, for clients and hosts on Linux and Windows.
3. **Toward 1.0, clients.** Streaming to a browser, in a repository of its own so that emberwave stays a codec: a host's screen encoded on its GPU, sent over the local network and decoded with WebGPU in a tab, its latency measured first. After that, emberwave's encoder offered to the hosts that stream PyroWave. 1.0 needs stable C and Rust APIs, one client in real use, and decoding at a stream's pace level with upstream's.

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
| `README.ko.md` | This README in Korean |
| `ARCHITECTURE.md` | Bitstream format, GPU pipeline and the decision log |
| `BENCHMARKS.md` | How decode times are measured, and the results |
| `CONTRIBUTING.md` | How to build, test and change the code |
| `CHANGELOG.md` | What each release changed |
| `LICENSE`, `THIRD_PARTY_NOTICES.md` | The MIT license; PyroWave's license and what the license does not cover |

## Development

```sh
cargo build --workspace
cargo test --workspace
```

The GPU tests skip when there is no adapter, unless `EMBERWAVE_REQUIRE_GPU` is set. [`CONTRIBUTING.md`](CONTRIBUTING.md) explains the test vectors, the conventions and how changes get in.

## Relationship to upstream

emberwave follows PyroWave's algorithm structure and its [bitstream definition](https://github.com/Themaister/pyrowave/blob/7fe0e2667cd3076b8b2bf04bcb80bc8d691b800b/bitstream/bitstream.md), but contains no code copied from the upstream repository. Test vectors and reference output come from upstream's own encoder and decoder; see [`testdata/README.md`](testdata/README.md).

Upstream also ships an Apple-only Metal port. emberwave aims at one portable code base that additionally covers Vulkan, DX12 and the browser.

## License

MIT. See [`LICENSE`](LICENSE). [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) has PyroWave's copyright notice and license, and what the license does not cover: the frame of Crimson Desert in `docs/images/quality.png`, © Pearl Abyss, which the source archives of releases leave out.
