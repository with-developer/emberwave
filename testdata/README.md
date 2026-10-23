# Test data

Test vectors are encoded by the **original PyroWave encoder**, and the reference pictures they are compared against are decoded by the **original PyroWave decoder**. Vectors are never hand-made: an invented stream that "passes" proves nothing about compatibility.

## Pinned upstream revision

| | |
|---|---|
| Repository | <https://github.com/Themaister/pyrowave> |
| Commit | `79a3232a2199281f33f7a2e2659fc4fac62b022a` (2026-10-21, "Added Metal P010 support") |
| Bitstream | v1, frozen 2026-10-03 (`bitstream/bitstream.md`) |

## Building the upstream reference

### macOS: upstream Metal port (verified)

No Granite and no Vulkan SDK are needed. The MSL shaders are committed upstream and compiled by the OS at run time.

```sh
git clone https://github.com/Themaister/pyrowave.git
git -C pyrowave checkout 79a3232a2199281f33f7a2e2659fc4fac62b022a
cmake -S pyrowave/metal -B build/pyrowave-metal -DCMAKE_BUILD_TYPE=Release
cmake --build build/pyrowave-metal -j
```

This produces `build/pyrowave-metal/libpyrowave-metal.dylib`. Its C API is declared in `pyrowave/metal/pyrowave_metal.h`.

Requirements: Apple Silicon (Apple7 GPU family or newer), Xcode command line tools, CMake 3.20 or newer.

Verified on 2026-10-23 with an M3 Max, macOS 27.0.1, Xcode 26.4.1 (Apple clang 21.0.0) and CMake 4.3.1. Besides building, a smoke test created a device, a 256x256 4:2:0 decoder and a 256x256 4:2:0 encoder successfully.

### Linux / Windows: upstream Vulkan implementation (not verified here)

Follow the upstream README, section "Local development and CLI": check out Granite, configure with `-DPYROWAVE_DEVEL=ON`, then use the `pyrowave-encode` and `pyrowave-decode` command line tools. Treat these steps as unverified until this project has exercised them.

## File format

Vectors use upstream's raw disk format, PWV1 (`.pwv1`): a `PWV1Header`, then for every frame a `u32` size followed by that many bytes. It is defined in upstream's `bitstream/bitstream.md`, section "Raw disk format specification". `ARCHITECTURE.md` documents the format as implemented here.

## Vectors

`scripts/gen_testdata.sh` builds the pinned upstream revision, encodes the vectors below with the upstream encoder and decodes each of them twice with the upstream decoder. It needs Apple Silicon macOS and writes into this directory; the upstream checkout and build live in the git-ignored `testdata/_cache/`.

| Vector | Picture | Frames | Budget per frame | Purpose |
|---|---|---|---|---|
| `solid_160x96_420` | 160x96, 4:2:0 | 3 | 40000 B | Constant colour, padding to 160x128, trivial blocks |
| `gradient_200x120_420` | 200x120, 4:2:0 | 3 | 60000 B | Smooth content, picture size not a multiple of 32 |
| `noise_192x136_420` | 192x136, 4:2:0 | 3 | 48000 B | High entropy, rate control active |
| `noise_192x136_420_lowrate` | 192x136, 4:2:0 | 3 | 12000 B | Coarse quantisation, non-trivial quantiser codes |
| `noise_131x67_444` | 131x67, 4:4:4 | 2 | 20000 B | Odd dimensions, 4:4:4 |

Pictures are synthesised deterministically by `scripts/refgen/refgen.mm`, a small tool that only writes the pictures into upstream's encoder and reads decoded pictures out of upstream's decoder. All bitstream production and decoding is upstream's.

Each vector comes as:

| File | Content |
|---|---|
| `NAME.pwv1` | The bitstream, in the PWV1 format |
| `NAME.ref.y4m` | Upstream decoder output at its default precision (FP16 wavelet storage), 8-bit planar YUV |
| `NAME.ref-fp32.y4m` | Upstream decoder output at FP32 precision (`PYROWAVE_PRECISION=2`) |

Tests decode a vector and compare plane by plane: at least 60 dB PSNR against `ref.y4m` and at least 80 dB against `ref-fp32.y4m`. See `ARCHITECTURE.md`, decision D9. Set `EMBERWAVE_REFERENCE_DIR` to compare against decodes made elsewhere, for example by the upstream Vulkan decoder.

The upstream encoder does not write byte-for-byte reproducible bitstreams (the unused bits of a block's last sign byte are leftover GPU memory), so regenerating changes the committed `.pwv1` files. Regenerate only when adding a vector or changing its parameters.

## Large streams

Streams of real size are never committed. Two scripts generate them into `testdata/generated/`, which git ignores. The picture is a synthetic scene that `refgen` feeds to the upstream encoder: colour bars, a gradient, a moving shaded disc, a fine diagonal grid and a little film grain. Like `gen_testdata.sh` they need Apple Silicon macOS.

| Script | What it does |
|---|---|
| `scripts/verify_4k.sh [frames]` | The 4K acceptance check. Encodes a 4K scene, decodes it with the upstream decoder at both precisions and with `emberwave-cli` (to PNG and to Y4M), and requires 60 dB against the default precision and 80 dB against the FP32 decode. See `ARCHITECTURE.md`, decision D20 |
| `scripts/gen_bench_data.sh` | 60-frame scene streams at 1080p, 1440p and 4K for the benchmarks, and with `--noise` a 4K stream of white noise. `scripts/bench.sh` times the decoder on them; see `BENCHMARKS.md` |
