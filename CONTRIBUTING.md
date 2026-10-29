# Contributing to emberwave

This page says how to build and test emberwave, which rules the code follows and how a change gets in. [`ARCHITECTURE.md`](ARCHITECTURE.md) explains the design and records the decisions behind it.

## Setting up

- [Rust](https://rustup.rs) 1.87 or newer, with `rustfmt` and `clippy`.
- For the GPU tests, a GPU with Vulkan, Metal or DX12, or a software driver: Mesa's lavapipe on Linux (`mesa-vulkan-drivers` and `libvulkan1` on Debian and Ubuntu), WARP on Windows (built in).
- Only for new test vectors, benchmark streams and the 4K check: Apple Silicon macOS with the Xcode command line tools and CMake 3.20 or newer, because upstream's Metal port is the reference encoder and decoder ([`testdata/README.md`](testdata/README.md)).

Before every commit:

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Tests

| Suite | Where | What it checks |
|---|---|---|
| Unit tests | `src/` of each crate | Parser, CPU decoder, colour conversion, CLI helpers |
| Vectors | `crates/emberwave-core/tests/vectors.rs` | The CPU decoder against upstream's decodes of the committed vectors |
| Robustness | `crates/emberwave-core/tests/robustness.rs` | Malformed input never panics (proptest) |
| Shaders | `crates/emberwave-core/tests/shaders.rs` | The WGSL validates with naga without any optional capability, and the workgroup variables are the reviewed ones, without a GPU |
| GPU | `crates/emberwave-core/tests/gpu.rs` | The GPU decoder against the CPU decoder and against upstream, on a real adapter |
| CLI | `crates/emberwave-cli/tests/cli.rs` | The `emberwave-cli` binary end to end |
| C API | `crates/emberwave-ffi/tests/api.rs`, `tests/encoder.rs`, `tests/header.rs` | The C functions called through raw pointers against the Rust decoder and encoder; the committed header against cbindgen's output |
| Encoder | `crates/emberwave-core/tests/encoder.rs`, `src/encode/` | The encoder makes the committed bytes again, its frames fit their budgets, and emberwave decodes them as upstream does |
| GPU encoder | `crates/emberwave-core/tests/gpu_encoder.rs`, `src/gpu/encoder/tests.rs` | Each stage of the GPU encoder given the CPU encoder's input, against the CPU's output; whole frames against the committed vectors, on a real adapter; pictures in textures against the same pictures in CPU memory |
| C example | `scripts/check_c_example.sh` | The C examples, built with CMake against the shared library: one decodes every vector, the other encodes two of the encoder's sources and decodes its packets again (CI runs it after the tests) |
| iPhone app | `scripts/check_ios_app.sh` | The benchmark app of `examples/ios`, built for iPhones and for the iOS Simulator, where it runs a short benchmark and reads a texture of `emberwave_decoder_decode_texture` back through Metal (CI's macOS job `XCFramework`) |
| XCFramework | `scripts/check_xcframework.sh` | The C examples and the Swift example, linked with each static library of the XCFramework, on the Mac and in an iOS Simulator, as `check_c_example.sh` runs them; for iPhones and iPads they are linked only (CI's macOS job `XCFramework`) |

Environment variables:

- `EMBERWAVE_REQUIRE_GPU=1` turns a missing adapter into a failure instead of a skip. CI sets it on Linux, where lavapipe is installed.
- `EMBERWAVE_BACKEND=vulkan|metal|dx12` picks the backend, for the tests and for the CLI.
- `EMBERWAVE_REFERENCE_DIR=DIR` compares against reference decodes made elsewhere, for example by upstream's Vulkan decoder.
- `PROPTEST_CASES=100000` runs a deeper robustness pass than the default 256 cases.

On Linux with a Vulkan loader from 1.3.284 to 1.4.344 (Debian 13, Ubuntu 25.04 to 26.04) and several Vulkan drivers installed, the GPU test binaries can crash when their tests run in parallel: the loader unloads the drivers that find no device while another thread reads their list (decision D31; loader 1.4.345 fixed it). Load only the driver you test with, as CI does, for example `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json` for lavapipe (the file name varies between distributions), or run the tests on one thread with `cargo test --workspace -- --test-threads=1`.

After a deliberate change to what the encoder writes, `scripts/gen_encoder_vectors.sh` makes its vectors again (macOS), and `scripts/gen_encoder_weights.py` writes its weight table, which is generated rather than written by hand. After a change to the C API, `EMBERWAVE_BLESS=1 cargo test -p emberwave-ffi --test header` writes `crates/emberwave-ffi/include/emberwave.h` anew; commit it with the change. `scripts/check_c_example.sh` needs CMake and a C compiler, and `cargo build -p emberwave-ffi -p emberwave-cli` first. `scripts/build_xcframework.sh` builds the XCFramework for iOS and macOS on a Mac with Xcode, after `rustup target add aarch64-apple-ios aarch64-apple-ios-sim`; `scripts/build_xcframework.sh macos-arm64` builds the Mac's library alone, which needs no extra target. `scripts/check_xcframework.sh target/debug/emberwave-cli` then checks what was built, the iOS Simulator's library in a simulator, which needs an iOS Simulator runtime (Xcode > Settings > Components).

`scripts/build_web.sh` builds the browser player into `web/` (it needs `rustup target add wasm32-unknown-unknown` and `cargo install wasm-bindgen-cli` at the version of the `wasm-bindgen` crate in `Cargo.lock`), and `node scripts/check_web.mjs` then checks that the module instantiates, as CI does; serve `web/` with any static file server, for example `python3 -m http.server -d web 8000`, and open it in a browser with WebGPU. `scripts/verify_4k.sh` is the 4K acceptance check, and [`BENCHMARKS.md`](BENCHMARKS.md) explains the benchmark scripts. Two leak checks run on request, both on the 4K stream of `scripts/gen_bench_data.sh`: `cargo test --release -p emberwave-core --test gpu -- --ignored` counts the GPU resources over 1000 frames, and `scripts/soak.sh` watches the memory of `emberwave-cli` over as many. `scripts/readme_figures.py` draws the README's banner and charts into `docs/images/`, and CI checks that the files are what it draws. Their numbers are copied from `BENCHMARKS.md`, as are those of the README's highlights and of the figures' alt texts: when a benchmark run changes them, change them in all three places and run the script again. `scripts/quality_figure.py` makes the README's picture of the quality, `docs/images/quality.png`, and the numbers of [`BENCHMARKS.md`](BENCHMARKS.md#picture-quality), by hand: it needs the frame, which the repository does not have, and numpy and Pillow.

## Test vectors and acceptance bars

- Streams come from upstream's encoder, never from emberwave or from hand-made bytes: a stream that only emberwave understands proves nothing. Unit tests may build single blocks by hand to reach specific code paths.
- `scripts/gen_testdata.sh` regenerates the committed vectors and their reference decodes. Upstream's encoder is not byte-for-byte reproducible, so regenerate only when a vector is added or changed.
- Large streams are generated into `testdata/generated/` and never committed.
- Decoded frames must reach 60 dB PSNR per plane against upstream's default precision and 80 dB against its FP32 mode (decision D9). Dequantised coefficients must equal the CPU's bit for bit. Never lower a bar to make a test pass: find out why it fails.

## Rules for the code

- **Portable GPU code.** The shaders use only what wgpu guarantees on Vulkan, Metal, DX12 and WebGPU: no subgroup operations, no `f16`, no 8-bit or 16-bit storage, no push constants, at most 256 invocations and 16 KiB of workgroup memory per workgroup. `tests/shaders.rs` enforces it.
- **Every shader starts with a comment** naming the stage of the upstream algorithm it implements.
- **Constant indices for local arrays in WGSL.** An array local to an invocation that is indexed by a loop variable can end up in memory instead of registers; that made the inverse transform 4.5 times slower (decision D24). Write such code out, and measure with `scripts/bench.sh` before and after.
- **Write workgroup memory before reading it.** The pipelines do not zero workgroup memory (decision D25), so a shader that reads an element no invocation of its workgroup has written gets whatever an earlier workgroup left there. Every workgroup variable is written in full, followed by a barrier, before it is read. Adding, removing or resizing one makes `tests/shaders.rs` fail until its list is updated; check the writes and reads before you update it. The GPU tests compare every level with the CPU exactly, which catches a slip in practice, but only if the stale values differ from the right ones.
- **No code from upstream.** Understand upstream's algorithm and write it again; only the algorithm and the bitstream format are shared. Bitstream constants get names, with a comment saying where upstream defines them.
- **No `unsafe`.** The workspace forbids it.
- **Public items have doc comments.**
- **No "fix later" comments.** Open an issue instead.
- **New dependencies** need a reason in the pull request.

## Commits and pull requests

- Work on a branch and open a pull request against `main`. It is merged with a merge commit once CI passes on Linux, macOS and Windows.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org) (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `perf:`, `refactor:`) and say why in the body.
- Code, commit messages and documentation are in English. The project's own issues and pull requests use a Korean template (decision D6) with the sections 목적 (purpose), 변경 요약 (changes), 테스트 방법 (how it was tested), 벤치 영향 (effect on benchmarks) and 관련 Issue (`Closes #N`). Outside contributions may write the same sections in English.
- A design decision goes into the Decisions section of `ARCHITECTURE.md`, with its date.
- A benchmark number is only published with its environment. `scripts/bench.sh` records it with the numbers.

## Licence

emberwave is MIT licensed, and contributions are accepted under the same licence.
