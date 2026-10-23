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

Environment variables:

- `EMBERWAVE_REQUIRE_GPU=1` turns a missing adapter into a failure instead of a skip. CI sets it on Linux, where lavapipe is installed.
- `EMBERWAVE_BACKEND=vulkan|metal|dx12` picks the backend, for the tests and for the CLI.
- `EMBERWAVE_REFERENCE_DIR=DIR` compares against reference decodes made elsewhere, for example by upstream's Vulkan decoder.
- `PROPTEST_CASES=100000` runs a deeper robustness pass than the default 256 cases.

`scripts/build_web.sh` builds the browser player into `web/` (it needs `rustup target add wasm32-unknown-unknown` and `cargo install wasm-bindgen-cli` at the version of the `wasm-bindgen` crate in `Cargo.lock`); serve `web/` with any static file server, for example `python3 -m http.server -d web 8000`, and open it in a browser with WebGPU. `scripts/verify_4k.sh` is the 4K acceptance check, and [`BENCHMARKS.md`](BENCHMARKS.md) explains the benchmark scripts. Two leak checks run on request, both on the 4K stream of `scripts/gen_bench_data.sh`: `cargo test --release -p emberwave-core --test gpu -- --ignored` counts the GPU resources over 1000 frames, and `scripts/soak.sh` watches the memory of `emberwave-cli` over as many.

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
