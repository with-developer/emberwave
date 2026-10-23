# Architecture

> **Status: skeleton (M0).** The bitstream format is documented in M1 and the GPU pipeline in M2. The decision log at the end is kept up to date from the start.

## Overview

emberwave decodes PyroWave bitstreams. The work is split the way upstream splits it:

```
bytes ──► [CPU] packet parser ──► block table + payload words
                                       │
                                       ▼
                         [GPU] dequantisation (coefficients → float)
                                       │
                                       ▼
                         [GPU] inverse DWT, level by level (5 levels)
                                       │
                                       ▼
                         [GPU] Y / Cb / Cr planes ──► (M3) YUV → RGB
```

Steps that upstream runs on the CPU stay on the CPU, and steps that upstream runs on the GPU stay on the GPU. Parity with the reference comes before any optimisation.

## Bitstream format

To be written in M1.

## GPU pipeline

To be written in M2.

## Decisions

Short-form ADRs, newest last. Each entry is dated.

### 2026-10-23 — D1: Reference implementation and test vector source

**Context.** `CLAUDE.md` assumes upstream is Vulkan-only and plans to fall back to MoltenVK on macOS. Upstream now also contains a Metal port (`metal/`, MIT, with an encoder and a decoder behind a C API). Its README describes it as an AI-assisted port that converts the upstream shaders to MSL.

**Decision.** Vectors and reference output are produced on the macOS development machine with upstream's Metal encoder and decoder, built from the pinned revision in `testdata/README.md`. The build and basic object creation were verified on 2026-10-23. Output of the upstream Vulkan implementation (for example from the Windows host in `CLAUDE.md` §8) is a welcome cross-check once someone provides it.

**Consequence.** The Metal port is itself a port. If it ever disagrees with the Vulkan implementation, that is investigated and reported rather than averaged away.

### 2026-10-23 — D2: Container format and file extension

**Decision.** Files use upstream's PWV1 raw disk format with the extension `.pwv1`. The `.pyro` extension in `CLAUDE.md` was a placeholder ("or the original format's extension").

### 2026-10-23 — D3: Acceptance criterion for decoder output

**Context.** Upstream's `bitstream.md` states that the decoding process is not bit-exact: the CDF 9/7 transform is defined in floating point and implementations may store intermediates in FP16.

**Decision.** A decoded frame passes when its PSNR against the reference decoder's output is at least 60 dB, measured per plane (Y, Cb, Cr) at the output bit depth. The threshold is never lowered to make a test pass.

### 2026-10-23 — D4: Portability constraints for the shaders

**Context.** Upstream's dequantisation shader needs the Vulkan subgroup operations basic, vote, ballot, arithmetic and shuffle-relative plus 8-bit and 16-bit storage buffer access (with texel-buffer fallbacks), and its decoder refuses to initialise without the subgroup features. None of these belong to the feature set wgpu guarantees across backends.

**Decision.** The WGSL implementation uses none of them. Prefix sums use workgroup memory and `workgroupBarrier()`, the payload is read as `array<u32>` with bytes and half-words extracted by shifts, and storage textures stick to formats in the portable baseline. Subgroup operations are not needed, so stop condition 5 in `CLAUDE.md` does not apply. An optional subgroup fast path may be considered after parity with the reference is reached.

### 2026-10-23 — D5: Toolchain and lints

**Decision.** Rust edition 2024, resolver 3, `rust-version` 1.85 (to be revisited when wgpu is added). Workspace lints: `unsafe_code = "forbid"` (the FFI crate will define its own lint table when it needs `unsafe`) and `missing_docs = "warn"`. CI turns warnings into errors through `clippy -D warnings`. Crates are `publish = false` because releases are GitHub Releases.

### 2026-10-23 — D6: Workflow conventions

**Decision.** Pull requests are merged with a merge commit, so each milestone shows up as one entry in the first-parent history. Commit messages, code and documentation are in English. Issues and pull request descriptions follow the Korean template from `CLAUDE.md`. Third-party GitHub Actions are pinned by commit SHA; first-party actions use a major version tag.

### 2026-10-23 — D7: Hosting

**Decision.** The repository is public at `with-developer/emberwave`, chosen by the repository owner on 2026-10-23, with default branch `main`. The empty root commit was pushed directly to give pull requests a base. Everything after it goes through pull requests.
