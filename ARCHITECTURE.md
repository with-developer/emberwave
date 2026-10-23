# Architecture

> **Status (M2).** The bitstream format is documented below and implemented by `emberwave_core::bitstream`, with a slow CPU reference decoder in `emberwave_core::cpu` and the wgpu/WGSL decoder in `emberwave_core::gpu`. Colour conversion and the CLI come in M3. The decision log at the end is kept up to date.

## Overview

emberwave decodes PyroWave bitstreams. The work is split the way upstream splits it:

```
bytes ──► [CPU] packet parser ──► block table + payload words
                                       │
                                       ▼
                         [GPU] dequantisation (bit-planes + signs → float coefficients)
                                       │
                                       ▼
                         [GPU] inverse DWT, level by level (5 levels)
                                       │
                                       ▼
                         [GPU] Y / Cb / Cr planes ──► (M3) YUV → RGB
```

Steps that upstream runs on the CPU stay on the CPU, and steps that upstream runs on the GPU stay on the GPU. Parity with the reference comes before any optimisation.

| Module | What it is |
|---|---|
| `bitstream` | Pure Rust, GPU independent: headers, block layout, packet assembly, PWV1 files. Never panics on malformed input. This is the CPU stage. |
| `cpu` | Dequantisation and inverse DWT as plain sequential Rust, following `bitstream.md` literally. A test oracle and the yardstick for shader stages, not a product feature. |
| `gpu` | wgpu pipelines and WGSL shaders for the GPU stages (cargo feature `gpu`, on by default). |

## Bitstream format

This is PyroWave bitstream v1, which upstream froze on 2026-10-03. It is described by upstream's `bitstream/bitstream.md`; this section restates it in the form the code uses and adds what the document leaves out. `bitstream.md` itself says that where it and the implementation disagree the implementation is authoritative, so everything here was checked against output of the real encoder (see [Verification](#verification)).

"Upstream" is the PyroWave repository at the revision pinned in `testdata/README.md`.

### Conventions

- Byte oriented, little endian. A *word* is a little-endian `u32`.
- Bit-fields start at the least significant bit, in the order they are listed.
- Coefficient blocks are the unit of coding: every 32x32 block of a sub-band is coded in isolation, with no prediction or context. A missing block decodes as zeros.

### Frames, packets and the PWV1 file

A *frame* is one picture. It consists of a start of frame header (8 bytes) and the coded 32x32 blocks, back to back. For network use a frame is cut into packets at block boundaries: a packet is any number of whole blocks, optionally preceded by the start of frame header, which may travel in any packet.

The PWV1 file is the on-disk form:

```
┌───────────────────────┬───────────────┬─────────────────┬───────────────┬─────────────────┬─────
│ PWV1 header, 24 bytes │ u32 size (LE) │ frame 0         │ u32 size (LE) │ frame 1         │ ...
│                       │               │ `size` bytes    │               │ `size` bytes    │
└───────────────────────┴───────────────┴─────────────────┴───────────────┴─────────────────┴─────
                                          └ start of frame header + blocks ┘
```

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 4 | `magic` | the bytes `PWV1` |
| 4 | 8 | `stream` | the same 8 bytes as a start of frame header, carrying picture size, chroma format and colour signalling. `extended`, `sequence`, `total_blocks` and `code` are reserved and zero here |
| 12 | 4 | `frame_rate_num` | 0 together with a zero denominator means unrelated images |
| 16 | 4 | `frame_rate_den` | |
| 20 | 1 | `reference_bit_depth` | 0 (arbitrary), 8, 10 or 16 |
| 21 | 1 | `header_version` | must be 1 |
| 22 | 2 | padding | zero |

### Picture geometry

Dimensions are padded so the five-level transform divides evenly:

```
AlignedWidth  = max(align(Width, 32),  128)
AlignedHeight = max(align(Height, 32), 128)
```

`Width` and `Height` are in `1..=16384`, and 4:2:0 needs both to be even. Sub-band sizes follow from the padded size. Level 0 is the finest, level 4 the coarsest:

```
band width  at level L = (AlignedWidth  / 2) >> L
band height at level L = (AlignedHeight / 2) >> L
```

Only one LL band is transmitted, the one at level 4. The other LL bands are reconstructed by the inverse transform. A band is cut into 32x32 blocks, row-major; blocks at the right and bottom edge may be partial, and so may the 8x8 blocks inside them.

```
┌─────┬─────┬───────────────┬─────────────────────────────┐
│ LL4 │ HL4 │               │                             │
├─────┼─────┤      HL3      │                             │
│ LH4 │ HH4 │               │                             │
├─────┴─────┼───────────────┤            HL2              │
│           │               │                             │
│    LH3    │      HH3      │                             │
│           │               │                             │
├───────────┴───────────────┼─────────────────────────────┤
│                           │                             │
│            LH2            │            HH2              │
│                           │                             │
└───────────────────────────┴─────────────────────────────┘
        ... levels 1 and 0 continue the pattern ...
```

Each level has four bands. `HL` is high-pass horizontally and low-pass vertically, `LH` the other way round.

For 4:2:0 the chroma planes have half the luma size and no level 0 bands. Chroma is reconstructed from level 4 down to level 1, and the result of level 1 is the chroma plane. 4:4:4 chroma is coded like luma. Example, 1920x1080 in 4:2:0 (padded to 1920x1088):

| Level | Band size | 32x32 blocks per band | Bands | Components | Blocks |
|---|---|---|---|---|---|
| 4 | 60 x 34 | 2 x 2 | 4 (LL, HL, LH, HH) | Y, Cb, Cr | 48 |
| 3 | 120 x 68 | 4 x 3 | 3 | Y, Cb, Cr | 108 |
| 2 | 240 x 136 | 8 x 5 | 3 | Y, Cb, Cr | 360 |
| 1 | 480 x 272 | 15 x 9 | 3 | Y, Cb, Cr | 1215 |
| 0 | 960 x 544 | 30 x 17 | 3 | Y only | 1530 |
| | | | | **Total** | **3261** |

### Block numbering

Every possible 32x32 block has a linear index, `block_index`, in this order:

```
for level in 4 down to 0:
    for component in Y, Cb, Cr:
        skip if level == 0 and component != Y and chroma is 4:2:0
        for band in (LL, HL, LH, HH if level == 4 else HL, LH, HH):
            for y in block rows of the band:
                for x in block columns of the band:
                    assign the next index
```

`FrameLayout::new` builds this table and `FrameLayout::locate` inverts it.

### Headers

Both headers are 8 bytes (two words). Bit 31 of word 0, `extended`, tells them apart.

Block header (`extended = 0`):

| Word | Bits | Field | Meaning |
|---|---|---|---|
| 0 | 0–15 | `ballot` | bit *i* set: the 8x8 block at row-major position *i* of this 32x32 block carries data |
| 0 | 16–27 | `payload_words` | size of the whole block in words, header included |
| 0 | 28–30 | `sequence` | frame counter, modulo 8 |
| 0 | 31 | `extended` | 0 |
| 1 | 0–7 | `quant_code` | block quantiser scale, see [Dequantisation](#dequantisation) |
| 1 | 8–31 | `block_index` | linear block index |

Start of frame header (`extended = 1`):

| Word | Bits | Field | Meaning |
|---|---|---|---|
| 0 | 0–13 | `width_minus_1` | |
| 0 | 14–27 | `height_minus_1` | |
| 0 | 28–30 | `sequence` | frame counter, modulo 8 |
| 0 | 31 | `extended` | 1 |
| 1 | 0–23 | `total_blocks` | number of non-zero 32x32 blocks the frame carries |
| 1 | 24–25 | `code` | 0 = start of frame; other values are reserved |
| 1 | 26 | `chroma_resolution` | 0 = 4:2:0, 1 = 4:4:4 |
| 1 | 27 | `color_primaries` | 0 = sRGB, 1 = BT.2020 |
| 1 | 28 | `transfer_function` | 0 = sRGB, 1 = PQ |
| 1 | 29 | `ycbcr_transform` | 0 = BT.709, 1 = BT.2020 (non-constant luminance) |
| 1 | 30 | `ycbcr_range` | 0 = full, 1 = limited |
| 1 | 31 | `chroma_siting` | 0 = centre, 1 = left |

The last five fields are signalling only; they do not change how a frame is decoded.

### Block payload

A block is a header followed by tables, bit-planes and sign bits. `N` is the number of set bits in `ballot`, `P` the total number of bit-plane bytes and `S` the number of non-zero coefficients:

```
byte 0        ┌──────────────────────────────────────────┐
              │ header                          8 bytes  │
byte 8        ├──────────────────────────────────────────┤
              │ control[0..N)     u16 each      2N bytes │
              ├──────────────────────────────────────────┤
              │ qscale[0..N)      u8 each       N bytes  │
byte 8 + 3N   ├──────────────────────────────────────────┤
              │ bit-planes                      P bytes  │
byte 8+3N+P   ├──────────────────────────────────────────┤
              │ sign bits                       ceil(S/8)│
              ├──────────────────────────────────────────┤
              │ padding to a 32-bit boundary             │
              └──────────────────────────────────────────┘
```

So `payload_words = ceil((8 + 3N + P + ceil(S/8)) / 4)`. The padding bytes and the unused bits of the last sign byte have no meaning and are ignored.

The active 8x8 blocks (those with a `ballot` bit) follow each other in row-major order, and entry *i* of `control` and `qscale` belongs to the *i*-th of them.

An 8x8 block is made of eight 4x2 subblocks, and the values of a subblock are numbered down the columns:

```
8x8 block: subblock numbers              subblock: value numbers

  x:  0 1 2 3 | 4 5 6 7                    x:  0 1 2 3
     ┌─────────┬─────────┐                    ┌─────────┐
y 0-1│    0    │    4    │                 y 0│ 0 2 4 6 │
y 2-3│    1    │    5    │                 y 1│ 1 3 5 7 │
y 4-5│    2    │    6    │                    └─────────┘
y 6-7│    3    │    7    │
     └─────────┴─────────┘
```

Fields of an active 8x8 block:

- `control` holds eight 2-bit fields; field *s* (bits 2*s* and 2*s*+1) is the number of extra bit-planes of subblock *s*.
- `qscale` holds, in its low nibble, the number of *base* bit-planes `q`, and in its high nibble a 4-bit scale code for the 8x8 block.
- Subblock *s* has `q + control field s` bit-planes.

Decoding a block, in the order the data is stored:

1. Visit the active 8x8 blocks in row-major order, and in each the subblocks 0 to 7.
2. For each subblock read its bit-planes, most significant plane first. A plane is one byte, and bit *v* of the byte is bit *plane index* of the magnitude of value *v*. The planes of all subblocks are stored back to back, so the offset of a subblock is the sum of the sizes before it.
3. Every value with a non-zero magnitude takes the next bit of the sign area. Signs are numbered in the same order as the values (8x8 block, subblock, value) and packed from the least significant bit of each byte; 1 means negative.
4. A value is placed at `x = 32 * block_x + 8 * (block8 mod 4) + 4 * (subblock / 4) + value / 2` and `y = 32 * block_y + 8 * (block8 / 4) + 2 * (subblock mod 4) + value mod 2` inside its band. Values that fall outside the band are decoded, because they still take a sign bit, and then discarded.

An 8x8 block with an all-zero `control` word decodes to nothing in the reference decoder, whatever its `qscale` says. The encoder never writes one because it clears the `ballot` bit for such blocks.

Worked example, which is also the unit test `decodes_a_hand_built_block`: one active 8x8 block, `control = 0b1001` (subblock 0 has 1 plane, subblock 1 has 2), base planes 0. Subblock 0 stores the plane byte `0b00101001`, so values 0, 3 and 5 have magnitude 1. Subblock 1 stores `0b01000010` then `0b01000100`, so value 1 is `0b10` = 2, value 2 is `0b01` = 1 and value 6 is `0b11` = 3. The six sign bits follow, in order sub 0 value 0, 3, 5, then sub 1 value 1, 2, 6.

### Dequantisation

A decoded integer magnitude `m` (with sign) becomes a float:

```
coefficient = ±(m + 0.5) * scale32 * scale8        for m != 0, otherwise 0

scale32 = (8 + (quant_code & 7)) * 2^(1 - (quant_code >> 3))
scale8  = code / 8 + 0.25                           code: the high nibble of qscale
```

The `+ 0.5` is the reconstruction point of the deadzone quantiser. `scale32` is exact in `f32`; it equals upstream's `(1 / 2^23) * ((8 + m) << (20 + e))` with `e = 4 - (quant_code >> 3)`. The product is evaluated as `(m + 0.5) * (scale32 * scale8)`, the same order as the reference shader, so a CPU or GPU implementation can reproduce it bit for bit.

| `quant_code` | 0 | 7 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| `scale32` | 16 | 30 | 8 | 4 | 1 | 1/16 |

### Inverse transform and output

The transform is the CDF 9/7 filter of JPEG 2000 in lifting form (ITU-T T.800, section F.4.8.2). One level combines the four bands of that level into the LL band of the next finer level:

1. Interleave: LL goes to even columns and even rows, HL to odd columns and even rows, LH to even columns and odd rows, HH to odd columns and odd rows. The result is twice as wide and twice as high as a band.
2. Filter every row, then every column, with the 1-D inverse below. The two directions commute mathematically.

1-D inverse of an interleaved line, low-pass samples at even positions and high-pass samples at odd positions:

```
even samples  *= K          odd samples  *= 1/K
even          -= DELTA * (left + right)
odd           -= GAMMA * (left + right)
even          -= BETA  * (left + right)
odd           -= ALPHA * (left + right)

ALPHA = -1.586134342059924   BETA = -0.052980118572961   GAMMA = 0.882911075530934
DELTA =  0.443506852043971   K    =  1.230174104914001
```

The signal is extended at both ends by whole-sample symmetric mirroring (`x[-1] = x[1]`, `x[n] = x[n-2]`), as in JPEG 2000, before filtering; four samples of extension are enough for four lifting steps. Levels are processed from 4 down to 0 (down to 1 for 4:2:0 chroma). The result is cropped to the plane size, shifted by `+0.5` and clamped to `[0, 1]`.

Converting the float planes to integers is not part of the bitstream definition. The reference decoder writes UNORM storage, i.e. `round_to_nearest_even(v * (2^bits - 1))`, and so does the CPU decoder here.

Decoding is not bit-exact between implementations: the transform is floating point and the reference decoder may store intermediates as FP16.

### Packet assembly (the CPU stage)

`FrameAssembler` follows upstream's `push_packet`:

- A packet is read header by header. A start of frame header sets `total_blocks`; a block header followed by `payload_words * 4` bytes is a block.
- The sequence number decides which frame a header belongs to. With `d = (new - last) mod 8`: `d = 0` is the current frame, `1 <= d <= 3` starts a newer frame and discards what was collected, `4 <= d <= 7` is an older frame; it is ignored and the rest of the packet is dropped. The first packet always starts a frame.
- The first copy of a block wins; later copies of the same `block_index` are ignored.
- A header or block is validated before anything changes, and one that is rejected changes nothing, even if it belongs to a newer frame. Upstream does it the other way round: it starts the new frame as soon as a non-stale header arrives and validates afterwards, so a corrupt packet can discard the frame being assembled.
- The output is a table with the word offset of every block (`u32::MAX` for blocks that were not received) and the received blocks concatenated, headers included. This is exactly what the GPU dequantisation stage consumes.
- A frame is ready when `received >= total_blocks`. A frame that lost packets may be decoded anyway if the coarsest bands are complete and more than 90 percent of the blocks arrived; the thresholds are parameters.

### Where the implementation differs from or goes beyond `bitstream.md`

| Topic | Upstream | emberwave |
|---|---|---|
| Order of restart and validation | A non-stale header starts the new frame first and is validated afterwards, so a rejected header or block of a newer frame has already discarded the old frame | Validated first; a rejected header or block changes nothing, and an old frame stays available until a valid packet of a newer one arrives |
| Block internals | Not validated on the CPU. The GPU reads whatever the header declares | Tables and bit-planes must fit in `payload_words`, otherwise the block is rejected and the assembler is left unchanged. A sign area that is too short is detected while decoding |
| Colour signalling in the frame | The encoder leaves colour and range bits of the in-frame start of frame header at zero (checked on real vectors: range reads "full" for a limited-range stream) | Colour information is taken from the PWV1 header, which the file writers fill in |
| Odd 4:2:0 sizes | Required to be even, but not checked | Rejected |
| `quant_code >= 200` | The integer shift `1 << (20 + e)` is undefined | The scale formula is extended to smaller values |
| Reproducibility | The encoder's output is not byte-for-byte reproducible: the unused bits of a block's last sign byte are leftover GPU memory | Tests use committed bitstreams and compare decoded pictures, never bitstreams |
| A duplicate block that is malformed | Silently skipped, because the checks come after the duplicate test | Validated like any other block, so it is rejected |
| An old header of an unknown kind | Ignored with the rest of the packet | The same; a current or newer one is an error |

### Verification

Measured on 2026-10-23 with the committed vectors (five streams, 3 frames each except one with 2: solid, gradient, noise at two rates, and 4:4:4 noise with odd dimensions), decoding every frame with `emberwave_core::cpu` and comparing to the original decoder per plane:

| Reference | Worst PSNR | Largest difference | Notes |
|---|---|---|---|
| Default precision (FP16 wavelet storage) | 64.83 dB | 1 LSB | Above the 60 dB bar |
| FP32 precision (`PYROWAVE_PRECISION=2`) | 86.28 dB | 1 LSB | Most planes are bit-identical; the rest differ in a single pixel |

The gap to the default precision is the reference decoder's own FP16 storage. In addition, for every block of every vector, `payload_words` equals the formula above exactly, which pins down the layout of planes and signs.

## GPU pipeline

`gpu::Decoder` decodes frames of one size and chroma format. All GPU memory is allocated when it is created and reused for every frame. The payload buffer is the one exception: it grows when a frame needs more room.

### Stages

A frame takes seven dispatches in three compute passes, whatever its size:

| Stage | Shader | Dispatch | Reads | Writes |
|---|---|---|---|---|
| Dequantisation | `dequant.wgsl` | 1: one workgroup of 128 invocations per block index | payload, block offsets | the pyramid |
| Inverse DWT | `idwt.wgsl` | 5: one per level, coarsest first; a workgroup per 32x32 output tile and component | the pyramid | the pyramid (next LL band) or the float planes |
| Packing | `pack.wgsl` | 1: one invocation per output word | float planes | packed 8-bit planes |

The packed planes are copied to a mappable buffer and returned as a `Frame`.

### Memory

| Buffer | Content | Size for 4K 4:2:0 |
|---|---|---|
| payload (`array<u32>`) | received blocks, padded with four zero words | grows on demand, 64 KiB at least |
| block offsets (`array<u32>`) | word offset of every block, `u32::MAX` if missing | 49 KiB |
| pyramid (`array<f32>`) | the four bands of every level of every component, back to back | 63.6 MiB |
| float planes (`array<f32>`) | the finished planes, in `[0, 1]` | 47.5 MiB |
| packed planes (`array<u32>`) | 8-bit planes, four pixels per word, rows padded to words | 11.9 MiB |
| readback (`MAP_READ`) | copy of the packed planes for the CPU | 11.9 MiB |

The pyramid holds LL, HL, LH and HH for each level, so the LL bands of levels 3 to 0 have room where the inverse transform of the level above writes them. Every band is rewritten for every frame, with zeros for blocks that were not received, so nothing needs clearing. The sizes of the pyramid and the planes are checked against the device limits when the decoder is created. The payload buffer is checked the same way every time it grows: a frame whose payload does not fit gives `GpuError::LimitExceeded` instead of an allocation that fails. With the WebGPU default of 128 MiB that can only happen for a stream of maximal blocks (a 4K frame of them is about 200 MiB); real frames are a few MiB.

Per-dispatch parameters (band table, tile sizes, buffer offsets) depend only on the picture size, so they are computed once and live in small uniform buffers.

### Dequantisation

One workgroup per block index, found by a lookup in a table of the (at most 48) transmitted bands. Each invocation decodes the 8 values of one 4x2 subblock. If a block was not received, a value that comes through `workgroupUniformLoad` makes the whole workgroup write zeros and return, so the later barriers stay in uniform control flow.

1. Sixteen invocations read the control word and scale byte of one 8x8 block each and compute its number of bit-plane bytes into workgroup memory.
2. After a barrier every invocation adds up the sizes before its own 8x8 block, and the sizes of the preceding subblocks of its block, to find its bit-planes, and decodes its magnitudes.
3. Every non-zero value takes one sign bit. Each invocation writes its count to workgroup memory; after a barrier sixteen invocations total the counts of one 8x8 block each; after a second barrier every invocation adds the totals of the earlier 8x8 blocks and the counts of the earlier subblocks of its block to find its first sign bit.
4. Values are computed as `(m + 0.5) * (scale32 * scale8)`, the same expression and order as the CPU decoder, and stored with bounds checks.

Without subgroup operations the prefix sum costs two barriers and about 20 workgroup memory reads per invocation.

### Inverse transform

A workgroup of 256 invocations produces a 32x32 tile of one level's output:

1. It loads a 40x40 window of the interleaved image into workgroup memory (6400 bytes). Each window coordinate is mirrored about the level's true size, using whole-sample symmetry, which preserves the parity of the coordinate; the parity picks the band (LL, HL, LH or HH) and the mirrored coordinate halved picks the sample. Four samples of extension on each side are what four lifting steps need.
2. Horizontal pass: 160 invocations each take 16 samples of a window row, apply the gain and the four lifting steps in registers and keep the 8 valid results (workgroup memory, 5120 bytes). The apron of a segment is recomputed rather than shared.
3. Vertical pass: 128 invocations do the same down a column and store the 8 results: into the LL band of the next level, or, at the finest level of a component, shifted by +0.5 and clamped into the float plane.

Total workgroup memory is 11520 bytes. Tiles that stick out of the image (the coarsest levels can be 8x8) are computed in full and stored with bounds checks, and mirroring always uses the true level size, never the picture size. The component is the z index of the dispatch; at level 0 a 4:2:0 frame has luma only.

### Packing

Float planes become 8-bit samples with `round(v * 255)`. WGSL's `round` rounds ties to even on every backend (naga emits `rint`, `round` and `RoundEven` for Metal, HLSL and SPIR-V), which is what a UNORM storage write does. Four pixels share a word, first pixel in the low byte, and rows are padded to whole words; the CPU side cuts the padding off.

### Portability

The shaders use only what WebGPU guarantees: no subgroup operations, no 8-bit or 16-bit storage, no `f16`, no immediates (push constants), no storage textures. All buffers are plain storage or uniform buffers and every workgroup stays within 256 invocations and 16 KiB. Two tests hold this to account:

- `tests/shaders.rs` validates the WGSL with naga using an empty capability set, without a GPU, and checks the workgroup sizes and memory.
- `tests/gpu.rs` creates a device with `InstanceFlags::STRICT_WEBGPU_COMPLIANCE` and the default limits and decodes a vector on it.
- The rounding rule of the packing stage is checked on the real backend against the CPU's, on inputs that are exact ties (`gpu::decoder` unit test).

What upstream uses and what replaces it:

| Upstream (Vulkan) | emberwave (WGSL) |
|---|---|
| Subgroup scans in the dequantiser | Workgroup memory sums, two barriers |
| `uint8_t` / `uint16_t` buffer access | `array<u32>` with shifts |
| `textureGather` with a mirrored sampler | Explicit mirroring, then a load from the band buffer |
| FP16 storage images, `PYROWAVE_PRECISION` modes | FP32 buffers |
| 8-bit UNORM storage images for the output | A packing pass into `array<u32>` |
| Dozens of dispatches, one per band | One dequantisation dispatch and one per level |

### Verification

Run on 2026-10-23 with the committed vectors on an Apple M3 Max (Metal):

| Check | Result |
|---|---|
| Dequantised coefficients, GPU against CPU, all vectors | Bit-identical |
| The same on synthetic frames with every quantiser code, sparse ballots, all-zero control words and missing blocks | Bit-identical |
| LL bands after each inverse level, GPU against CPU | Largest difference 8.0e-7 (the tolerance is 1e-5) |
| Decoded planes against the default-precision reference | Worst 64.83 dB (needs 60), largest difference 1 LSB |
| Decoded planes against the FP32 reference | Worst 86.28 dB (needs 80), most planes bit-identical |

The GPU numbers equal the CPU decoder's, which is the point: the GPU pipeline adds no error of its own beyond float rounding.

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

**Decision.** A decoded frame passes when its PSNR against the reference decoder's output is at least 60 dB, measured per plane (Y, Cb, Cr) at the output bit depth. The threshold is never lowered to make a test pass. (Refined by D9.)

### 2026-10-23 — D4: Portability constraints for the shaders

**Context.** Upstream's dequantisation shader needs the Vulkan subgroup operations basic, vote, ballot, arithmetic and shuffle-relative plus 8-bit and 16-bit storage buffer access (with texel-buffer fallbacks), and its decoder refuses to initialise without the subgroup features. None of these belong to the feature set wgpu guarantees across backends.

**Decision.** The WGSL implementation uses none of them. Prefix sums use workgroup memory and `workgroupBarrier()`, the payload is read as `array<u32>` with bytes and half-words extracted by shifts, and storage textures stick to formats in the portable baseline. Subgroup operations are not needed, so stop condition 5 in `CLAUDE.md` does not apply. An optional subgroup fast path may be considered after parity with the reference is reached.

### 2026-10-23 — D5: Toolchain and lints

**Decision.** Rust edition 2024, resolver 3, `rust-version` 1.85 (to be revisited when wgpu is added). Workspace lints: `unsafe_code = "forbid"` (the FFI crate will define its own lint table when it needs `unsafe`) and `missing_docs = "warn"`. CI turns warnings into errors through `clippy -D warnings`. Crates are `publish = false` because releases are GitHub Releases.

### 2026-10-23 — D6: Workflow conventions

**Decision.** Pull requests are merged with a merge commit, so each milestone shows up as one entry in the first-parent history. Commit messages, code and documentation are in English. Issues and pull request descriptions follow the Korean template from `CLAUDE.md`. Third-party GitHub Actions are pinned by commit SHA; first-party actions use a major version tag.

### 2026-10-23 — D7: Hosting

**Decision.** The repository is public at `with-developer/emberwave`, chosen by the repository owner on 2026-10-23, with default branch `main`. The empty root commit was pushed directly to give pull requests a base. Everything after it goes through pull requests.

### 2026-10-23 — D8: A CPU reference decoder in `emberwave-core`

**Context.** The original decoders expose only finished pictures, but the M1 definition of done asks for a parser that is checked against real encoder output. Pictures are the only thing that can be compared, and they depend on dequantisation and the inverse transform as well as on the parser.

**Decision.** Dequantisation and the inverse DWT are implemented a second time on the CPU in `emberwave_core::cpu`, as literally and simply as possible. It is a test oracle: it validates the parser against the original decoder in M1, and in M2 it is what shader stages are compared against, coefficient by coefficient. It is not the product decode path and is not optimised.

**Consequence.** The product path keeps upstream's split (parse on the CPU, everything else on the GPU). The oracle is exact where it can be: its dequantised coefficients can be compared bit for bit with the GPU's.

### 2026-10-23 — D9: Two reference precisions

**Context.** The first comparison of the CPU oracle with the original decoder gave 64.8 dB or better at a maximum error of 1 LSB. Against the original decoder run in FP32 (`PYROWAVE_PRECISION=2`) the same output was bit-identical on most planes and at least 86 dB everywhere. The 65 dB is therefore the original's own FP16 storage, not an error of the oracle.

**Decision.** `scripts/gen_testdata.sh` writes two reference decodes per vector, `NAME.ref.y4m` (default precision) and `NAME.ref-fp32.y4m`. Decoders must reach 60 dB against the first, as D3 says, and 80 dB against the second. The second bar is stricter, not looser: an implementation with a small error can still pass at 60 dB, but not at 80 dB. Neither bar is lowered to make a test pass.

**Consequence.** `EMBERWAVE_REFERENCE_DIR` redirects the tests to other decodes of the same vectors, for example from the upstream Vulkan decoder.

### 2026-10-23 — D10: Stricter block validation than upstream

**Context.** Upstream's CPU parser checks the packet bounds, the block index and `payload_words >= 2`, and leaves the inside of a block to the GPU, which then reads wherever the header points.

**Decision.** `FrameAssembler` also checks that a block's tables and bit-planes fit in its declared size, and a header or block that fails any check is rejected without changing the assembler, which includes the restart that a header of a newer frame would otherwise cause. Upstream restarts first and validates afterwards. The sign area depends on the decoded magnitudes, so its size is checked while decoding. Behaviour on streams from a real encoder is identical to upstream's.

**Consequence.** Malformed input produces errors, never reads outside a block, and never leaves the assembler inconsistent. Property tests check that.

### 2026-10-23 — D11: New dependencies

**Decision.** `thiserror` for the error type (`CLAUDE.md` lists it for M5; adding it now avoids rewriting the error type later). `proptest` as a dev-dependency for the "must not panic on malformed input" requirement: unlike `cargo fuzz` it runs on stable inside `cargo test`, so CI exercises it on every push, and `PROPTEST_CASES` scales it up for a deeper pass.

### 2026-10-23 — D12: One dequantisation dispatch for the whole frame

**Context.** Upstream dispatches the dequantiser once per band, about 50 dispatches for a frame, each with its own push constants. wgpu has no push constants in the portable set.

**Decision.** One dispatch with one workgroup per block index. The workgroup finds its band in a table of at most 48 entries (a uniform block) and its block's offset in the block table. The grid is two-dimensional once there are more than 65535 blocks. The prefix sum over sign bits uses per-8x8-block totals (two barriers) instead of a scan over 128 invocations.

**Consequence.** A frame is seven dispatches. Parameters never change between frames, so they are written once.

### 2026-10-23 — D13: Own inverse transform kernel, FP32 buffers

**Context.** Upstream loads the inverse transform's input through `textureGather` and a mirrored sampler, and stores the wavelet pyramid as FP16 or FP32 storage images. `r16float` storage images are not in the portable set, and the sampler's mirroring is not the whole-sample symmetric extension the format needs without per-coordinate fix-ups.

**Decision.** Explicit mirroring of the interleaved coordinates and plain `f32` storage buffers. A 32x32 output tile per workgroup with a 40x40 window in workgroup memory, as described above. FP16 packing of the pyramid is an optimisation for later, to be weighed against the accuracy it costs (the original's default FP16 storage is why it sits about 20 dB below its FP32 mode).

### 2026-10-23 — D14: Output as packed 8-bit buffers

**Decision.** The planes are packed to `u32` words by a separate pass because `r8unorm` storage textures are outside the portable set. The same buffer serves the readback now and will feed the colour conversion in M3.

### 2026-10-23 — D15: What is compared exactly

**Decision.** Dequantised coefficients are compared bit for bit between GPU and CPU: they come from integer decoding and one multiplication, so they have no excuse to differ. The inverse transform is compared with a tolerance (1e-5; 8.0e-7 observed), because a GPU compiler may fuse a multiply and an add, which a CPU build does not, and WGSL cannot switch that off. Whole frames are compared with the PSNR bars of D9.

### 2026-10-23 — D16: wgpu 30, the `gpu` feature and its dependencies

**Decision.** `wgpu` 30.0.1 with the `vulkan`, `metal`, `dx12`, `wgsl`, `std` and `parking_lot` features, without GLES (which would need EGL) and without the browser backend (the wasm build gets its own crate later). The `rust-version` is raised to 1.87, wgpu's. The GPU code is behind the cargo feature `gpu`, on by default, so the parser and the CPU decoder can be used without wgpu.

New crates: `wgpu`; `pollster` to block on wgpu's futures in the synchronous API; `bytemuck` (with `derive`) to hand `u32` and parameter structs to buffers without `unsafe`, which the workspace forbids; `naga` as a dev-dependency for the offline shader validation, at the version wgpu uses.

The GPU tests skip when there is no adapter, unless `EMBERWAVE_REQUIRE_GPU` is set; CI sets it on Linux, where it installs lavapipe, a software Vulkan driver.
