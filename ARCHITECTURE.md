# Architecture

> **Status (v0.2.0).** The bitstream format is documented below and implemented by `emberwave_core::bitstream`, with a slow CPU reference decoder in `emberwave_core::cpu`, the wgpu/WGSL decoder in `emberwave_core::gpu` and the colour conversion in `emberwave_core::color`. `emberwave-cli` decodes files with them. Decode times are in `BENCHMARKS.md`. The decision log at the end is kept up to date.

## Overview

emberwave decodes PyroWave bitstreams. The work is split the way upstream splits it:

```
bytes ──► [CPU] packet parser ──► block table + payload words
                                       │
                                       ▼
                         [GPU] dequantisation (bit-planes + signs → float coefficients)
                                       │
                                       ▼
                         [GPU] inverse DWT, level by level (5 levels);
                               the last level writes 8-bit Y / Cb / Cr planes
                                       │
                                       ▼
                         [GPU] YUV → RGB (optional)
```

Steps that upstream runs on the CPU stay on the CPU, and steps that upstream runs on the GPU stay on the GPU. Parity with the reference comes before any optimisation.

| Module | What it is |
|---|---|
| `bitstream` | Pure Rust, GPU independent: headers, block layout, packet assembly, PWV1 files. Never panics on malformed input. This is the CPU stage. |
| `cpu` | Dequantisation and inverse DWT as plain sequential Rust, following `bitstream.md` literally. A test oracle and the yardstick for shader stages, not a product feature. |
| `color` | The YCbCr to RGB conversion: matrices, range handling and chroma siting, with a CPU implementation that the shader is tested against. |
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

A frame takes six dispatches in two compute passes, whatever its size, and a seventh dispatch in a third pass when it is converted to RGB:

| Stage | Shader | Dispatch | Reads | Writes |
|---|---|---|---|---|
| Dequantisation | `dequant.wgsl` | 1: one workgroup of 128 invocations per block index | payload, block offsets | the pyramid |
| Inverse DWT | `idwt.wgsl` | 5: one per level, coarsest first; a workgroup per 32x32 output tile and component | the pyramid | the pyramid (next LL band) or, at the last level of a component, the packed 8-bit planes |
| Colour conversion (optional) | `yuv2rgb.wgsl` | 1: one invocation per pixel | packed planes | RGBA words |

The packed planes are copied to a mappable buffer and returned as a `Frame`. When the frame is converted to RGB, the RGBA words are copied instead, as an `RgbFrame`, and the packed planes stay in GPU memory.

### Memory

| Buffer | Content | Size for 4K 4:2:0 |
|---|---|---|
| payload (`array<u32>`) | received blocks, padded with four zero words | grows on demand, 64 KiB at least |
| block offsets (`array<u32>`) | word offset of every block, `u32::MAX` if missing | 49 KiB |
| pyramid (`array<f32>`) | the four bands of every level of every component, back to back | 63.6 MiB |
| packed planes (`array<u32>`) | 8-bit planes, four pixels per word, rows padded to words | 11.9 MiB |
| readback (`MAP_READ`) | copy of the packed planes for the CPU | 11.9 MiB |
| RGBA words (`array<u32>`) | one word per pixel, created on the first RGB decode | 31.6 MiB |
| RGB readback (`MAP_READ`) | copy of the RGBA words, likewise | 31.6 MiB |

The pyramid holds LL, HL, LH and HH for each level, so the LL bands of levels 3 to 0 have room where the inverse transform of the level above writes them. Every band is rewritten for every frame, with zeros for blocks that were not received, so nothing needs clearing. The sizes of the pyramid and the planes are checked against the device limits when the decoder is created, and that of the RGBA buffer when it is created. The payload buffer is checked the same way every time it grows: a frame whose payload does not fit gives `GpuError::LimitExceeded` instead of an allocation that fails. With the WebGPU default of 128 MiB that can only happen for a stream of maximal blocks (a 4K frame of them is about 200 MiB); real frames are a few MiB.

Per-dispatch parameters (band table, tile sizes, buffer offsets) depend only on the picture size, so they are computed once and live in small uniform buffers.

No GPU resource outlives a frame. The decoder's own buffers are created once (the payload buffer grows when a frame needs more room), and the only allocations of a frame are the staging buffers wgpu uses for `Queue::write_buffer`, three or four of them, which it frees when the frame is done. The test `decoding_frame_after_frame_allocates_nothing` compares every count wgpu keeps of live objects and memory (buffers, textures, bind groups, pipelines, command encoders, query sets and so on; Metal does not count memory) after a first pass over every frame and every way of decoding with the counts after 48 more decodes. The same check over 1000 frames of a 4K stream, `a_thousand_4k_frames_leave_no_trace`, which runs on request, found 19 buffers, 8 bind groups, 3 command encoders and 1 query set before and after on an Apple M3 Max (2026-10-23, v0.2.0). `scripts/soak.sh` found the resident memory of `emberwave-cli` flat at 178 MiB while it decoded 1000 4K frames to RGB and read them back.

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
3. Vertical pass: 128 invocations do the same down a column and store the 8 results into the LL band of the next level. At the last level of a component they are 8-bit samples instead, packed as described next.

The lifting, and the loads and stores around it, are written out with constant indices, which keeps the samples in registers (D24). Total workgroup memory is 11520 bytes, and it is not zeroed first: the load and the two passes write every element before it is read (D25). Tiles that stick out of the image (the coarsest levels can be 8x8) are computed in full and stored with bounds checks, and mirroring always uses the true level size, never the picture size. The component is the z index of the dispatch; at level 0 a 4:2:0 frame has luma only.

### 8-bit output

The last level of each component writes the decoded plane as 8-bit samples, which is what upstream's UNORM storage images hold. A sample is `round(clamp(v + 0.5, 0, 1) * 255)` (`to_unorm8` in `idwt.wgsl`); the float picture never goes to memory. The vertical pass leaves the 8-bit levels of its 32x32 tile in the window array, which is free by then, and after one barrier each of the 256 invocations writes one word: four samples of a tile row. Every level waits at that barrier, though only the last needs it. Inside `if (is_final)` the barrier would be legal, since `is_final` is the same for the whole workgroup, but there it crashed WARP, Direct3D's software rasteriser, in CI; a bisection of the commits pinned the crash on it.

WGSL's `round` rounds ties to even on every backend (naga emits `rint`, `round` and `RoundEven` for Metal, HLSL and SPIR-V), which is what a UNORM storage write does. Four pixels share a word, first pixel in the low byte, and rows are padded to whole words with zero samples; the CPU side cuts the padding off.

### Colour conversion

The bitstream ends at the Y, Cb and Cr planes; turning them into RGB is left to the application. Upstream does it in its viewer (`shaders/yuv2rgb.frag`), not in the codec. emberwave offers the step as an option of the decoder (`Decoder::decode_rgb`): one more pass over the packed planes, so what gets converted is the 8-bit picture.

Per pixel, with `y`, `cb` and `cr` the 8-bit samples times 1/255:

```
cb -= 128/255;  cr -= 128/255
if limited range:
    y  = clamp((y - 16/255) * 255/219, 0, 1)
    cb = clamp(cb * 255/224, -0.5, 0.5)            (cr the same)
R = y + R_Cr * cr
G = y - G_Cb * cb - G_Cr * cr
B = y + B_Cb * cb
output = round_ties_even(clamp(v, 0, 1) * 255)
```

The four factors come from the luma weights `Kr` and `Kb` of the standard, with `Kg = 1 - Kr - Kb`: `R_Cr = 2 (1 - Kr)`, `B_Cb = 2 (1 - Kb)`, `G_Cb = 2 Kb (1 - Kb) / Kg` and `G_Cr = 2 Kr (1 - Kr) / Kg`. They are computed in `f64` and rounded to `f32` once.

| Matrix | `Kr` | `Kb` | `R_Cr` | `G_Cb` | `G_Cr` | `B_Cb` |
|---|---|---|---|---|---|---|
| BT.601 | 0.299 | 0.114 | 1.402 | 0.344136 | 0.714136 | 1.772 |
| BT.709 | 0.2126 | 0.0722 | 1.5748 | 0.187324 | 0.468124 | 1.8556 |
| BT.2020 (non-constant luminance) | 0.2627 | 0.0593 | 1.4746 | 0.164553 | 0.571353 | 1.8814 |

A stream names BT.709 or BT.2020. BT.601 is there for the caller to choose, and it matters because of what upstream's viewer does (decision D17).

4:2:0 chroma is upsampled bilinearly at the position the siting gives. Horizontally, centre siting puts a chroma sample between two luma columns, so luma column `p` sits at chroma coordinate `p/2 - 1/4`; left siting puts it on the even column, at `p/2`. Vertically chroma is always centred, as in upstream's viewer, which shifts only the horizontal coordinate. Every weight is then a multiple of 1/4:

| Luma position `p` | Centre: first sample, weight of the next | Left |
|---|---|---|
| even | `p/2 - 1`, 3/4 | `p/2`, 0 |
| odd | `p/2`, 1/4 | `p/2`, 2/4 |

(`p/2` is an integer division.) Indices outside the plane are clamped to the edge sample. The four taps are summed in integers with weights that add up to 16 and the sum is scaled by 1/16, so the interpolation is exact. Filtering a texture in hardware would round the weights to a fixed-point format of the GPU's choosing.

`color::yuv_to_rgb8` is the CPU implementation and evaluates the same expressions in the same order. The shader's output equals it to within one 8-bit step. A GPU compiler may fuse a multiply and an add, which moves a rounding in rare cases: at most 0.003 % of the samples differ on the test vectors.

What the stage leaves alone: the result is the encoded, non-linear R'G'B' of the stream. A stream that signals PQ or BT.2020 primaries is reported by `info`, but no tone mapping, transfer function or gamut conversion is applied; that belongs to the display pipeline the picture is handed to.

### Timing

`Decoder::time_frame` decodes a frame and waits for the GPU without reading the picture back, which is what a player does: the picture stays in GPU memory. It returns where the time went:

| Field | Meaning |
|---|---|
| `prepare` | CPU time to assemble the frame, upload the payload and the block table and, for RGB, write the conversion parameters |
| `gpu_wall` | Wall-clock time from creating the command encoder until the GPU is idle again: encoding, submission, execution and the wait. With a readback it includes that |
| `stages` | GPU time per pass, from timestamp queries: dequantisation, the inverse transform (all five levels together, including the 8-bit output) and, for RGB, the colour conversion |

Timestamps are written at the beginning and the end of each compute pass and resolved into a small buffer that is read after the frame. That needs only `Features::TIMESTAMP_QUERY`, which the browser offers too. Writes between dispatches (`TIMESTAMP_QUERY_INSIDE_PASSES`) are not available on Apple GPUs, so a pass is the finest granularity, which is why the five levels of the inverse transform are timed together. Where the feature is missing `stages` is `None` and the tools print the CPU and wall-clock numbers only.

The wall-clock time exceeds the sum of the stages, because submitting work and noticing that it has finished take time of their own. For the 192x136 test vector, decoded to RGB on an Apple M3 Max (Metal, wgpu 30.0.1, 2026-10-24), the stages add up to about 0.05 ms and `gpu_wall` is about 0.30 ms, with 0.03 ms of `prepare` on top (v0.2.0; v0.1.0 took 0.15 and 0.42 ms). Both are reported: the first says how long the GPU works, the second how long the application waits for its picture.

### Portability

The shaders use only what WebGPU guarantees: no subgroup operations, no 8-bit or 16-bit storage, no `f16`, no immediates (push constants), no storage textures. All buffers are plain storage or uniform buffers and every workgroup stays within 256 invocations and 16 KiB. Two tests hold this to account:

- `tests/shaders.rs` validates the WGSL with naga using an empty capability set, without a GPU, and checks the workgroup sizes and memory.
- `tests/gpu.rs` creates a device with `InstanceFlags::STRICT_WEBGPU_COMPLIANCE` and the default limits and decodes a vector on it.
- The rounding rule of the last level is checked on the real backend against the CPU's, on inputs that are exact ties: a unit test of `gpu::decoder` cuts `to_unorm8` out of `idwt.wgsl` and runs it.

What upstream uses and what replaces it:

| Upstream (Vulkan) | emberwave (WGSL) |
|---|---|
| Subgroup scans in the dequantiser | Workgroup memory sums, two barriers |
| `uint8_t` / `uint16_t` buffer access | `array<u32>` with shifts |
| `textureGather` with a mirrored sampler | Explicit mirroring, then a load from the band buffer |
| FP16 storage images, `PYROWAVE_PRECISION` modes | FP32 buffers |
| 8-bit UNORM storage images for the output | The last inverse transform level packs four samples per `array<u32>` word |
| Dozens of dispatches, one per band | One dequantisation dispatch and one per level |
| Hardware bilinear filtering of the chroma textures (viewer) | Integer taps and weights in the shader |

### Verification

Run on 2026-10-23 with the committed vectors on an Apple M3 Max (Metal):

| Check | Result |
|---|---|
| Dequantised coefficients, GPU against CPU, all vectors | Bit-identical |
| The same on synthetic frames with every quantiser code, sparse ballots, all-zero control words and missing blocks | Bit-identical |
| LL bands after each inverse level, GPU against CPU | Largest difference 6.6e-7 (the tolerance is 1e-5) |
| Decoded planes against the default-precision reference | Worst 64.83 dB (needs 60), largest difference 1 LSB |
| Decoded planes against the FP32 reference | Worst 86.28 dB (needs 80), most planes bit-identical |
| RGB output, GPU against the CPU conversion: all vectors, three matrices, both ranges, both sitings | Largest difference 1; at most 0.003 % of the samples differ |
| 4K stream decoded by `emberwave-cli`, planes against the default-precision reference | Worst 65.64 dB (needs 60), largest difference 1 LSB |
| The same stream against the FP32 reference | Worst 96.56 dB (needs 80), largest difference 1 LSB |

The GPU numbers equal the CPU decoder's, which is the point: the GPU pipeline adds no error of its own beyond float rounding.

The last two rows come from `scripts/verify_4k.sh`: it encodes two 3840x2160 frames of a synthetic scene (1.2 MB each) with the upstream encoder, decodes them with the upstream decoder at both precisions and with `emberwave-cli` to PNG and Y4M, and fails unless the bars of D9 are met.

## Command line tool

`emberwave-cli` (`crates/emberwave-cli`) is a thin front end: the decoder, the colour conversion and the timing API are the library's, and the tool adds file formats and reporting.

| Command | What it does |
|---|---|
| `decode FILE --out-dir DIR [--format png\|y4m] [--frames N]` | Decodes the frames of a PWV1 file and writes `STEM_000000.png`, ... or one `STEM.y4m` with the unconverted planes. Prints the time of every frame and a summary |
| `bench FILE [--output yuv\|rgb] [--readback] [--frames N] [--warmup N] [--warmup-seconds S] [--json]` | Decodes in a loop and prints mean, median, 99th percentile, minimum and maximum of the numbers described under [Timing](#timing); `--json` prints them as JSON. The frames of the file are used in turn when more are asked for than it holds. Warm-up lasts for at least N frames and at least S seconds (10 and 1 by default, see D19). `--readback` copies the picture back and reports the total only |
| `info FILE [--verbose]` | Prints the PWV1 header, the colour signalling and the frame sizes, without a GPU |
| `compare A.y4m B.y4m [--min-psnr DB]` | Prints the PSNR and the largest difference per plane of every frame; with `--min-psnr` the exit status tells whether every plane reached it |

`decode` and `bench` choose the GPU with `--backend auto|vulkan|metal|dx12` (`auto` honours `EMBERWAVE_BACKEND`, otherwise any backend the machine has) and `--software`, which insists on a software adapter such as lavapipe or WARP. The adapter in use is printed to stderr. `--matrix auto|bt601|bt709|bt2020` and `--range auto|full|limited` set the colour conversion; `auto` takes the PWV1 header's signalling. They belong to the RGB conversion, so `decode --format y4m`, which keeps the planes as decoded, refuses them. Options that cannot work (`--frames 0`) are refused before anything is read or written. An error is printed as `error: ...` and ends the program with status 1.

PNG output is 8-bit RGB with fast compression. The Y4M writer takes range, chroma format and chroma siting from the stream: `XCOLORRANGE`, and `C420jpeg` (centred chroma), `C420mpeg2` (chroma on the left luma column) or `C444`. A stream without a frame rate gets 30:1, which Y4M needs, and a warning says so. The Y4M reader of `compare` takes 8-bit 4:2:0 and 4:4:4 only, and a file whose header names a picture size that overflows, or that has no frames, is an error rather than a pass. Percentiles are nearest-rank.

## Browser player

`crates/emberwave-web` builds the decoder for the browser (WebAssembly and WebGPU), and `web/` is a page that plays `.pwv1` files with it: a file opened or dropped on the page, or one of the test vectors. `scripts/build_web.sh` builds the module into `web/pkg/` and copies the vectors to `web/samples/`, and `.github/workflows/pages.yml` publishes `web/` to GitHub Pages on every push to `main`.

Nothing in a browser may block, so the player uses only the parts of the decoder that do not wait:

- `GpuContext::from_adapter` creates the device on an adapter that can draw on the canvas. `GpuContext::new`, which blocks, does not exist in the browser build.
- `Decoder::encode_rgb` records the decode and the colour conversion into the player's command encoder and returns. A render pass in the same encoder copies the RGBA words to the canvas (`blit.wgsl`: one triangle over the canvas, which has the size of the picture, so a fragment reads one word). One submission per frame, and the picture never leaves the GPU.
- Error scopes cannot be waited for, so in the browser build `with_error_scope` just runs its work. The player catches what the browser reports instead (`on_uncaptured_error`) and shows the first error on the page with the next frame. Pictures wider or taller than the browser can draw (`max_texture_dimension_2d`, 8192 by default) are refused when the file is loaded.
- The methods that read a frame back or time it wait for the GPU, and are not for the browser.
- In a browser, dropping a wgpu buffer only lets go of the JavaScript object, and the GPU memory waits for the garbage collector. `Decoder` therefore destroys its large buffers when it is dropped, and the player drops the old stream before it loads the next file: about 140 MiB for a 4K stream.

The page paces frames by the stream's frame rate, keeping the remainder of each interval, so 25 frames per second play as 25 on a 60 Hz display. It shows the frames per second, the CPU time to record and submit a frame, and the time from submitting a frame until the browser reports the GPU done with it (`onSubmittedWorkDone`): decoding and drawing, plus the delay of the report, averaged over the frames reported each second. In headless Chrome 155 on an Apple M3 Max (2026-10-24) the 1080p benchmark stream took about 0.6 ms of CPU and 2.2 ms from submission to done, the 4K one about 1.2 and 3.8 ms, both playing at their 30 frames per second.

## C API

`crates/emberwave-ffi` is the decoder for C programs. It builds a shared library (and, for releases, a static one) with the functions of `include/emberwave.h`, which cbindgen generates from `src/lib.rs`; `tests/header.rs` fails when the committed header falls behind the source. [`crates/emberwave-ffi/README.md`](crates/emberwave-ffi/README.md) is the guide for C programmers. The objects are those of the Rust API:

| C | Rust | Role |
|---|---|---|
| `EmberwaveContext` | `GpuContext` | A GPU device and the compiled shaders, shared by its decoders |
| `EmberwaveDecoder` | `Decoder` | Assembles the packets of one stream and decodes its frames |

A decoder holds the device it was made on, so it may outlive its context. The functions:

| Function | What it does |
|---|---|
| `emberwave_probe_pwv1`, `emberwave_probe_packet` | Picture size, chroma format and colour signalling from a PWV1 file header or from a packet that starts with a start of frame header |
| `emberwave_context_create`, `_destroy`, `_adapter` | Choose a GPU (backends, software only) and name it |
| `emberwave_decoder_create`, `_destroy`, `_info` | One decoder per picture size and chroma format; the sizes of its planes |
| `emberwave_decoder_push_packet`, `_is_ready`, `_clear` | Feed packets as they arrive; ask whether the frame, or enough of it, is there |
| `emberwave_decoder_decode_yuv`, `_decode_rgb` | Decode and copy the planes, or RGB, into the caller's memory, row by row with the caller's strides |
| `emberwave_version`, `emberwave_last_error` | The version, and the message of the last failure on the calling thread |

Every function returns a status, or a value that cannot fail. A Rust panic is caught at the boundary, because unwinding into C is not allowed, and becomes `EMBERWAVE_STATUS_INTERNAL`, or `EMBERWAVE_STATUS_DEVICE_LOST` when the device is lost: wgpu panics when the device fails while it is waited for (D22), after recording the loss. Arguments are checked before the frame is used: a `NULL` pointer, a stride shorter than a row, a picture size the format cannot have or an unknown constant is `EMBERWAVE_STATUS_INVALID_ARGUMENT`, and the frame can still be decoded with correct ones. What the caller passes in is a plain integer, never a Rust enum, so no value from C can be an invalid one.

`unsafe` code lives in this crate only. It has its own lint table instead of the workspace's `forbid`: every unsafe operation sits in an `unsafe` block of its own with a comment saying why it is sound, and every public function documents what the caller must guarantee.

The C API copies the picture to the CPU, which costs more than decoding it: a 4K frame to RGB takes 7.5 ms with the copy and 1.4 ms without (`BENCHMARKS.md`). Handing the picture to the caller's graphics API on the GPU would need the Vulkan, Metal or Direct3D 12 object behind a wgpu buffer and a way for each API to import it; that is left for later.

Frames from upstream's Metal encoder signal BT.709, full range and centred chroma in their start of frame headers whatever the pictures are, because its API has no way to set the signalling; `testdata/README.md` has the details. `emberwave_decoder_decode_rgb` with a `NULL` colour follows the frame, so a caller with better knowledge, such as the colours of a PWV1 file header, passes them.

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

**Decision.** (Superseded by D24: the last inverse transform level packs, there is no separate pass.) The planes are packed to `u32` words by a separate pass because `r8unorm` storage textures are outside the portable set. The same buffer serves the readback now and will feed the colour conversion in M3.

### 2026-10-23 — D15: What is compared exactly

**Decision.** Dequantised coefficients are compared bit for bit between GPU and CPU: they come from integer decoding and one multiplication, so they have no excuse to differ. The inverse transform is compared with a tolerance (1e-5; 8.0e-7 observed), because a GPU compiler may fuse a multiply and an add, which a CPU build does not, and WGSL cannot switch that off. Whole frames are compared with the PSNR bars of D9.

### 2026-10-23 — D16: wgpu 30, the `gpu` feature and its dependencies

**Decision.** `wgpu` 30.0.1 with the `vulkan`, `metal`, `dx12`, `wgsl`, `std` and `parking_lot` features, without GLES (which would need EGL) and without the browser backend (the wasm build gets its own crate later). The `rust-version` is raised to 1.87, wgpu's. The GPU code is behind the cargo feature `gpu`, on by default, so the parser and the CPU decoder can be used without wgpu.

New crates: `wgpu`; `pollster` to block on wgpu's futures in the synchronous API; `bytemuck` (with `derive`) to hand `u32` and parameter structs to buffers without `unsafe`, which the workspace forbids; `naga` as a dev-dependency for the offline shader validation, at the version wgpu uses.

The GPU tests skip when there is no adapter, unless `EMBERWAVE_REQUIRE_GPU` is set; CI sets it on Linux, where it installs lavapipe, a software Vulkan driver.

### 2026-10-23 — D17: Colour conversion uses the standard matrices, not upstream's viewer's

**Context.** The stream signals `ycbcr_transform` (BT.709 or BT.2020 non-constant luminance), `ycbcr_range` and `chroma_siting`, and `bitstream.md` leaves their definitions "to the respective specifications". Upstream's only conversion is the viewer's `shaders/yuv2rgb.frag`. Its matrix `yuv2rgb_bt709` is BT.709 (1.5748, 1.8556), but `yuv2rgb_bt2020` holds the BT.601 coefficients: 1.402 and 1.772, and green factors written as `0.202008 / 0.587` and `0.419198 / 0.587`, where 0.587 is BT.601's `Kg`. The factors of BT.2020 are 1.4746 and 1.8814. Nothing else in upstream converts to or from BT.2020 (checked by searching the pinned revision).

**Decision.** `ColorMatrix::Bt2020` is BT.2020 as the standard defines it, derived from `Kr = 0.2627` and `Kb = 0.0593`, so a stream that signals BT.2020 is converted the way the signalling and `bitstream.md` say, not the way the viewer does it. BT.601 is available as a choice (`--matrix bt601`), and it is exactly what the viewer's "BT.2020" branch computes, so upstream's picture can be reproduced on request. The conversion works on the 8-bit planes with integer chroma weights, and applies no tone mapping or transfer function.

**Consequence.** On BT.2020 streams the RGB differs from upstream's viewer; the viewer's matrix is mislabelled, which is a matter for upstream. Streams that signal BT.709 agree up to the viewer's texture filtering. The matrices are unit-tested against the published coefficients.

### 2026-10-23 — D18: The command line tool and its dependencies

**Decision.** One binary with the subcommands `decode`, `bench`, `info` and `compare` (see [Command line tool](#command-line-tool)). `decode` has no default output location: `--out-dir` is required, so the tool never chooses where to write. `compare` is part of the tool, so the acceptance check needs no external program and works on every OS.

New crates: `clap` with `derive`, for argument parsing, help texts and errors that stay in step with the code; `png`, for PNG output, with its default features; `serde_json` as a dev-dependency only, so that the end-to-end tests can check that `bench --json` is valid JSON. The JSON is written by hand because its shape is small and flat, which keeps a serialisation framework out of the binary. The Y4M reader and writer and the statistics are small enough to live in the tool.

### 2026-10-23 — D19: What the timings mean

**Context.** Two things make a single number misleading. A GPU finishes a small frame long before the application finds out: for the 192x136 vector the pass timestamps add up to 0.15 ms and the wall clock shows 0.42 ms. And the first moments of a run are slow: with a warm-up of ten frames, the same 200x120 vector measured 0.52 ms of GPU time over 30 counted frames, 0.21 ms over 300 and 0.15 ms over 1500. That looks like the GPU raising its clock under load; the cause was not isolated.

**Decision.** `bench` reports the CPU time to prepare, the wall-clock time until a finished frame is in GPU memory (`gpu wall`), their sum (`total`) and, where the device can, per-pass GPU times from timestamps. `total` is the number to quote for a frame, since it is what an application waits for; the pass times show where it goes. Reading the picture back is optional (`--readback`) because a player keeps it on the GPU. Mean, median, 99th percentile, minimum and maximum are all printed: for a stream a late frame matters more than the average.

Warm-up is both a number of frames and a time: at least 10 frames and at least one second (`--warmup`, `--warmup-seconds`) are decoded and not counted. With that, the 192x136 vector gave 0.148 to 0.149 ms of GPU time in four runs of 300 or 30 counted frames, and `gpu wall` agreed within 5 %.

**Consequence.** Numbers are only quoted together with the environment (the adapter, the backend, the date and the wgpu version), as `CLAUDE.md` demands. A slow software renderer spends its ten frames on warm-up before it counts anything, so `--warmup` can be lowered there.

### 2026-10-23 — D20: The 4K acceptance check generates its stream

**Context.** The definition of done asks for a 4K stream decoded by the CLI and matched to the reference at the PSNR criterion. A 4K frame needs a source picture, and a committed 4K stream is megabytes.

**Decision.** `scripts/verify_4k.sh` creates the stream when it runs. The input picture is a synthetic scene (colour bars, a gradient, a moving shaded disc, a fine diagonal grid and a little film grain, defined in `scripts/refgen/refgen.mm`), encoded by upstream's own encoder, so the bitstream is still the original encoder's output and only the picture is ours. The same scene feeds the benchmark streams in M4. Generated files go to `testdata/generated/`, which git ignores. Like the vectors, the script needs Apple Silicon macOS, because the reference decoder is upstream's Metal one (D1).

### 2026-10-23 — D21: How benchmarks are run and published

**Context.** `CLAUDE.md` asks for the mean and 99th percentile decode time of 60-frame sequences at 1080p, 1440p and 4K on Metal, written into `BENCHMARKS.md` by a script and always with the environment, and for lavapipe numbers if possible. A number alone does not say whether it is good.

**Decision.**
- `scripts/bench.sh` measures and then rewrites the generated blocks of `BENCHMARKS.md` and of the README (between `bench` markers) with `scripts/bench_report.py`, which needs only Python's standard library. The same run records the environment: computer, OS, power source, wgpu and Rust versions, commit and date. The raw JSON stays in `target/bench/`.
- Each number is over 600 frames, ten passes over the 60-frame sequence, so that the 99th percentile is not just the slowest frame.
- Upstream's own decoder is the yardstick: `refgen bench` times its Metal port on the same streams in the same way, at both of its precisions. The first run puts emberwave's GPU time at about seven times upstream's FP32 at 4K, which issue #11 takes up.
- lavapipe is measured in a Linux container on the same computer (`scripts/bench_lavapipe.sh`, which needs Docker) and labelled as a reference.
- Text that interprets the numbers is written by hand outside the generated blocks and names the run it is about.

**Consequence.** Running the script again updates every published number at once.

### 2026-10-23 — D22: Errors instead of panics, and the panics that stay

**Context.** M5 asks for clean error types and no panics.

**Decision.**
- Everything that input or the environment can cause is an error value, made with `thiserror`: `bitstream::Error` for streams, `GpuError` for adapters, devices and limits, and `ColorError` for planes smaller than the picture size says. `color::yuv_to_rgb8` used to panic in that case and now returns a `Result`. The CLI prints errors and exits with status 1.
- All GPU work runs inside wgpu error scopes for validation, out-of-memory and internal errors, the last being what a driver's failure to compile a shader or build a pipeline becomes. That now includes the uploads, the parameter writes and the mapping and unmapping of readback buffers of each frame, and `Decoder::read_band`. Such an error becomes `GpuError::Device` or `GpuError::Readback` instead of reaching wgpu's default handler, which panics. An out-of-memory error can cost the device itself.
- A lost device (a driver reset, a GPU that went away, a destroyed device) reports through a callback, not through the scopes. The context records it, and a decoder checks the record before and after each frame: a frame on a lost device is `GpuError::DeviceLost`, never a stale result, and so is every later frame. Before this a timed decode on a destroyed device returned plausible numbers. The context and its decoders have to be created again.
- One path is out of reach: wgpu 30 treats a device error while waiting in `Device::poll` as fatal and panics, whatever the error scopes. A timeout comes back as an error.
- Three `expect`s stay, in `gpu::layout`. They assert that the band tables of a `FrameLayout` and of the `BufferLayout` built from it agree, which no input can change. A property test builds the parameter blocks for random picture sizes across the whole range the format allows, in both chroma formats, to back that up. If one of them fired it would be a bug, which is what a panic is for.
- Malformed streams are covered by the property tests of M1, which feed random and damaged packets to the parser.

**Consequence.** `ColorError`, `GpuError::DeviceLost` and `GpuContext::lost` are new public API.

### 2026-10-23 — D23: Releases

**Decision.** A tag `vX.Y.Z` runs `.github/workflows/release.yml`. It first checks that the tag names the version in `Cargo.toml` and a commit on `main`. Then it builds `emberwave-cli` on GitHub's runners for x86_64 Linux, Apple Silicon macOS and x86_64 Windows, packs each binary with the README, the licence and the changelog (`.tar.gz`, `.zip` on Windows), and publishes a GitHub release with SHA-256 checksums, whose notes are the version's section of `CHANGELOG.md`. The release workflow uses only first-party actions and the `gh` CLI. The binaries are not signed or notarised; the README says how to run the macOS one.

**Consequence.** Intel Macs and ARM Linux build from source for now.

### 2026-10-23 — D24: Decode speed: registers, 8-bit output in the transform, a cheaper block check

**Context.** v0.1.0 decoded a 4K frame in 4.28 ms, 3.58 ms of it on the GPU, on an Apple M3 Max, where upstream's Metal decoder takes 0.78 ms (0.49 ms on the GPU) at FP32 (issue #11). The inverse transform was 82 % of the GPU time.

**Decision.**
- Arrays local to an invocation are indexed with constants only. The inverse transform loaded 16 samples into a local array, lifted them and stored 8, all with loops over a variable index, and passed the array to `lift()` by pointer; the compiler kept the array in memory rather than in registers. Written out sample by sample, the transform takes 0.65 ms instead of 2.93 ms. The same change to the eight values of a subblock in the dequantiser takes it from 0.45 to 0.36 ms (0.72 to 0.51 ms on the noise stream).
- The last level of the inverse transform writes the 8-bit planes itself (see [8-bit output](#8-bit-output)), which replaces the float planes and the packing pass of D14: a 0.19 ms pass and 47.5 MiB of GPU memory at 4K go away. Its barrier sits outside the branch for the last level, because a barrier inside it crashed WARP.
- The dequantiser finds the band of its block by a binary search of the band table instead of scanning all of it: every invocation read up to 48 entries from the uniform block. Dequantisation then takes 0.17 ms instead of 0.34 ms.
- The frame assembler validates incoming blocks with `BlockStructure::check_bytes`, the checks of building the full structure without building it: 0.057 ms instead of 0.161 ms for a 4K frame, which is what upstream's `push_packet` takes.
- Two other ideas were measured and dropped. Padding the workgroup arrays against bank conflicts gained nothing, and on the faster transform cost 7 %. Writing dequantised blocks through workgroup memory so that their stores are coalesced cost 13 %: the caches of Apple GPUs absorb the scattered stores.

**Consequence.** A 4K frame decodes to planes in 1.38 ms, 0.80 ms of it on the GPU: 1.8 and 1.7 times upstream's FP32 times on the same machine. With the conversion to RGB it takes 1.63 ms instead of 4.53 ms; `BENCHMARKS.md` has the numbers. `StageTimings` lost its `pack` field, so the next release is 0.2.0. Shaders in this project index function-scope arrays with constants only (`CONTRIBUTING.md`).

### 2026-10-24 — D25: Workgroup memory is not zeroed

**Context.** WebGPU requires workgroup memory to start out as zeros, and wgpu zeroes it unless a pipeline is created with `zero_initialize_workgroup_memory: false`. For Metal and DX12, naga puts the zeroing at the top of the shader: invocation 0 of each workgroup stores every element, one after the other, while the others wait at a barrier. In the inverse transform that is 2,880 floats before each 32x32 tile. On Vulkan wgpu leaves the zeroing to the driver when the device offers it (Vulkan 1.3 or `VK_KHR_zero_initialize_workgroup_memory`). On Windows CI, where DX12 runs on WARP, creating the pipelines took 6 s and the first dispatch on each device 85 to 100 s, so the tests took 11 minutes (issue #12).

**Decision.** The three pipelines are created with `zero_initialize_workgroup_memory: false`, on every backend and including the Vulkan driver's zeroing, so that a mistake shows the same way everywhere. The shaders write every workgroup variable before they read it. The comments next to the variables say how, `CONTRIBUTING.md` makes it a rule, and `tests/shaders.rs` fails when a workgroup variable is added, removed or resized, so that the writes and reads get checked again. The GPU tests run the same pipelines as the library.

Two alternatives were rejected. Zeroing in WGSL with all invocations would bring back part of the cost and still hide a missed write, since zero is the right value for missing coefficients and for padding. Zeroing only in debug builds would test something other than what ships. The browser build cannot opt out, as browsers always zero workgroup memory, so there a mistake would read zeros where the native build reads stale values.

**Consequence.** On an Apple M3 Max, measured alternately with and without zeroing (three rounds of 600 frames of the 4K stream), the inverse transform takes 0.39 ms instead of 0.65 ms and the GPU passes 0.55 ms instead of 0.82 ms. A frame decoded to planes takes 1.12 ms instead of 1.40 ms, and 1.39 ms instead of 1.63 ms with the conversion to RGB. On WARP the pipelines are created in 0.4 s, the first dispatch is no slower than the next, and the Windows CI job takes 1.5 minutes instead of 14. lavapipe, which zeroes in the driver, stays within the noise of its ten-frame runs: 44 ms for a 4K frame, 46 ms before. A shader that reads workgroup memory before writing it gets what an earlier workgroup left there. The GPU tests compare every level with the CPU exactly and catch that, unless the stale values happen to be the right ones.

### 2026-10-24 — D26: The browser player

**Context.** `CLAUDE.md` lists a WebGPU demo among the things to do after the release: a wasm build and a static page on GitHub Pages that plays a dropped file (issue #15). The decoder's API waited for the GPU everywhere: creating a context, popping error scopes, reading a picture back, timing it. A browser does not allow any of that.

**Decision.**
- A crate `emberwave-web`, made with wasm-bindgen. It has content only for `wasm32` targets, so the workspace builds everywhere and native builds pull in no browser crates. New crates for it: `wasm-bindgen`, `wasm-bindgen-futures` and `web-sys`, at the versions wgpu 30 itself requires.
- The decoder gains a path that does not wait. `Decoder::encode_rgb` records into the caller's encoder, as upstream's `decode_gpu_buffer` does, `Decoder::rgba_buffer` gives the result, and `GpuContext::from_adapter` lets the caller pick an adapter that can draw on its surface. Native players can use the same path.
- The picture reaches the canvas through a render pass that reads the RGBA storage buffer, with the canvas at the size of the picture: no texture copy, no readback.
- The page is plain HTML and JavaScript, without a bundler or framework.
- CI checks the browser build with `cargo clippy --target wasm32-unknown-unknown`. The page itself was checked by script in headless Chrome with WebGPU (a sample and the 1080p and 4K benchmark streams played, and the screenshots showed the right pictures); that check needs a GPU and is not part of CI.

**Consequence.** Any browser with WebGPU can play a stream without uploading it anywhere. `GpuContext::new` is not part of the browser build.

### 2026-10-24 — D27: The C API

**Context.** `CLAUDE.md` asks for a C FFI, `emberwave-ffi` with a header made by cbindgen, for Moonlight-style clients (issue #16): creating and destroying a decoder, pushing packets, asking whether a frame is ready, and getting the decoded frame as YUV planes or RGB in the caller's buffers, with documented error codes and thread rules, `unsafe` only in that crate (D5), and a C example that decodes the test vectors in CI on all three systems.

**Decision.**
- The objects follow the Rust API and upstream's own C API for Metal: a context per GPU, which compiles the shaders once, and decoders on it, one per stream. See [C API](#c-api).
- Values that come in from C are integers with `#define`d names (chroma format, backends, colours, flags), and the status is the only enum, which only goes out. A C caller cannot hand Rust an invalid enum value.
- Errors are a status plus a message for the calling thread (`emberwave_last_error`), the way `errno` and `dlerror` work, so that a failed `*_create`, which has no object yet, can still say why.
- Pictures are copied into the caller's memory with the caller's strides; nothing the caller has to free comes out of the library.
- The header is generated by cbindgen inside a test and committed, so that C programmers read it without Rust tools and CI notices when it is stale. cbindgen is a dev-dependency only (it brings `toml` and a few crates of its own); `EMBERWAVE_BLESS=1` rewrites the header.
- The shared library carries a name that lets it move with the program: `@rpath/libemberwave_ffi.dylib` on macOS, the SONAME `libemberwave_ffi.so` on Linux (`build.rs`).
- Everyday builds make the shared library only; releases also build the static one with `cargo rustc --crate-type staticlib` and ship the list of system libraries it needs.
- CI builds `examples/c` with CMake against the shared library on Linux (lavapipe), macOS (Metal) and Windows (WARP), and decodes every test vector with it to at least 80 dB against upstream's FP32 decode (`scripts/check_c_example.sh`). `tests/api.rs` calls the functions through raw pointers, as C does, and compares with the Rust decoder.
- The release workflow can be started by hand to rehearse: it then builds and packages everything, the C library included, and publishes nothing.

**Consequence.** C and C++ programs can decode PyroWave on all three systems with a prebuilt library. The picture goes through the CPU; sharing it with the caller's graphics API on the GPU is a separate piece of work.
