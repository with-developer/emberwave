# Architecture

> **Status (v0.11.0).** The bitstream format is documented below and implemented by `emberwave_core::bitstream`, with a slow CPU reference decoder in `emberwave_core::cpu`, the wgpu/WGSL decoder in `emberwave_core::gpu` and the colour conversion in `emberwave_core::color`. The encoder runs on the CPU (`emberwave_core::encode`) and on the GPU (`emberwave_core::gpu`). `emberwave-cli` decodes and encodes files with them, `emberwave-web` plays them in a browser and `emberwave-ffi` offers the decoder and the GPU encoder to C, also as an XCFramework for iPhone, iPad and Mac apps, and hands Metal clients the decoded picture in a texture, or its planes in three, also into the client's own textures and on its own queue without waiting. Decode and encode times are in `BENCHMARKS.md`. The decision log at the end is kept up to date.

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
- A frame is ready when `received >= total_blocks`. A frame that lost packets may be decoded anyway if the coarsest bands are complete and more than 90 percent of the blocks arrived; the thresholds are parameters. As upstream's, the test of the coarsest bands counts a block that the encoder never sent, because it was all zeros, as missing: in the committed gradient stream 4 or 5 of the 9 coarsest detail blocks are never sent, and a frame of it that lost any other block is not ready for a partial decode either. A complete frame is not affected.

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
| A block of an older frame that is malformed | Ignored with the rest of the packet, as staleness is tested first | The same: staleness first, then the block's size, index and internals |

### Verification

Measured on 2026-10-23 with the committed vectors (five streams, 3 frames each except one with 2: solid, gradient, noise at two rates, and 4:4:4 noise with odd dimensions), decoding every frame with `emberwave_core::cpu` and comparing to the original decoder per plane:

| Reference | Worst PSNR | Largest difference | Notes |
|---|---|---|---|
| Default precision (FP16 wavelet storage) | 64.83 dB | 1 LSB | Above the 60 dB bar |
| FP32 precision (`PYROWAVE_PRECISION=2`) | 86.28 dB | 1 LSB | Most planes are bit-identical; the rest differ in a single pixel |

The gap to the default precision is the reference decoder's own FP16 storage. In addition, for every block of every vector, `payload_words` equals the formula above exactly, which pins down the layout of planes and signs.

## GPU pipeline

`gpu::Decoder` decodes frames of one size and chroma format. All GPU memory is allocated when it is created and reused for every frame. The payload buffer is the one exception: it grows when a frame needs more room. The encoder's GPU pipeline, `gpu::Encoder`, is described under [Encoder](#on-the-gpu).

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

WGSL's `round` rounds ties to even on every backend (naga emits `rint`, `round` and `RoundEven` for Metal, HLSL and SPIR-V), which is what a UNORM storage write does. Four pixels share a word, first pixel in the low byte, and rows are padded to whole words with zero samples. A decoder's rows are then padded with words it never writes to a multiple of 256 bytes, the rows a copy into a texture takes, and its planes start at multiples of 512 bytes (D41). The CPU side cuts the padding off.

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

`Encoder::encode_timed` reports the same three numbers for an encode, whose frame always comes back: `prepare` is the CPU time to check the planes, copy them where wgpu uploads them from and write the frame's parameters; `gpu_wall` runs from the submission until the frame is in CPU memory, the upload of the planes and the read-back included; `stages` are the forward transform, the quantisation, the rate control (its four passes and two prefix sums together) and the packing. Timestamps go to a `StageQueries` (`gpu/resources.rs`), which the decoder uses too. It resolves them in a submission of its own once the work is done: resolved in the work's command buffer, Metal gave the last stages the previous frame's times or an end before the beginning in a few frames of every hundred.

The wall-clock time exceeds the sum of the stages, because submitting work and noticing that it has finished take time of their own. For the 192x136 test vector, decoded to RGB on an Apple M3 Max (Metal, wgpu 30.0.1, 2026-10-24), the stages add up to about 0.05 ms and `gpu_wall` is about 0.30 ms, with 0.03 ms of `prepare` on top (v0.2.0; v0.1.0 took 0.15 and 0.42 ms). Both are reported: the first says how long the GPU works, the second how long the application waits for its picture.

### Portability

The shaders, the decoder's and the encoder's, use only what WebGPU guarantees: no subgroup operations, no 8-bit or 16-bit storage, no `f16`, no immediates (push constants), no storage textures. All buffers are plain storage or uniform buffers and every workgroup stays within 256 invocations and 16 KiB. The encoder reads pictures in textures as sampled `r8unorm` and `rg8unorm` textures with `textureGather` and a non-filtering sampler, which WebGPU has without any feature. The encoder's rate control adds with 32-bit integer atomics, which WGSL has without any feature. Two tests hold this to account:

- `tests/shaders.rs` validates the WGSL of every shader with naga using an empty capability set, without a GPU, and checks the entry points, the workgroup sizes and memory.
- `tests/gpu.rs` and `tests/gpu_encoder.rs` create a device with `InstanceFlags::STRICT_WEBGPU_COMPLIANCE` and the default limits and decode a vector on it, and encode pictures on it.
- The rounding rule of the last level is checked on the real backend against the CPU's, on inputs that are exact ties: a unit test of `gpu::decoder` cuts `to_unorm8` out of `idwt.wgsl` and runs it.

Outside those tests a context asks for the default limits with two exceptions. Buffers and textures may be as large as the adapter allows. No limit goes beyond the adapter's, since some adapters fall short of a default the shaders do not need. Metal gives the iOS Simulator, and Apple GPUs before the A11, 15 inter-stage variables, one fewer than WebGPU's default, and the shaders use none (D36).

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
| Subgroup operations in the encoder's quantisation and rate control | Workgroup memory, a histogram with 32-bit atomics and prefix sums in workgroup memory |
| 8-bit stores of the encoder's block bytes | Bytes ORed into words of workgroup memory with atomics, then copied out |
| The encoder's IOSurface input: three R8 surfaces, or NV12 with its chroma plane bound twice under two swizzles | Texture views, R8 or R8 and Rg8, packed into the planes by a dispatch of their own ahead of the transform's, 2x2 texels per `textureGather`; a uniform picks the channel, as WGSL has no swizzled views |

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

`emberwave-cli` (`crates/emberwave-cli`) is a thin front end: the decoder, the encoder, the colour conversion and the timing API are the library's, and the tool adds file formats and reporting.

| Command | What it does |
|---|---|
| `decode FILE --out-dir DIR [--format png\|y4m] [--frames N]` | Decodes the frames of a PWV1 file and writes `STEM_000000.png`, ... or one `STEM.y4m` with the unconverted planes. Prints the time of every frame and a summary |
| `bench FILE [--output yuv\|rgb] [--readback] [--frames N] [--warmup N] [--warmup-seconds S] [--json]` | Decodes in a loop and prints mean, median, 99th percentile, minimum and maximum of the numbers described under [Timing](#timing); `--json` prints them as JSON. The frames of the file are used in turn when more are asked for than it holds. Warm-up lasts for at least N frames and at least S seconds (10 and 1 by default, see D19). `--readback` copies the picture back and reports the total only |
| `info FILE [--verbose]` | Prints the PWV1 header, the colour signalling and the frame sizes, without a GPU |
| `compare A.y4m B.y4m [--min-psnr DB]` | Prints the PSNR and the largest difference per plane of every frame; with `--min-psnr` the exit status tells whether every plane reached it |
| `bench-encode FILE.y4m --bytes-per-frame B [--gpu [--textures]] [--frames N] [--warmup N] [--warmup-seconds S] [--json]` | Reads the frames of a Y4M video into memory, encodes them in turn and prints the statistics of the time of the encoder's call, as `bench` does for decoding; with `--gpu` also the numbers of `Encoder::encode_timed`. With `--textures` the GPU encodes the frames from textures, an R8 texture per plane of each frame made before the measurement, through `Encoder::encode_textures_timed`. The frames are one pass over the video unless `--frames` says otherwise |
| `encode FILE.y4m --out FILE.pwv1 --bytes-per-frame B [--frames N] [--matrix bt709\|bt2020] [--gpu]` | Encodes an 8-bit 4:2:0 or 4:4:4 Y4M video with the CPU encoder, or with `--gpu` the GPU encoder, each frame within B bytes, and prints the size and time of every frame and a summary. The time of a frame is that of the encoder's call: from the picture in memory to the frame in memory, with the transfers to and from a GPU. The summary counts every frame, the first too, which on a GPU takes several times as long as the rest; `bench`'s warm-up is not there. Range, chroma siting and frame rate come from the Y4M header; `--matrix` names the colours to signal. The file appears only when it is complete, and the video is read a frame at a time |

`decode`, `bench`, `encode --gpu` and `bench-encode --gpu` choose the GPU with `--backend auto|vulkan|metal|dx12` (`auto` honours `EMBERWAVE_BACKEND`, otherwise any backend the machine has) and `--software`, which insists on a software adapter such as lavapipe or WARP. The adapter in use is printed to stderr. `--matrix auto|bt601|bt709|bt2020` and `--range auto|full|limited` set the colour conversion; `auto` takes the PWV1 header's signalling. They belong to the RGB conversion, so `decode --format y4m`, which keeps the planes as decoded, refuses them. Options that cannot work (`--frames 0`, a budget below the 8 bytes of a frame header, a 4:2:0 picture of odd size, `--backend` or `--software` for an encoder without `--gpu`) are refused before anything is read or written. An error is printed as `error: ...` and ends the program with status 1.

PNG output is 8-bit RGB with fast compression. The Y4M writer takes range, chroma format and chroma siting from the stream: `XCOLORRANGE`, and `C420jpeg` (centred chroma), `C420mpeg2` (chroma on the left luma column) or `C444`. A stream without a frame rate gets 30:1, which Y4M needs, and a warning says so. The Y4M reader of `compare` takes 8-bit 4:2:0 and 4:4:4 only, and a file whose header names a picture size that overflows, or that has no frames, is an error rather than a pass. Percentiles are nearest-rank.

## Browser player

`crates/emberwave-web` builds the decoder for the browser (WebAssembly and WebGPU), and `web/` is a page that plays `.pwv1` files with it: a file opened or dropped on the page, or one of the test vectors. `scripts/build_web.sh` builds the module into `web/pkg/` and copies the vectors to `web/samples/`, and `.github/workflows/pages.yml` publishes `web/` to GitHub Pages on every push to `main`.

Nothing in a browser may block, so the player uses only the parts of the decoder that do not wait:

- `GpuContext::from_adapter` creates the device on an adapter that can draw on the canvas. `GpuContext::new`, which blocks, does not exist in the browser build.
- `Decoder::encode_rgb` records the decode and the colour conversion into the player's command encoder and returns. A render pass in the same encoder copies the RGBA words to the canvas (`blit.wgsl`: one triangle over the canvas, which has the size of the picture, so a fragment reads one word). One submission per frame, and the picture never leaves the GPU.
- Error scopes cannot be waited for, so in the browser build `with_error_scope` just runs its work. The player catches what the browser reports instead (`on_uncaptured_error`) and shows the first error on the page with the next frame. Pictures wider or taller than the browser can draw (`max_texture_dimension_2d`, 8192 by default) are refused when the file is loaded.
- The methods that read a frame back or time it wait for the GPU, and are not for the browser. The decoder makes its read-back buffers only when a decode first reads back, so the player has none: 43.5 MiB fewer at 4K.
- In a browser, dropping a wgpu buffer only lets go of the JavaScript object, and the GPU memory waits for the garbage collector. `Decoder` therefore destroys its large buffers when it is dropped, and the player drops the old stream before it loads the next file: about 110 MiB for a 4K stream.
- The player keeps one copy of the file in the module's memory and walks its frames in place. When it loads a file it reads the frames from the first up to one that is cut off, or shorter than the start of frame header every frame begins with; it plays those, and says why it does not play the rest. It loads files of up to 1 GiB and pictures of up to 8192x4320 pixels' worth (D39).
- A lost device ends the stream, and the next file or sample gets a new player with a new device. A panic in the module leaves its objects in use, so the page then asks to be reloaded and disables its controls.

The page paces frames by the stream's frame rate, keeping the remainder of each interval, so 25 frames per second play as 25 on a 60 Hz display. It shows the frames per second, the CPU time to record and submit a frame, and the time from submitting a frame until the browser reports the GPU done with it (`onSubmittedWorkDone`): decoding and drawing, plus the delay of the report, averaged over the frames reported each second. In headless Chrome 155 on an Apple M3 Max (2026-10-24) the 1080p benchmark stream took about 0.6 ms of CPU and 2.2 ms from submission to done, the 4K one about 1.2 and 3.8 ms, both playing at their 30 frames per second.

`web/bench.html` times the decoding on the device it runs on, a phone included, with the code of `crates/emberwave-web/src/bench.rs`. It makes test streams at 1080p, 1440p and 4K, four pictures each of a synthetic scene encoded by the CPU encoder in the page at the budgets of the native benchmarks, then decodes them in turn on the player's path, to RGB in GPU memory, and waits for each frame through `onSubmittedWorkDone` before the next: 300 frames after at least 30 and a second of warm-up, as `emberwave-cli bench --output rgb` does natively. wgpu's callback for that has to be `Send`, which a JavaScript function is not, so it sets a flag and wakes a future that `wasm_bindgen_futures` turns into the page's promise. Browsers coarsen `performance.now()`, to a millisecond in some, so the mean comes from the time of the whole run and the median and 99th percentile, which are per frame, are as fine as the timer, whose step the report states. The times include what WebGPU adds over native wgpu: the browser's checks, its process boundary and the delay of the report. A GPU error the browser reports during a size fails that size rather than give times for frames that were not decoded, a run during which the page was hidden is marked as not valid, and the benchmark lets go of its device when the run ends.

CI builds the module as Pages does and instantiates it in Node (`scripts/check_web.mjs`), so that a module whose glue does not load fails a pull request rather than the published page.

## C API

`crates/emberwave-ffi` is the decoder and the GPU encoder for C programs. It builds a shared library (and, for releases, a static one) with the functions of `include/emberwave.h`, which cbindgen generates from `src/lib.rs`; `tests/header.rs` fails when the committed header falls behind the source. [`crates/emberwave-ffi/README.md`](crates/emberwave-ffi/README.md) is the guide for C programmers. The objects are those of the Rust API:

| C | Rust | Role |
|---|---|---|
| `EmberwaveContext` | `GpuContext` | A GPU device and the compiled shaders, shared by its decoders and encoders |
| `EmberwaveDecoder` | `Decoder` | Assembles the packets of one stream and decodes its frames |
| `EmberwaveEncoder` | `gpu::Encoder` | Encodes the pictures of one stream into frames within a byte budget |

A decoder or an encoder holds the device it was made on, so it may outlive its context. The functions:

| Function | What it does |
|---|---|
| `emberwave_probe_pwv1`, `emberwave_probe_packet` | Picture size, chroma format and colour signalling from a PWV1 file header or from a packet that starts with a start of frame header |
| `emberwave_context_create`, `_destroy`, `_adapter` | Choose a GPU (backends, software only) and name it |
| `emberwave_decoder_create`, `_destroy`, `_info` | One decoder per picture size and chroma format; the sizes of its planes |
| `emberwave_decoder_push_packet`, `_is_ready`, `_clear` | Feed packets as they arrive; ask whether the frame, or enough of it, is there |
| `emberwave_decoder_decode_yuv`, `_decode_rgb` | Decode and copy the planes, or RGB, into the caller's memory, row by row with the caller's strides |
| `emberwave_texture_create`, `_destroy`, `emberwave_decoder_decode_texture` | A texture of the picture's size, RGBA or BGRA, on the decoder's device, and a decode that draws the RGB picture into it and waits |
| `emberwave_texture_create_plane`, `emberwave_decoder_decode_plane_textures` | A one-channel texture for the Y, the Cb or the Cr plane, and a decode that copies the planes into three and waits |
| `emberwave_texture_metal`, `emberwave_context_metal_device` | The `id<MTLTexture>` and `id<MTLDevice>` behind a texture and a context, borrowed, or `NULL` off Metal |
| `emberwave_context_create_metal`, `emberwave_decoder_submit_texture`, `_submit_plane_textures` | A context on the client's own Metal device and queue, and decodes into textures that return without waiting, which only such a context takes |
| `emberwave_texture_from_metal`, `_from_metal_plane` | The client's own `id<MTLTexture>`, such as a `CVPixelBuffer`'s, as a texture every decode above fills |
| `emberwave_encoder_create`, `_destroy`, `_info` | One encoder per picture size and chroma format, with the colours its frames signal; the sizes of its planes |
| `emberwave_encoder_encode` | Encode planes from the caller's memory, with the caller's strides, into a frame of at most the budget, written into the caller's buffer of that size |
| `emberwave_frame_packets` | Where a frame splits into packets of a given size, at its blocks |
| `emberwave_version`, `emberwave_last_error` | The version, and the message of the last failure on the calling thread |

Every function returns a status, or a value that cannot fail. A Rust panic is caught at the boundary, because unwinding into C is not allowed, and becomes `EMBERWAVE_STATUS_INTERNAL`, or `EMBERWAVE_STATUS_DEVICE_LOST` when the device is lost: wgpu panics when the device fails while it is waited for (D22), after recording the loss. Arguments are checked before the frame is used: a `NULL` pointer, a stride shorter than a row, a picture size the format cannot have or an unknown constant is `EMBERWAVE_STATUS_INVALID_ARGUMENT`, and the frame can still be decoded with correct ones. What the caller passes in is a plain integer, never a Rust enum, so no value from C can be an invalid one.

`unsafe` code lives in this crate only. It has its own lint table instead of the workspace's `forbid`: every unsafe operation sits in an `unsafe` block of its own with a comment saying why it is sound, and every public function documents what the caller must guarantee.

For apps on iPhones, iPads and Macs, `scripts/build_xcframework.sh` bundles the static library into `Emberwave.xcframework`, built for devices, for the iOS Simulator and for Macs, all with Apple silicon. It has the header and a module map that Swift imports as `Emberwave` (D35). `examples/swift` is the decoding example in Swift, a SwiftPM package that links the XCFramework.

The C API copies the picture to the CPU, which costs about as much as decoding it: a 4K frame to RGB takes 2.7 ms with the copy and 1.2 ms without on an Apple M3 Max (D37). For Metal clients, `emberwave_decoder_decode_texture` puts it into a texture on the GPU instead, whose `id<MTLTexture>` the client reads: 1.3 to 1.4 ms for the 4K frame (D40, D42). `emberwave_decoder_decode_plane_textures` copies the Y, Cb and Cr planes into three textures for a client that converts the colours itself, in 1.0 to 1.1 ms (D41). On a context made on the client's own device and queue, the submit functions do the same without waiting, and Metal orders the client's reads after them (D43). The client's own textures, such as those of `CVPixelBuffer`s for an `AVSampleBufferDisplayLayer`, take the decodes as well (D44). Vulkan and Direct3D 12 handles are later work.

Frames from upstream's Metal encoder signal BT.709, full range and centred chroma in their start of frame headers whatever the pictures are, because its API has no way to set the signalling; `testdata/README.md` has the details. `emberwave_decoder_decode_rgb` with a `NULL` colour follows the frame, so a caller with better knowledge, such as the colours of a PWV1 file header, passes them.

## Encoder

`emberwave_core::encode` turns 8-bit pictures into frames of bitstream v1 within a byte budget, on the CPU, and `gpu::Encoder` does the same on the GPU ([On the GPU](#on-the-gpu)); `emberwave-cli encode` runs either on a Y4M file. The CPU encoder is the reference for the GPU's, as `emberwave_core::cpu` is for the GPU decoder, and like that it is plain sequential Rust. Its frames decode with any decoder of the format: the tests check emberwave's against upstream's, and the vectors in `testdata/encoder/` carry upstream's decode of emberwave's streams.

The format fixes the transform and what the decoder does with the numbers; the rest is the encoder's choice:

| Step | Fixed by the format | emberwave's choice |
|---|---|---|
| Transform | CDF 9/7 lifting and gains, whole-sample symmetric extension at every level, the DC shift | The picture is padded to the aligned size by repeating its last column and row (`fdwt.rs`) |
| Quantisation | Reconstruction `(m + 0.5) * block_scale * block8_scale` | A band's coefficients are scaled by `2^b`, `b = 8 + (2 for LL, 1 for HL and LH) + level − 1 for chroma`, at most 12, so that a step costs about the same in the picture in every band. An 8x8 block takes the finest scale that needs no more bit-planes than scale 1; magnitudes are `trunc(|t| / scale)`, a dead zone around zero (`quant.rs`) |
| Bit-planes | Base planes plus 0 to 3 per 4x2 subblock, most significant first | The base is all but the top three planes of the block's largest magnitude |
| Rate | — | Each 32x32 block drops 0 to 15 of its least significant planes; which, see below (`rate.rs`) |
| Framing | Start of frame header, then blocks; `total_blocks` counts the blocks sent | Blocks in block index order, unused sign bits zero, and the stream's colours in every start of frame header (`pack.rs`) |

Rate control. For every block and every number of dropped planes the encoder knows the block's size in words and its distortion: the squared difference between coefficients and reconstruction, weighted per band by the contrast sensitivity of the eye at the band's frequency and by the energy the inverse transform gives the band (`weights.rs`, generated by `scripts/gen_encoder_weights.py`). A block can be sent at a few sizes, and at each it takes the number of dropped planes with the least distortion: dropping a plane sometimes costs no word and still lands the reconstruction nearer. Going from one size to the next smaller is a step, which saves words and adds distortion. A step's key is its added distortion per saved word, rounded down to one of 512 thresholds a quarter of a power of two apart; a step's key is at least that of the step before it, so a block's steps keep their order. All steps of the frame are applied in key order, ties by block index, until the frame fits. A frame that cannot fit even with every block dropping all it may is an error rather than an overshoot; that takes a budget below the 8 bytes of the header.

The keys are integers found by exact comparisons, with no division and no logarithm, the weights are literals, and every sum runs in a fixed order without fused multiply-adds. The encoder therefore makes the same bytes on every system, which `tests/encoder.rs` checks against the committed vectors, and a GPU given the same sizes and distortions can make the same choices.

At the same budget and frame size, decoded by upstream's FP32 decoder (Apple M3 Max, 2026-10-24, `refgen` pictures, one frame each), emberwave's encoder reaches the luma PSNR of upstream's or more:

| Picture | Budget | Upstream | emberwave |
|---|---|---|---|
| 1280x720 scene, 4:2:0 | 60 kB | 33.96 dB | 37.32 dB |
| 1280x720 scene, 4:2:0 | 150 kB | 42.32 dB | 45.04 dB |
| 1280x720 scene, 4:2:0 | 400 kB | 50.11 dB | 50.27 dB |
| 1920x1080 scene, 4:2:0 | 100 kB | 32.20 dB | 32.77 dB |
| 640x360 scene, 4:4:4 | 40 kB | 39.49 dB | 42.56 dB |
| 1280x720 gradient, 4:2:0 | 40 kB | 57.88 dB | 62.61 dB |
| 640x360 noise, 4:2:0 | 60 and 200 kB | 11.90 and 25.79 dB | 11.95 and 25.78 dB |

Noise, which no budget compresses, comes out the same. Without the contrast weighting the encoder would gain another 0 to 6.5 dB of PSNR (decision D28).

### On the GPU

`gpu::Encoder` runs the same steps as compute shaders, so that a picture goes to the GPU once and its frame comes back once. The CPU writes the planes into an upload buffer the encoder keeps from frame to frame and maps again before each, the passes below run in one submission after the copy of the planes, and the summary of the rate control and the payload come back in one read-back. The CPU writes the start of frame header in front of the payload. A picture that is in GPU memory already, in textures (`Encoder::encode_textures`), takes the place of the upload: `gather.wgsl` packs the textures into the same planes at the start of the transform's pass, so the frame is the same bytes (decision D32).

| Stage | Shaders | Dispatches | What it does |
|---|---|---|---|
| Input from textures | `gather.wgsl` | 1, in the transform's pass: a workgroup per 64x16 samples of a plane, the component in z | Each sample of the R8 or Rg8 textures, its byte back from the unorm float, packed four to a word with rows padded to whole words: the words the upload writes for the same picture |
| Forward transform | `fdwt.wgsl` | 5: one per level, finest first; a workgroup per 32x32 tile of the level's image and component | The inverse transform's shape and parameters (`TransformLevel`), the other way round: a 40x40 window of the image, rows and then columns lifted in registers, and the four bands written. At a component's finest level the image is the 8-bit plane, its last column and row repeated up to the aligned size |
| Quantisation | `quantise.wgsl`, `equant.wgsl` | 1: a workgroup of 128 invocations per block | Scales, magnitudes and bit-planes of `quant.rs`, then the block's size and distortion for 0 to 15 dropped planes, its cost table. Each invocation adds up the squared errors of its own 4x2 subblock in registers, and the subblocks are added up per 8x8 block, in the grouping the CPU adds in. Each 8x8 block is measured only up to the drop that leaves nothing of it; beyond, its error is the same sum of the same terms |
| Rate control | `rate.wgsl`, `scan.wgsl` | 4, and two prefix sums of 3 | Each block's steps, and their saved words added into a histogram of the keys; the key at which the frame fits; what each block's steps of that key save, summed in block order; each block's drop and size, and their sums in block order, which are the blocks' places in the payload |
| Packing | `quantise.wgsl`, `pack.wgsl` | 1: a workgroup per block | The block quantised again, the chosen planes dropped, its bytes assembled in workgroup memory and written at its place |

A workgroup past the last block, which a grid of more than 65,535 workgroups has, quantises the last block again and writes nothing, rather than return at the top. A return in front of the barriers of `equant.wgsl` and of `scan_chunks` in `scan.wgsl` crashed WARP, Direct3D's software adapter, in CI, though no workgroup took it; shader variants, each run in a process of its own on WARP, pinned the crash on the return. `pack.wgsl` and the decoder's `dequant.wgsl` return early in the same way and run on WARP.

Nothing is sorted. The CPU applies the steps of all blocks in key order, ties by block index, until the frame fits. On the GPU every step below the cutoff key is applied and none above it, and the steps of the cutoff key are applied in block order while the frame does not fit yet, which is where the CPU stops. Quantisation and packing share `quantise.wgsl`, compiled in front of each, so that what the rate control measures is what the packing writes.

The GPU cannot repeat the CPU's floating point exactly, so the contract between the two is per stage (decision D29), and the tests feed each stage the CPU's input:

| Stage | Given the CPU's input, the GPU's output is |
|---|---|
| Forward transform | Within 1e-5 of the CPU's coefficients, relative for coefficients above 1 and absolute below (8.3e-7 at most on an Apple M3 Max) |
| Quantisation | The same sizes; the same distortions up to fused multiply-adds, within 1e-5 of each (2.5e-7 at most) |
| Rate control | The same drops |
| Packing | The same bytes |

A frame is therefore the CPU encoder's, or one that differs in the blocks whose coefficients sit on a rounding boundary. On an Apple M3 Max all 25 frames of the committed vectors come out identical. Of `refgen`'s scene at 1080p, 1440p and 4K about one sent block in 200 differs, mostly at the same size, and the PSNR is the same to a thousandth of a dB; of pictures of flat rectangles, thin strokes or squares, between none and one in 200. The tests allow differing frames as long as they are as good (within 0.05 dB). The CPU encoder stays the one that makes the same bytes on every system.

[`BENCHMARKS.md`](BENCHMARKS.md#encode-next-to-upstream) has the encoder's times next to upstream's, measured as the decoder's are: on an Apple M3 Max a 4K frame takes 1.91 ms from the planes in CPU memory to the frame in CPU memory, against 3.39 ms for upstream's encoder at the same FP32 precision and 218 ms for the CPU encoder (decision D30).

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

### 2026-10-24 — D28: An encoder, on the CPU first

**Context.** Issue #14 asks for an encoder: forward transform, quantisation, bit-plane packing and rate control to a byte budget, interoperable with upstream's decoder and encoder in both directions, its PSNR at equal bytes measured against upstream's, and round trips in CI on three systems. Upstream's encoder runs entirely on the GPU with subgroup operations, 8-bit stores and atomics; a study of it, checked against its output byte for byte, describes its algorithm without its code.

**Decision.**
- A CPU encoder comes first, in `emberwave_core::encode`, as the reference for a GPU encoder, like the CPU decoder of D8. The GPU encoder is to be compared with it stage by stage: coefficients within a tolerance, everything after them exact given the same inputs. Its rate control is to run on the GPU too: reading costs back for the CPU to choose would put a stall in every frame.
- The algorithm is upstream's in kind, written anew: the same band exponents and the same idea for the 8x8 scale, dead-zone magnitudes, and drop-the-least-significant-planes rate control weighted by the eye's contrast sensitivity, so that the comparison with upstream is of like with like. The rate control is emberwave's own: steps keyed by their marginal distortion per word, monotone per block, applied in one frame-wide order, where upstream keys each step by its cumulative distortion since no plane was dropped, in buckets of 1.5 dB.
- Determinism over bit-exactness with upstream: integer keys from exact comparisons, weights as `f32` literals generated once, fixed sums, zero padding where upstream leaves leftover memory. One committed stream per vector is compared byte for byte on every system.
- Tests: the encoder's vectors live in `testdata/encoder/` with upstream's FP32 decode of them (`scripts/gen_encoder_vectors.sh`, macOS); emberwave's decoders must match it at 80 dB (D9). The decoder's conformance vectors stay upstream-encoded. Round trips in CI use small pictures; quality against upstream's encoder is measured locally and recorded, not a CI gate.
- New tooling: `refgen encode --source` writes the pictures it synthesises, and `--input` encodes a Y4M file with upstream's encoder, so that both encoders can be given the same pictures.

**Consequence.** emberwave can make streams, at the PSNR of upstream's encoder or above at equal bytes (see [Encoder](#encoder)); a 640x360 frame takes about 7 ms on one CPU core. Turning off the contrast weighting was tried once: PSNR rose by 0 dB on the gradient, 1.6 to 4.4 dB on the 1280x720 scene, 2.8 dB at 4:4:4 and 6.5 dB on the 1920x1080 scene, as a pure squared-error allocation keeps the fine detail the weighting gives up first. The weighting stays, as upstream's does. Next is the GPU encoder, measured against this one.

### 2026-10-24 — D29: The GPU encoder, and what it shares with the CPU's

**Context.** D28 made the CPU encoder the reference for a GPU encoder, to be compared stage by stage. A GPU cannot repeat the CPU's floating-point results exactly: WGSL has no way to forbid fusing a multiply and an add, and Metal compiles with fast math, so the coefficients of the forward transform can differ from the CPU's in the last place, and so can the sums of squared errors.

**Decision.**
- `gpu::Encoder` follows the CPU encoder step for step, its operations in the same order. Everything that can be exact is: given the same coefficients, the scales, magnitudes, bit-planes and block sizes come from exact comparisons and are the CPU's; the distortions are the same sums in the same order and differ only where the compiler fuses; given the same cost tables, the rate control chooses the same drops; given the same drops, the packing writes the same bytes. The tests feed each stage the CPU's input and compare ([On the GPU](#on-the-gpu)).
- A frame is therefore the CPU encoder's or differs in some blocks: about one sent block in 200 of a natural picture on an Apple M3 Max, mostly at the same size. Where frames differ, the tests require the same PSNR within 0.05 dB. Byte-identical streams on every system remain the CPU encoder's promise, not the GPU's.
- The rate control does not sort. A histogram of the words the steps of each key save, added with 32-bit atomics, which is exact in any order, gives the key at which the frame fits; a prefix sum in block order settles the steps of that key. Prefix sums use workgroup memory (D4), in two passes of 1024-value chunks, enough for the 786,432 blocks of the largest picture.
- Quantisation and packing share one WGSL file, compiled in front of each, and the packing quantises again rather than read what the quantisation would have had to store, 1024 magnitudes per block. Quantising is exact, so both see the same magnitudes.
- One upload and one read-back per frame. The read-back takes as many words as the budget allows, with the rate control's summary in front, so that the CPU waits once. The payload buffer grows to the budget in powers of two and stays. A frame over its budget, which the rate control never makes, is an error rather than returned, whatever a driver did.
- The new band table, `QuantParams`, looks bands up in the `BufferLayout` with an `expect` like the three D22 keeps, and the property test of D22 now builds it too.
- `GpuError` gains `Encode`, which carries the CPU encoder's `EncodeError`: the GPU encoder refuses what the CPU encoder refuses, with the same smallest frame size.

**Consequence.** On an Apple M3 Max a 4K frame encodes in a median 3.9 ms, the 12 MB upload and the read-back included, 60 times as fast as on one CPU core, and on that machine the committed vectors come out byte for byte as the CPU encoder made them. Most of the GPU time goes to the quantisation. Faster quantisation and a comparison with upstream's encoder are issue #29.

### 2026-10-24 — D30: The GPU encoder's speed: a mapped upload buffer, sums per subblock

**Context.** Measured as the decoder is (`BENCHMARKS.md`, issue #29), the GPU encoder took 3.37 to 3.66 ms for a 4K frame on an Apple M3 Max (three rounds of 600 frames) and upstream's encoder about 3.2 to 3.4 ms at the same FP32 precision. Two things took most of it. The CPU copied the 12 MB of planes into the staging memory that `Queue::write_buffer` allocates anew for every frame, 1.25 to 1.28 ms; into a buffer kept from frame to frame the same copy takes 0.22 ms. And the quantisation, 0.91 to 1.04 ms of 1.61 to 1.82 ms of GPU passes, added the 64 squared errors of each pair of an 8x8 block and a number of dropped planes one after the other, with two loads from workgroup memory per term.

**Decision.**
- The planes go through an upload buffer the encoder keeps, mapped for writing before each frame, which waits for nothing since every encode waits for its frame, and copied into the planes buffer at the start of the frame's command buffer. It keeps a second buffer of the planes' size, 12 MB at 4K, allocated between frames; the old path allocated as much for every frame, and the peak memory of a 4K encode did not rise.
- Each invocation of the quantisation adds up the squared errors of its own 4x2 subblock, for every number of dropped planes it needs, in registers; a pair adds up its 8 subblocks. The CPU encoder adds in the same grouping, each subblock's values in order and then the subblocks in order, instead of the 64 values in one run, so that the GPU's distortions remain the CPU's sums in the CPU's order (D29). No choice of the rate control moved: the committed vectors come out byte for byte as before.

**Consequence.** Measured alternately with the build before, a 4K frame encodes in 1.88 to 1.95 ms instead of 3.37 to 3.66 ms; the CPU's part is 0.22 to 0.23 ms and the quantisation's 0.59 to 0.61 ms. In the benchmarks that is 0.56 times the time of upstream's encoder at FP32 at 4K, and 1.12 and 0.82 ms at 1440p and 1080p against upstream's 1.53 and 0.93 ms. What remains is a second copy of the planes on the GPU, which an encode from pictures already in GPU memory would avoid (issue #33).

### 2026-10-24 — D31: A race in the Vulkan loader: contexts one at a time, lavapipe alone in CI

**Context.** In an aarch64 Debian 13 container (Vulkan loader 1.4.309, Mesa 25.0.7 with eight drivers, lavapipe the only one with a device), the GPU encoder tests crashed in 16 of 40 runs when they ran in parallel, and never on one thread (issue #36). gdb with the loader's symbols puts the crash in the loader. The first `vkEnumeratePhysicalDevices` of a new instance unloads the drivers that found no device and frees their entries (`unload_drivers_without_physical_devices`, under `loader_lock`), while another thread walks the drivers of every instance under a different mutex (`loader_get_icd_and_device`). That walk comes from `vkSetDebugUtilsObjectNameEXT`, which wgpu calls for every object it labels in debug builds, and from `vkGetDeviceProcAddr` while a device is created. A release build crashed in 1 of 40 runs, on the second path. Loaders 1.3.284 to 1.4.344 have the race, and upstream fixed it in 1.4.345 (commit 5ee27b30). Debian 13 and Ubuntu 25.04 to 26.04 LTS ship affected loaders, and their `mesa-vulkan-drivers` package installs every Mesa driver, most of which find no device on any one machine. Ubuntu 24.04, CI's `ubuntu-latest` today, ships 1.3.275, which unloads nothing.

**Decision.**
- `GpuContext::new`, and with it `emberwave_context_create`, makes one context at a time in a process, behind a mutex. In a release build both racing paths are inside it, and nothing emberwave does once a device exists walks the loader's lists outside `loader_lock`: destroying a device or an instance takes that lock. The mutex does not cover `new_async` or `from_adapter`, other users of Vulkan in the process, or debug builds, where object names can cross the creation of a context on another thread. Their documentation says so.
- CI's Linux job loads lavapipe alone through `VK_DRIVER_FILES`. With no driver to unload, the debug tests ran 40 times in the container without a crash. That keeps CI green once `ubuntu-latest` has an affected loader. `CONTRIBUTING.md` tells developers on affected systems to do the same, or to run the tests on one thread.

**Consequence.** In the same container with every driver loaded, alternating the builds in three blocks of 100 runs, release builds of the GPU encoder tests crashed in 6 of 300 runs before the mutex and in none of 300 with it. Debug builds crashed in 9 of 40 runs before and 2 of 40 with it, through object names. Contexts created on several threads at once wait for each other, one device creation each. The cure is a loader of 1.4.345 or later.

### 2026-10-24 — D32: Encoding pictures that are in textures

**Context.** `gpu::Encoder::encode` takes 8-bit planes in CPU memory, which the CPU writes into an upload buffer, 0.22 ms of a 1.9 ms encode at 4K on an Apple M3 Max (D30). A streaming host has the picture on the GPU already, from a screen capture, and would otherwise read it back to CPU memory first (issue #33). Upstream's Metal encoder takes three R8 IOSurfaces or one NV12 or P010 IOSurface and reads them as textures.

**Decision.**
- `Encoder::encode_textures` takes a `TexturePicture`: views of three `R8Unorm` textures, Y, Cb and Cr, or the layout of NV12, Y in `R8Unorm` and Cb and Cr interleaved in `Rg8Unorm`. Views rather than textures, so that a caller can pass the plane views of a multi-planar NV12 texture once a context can have that format, which wgpu 30 offers only on Vulkan and DX12 and behind a feature emberwave does not ask for; until then the check of the view's texture, its format, size, dimension, samples and usage, accepts R8 and Rg8 textures only, and an unsuitable one is `GpuError::Texture`. What the view shows cannot be checked, nor its device: an array view fails in wgpu's validation, as `GpuError::Device`, a view of another mip level goes unnoticed, and wgpu resolves a view of another wgpu instance by its id, which can name an unrelated view of this device. A 2D view of one layer of an array texture is fine.
- Contexts ask for the adapter's largest 2D textures, as they ask for its largest buffers, so that the textures can be as large as the pictures the encoder takes: 16384 on most GPUs, where WebGPU guarantees 8192. A context limited to WebGPU's guarantees keeps 8192.
- `gather.wgsl` packs the textures into the planes buffer, one invocation per word, the byte of each sample back from its unorm float by rounding, before the transform in the same compute pass. Everything after it is unchanged, so a picture in textures encodes to the bytes of the same picture in CPU memory: a test compares the planes buffer word for word, padding included, after the upload and after the gather, for all 256 byte values, on every backend CI runs. Rounding is exact as long as the float is within half a step of the byte, and the formats' conversion is far inside that.
- The bind group of a picture's textures is made for each frame and dropped with it, rather than cached: a cache would keep the caller's textures alive, and making one costs microseconds.

**Consequence.** In the benchmarks on the Apple M3 Max (`BENCHMARKS.md`, at commit `991f541`) the CPU's part of a 4K encode falls from 0.23 to 0.02 ms, and the transform, the gather included, takes 0.52 instead of 0.40 ms: the gather costs more GPU time than the copy from the upload buffer that it replaces. A 4K frame takes 1.80 ms from textures against 1.93 ms from CPU memory, 1.08 against 1.11 ms at 1440p, and 0.80 ms either way at 1080p. Footprints of 64x4 and 16x16 samples per workgroup, and a flat grid of words, did not make it faster; the loads of single 8-bit texels are the cost. Reading the textures in the transform's first level instead would save most of it (issue #40). A capture that would have to come back to CPU memory saves the read-back as well, which is not measured here.

### 2026-10-24 — D33: The encoder in the C API

**Context.** The C API had a decoder only, for clients; the encoder was Rust's alone, so a host written in C or C++ could not use it (issue #42). Upstream's Metal C API encodes planes with row strides or IOSurfaces, and its `pyrowave_encoder_packetize` writes the packets into a buffer the caller provides.

**Decision.**
- `emberwave_encoder_encode` writes the frame into a buffer of `budget` bytes that the caller provides, as the decoder writes pictures into the caller's memory: a frame never exceeds its budget, so that buffer always suffices, and the bytes stay the caller's while they are being sent. An encoder-owned buffer, valid until the next call, would be overwritten by a retry with a smaller budget while the previous frame is still in flight.
- A budget below the smallest frame is `EMBERWAVE_STATUS_INVALID_ARGUMENT`, with the smallest frame's size in the size out-parameter, so a host can retry without parsing the message. Nothing is encoded then, and the sequence number does not move.
- The planes come with their strides, which the GPU encoder takes itself (`Encoder::encode_strided`): its upload copies row by row already, so padded rows, as captures have them, are copied once rather than first packed by the C API.
- `emberwave_frame_packets` splits a frame at its blocks into packets of at most a given size, as `encode::packets` does, and returns offsets and sizes into the frame; with no array it counts. It is apart from the encoder, so it works on frames from files as well.
- `EmberwaveSignalling` carries all five fields of the start of frame header, primaries and transfer function included, with three reserved words that must be zero, so that the struct can grow without a new one. BT.601 is refused: the format has no code for it. A zeroed struct is therefore an error, and `NULL` is the default (BT.709, video range, centred chroma, sRGB).
- Only the GPU encoder: the context is a GPU context. Pictures in GPU memory (`encode_textures`) need handles of the caller's graphics API and wait for #27.

**Consequence.** `examples/c/encode.c` encodes a Y4M file, splits each frame into packets of 1,200 bytes, decodes them with the C decoder, last packet first, and checks the PSNR; CI runs it on Linux, macOS and Windows. Frames through the C API are the bytes of the Rust encoder, with tight or padded rows.

### 2026-10-24 — D34: The decoder's frame upload through a buffer it keeps

**Context.** A 4K frame decoded to planes in about 1.05 ms on an Apple M3 Max, upstream's decoder at FP32 in 0.79 ms. The GPU times were close, 0.55 against 0.50 ms; the difference was on the CPU, which took 0.23 ms to prepare a frame: 72 µs to assemble it and 174 µs to hand its 1.2 MB of blocks to `Queue::write_buffer` (medians of 300 frames), about 7 GB/s. That is what D30 found for the encoder: `write_buffer` takes new staging memory for every call, and the first writes to it cost more than the copy (issue #45).

**Decision.**
- The decodes that wait for their frame (`decode`, `decode_rgb`, `time_frame`, and the C API's) write the block table and the payload into an upload buffer the decoder keeps, mapped for writing before each frame, which waits for nothing since the frame before is done, and copied into the decoder's buffers at the start of the frame's command buffer. It is made on the first such decode, and made again when the payload buffer grows, so a decoder that only records into the caller's command encoder never has it.
- `encode_rgb`, which records the decode into the caller's command encoder without waiting, and so cannot know when the previous frame let go of the buffer, still writes through the queue. So does the browser player, where nothing may block.
- When a frame fails, the decoder and the encoder poll the device once, without waiting, while no loss is recorded, as `GpuContext::explain` does (D31's companion fix of #35): a destroyed device fails the mapping before anything else has polled it, and is reported as lost only after a poll.

**Consequence.** Measured alternately with the build before, three rounds of 600 frames each: a 4K frame decodes to planes in 0.87 to 0.91 ms instead of 1.05 to 1.07 ms, and to RGB in 1.20 to 1.21 ms instead of 1.34 to 1.40 ms; the CPU's part falls from 0.23 to 0.08 ms. At 1440p the planes take 0.50 to 0.54 ms instead of 0.61 to 0.63, at 1080p 0.39 ms instead of 0.46 to 0.51 in two rounds of three (the third had a slower GPU, after as before). A decoder keeps a second buffer of the payload's size, 1.2 MB at 4K.

### 2026-10-24 — D35: The C library for iPhone, iPad and Mac apps, as an XCFramework

**Context.** iPhones decode PyroWave in Safari through WebGPU (`BENCHMARKS.md`). A native app needs the C library in a form Xcode links, and the releases had it for macOS only (issue #52). The development Mac had neither Rust's standard library for iOS nor an iOS Simulator runtime, and an app for an iPhone has to be signed by its developer. GitHub's macOS runners have Xcode and iOS Simulator runtimes.

**Decision.**
- `scripts/build_xcframework.sh` builds the C library as static libraries, one per slice, and bundles them with `xcodebuild -create-xcframework` into `Emberwave.xcframework`:

  | Slice | Rust target | Oldest system |
  |---|---|---|
  | `ios-arm64`, iPhones and iPads | `aarch64-apple-ios` | iOS 12, the oldest Xcode 26 builds for |
  | `ios-arm64-simulator`, the iOS Simulator on Apple silicon | `aarch64-apple-ios-sim` | iOS 14, the first Simulator on Apple silicon |
  | `macos-arm64`, Macs with Apple silicon | `aarch64-apple-darwin` | macOS 11, the first on Apple silicon |

  wgpu checks at run time for what newer systems add, so the old deployment targets cost nothing.
- Static libraries only. A dynamic library in an iOS app has to be a framework bundle, embedded and signed. A static library needs none of that, and the linker drops what the app does not call.
- No Intel slices. The Simulator on Intel Macs and Intel Macs themselves would each add a build, for machines Apple no longer sells.
- The header sits in `Headers/Emberwave/` with a module map. Swift imports the module `Emberwave`, and C includes `<Emberwave/emberwave.h>`. Xcode copies the headers of every static XCFramework of an app into one directory, where module maps at the top would collide. The bundle takes the module's name, as XCFrameworks do and as a SwiftPM binary target named `Emberwave` expects.
- `examples/swift` is `decode.c` in Swift, a SwiftPM package that names the XCFramework as a binary target and the system libraries in its linker settings. Those are the same for all three slices: Metal, QuartzCore, CoreGraphics, Foundation, CoreFoundation, `libiconv` and `libobjc`.
- CI's macOS job `XCFramework` builds it and runs `scripts/check_xcframework.sh`. That script builds the C examples and the Swift example against each library, linked statically with the system libraries rustc names. On the Mac and in an iOS Simulator, the decoding examples decode every test vector, which `emberwave-cli` compares with upstream's FP32 decode at 80 dB (D9), and the encoding example encodes two sources. For devices the examples are only linked. A failing program in the Simulator must fail the check, and the script makes sure that it does.
- Releases carry it as `emberwave-ffi-VERSION-xcframework.zip`, with the system libraries of each slice, the FFI README, the licence and the changelog.

**Consequence.** An app adds one XCFramework and a few system frameworks, and calls the C API from C, Objective-C or Swift. The first run in the Simulator found that no context could be created there (D36). With that fixed, the C and Swift examples decoded every vector in the Simulator ("Apple iOS simulator GPU", an iPhone 17 on iOS 26.5) to the PSNR they reach on the runner's Mac, from 86.28 dB to bit-exact, and encoded both sources. The job takes about 11 minutes, 2 to 3 of them booting the Simulator. Nothing has run on an iPhone's own GPU through the XCFramework yet: the Simulator uses the Mac's GPU. (Later, on 2026-10-26: the benchmark app ran on an iPhone 17, and its checks passed on the A19's GPU. `BENCHMARKS.md` has its times.) Sharing the picture with the app's Metal code on the GPU is still issue #27.

### 2026-10-24 — D36: Limits: the defaults, or less where the adapter has less

**Context.** A context asked for WebGPU's default limits, with the adapter's buffer and texture sizes (D32). In the iOS Simulator, an iPhone 17 on iOS 26.5, every context failed: "Limit 'max_inter_stage_shader_variables' value 16 is better than allowed 15" (issue #54). wgpu's Metal backend gives 31 inter-stage variables to Apple GPUs of the Apple4 family (the A11 and later), 30 to Macs' other GPUs, and 15 to everything else: the Simulator and the A7 to A10, which iOS 12 to 15 run on.

**Decision.** A context asks for the default limits, or the adapter's limit where that is lower (`Limits::or_worse_values_from`). For limits that are minimums, such as alignments, that means the adapter's where it is larger. Buffers and textures stay as large as the adapter allows. `strict_webgpu` asks for the defaults exactly, as the portability tests need. A unit test checks the limits a context asks for without a GPU.

**Consequence.** The compute shaders have no inter-stage variables, and the browser player's draw passes none besides the position. An adapter short of a limit the shaders do need now fails where the pipelines are made, naming that limit, rather than when the device is. Adapters that meet the defaults get exactly the device they got before.

### 2026-10-24 — D37: Pictures read back straight from the mapped buffer

**Context.** A 4K frame decoded to RGB and copied to the CPU took 7.2 ms in `emberwave-cli` and 7.6 ms through the C API. Left in GPU memory it takes 1.2 ms (issue #56). Three copies made the difference:
- The mapped RGBA, 33 MB, was copied into a new vector.
- A copy per pixel turned that into RGB in another vector, 4.5 ms on its own.
- The C API copied the result into the caller's memory row by row.

Every native client waits for the C API's decode until the picture can stay on the GPU (#27).

**Decision.**
- `read_back_with` hands the mapped bytes to a closure. A guard unmaps the buffer afterwards, inside the read's error scope, also when the closure panics. Unmapping a buffer that is not mapped is an error, so the guard exists only once the range is mapped.
- RGBA turns into RGB in a loop over chunks of three and four bytes, which the compiler vectorises: 0.5 ms for a 4K frame in a micro-benchmark on the M3 Max, where the copy per pixel took 4.5 ms. The core forbids `unsafe`, so it has no SIMD intrinsics.
- `Decoder::decode_assembled_rows` and `decode_assembled_rgb_rows` hand out the picture a row at a time from that memory; RGB rows are converted into one reused row. The C API copies each row through a pointer. The caller's memory may be uninitialised, and a `&mut [u8]` must not refer to such memory, so the core's API takes no slice of it.
- Rows go out only if the device was not lost when the frame was waited for. The wait polls the device, and the poll records a loss. A loss noticed after the rows went out still fails the call. The C API's documentation says that such a call may leave some rows written.
- `decode_assembled` and `decode_assembled_rgb` fill their new frames from the mapped memory directly.

**Consequence.** Measured alternately with the build before on the Apple M3 Max, 300 frames and two rounds:

| | Before | After |
|---|---|---|
| C API, 4K to RGB | 7.64 to 7.65 ms | 2.62 to 2.71 ms |
| C API, 4K to planes | 1.57 to 1.59 ms | 1.16 to 1.18 ms |
| C API, 1080p to RGB | 2.25 ms | 0.84 ms |
| C API, 1080p to planes | 1.10 ms | 0.49 ms |
| `emberwave-cli`, 4K to RGB in CPU memory | 7.13 to 7.20 ms | 2.14 to 2.27 ms |

The copy that remains, 25 MB at 4K, goes into the caller's memory. Doing without it is the work of #27.

### 2026-10-24 — D38: The gather pass reads texels four at a time

**Context.** A picture in textures reaches the transform through `gather.wgsl`, which packs it into the planes buffer (D32). At 4K the pass cost about 0.13 ms of the transform's 0.52 ms, more than the copy from the upload buffer it replaces. Other workgroup shapes did not help: the cost was the single 8-bit texel loads, one per sample. Issue #40 proposed reading the textures in the transform's first level instead, with `textureGather` as an alternative.

**Decision.**
- Each invocation of the gather pass writes one word in each of two rows. It takes the eight samples with two `textureGather` calls, 2x2 texels each, where four `textureLoad` calls gave one word.
- The gather happens at the corner the four texels share. Those coordinates are half a texel from every texel centre, so rounding cannot pick other texels. The bytes come out as before: the test that compares the planes buffer word for word with the CPU's upload, for every byte value, passes unchanged.
- The sampler is nearest and clamped to the edge, made once per encoder. Texels past the edge of a texture repeat the edge, and the samples there are masked to the zeros of the padding as before.
- Reading the textures in the transform's first level is not done. It would save at most the 0.04 ms the pass now costs, with a second transform pipeline. It would also load each texel about 1.5 times, as the 40x40 windows of neighbouring tiles overlap.

**Consequence.** Measured alternately with the build before on the Apple M3 Max, 600 frames and three rounds: a 4K frame from textures encodes in 1.71 to 1.72 ms instead of 1.80 to 1.81 ms. The transform, gather included, takes 0.43 to 0.44 ms instead of 0.52 ms. At 1440p and 1080p the transform takes 0.21 and 0.12 to 0.14 ms instead of 0.24 and 0.14 to 0.15 ms, and the totals stay where they were. The encoder needs a sampler binding in the gather pass's layout.

### 2026-10-24 — D39: The browser pages, from their review

**Context.** A read-only review of the browser pages, with experiments in headless Chrome 155, found no high-severity problem. It found two medium ones:
- The benchmark registered no handler for the GPU errors the browser reports. With a buffer made invalid, it gave times for frames that were never decoded.
- The player copied every frame of a file into a vector of its own, so the module held the file about twice. A PWV1 header followed by 200 MB of zeros, 50 million empty frames, took 1.7 GiB of module memory. A little more overflowed a vector's capacity, which panicked; after that every call failed with "recursive use of an object" until the page was reloaded.

The low ones:
- A 36-byte file could ask for a 16384x16384 decoder, gigabytes of GPU memory.
- A lost device or a panic left the page unusable without telling the user.
- A file whose last frame is cut off played not at all.
- The benchmark left its device and its screen wake lock behind on some paths.
- It counted runs during which the page was hidden.
- It could block a page with a coarse timer for seconds.
- The decoder made read-back buffers the player never uses.

**Decision.**
- The player takes the file as it is handed over and walks its frames in place. A frame shorter than a start of frame header, 8 bytes, cannot decode, and ends the frames played, as a frame that is cut off does; a file without a playable frame is refused with the reason.
- The player loads files of up to 1 GiB, which the page checks before it hands one over. It takes pictures of up to 8192x4320 pixels' worth, 35 megapixels, the size of DCI 8K. The format allows 16384x16384, but no video format in use is larger than 8K, and the cap keeps a few bytes from asking for gigabytes.
- `Bench` records the first GPU error the browser reports, and a decode fails while one is recorded, as do the end of each size's frames. The page frees the `Bench`, and releases its wake lock, however the run ends; marks a run during which it was hidden; and gives the timer at most 50 ms to show its step.
- A player whose device was lost is replaced for the next file. A WebAssembly `RuntimeError`, which is how a panic surfaces, or the "recursive use" error that follows one, stops the page with a request to reload it. Errors from the browser show their first line, not the JavaScript stack inside them.
- The decoder makes its read-back buffers when a decode first reads back: 43.5 MiB of the 153 MiB of a 4K decoder are not made for a player, in a browser or natively.
- CI builds the module and instantiates it in Node on every pull request; until now only the Pages workflow built it, after a merge.

**Consequence.** In headless Chrome on the Apple M3 Max:
- The 200 MB of zeros are refused in 31 ms, the module growing by the 190 MiB of the file itself.
- Cut-off files play their whole frames.
- 8192x8192 and 16384x16384 pictures are refused.
- A 4K stream holds 109 MiB of GPU buffers instead of 153 MiB.
- After `device.destroy()` the next sample plays on a new device.
- The benchmark with the invalid buffer stops at 1080p with the browser's message.

Picture sizes between 8192x4320 and 16384x16384 that the native decoder takes are refused in the browser.

### 2026-10-24 — D40: Pictures on the GPU for Metal clients

**Context.** Through the C API a client got the picture in CPU memory: 2.7 ms for a 4K frame to RGB on an Apple M3 Max, against 1.2 ms when it stays on the GPU (issue #27). Upstream's Metal C API takes the app's `id<MTLDevice>` and records the decode into the app's command buffer. It writes Y, Cb and Cr into the app's textures, so the app's queue orders the work. wgpu submits on queues of its own and does not record into a foreign command buffer.

**Decision.**
- `Decoder::encode_rgb_to_texture` and `decode_assembled_to_texture` draw the RGB picture into a texture with a render pass. That pass is `draw.wgsl`, the browser player's blit, now in the core. A copy from the RGBA buffer would take rows of a multiple of 256 bytes, so it would need a second, padded layout of a buffer the read-back takes packed.
- The targets are 2D textures of the picture's size in `Rgba8Unorm` or `Bgra8Unorm`. sRGB forms are refused, as the samples are gamma encoded already. The browser player draws its canvas this way too.
- The C API gets an `EmberwaveTexture`:
  - It is made by emberwave on the decoder's device, in RGBA or BGRA; BGRA is for copies onto a display's drawable.
  - `emberwave_decoder_decode_texture` draws into it and waits for the GPU, as the other decodes do.
  - `emberwave_texture_metal` and `emberwave_context_metal_device` give the `id<MTLTexture>` and the `id<MTLDevice>` through wgpu's `as_hal`. The pointers are borrowed, and `NULL` off Metal.
  - The documentation states the bridging rules and how long the pointers live.
- Ordering. The call's return orders the client's reads after the draw. Nothing orders the next decode after those reads, so keeping three textures in turn is a stated rule, not a hint. (Later, D43: on a context of the client's own queue, Metal orders both ways, and one tracked texture does.)
- Contexts get a process-unique id. Every context makes its own `wgpu::Instance`, and `wgpu::Device` equality said that devices of two contexts were equal. A texture of another context then reached wgpu-core, which panicked on a view id missing from its registry; it could also have named an unrelated view. The C API compares context ids.
- Deferred to issues:
  - Decoding without waiting, with an `MTLSharedEvent` signalled after the draw, which wgpu-hal can encode (#62).
  - Contexts on the client's own `MTLDevice` and queue, and drawing into textures of the client's own, imported with `create_texture_from_hal` (#63).
  - Y, Cb and Cr planes in textures (#64).
  - Vulkan and Direct3D 12 handles.

  The libraries' Apple systems, iPhones, iPads and Macs with Apple silicon, have one GPU each, so the device a context takes is the system's.
- Grok argued the other side, in two rounds:
  - Waiting leaves the GPU idle until the app commits: #62.
  - Textures of the app's own, and BGRA for drawables: BGRA now, the app's own in #63.
  - The bridging rules and the rotation rule belong in the header: done.
  - The 256-byte remark in the shader overstated the case: corrected.
  - Requiring an explicit colour conversion: withdrawn after the second round. `NULL` follows the frame as for `decode_rgb`, and the header repeats the warning about upstream's Metal encoder.

**Consequence.** Measured on the Apple M3 Max through the C API, 300 frames and two rounds:

| Frame | Into a texture | RGB in CPU memory | Planes in CPU memory |
|---|---|---|---|
| 4K | 1.50 ms | 2.70 ms | 1.23 to 1.35 ms |
| 1080p | 0.72 ms | 0.84 ms | 0.52 ms |

A Metal client draws the texture without a copy. `tests/metal.rs` reads the textures through Metal, on a queue of its own as an app does, and finds the RGB picture in both formats. The browser player draws through the same pass, and its canvas is still bit-exact against the reference decodes.

### 2026-10-24 — D41: The planes in textures

**Context.** D40 hands Metal clients the RGB picture, converted with the colours the stream signals or the caller passes. A client that converts the colours itself needs the samples: for high dynamic range, for its display, or for a stream that signals the wrong colours, as upstream's Metal encoder's do. Upstream's Metal API writes Y, Cb and Cr into the app's textures. The planes also take less memory: 1.5 bytes a pixel in 4:2:0, where RGBA takes 4 (issue #64).

**Decision.**
- The textures:
  - Three textures of one unsigned normalised 8-bit channel, `R8Unorm`: Y at the picture's size, and Cb and Cr at the chroma planes' size, half in 4:2:0.
  - They hold the stream's samples, as `decode` gives them, byte for byte.
- Copies from the packed planes fill them:
  - A copy from a buffer into a texture takes rows of a multiple of 256 bytes, so a decoder's packed rows are now padded to such a multiple (`BufferLayout::for_decoder`).
  - The words of padding are never written, and nothing reads them. `idwt.wgsl`'s last level stops at the row's last sample, where before it stopped at the end of the row's words, which were the same words until now. The encoder's planes keep rows of whole words, which its tests compare word for word.
  - At 1080p the padding grows the planes by 7%. At 4K only chroma rows grow, and luma rows are 3840 bytes, a multiple of 256 already.
  - The planes the CPU reads back grow by as much, which the timings below do not show.
  - The planes start at multiples of 512 bytes, as Direct3D 12 copies only from those. From other offsets wgpu copies through a buffer it allocates during the copy, and panics if that allocation fails. Without the alignment that would happen to Cb in 4:4:4 with an odd height, and to Cr in 4:2:0 when the height leaves 2 over a multiple of 4.
- Rust: `Decoder::encode_to_plane_textures`, `decode_assembled_to_plane_textures` and `create_plane_texture`.
  - Targets are checked as in D40, with `COPY_DST` where the picture's texture needs `RENDER_ATTACHMENT`.
  - The three targets must also be different textures.
- The C API:
  - `emberwave_texture_create_plane` with `EMBERWAVE_PLANE_Y`, `_CB` or `_CR` makes a texture for one plane, and `emberwave_decoder_decode_plane_textures(decoder, y, cb, cr)` fills three.
  - A texture remembers its plane. In 4:4:4 the three are of one size, so only that catches textures passed in the wrong order.
  - A plane's texture does not take the RGB picture, nor the picture's texture a plane. `emberwave_texture_metal` and `_destroy` serve both kinds.
  - The header says where 4:2:0 chroma sits. Vertically it is halfway between two luma rows. Horizontally it follows the stream's siting.
- Alternatives:
  - Render passes that unpack each plane in a fragment shader. Measured first, they cost about 0.3 ms more per frame than copies, at 1080p and 4K alike: 1.27 ms at 4K and 0.76 ms at 1080p, even with Cb and Cr drawn as two attachments of one pass. The GPU did a third less work than for the RGB texture, but a frame waited for alone gained nothing at 1080p.
  - Compute shaders that write the textures: WebGPU cannot write `R8Unorm` storage textures.
  - Cb and Cr interleaved in one `Rg8Unorm` texture, as NV12 and CoreVideo's biplanar formats have them. That is the same memory plus an interleave, a layout unlike upstream's, and no match in 4:4:4.
  - An object that holds three textures, with functions of its own. That gives the same three `id<MTLTexture>`s, with every texture function doubled.
  - Plane formats among `emberwave_texture_create`'s. A format would then also say a size, and Y and Cb would share a format in 4:4:4.
- Grok argued the other side, in two rounds.
  - First round, on the interface. It chose the same one, a texture per plane that remembers its plane, for the reasons above, and added three points:
    - The unorm round trip is exact.
    - 4:2:0 sizes are even, so chroma is exactly half.
    - Chroma siting belongs in the header.
  - Second round, a review of the copies. Three findings were confirmed and fixed:
    - The last level wrote zeros into the new padding, as far as its tiles reached. It now stops at the samples, and the padding test checks that the rest keeps its marker.
    - Direct3D 12's 512-byte offsets are needed in 4:2:0 too, and a failed allocation of wgpu's own buffer panics. Planes now start at multiples of 512 bytes.
    - The Metal read-back test covered 4:2:0 only. It now covers 4:4:4 too, whose planes only their contents tell apart.

**Consequence.** On Metal the textures equal `decode`'s planes byte for byte, in 4:2:0 and in 4:4:4 of an odd size. CI checks the same on lavapipe and WARP. `tests/metal.rs` reads them through Metal on a queue of its own and compares them with `emberwave_decoder_decode_yuv`. Measured on the Apple M3 Max through the C API, 300 frames and six rounds over two runs:

| Frame | Planes into textures | RGB into a texture | Planes in CPU memory | RGB in CPU memory |
|---|---|---|---|---|
| 4K | 1.02 to 1.07 ms | 1.47 to 1.49 ms | 1.19 to 1.20 ms | 2.54 to 2.72 ms |
| 1080p | 0.43 to 0.45 ms | 0.70 to 0.71 ms | 0.51 to 0.52 ms | 0.80 to 0.82 ms |

One round of the planes in CPU memory at 1080p took 0.81 ms, an outlier left out of the table.

The planes are the quickest way to the GPU at both sizes. The RGB texture's render pass probably costs as much as the planes' did, so an RGBA target whose rows are a multiple of 256 bytes could be filled by a copy too.

### 2026-10-24 — D42: The RGB picture into textures by copy

**Context.** D41 found that a render pass that only moves bytes into a texture cost about 0.3 ms more a frame than a copy on an Apple M3 Max. D40 puts the RGB picture into textures with such a pass, `draw.wgsl` (issue #70).

**Decision.**
- When the target texture has the `COPY_DST` usage and its rows are a multiple of 256 bytes, a width of a multiple of 64 pixels, the colour conversion's words are copied into it. 720p, 1080p, 1440p and 4K are such widths.
- For a BGRA texture `yuv2rgb.wgsl` writes blue in the low byte, a flag in its parameters, since a copy cannot swap channels. `Decoder::rgba_buffer` then holds BGRA words until the next decode, as its documentation says.
- Other textures keep the draw. `RENDER_ATTACHMENT` is still required of every target, so any target can fall back to the draw. (Later, D44: a target that takes a copy needs no render target.)
- `create_texture`, and with it `emberwave_texture_create`, adds `COPY_DST`.
- The browser player keeps drawing its canvas. WebGPU allows a canvas with `COPY_DST`, but wgpu 30's WebGPU backend reports `RENDER_ATTACHMENT` as a canvas's only usage. A player is not waiting on a frame the way a client in a stream is, and the change could not be tried in browsers here.
- Not done:
  - Padding the RGBA rows to 256 bytes for other widths: the read-back to the CPU and the draw take them packed, and the common widths need no padding.
  - Writing the texture from the colour conversion with a storage texture: WebGPU writes `rgba8unorm` storage textures but not `bgra8unorm` ones without an optional feature, and every target would need a bind group of its own.

**Consequence.** Through the C API on the Apple M3 Max, with a BGRA texture, 300 frames and three rounds: a 4K frame takes 1.31 to 1.38 ms instead of 1.46 to 1.52 ms, and a 1080p frame 0.56 to 0.57 ms instead of 0.71 to 0.72 ms. The planes of D41 stay faster, at 1.02 ms and 0.43 to 0.44 ms. `tests/gpu.rs` puts a 192-pixel-wide picture into RGBA and BGRA textures, with and without `COPY_DST`, and finds the RGB decode in each. With the BGRA flag broken, the test fails on the BGRA texture that takes copies. The same picture comes out either way, so a unit test pins which way each texture takes.

Grok reviewed the change and found no defect in the copy path: the flag and the way stay in step, and the CPU paths get RGB. It made three points:
- wgpu does not offer copies into a canvas. The browser change was dropped, as above.
- The documentation said an sRGB view fails validation. A copy does not go through the view, and the documentation now says so.
- The texture test cannot tell a copy from a draw. The unit test above now does.

### 2026-10-24 — D43: Decoding on the client's own Metal queue

**Context.** D40 to D42 decode into textures on a queue of emberwave's own and wait for the GPU, so that the client's command buffers on its queue read finished pictures. The wait holds the decoding thread about as long as the GPU works, 1.3 ms at 4K on an Apple M3 Max. Nothing orders the next decode after the client's reads either, so the client keeps three textures in turn. Upstream's Metal API records the decode into the client's own command buffer, where Metal orders it (issues #62 and #63).

**Decision.**
- `emberwave_context_create_metal(device, queue)` makes a context on the client's own `MTLDevice` and `MTLCommandQueue`:
  - wgpu's Metal adapter whose raw device is that device;
  - `wgpu::hal::metal::Device::device_from_raw` and `Queue::queue_from_raw`;
  - `Adapter::create_device_from_hal`;
  - in the core, `GpuContext::device_descriptor` and `from_device`.

  emberwave's command buffers then go onto the client's queue.
- `emberwave_decoder_submit_texture` and `_submit_plane_textures` submit the decode of the assembled frame and return without waiting.
  - In Rust they are `Decoder::submit_assembled_to_texture` and `_to_plane_textures`.
  - The frame goes up through the queue's staging memory. The decoder's own mapped upload buffer can take a frame only after the previous one is done (D34).
  - One poll that does not wait, after the submit, reclaims finished work and notices a lost device.
- Metal orders the work by itself. In wgpu 30, wgpu-hal makes its buffers and textures with Metal's default, tracked hazards. Untracked ones are a TODO in its code. Metal then orders the command buffers of one queue that use the same resource by when they were committed:
  - the client's draw committed after a decode reads its picture;
  - the next decode waits for that draw.

  So one texture takes every frame. `tests/metal.rs` checks that the textures are tracked, so a wgpu that changes this fails there.
- The submit functions refuse decoders of other contexts. There only a wait or a shared event would order the client's reads, and the event (`add_signal_event`) is queue-wide in wgpu, where two decoders could take each other's signal.
- The client's own textures, a drawable's or an IOSurface's, are left for later (#63). They are where an untracked heap texture could slip in.

**Consequence.** `tests/metal.rs` decodes every frame of a stream into the same textures without waiting. It commits a copy out of them to the queue after each decode, and waits only at the end; each copy finds its frame. This holds for RGB and planes, and for a picture copied and one drawn. `tests/gpu.rs` covers the submitted decodes on every backend, each followed by a read.

Measured from the call until a command buffer of the client's that reads the picture has completed, on the Apple M3 Max, 300 frames and three rounds:

| Frame | The call, waiting | The call, submitting | Until the read, waiting | Until the read, submitting |
|---|---|---|---|---|
| 4K | 1.29 to 1.38 ms | 0.26 to 0.27 ms | 1.44 to 1.59 ms | 1.44 to 1.45 ms |
| 1080p | 0.55 ms | 0.12 ms | 0.69 to 0.71 ms | 0.58 to 0.61 ms |

The decoding thread is free 1 ms sooner at 4K. The client's draw completes as early as after the waiting decode at 4K, and 0.1 ms earlier at 1080p.

Grok argued the other side before the work began. It agreed that tracked hazards order the work, and named their conditions: a direct use of the texture, commits after the call, and no heap. It also advised refusing the other contexts and leaving the client's textures for later.

Grok's review of the change found one defect, now fixed. A submitted decode counted its frame as decoded once recorded, before finishing the commands, which can still fail: on a view in a format the draw does not take, through the Rust API, or for want of memory. The frame was then used up although nothing ran. It now counts only once submitted, and `tests/gpu.rs` checks that such a view leaves the frame. A queue with room for fewer than four command buffers in flight can stall for good, since emberwave makes its two before committing them; the header says so.

### 2026-10-24 — D44: The client's own Metal textures

**Context.** D40 to D43 decode into textures that emberwave makes. A Metal client often has textures of its own to fill. One case is the BGRA texture of a `CVPixelBuffer`, made by a `CVMetalTextureCache`, whose pixel buffer goes to an `AVSampleBufferDisplayLayer`, as Moonlight on iOS shows its frames. Other cases are a texture from a pool of its own, or a drawable. Without a way in, such a client copies every picture once more (the rest of issue #63).

**Decision.**
- `emberwave_texture_from_metal` and `_from_metal_plane` wrap the client's `id<MTLTexture>`, with wgpu-hal's `metal::Device::texture_from_raw` and `Device::create_texture_from_hal`. The result is an `EmberwaveTexture` like those emberwave makes, so every decode takes it, waiting or not. emberwave holds a reference to the `MTLTexture` while the wrapper lives. The client keeps the object behind it, such as the `CVMetalTextureRef`, until the GPU has written it.
- Checked at import:
  - the device of the decoder's context;
  - 2D, of one level, layer and sample;
  - the picture's or the plane's size;
  - BGRA8 or RGBA8, or R8 for a plane;
  - not a view of another texture, not memoryless.
- What wgpu may do with the texture follows from the `MTLTexture`:
  - copies unless it is framebuffer-only;
  - draws with `MTLTextureUsageRenderTarget`;
  - sampling with `ShaderRead`.

  A texture of a `CVMetalTextureCache` is not framebuffer-only, so a blit may write it whatever usage the client asks for with `kCVMetalTextureUsage`. On macOS its default usage is read, write and render target. At the usual widths a copy fills it. A drawable is framebuffer-only and takes a draw. A texture that neither a copy nor a draw can fill is refused at import, as is a framebuffer-only plane, since planes are copied.
- In the core, a picture's target needs `RENDER_ATTACHMENT` only when a draw fills it. A target that takes a copy needs `COPY_DST` and rows of a multiple of 256 bytes, and nothing more.
- A decode that does not wait takes only textures whose hazards Metal tracks, as D43's ordering rests on them. A `CVMetalTextureCache`'s are tracked; a heap's are not by default. The waiting decodes take any, as the wait orders the client's reads.
- Not done: NV12, Cb and Cr in one `Rg8Unorm` plane, as `420v` pixel buffers have them. The BGRA picture and the three R8 planes are what the decoder gives.

**Consequence.** `tests/metal.rs` makes textures as a `CVMetalTextureCache` does: shared storage, and the narrowest usage a client can ask for, read only, or a render target where only a draw can fill them. It decodes into them, waiting and not, the picture at a width a copy takes and at one that needs the draw, and the planes. It reads them back through Metal and finds the decodes to CPU memory. It also checks what is refused:
- an untracked texture by a submit only;
- the wrong size or pixel format;
- a picture's texture that neither way can fill.

Grok argued the other side before the work. It kept the two functions and the usages read from the texture, and named the facts above about `CVMetalTextureCache` textures and drawables. It added the refusal of views, of memoryless textures and of framebuffer-only planes, and suggested leaving NV12 out.

Grok's review of the change found no defect. It traced the reference taken at import to wgpu's release of it, and the device that the wrapper keeps alive. It checked the declared usages against wgpu-core's checks of copies and draws, and the browser's canvas, which keeps drawing. It also confirmed that the tests fail without the choice of a copy, the refusal of a draw without its usage, or the check of tracking.

A review of the whole texture C API, before 0.11.0, read the four decisions together. It found no memory-safety or crash defect. It checked real `CVMetalTextureCache` textures on macOS with scratch programs:
- by default their usage is read, write, render target and pixel format views, and they are tracked and shared;
- they decode right at 192 and 200 pixels wide;
- 5000 rounds of import, decode and destruction did not grow the process.

It did find the contract text behind the code. Fixed:
- The ordering rules were stated three ways. `EmberwaveTexture`'s documentation now sets out every case, and the decodes point to it:
  - on emberwave's own queue, textures in turn;
  - on the client's queue with a tracked texture, one texture, for the decodes that wait as well;
  - an untracked texture only with the decodes that wait, in turn;
  - other readers of the memory, such as a display layer, only after the GPU is done.
- The README's pixel-buffer sample said that the submit could replace the waiting decode where it could not. The submit now has its own sample, which hands the buffer on in the completed handler of a command buffer that uses the texture. One that does not use it may run beside the decode and finish first.
- `emberwave_texture_from_metal_plane` takes the plane before the Metal texture, as `emberwave_texture_create_plane` does. No release had it yet.
- One Metal texture wrapped for two planes slipped past the check for three different textures, which compares wgpu's. The plane decodes now compare the Metal textures behind them.
- The documentation now covers the newer functions:
  - `emberwave_texture_destroy` and `emberwave_context_destroy` name the newer constructors;
  - the refusals of each decode are listed in full, and the statuses described as returned;
  - the `# Safety` sections of the functions that take a decoder ask that no other thread uses it.

### 2026-10-26 — D45: HEVC beside emberwave, in the benchmark app

**Context.** emberwave's times had nothing beside them on the same devices but upstream's decoder. A client would replace Moonlight's hardware HEVC decoder, so a reader asks how long that path takes on the same Mac and iPhone (issue #80). Apple's devices encode and decode HEVC through VideoToolbox. They decode AV1 too but do not encode it, and the Mac had no AV1 encoder, so AV1 waits for streams made elsewhere (#82).

**Decision.**
- The benchmark app times HEVC through VideoToolbox in the same run as emberwave, on the same pictures (`VideoToolbox.swift`). A command-line front-end, `EmberwaveBenchMac`, runs the same code on a Mac, linked with the XCFramework's library for the Mac (`scripts/bench_app_on_mac.sh`). The Mac and the phone thus share one harness, and both codecs one process.
- The device encodes the pictures in turn into one stream: a keyframe, then a frame for each one timed, each predicted from the one before, keyframes only where forced, no B-frames. The bit rate is an average. It is Moonlight's default for the size at 60 frames per second (moonlight-qt's `getDefaultBitrate`: 20, 40 and 80 Mbit/s), and, apart, emberwave's bytes per frame.
- The encoder runs at its fastest: real time off, power efficiency off, speed before quality. A prototype on the M3 Max tried other settings at 4K, frame after frame. Its runs were not kept:
  - Apple's settings for low latency, low-latency rate control with real time, chose another encoder, `hevc.rtvc`, which took 22 ms a frame;
  - real time at 60 frames per second took 20 ms, also with frames handed over 60 a second;
  - real time off took 12.4 ms.
- The prototype also tried a cap per frame through `DataRateLimits`, which was dropped. With one frame's bytes per 1/60 s, the encoder spent 43 % of the rate and still made a frame of 372 kB against 167 kB.
- Decoding takes the frame's bytes as a client receives them. They are wrapped without a copy in a block buffer and a sample buffer, with a format made once from the parameter sets. VideoToolbox decodes asynchronously, as Moonlight has it. The time ends in the output handler, once the picture is wrapped in Metal textures by a `CVMetalTextureCache`:
  - NV12 as an R8 and an RG8 texture, the counterpart of emberwave's Y, Cb and Cr textures;
  - BGRA as one texture, that of its RGB texture.

  In the prototype, a synchronous decode returned 0.35 ms after its handler on average, at most 5.6 ms, so its return would have counted against VideoToolbox. HEVC's means are those of its frames' own times, keyframes' too.
- Keyframes are not among the timed frames, as a stream has them only at its start and after a loss. The same keyframe decoded over again times them apart.
- Both codecs' decodes into textures are timed in two ways: frame after frame, as before, and as a stream's frames arrive, 60 a second (`FrameClock`).
  - Frames are due on a fixed schedule. A strict timer without leeway wakes the sleeping thread; a sleep woke it about 4 ms late, which made the first runs' cadence about 47 frames a second.
  - With `EMBERWAVE_BENCH_SPIN=1` the thread spins instead, which tells the GPU's part of a paced frame from the CPU's.
- One VideoToolbox session is open at a time. A frame that takes more than 10 s ends its session and fails the run.
- The luma PSNR of the last frames decoded, against the pictures, checks the stream, and a run fails below 30 dB. It is not a comparison of quality: the scene's grain, fresh in every picture, dominates it.
- A session that cannot be made or set up means a device without such an encoder or decoder. That is reported, and `-autorun` passes, as in a simulator or a virtual machine in CI. A failure after that fails the run.

**Consequence.** On the M3 Max, three runs (`BENCHMARKS.md`):
- Frame after frame, emberwave decodes into the planes 2.2 to 3.9 times as quickly as HEVC into NV12: 1.00 against 2.64 to 3.38 ms at 4K.
- At 60 frames a second the two take about as long: emberwave from 17 % less to 12 % more than HEVC, run by run. emberwave's decodes slow down 2.8 to 5.1 times, HEVC's 1.1 to 2.1 times.
- With the CPU kept awake, HEVC's paced decodes are no slower than frame after frame, and emberwave's still are, 2.4 to 3.4 times. The slowdown is the GPU's own, idle between frames. emberwave then takes 3 to 60 % longer than HEVC.
- HEVC's decoding grows with its bytes: 12.3 to 12.7 ms at 4K at emberwave's. Its keyframes take 5.6 to 7.8 ms.

So a client gains decoding time from emberwave only while its GPU stays busy. How long a client's decodes take while it draws every display refresh is issue #81.

Grok argued the other side twice:
- **Before the work.** It asked for both cadences for both codecs, the Metal wrap in the time, asynchronous decoding, one session at a time, keyframes apart and the encoder's identity in the report, all taken. Three of its points did not hold up when measured:
  - its explanation of the slow real-time encoder by the frames' time stamps failed the paced test;
  - a cap per frame did not make the encoder steadier;
  - a keyframe every few seconds is not what Moonlight's hosts send.
- **After the first runs.** It warned against claiming that an emberwave client is quicker, as no presenting client was timed, hence #81. It suspected that holding the decoded pictures slowed VideoToolbox frame after frame. A run without holding them was no quicker, 3.65 against 3.32 ms at 4K.

A review of the change found no defect of the harness's correctness:
- It checked the lifetimes of the frames' bytes against asynchronous decoding, and every path of the output handlers. Corrupt, truncated and reference-less frames fail cleanly.
- It checked the mapping of decoded pictures to their sources for the PSNR check.

It found that the paced cadence ran at about 47 frames a second. Its own A/B at an exact 60 changed the means by about 1 %. The runs above were made again with the fixed clock. The first runs had shown emberwave 11 to 20 % quicker than HEVC at 4K at that cadence, and these do not: paced times vary more from session to session than within one. Fixed besides:
- the keyframes' and the means' time basis;
- failures in setting up a session, all reported as an encoder or decoder missing;
- the waits, now bounded;
- frame sizes copied wrong into `BENCHMARKS.md`.

The figures of the prototype's runs above are marked as not kept.

### 2026-10-29 — D47: The README's figures, drawn from copied numbers

**Context.** The README opened with a line of text and a long status paragraph (#87). Charts of the decode and encode times and a figure of the pipeline show at a glance what emberwave does, and GitHub shows a README's images in its light or dark theme. Their numbers come from several sections of `BENCHMARKS.md`: some written by `scripts/bench.sh`, others by hand, about the iPhone, the browsers and HEVC.

**Decision.**
- `scripts/readme_figures.py` draws the banner and the figures as SVG, with the Python standard library only. Each chart is drawn for GitHub's light and dark themes and picked with `<picture>`. It has a background of its own, so that either file stays legible in the other theme. The colours are a two-series palette that a colour-vision checker passed against both themes' backgrounds.
- The numbers are copied into the script, each with the section it comes from, rather than parsed from `BENCHMARKS.md`, whose hand-written sections have no fixed format. The README's highlights and the figures' alt texts carry some of them too.
- CI checks that the committed SVG files are what the script draws. Unlike the tables (D21), running `scripts/bench.sh` again does not update the figures: after a run that changes their numbers, they are changed by hand in the script, the highlights and the alt texts.
- Each figure states its devices and dates. The decode chart also says that frames one after the other are not a stream's pace, and by how much the native decodes slowed down at 60 frames a second.

**Consequence.** The README shows the speed, the platforms and the pipeline before its first heading. The figures lag a new benchmark run until they are updated by hand.

