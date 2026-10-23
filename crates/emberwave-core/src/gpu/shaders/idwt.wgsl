// Stage: inverse CDF 9/7 wavelet transform, one decomposition level.
//
// Upstream counterpart: shaders/idwt.comp. This is a reimplementation of the
// transform described in bitstream.md ("Wavelet transform", "Inverse DC
// shift"). It does not use upstream's textureGather and mirrored sampler
// trick: interleaved coordinates are mirrored explicitly (whole-sample
// symmetric extension, which keeps the parity of an index) and the sample is
// then fetched from the band its parity selects.
//
// One dispatch combines the LL, HL, LH and HH bands of a level into the LL band
// of the next finer level (or into the final plane). A workgroup produces a
// 32x32 tile of the output: it loads the 40x40 interleaved neighbourhood
// (4 samples of extension on each side, one per lifting step), filters
// horizontally and then vertically in workgroup memory, and stores the tile.
// At the last level of a component the tile is stored as 8-bit samples,
// packed four to a word, which is what upstream's UNORM storage images hold;
// the float picture never goes to memory. Workgroup z selects the colour
// component.

struct CompParams {
    // Offsets of the LL, HL, LH and HH bands in the pyramid, in floats.
    band_offset: vec4<u32>,
    // Where the output goes: the LL band of the next level in the pyramid
    // (offset and row length in floats), or at the last level the packed
    // 8-bit plane (offset and row length in words).
    dst_offset: u32,
    dst_stride: u32,
    // Size of the plane, which the last level crops its output to.
    crop_width: u32,
    crop_height: u32,
    // 1: the last level, which writes the packed plane; 0: an earlier level.
    is_final: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct IdwtParams {
    band_width: u32,
    band_height: u32,
    out_width: u32,
    out_height: u32,
    comps: array<CompParams, 3>,
}

@group(0) @binding(0) var<storage, read_write> pyramid: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed: array<u32>;
@group(0) @binding(2) var<uniform> params: IdwtParams;

// Lifting step constants of the CDF 9/7 filter (ITU-T T.800, F.4.8.2).
const ALPHA: f32 = -1.586134342059924;
const BETA: f32 = -0.052980118572961;
const GAMMA: f32 = 0.882911075530934;
const DELTA: f32 = 0.443506852043971;
// Gain of the low-pass band; the high-pass band is scaled by its inverse.
const K: f32 = 1.230174104914001;
const INV_K: f32 = 0.8128930661159609;

// Output tile and the interleaved window that feeds it.
const TILE: u32 = 32u;
const WINDOW: u32 = 40u;
const APRON: i32 = 4;

// Shifts the DC level back (bitstream.md, "Inverse DC shift").
const DC_SHIFT: f32 = 0.5;

// Workgroup memory is not zeroed for this pipeline (decision D25): every
// element is written before it is read. The load fills all of `window`, the
// horizontal pass all of `horizontal`, and at the last level the vertical
// pass rewrites the 32x32 corner of `window` that the packing reads.
var<workgroup> window: array<f32, 1600>;
var<workgroup> horizontal: array<f32, 1280>;

// Whole-sample symmetric mirroring: -1 -> 1, n -> n - 2.
fn mirror(index: i32, n: i32) -> i32 {
    var m = index;
    if (m < 0) {
        m = -m;
    }
    if (m > n - 1) {
        m = 2 * (n - 1) - m;
    }
    return clamp(m, 0, n - 1);
}

// Inverse lifting of 16 interleaved samples whose first sample is at an even
// position. The gains are applied here. The 8 samples in the middle (4..12)
// are the valid result.
//
// This, and the loads and stores around it, is written out sample by sample
// on purpose. With loops over a variable index the compiler kept the array in
// memory instead of registers, and the whole transform took 4.5 times as long
// on an Apple GPU (ARCHITECTURE.md, decision D24).
fn lift(v: ptr<function, array<f32, 16>>) {
    (*v)[0] = (*v)[0] * K;
    (*v)[1] = (*v)[1] * INV_K;
    (*v)[2] = (*v)[2] * K;
    (*v)[3] = (*v)[3] * INV_K;
    (*v)[4] = (*v)[4] * K;
    (*v)[5] = (*v)[5] * INV_K;
    (*v)[6] = (*v)[6] * K;
    (*v)[7] = (*v)[7] * INV_K;
    (*v)[8] = (*v)[8] * K;
    (*v)[9] = (*v)[9] * INV_K;
    (*v)[10] = (*v)[10] * K;
    (*v)[11] = (*v)[11] * INV_K;
    (*v)[12] = (*v)[12] * K;
    (*v)[13] = (*v)[13] * INV_K;
    (*v)[14] = (*v)[14] * K;
    (*v)[15] = (*v)[15] * INV_K;
    (*v)[2] = (*v)[2] - DELTA * ((*v)[1] + (*v)[3]);
    (*v)[4] = (*v)[4] - DELTA * ((*v)[3] + (*v)[5]);
    (*v)[6] = (*v)[6] - DELTA * ((*v)[5] + (*v)[7]);
    (*v)[8] = (*v)[8] - DELTA * ((*v)[7] + (*v)[9]);
    (*v)[10] = (*v)[10] - DELTA * ((*v)[9] + (*v)[11]);
    (*v)[12] = (*v)[12] - DELTA * ((*v)[11] + (*v)[13]);
    (*v)[14] = (*v)[14] - DELTA * ((*v)[13] + (*v)[15]);
    (*v)[3] = (*v)[3] - GAMMA * ((*v)[2] + (*v)[4]);
    (*v)[5] = (*v)[5] - GAMMA * ((*v)[4] + (*v)[6]);
    (*v)[7] = (*v)[7] - GAMMA * ((*v)[6] + (*v)[8]);
    (*v)[9] = (*v)[9] - GAMMA * ((*v)[8] + (*v)[10]);
    (*v)[11] = (*v)[11] - GAMMA * ((*v)[10] + (*v)[12]);
    (*v)[13] = (*v)[13] - GAMMA * ((*v)[12] + (*v)[14]);
    (*v)[4] = (*v)[4] - BETA * ((*v)[3] + (*v)[5]);
    (*v)[6] = (*v)[6] - BETA * ((*v)[5] + (*v)[7]);
    (*v)[8] = (*v)[8] - BETA * ((*v)[7] + (*v)[9]);
    (*v)[10] = (*v)[10] - BETA * ((*v)[9] + (*v)[11]);
    (*v)[12] = (*v)[12] - BETA * ((*v)[11] + (*v)[13]);
    (*v)[5] = (*v)[5] - ALPHA * ((*v)[4] + (*v)[6]);
    (*v)[7] = (*v)[7] - ALPHA * ((*v)[6] + (*v)[8]);
    (*v)[9] = (*v)[9] - ALPHA * ((*v)[8] + (*v)[10]);
    (*v)[11] = (*v)[11] - ALPHA * ((*v)[10] + (*v)[12]);
}

// Stores one sample of the next level's LL band, if it lies inside the output.
fn store_ll(comp: CompParams, x: i32, y: i32, value: f32) {
    if (x < i32(params.out_width) && y < i32(params.out_height)) {
        pyramid[comp.dst_offset + u32(y) * comp.dst_stride + u32(x)] = value;
    }
}

// A final sample as an 8-bit level: the DC shift is added by the caller, the
// result clamped to [0, 1] and rounded like a UNORM store. WGSL's round()
// rounds ties to even on every backend. Kept as a float so that it can wait in
// the window array; it is a whole number from 0 to 255.
fn to_unorm8(v: f32) -> f32 {
    return round(clamp(v, 0.0, 1.0) * 255.0);
}

// The 8-bit level of tile sample `at` collected in the window array, or 0 for
// a padding sample past the right edge of the plane.
fn level_at(at: u32, x: u32, width: u32) -> u32 {
    return select(0u, u32(window[at]), x < width);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let comp = params.comps[wg.z];
    let out_width = i32(params.out_width);
    let out_height = i32(params.out_height);
    let tile_x = i32(wg.x * TILE);
    let tile_y = i32(wg.y * TILE);

    // Load the interleaved window. The low bit of a mirrored coordinate says
    // which band the sample comes from: LL, HL (odd column), LH (odd row), HH.
    for (var i = tid; i < WINDOW * WINDOW; i = i + 256u) {
        let row = i32(i / WINDOW);
        let col = i32(i % WINDOW);
        let x = mirror(tile_x + col - APRON, out_width);
        let y = mirror(tile_y + row - APRON, out_height);
        let band = u32(x & 1) | (u32(y & 1) << 1u);
        let index = comp.band_offset[band] + u32(y >> 1u) * params.band_width + u32(x >> 1u);
        window[i] = pyramid[index];
    }
    workgroupBarrier();

    // Horizontal pass: one invocation per segment of a window row. Window
    // columns 8s..8s+16 give the filtered columns 8s+4..8s+12.
    if (tid < 160u) {
        let row = tid / 4u;
        let segment = tid % 4u;
        var v: array<f32, 16>;
        let base = row * WINDOW + 8u * segment;
        v[0] = window[base + 0u];
        v[1] = window[base + 1u];
        v[2] = window[base + 2u];
        v[3] = window[base + 3u];
        v[4] = window[base + 4u];
        v[5] = window[base + 5u];
        v[6] = window[base + 6u];
        v[7] = window[base + 7u];
        v[8] = window[base + 8u];
        v[9] = window[base + 9u];
        v[10] = window[base + 10u];
        v[11] = window[base + 11u];
        v[12] = window[base + 12u];
        v[13] = window[base + 13u];
        v[14] = window[base + 14u];
        v[15] = window[base + 15u];
        lift(&v);
        let out = row * TILE + 8u * segment;
        horizontal[out + 0u] = v[4];
        horizontal[out + 1u] = v[5];
        horizontal[out + 2u] = v[6];
        horizontal[out + 3u] = v[7];
        horizontal[out + 4u] = v[8];
        horizontal[out + 5u] = v[9];
        horizontal[out + 6u] = v[10];
        horizontal[out + 7u] = v[11];
    }
    workgroupBarrier();

    // Vertical pass: one invocation per segment of a tile column. Earlier
    // levels store the LL band of the next level; the last level collects its
    // 8-bit samples for packing. One workgroup belongs to one component, so
    // the branch is the same for all its invocations.
    let is_final = comp.is_final != 0u;
    if (tid < 128u) {
        let column = tid % 32u;
        let segment = tid / 32u;
        var v: array<f32, 16>;
        let base = 8u * segment * TILE + column;
        v[0] = horizontal[base + 0u * TILE];
        v[1] = horizontal[base + 1u * TILE];
        v[2] = horizontal[base + 2u * TILE];
        v[3] = horizontal[base + 3u * TILE];
        v[4] = horizontal[base + 4u * TILE];
        v[5] = horizontal[base + 5u * TILE];
        v[6] = horizontal[base + 6u * TILE];
        v[7] = horizontal[base + 7u * TILE];
        v[8] = horizontal[base + 8u * TILE];
        v[9] = horizontal[base + 9u * TILE];
        v[10] = horizontal[base + 10u * TILE];
        v[11] = horizontal[base + 11u * TILE];
        v[12] = horizontal[base + 12u * TILE];
        v[13] = horizontal[base + 13u * TILE];
        v[14] = horizontal[base + 14u * TILE];
        v[15] = horizontal[base + 15u * TILE];
        lift(&v);

        if (is_final) {
            // The window array is free now: the horizontal pass is done.
            let at = 8u * segment * TILE + column;
            window[at + 0u * TILE] = to_unorm8(v[4] + DC_SHIFT);
            window[at + 1u * TILE] = to_unorm8(v[5] + DC_SHIFT);
            window[at + 2u * TILE] = to_unorm8(v[6] + DC_SHIFT);
            window[at + 3u * TILE] = to_unorm8(v[7] + DC_SHIFT);
            window[at + 4u * TILE] = to_unorm8(v[8] + DC_SHIFT);
            window[at + 5u * TILE] = to_unorm8(v[9] + DC_SHIFT);
            window[at + 6u * TILE] = to_unorm8(v[10] + DC_SHIFT);
            window[at + 7u * TILE] = to_unorm8(v[11] + DC_SHIFT);
        } else {
            let x = tile_x + i32(column);
            let y = tile_y + i32(8u * segment);
            store_ll(comp, x, y + 0, v[4]);
            store_ll(comp, x, y + 1, v[5]);
            store_ll(comp, x, y + 2, v[6]);
            store_ll(comp, x, y + 3, v[7]);
            store_ll(comp, x, y + 4, v[8]);
            store_ll(comp, x, y + 5, v[9]);
            store_ll(comp, x, y + 6, v[10]);
            store_ll(comp, x, y + 7, v[11]);
        }
    }

    // At the last level every invocation writes one word of the tile: four
    // samples of a row, the first in the low byte. Rows of a packed plane are
    // padded to whole words, and the padding samples are zero.
    //
    // The barrier is outside the branch on purpose: every level waits, though
    // only the last needs it. Inside `if (is_final)`, uniform as that is, it
    // crashed WARP, Direct3D's software rasteriser.
    workgroupBarrier();
    if (is_final) {
        let row = tid / 8u;
        let word = tid % 8u;
        let y = u32(tile_y) + row;
        let word_x = u32(tile_x) / 4u + word;
        if (y < comp.crop_height && word_x < comp.dst_stride) {
            let x = 4u * word_x;
            let at = row * TILE + 4u * word;
            let width = comp.crop_width;
            packed[comp.dst_offset + y * comp.dst_stride + word_x] = level_at(at, x, width) |
                (level_at(at + 1u, x + 1u, width) << 8u) |
                (level_at(at + 2u, x + 2u, width) << 16u) |
                (level_at(at + 3u, x + 3u, width) << 24u);
        }
    }
}
