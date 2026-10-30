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
// 32x32 tile of the output from its 40x40 interleaved neighbourhood (4 samples
// of extension on each side, one per lifting step): it lifts the rows of the
// neighbourhood, read straight from the bands, into workgroup memory, then the
// columns of the tile from there, and stores the tile.
// At the last level of a component the tile becomes 8-bit samples, which is
// what upstream's UNORM storage images hold; the float picture never goes to
// memory. Workgroup z selects the colour component.
//
// This file is the front of two shaders, which differ in where the last level
// stores its samples: `store_plane` of idwt_packed.wgsl packs them four to a
// word into a buffer, and that of idwt_planes.wgsl stores them straight into
// a texture per plane, where the adapter allows it (decision D52).

// The parameters of a level are shared with fdwt.wgsl, which goes the other
// way between the same places (gpu/layout.rs, TransformLevel).
struct CompParams {
    // Offsets of the LL, HL, LH and HH bands in the pyramid, in floats.
    band_offset: vec4<u32>,
    // Where the output goes, the level's image: the LL band of the next level
    // in the pyramid (offset and row length in floats), or at the last level
    // the packed 8-bit plane (offset and row length in words), if it is packed.
    image_offset: u32,
    image_stride: u32,
    // Size of the plane, which the last level crops its output to.
    plane_width: u32,
    plane_height: u32,
    // 1: the last level, which writes the plane; 0: an earlier level.
    is_plane: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct LevelParams {
    band_width: u32,
    band_height: u32,
    // Size of the output, twice that of the bands.
    image_width: u32,
    image_height: u32,
    comps: array<CompParams, 3>,
}

@group(0) @binding(0) var<storage, read_write> pyramid: array<f32>;
@group(0) @binding(2) var<uniform> params: LevelParams;

// Lifting step constants of the CDF 9/7 filter (ITU-T T.800, F.4.8.2).
const ALPHA: f32 = -1.586134342059924;
const BETA: f32 = -0.052980118572961;
const GAMMA: f32 = 0.882911075530934;
const DELTA: f32 = 0.443506852043971;
// Gain of the low-pass band; the high-pass band is scaled by its inverse.
const K: f32 = 1.230174104914001;
const INV_K: f32 = 0.8128930661159609;

// Output tile, and the extension of its neighbourhood on each side.
const TILE: u32 = 32u;
const APRON: i32 = 4;

// Shifts the DC level back (bitstream.md, "Inverse DC shift").
const DC_SHIFT: f32 = 0.5;

// Workgroup memory is not zeroed for this pipeline (decision D25): every
// element is written before it is read. The horizontal pass fills all of
// `horizontal`, and at the last level the vertical pass all of `levels`,
// which the store reads.
var<workgroup> horizontal: array<f32, 1280>;
// The last level's tile, as 8-bit levels.
var<workgroup> levels: array<f32, 1024>;

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
    if (x < i32(params.image_width) && y < i32(params.image_height)) {
        pyramid[comp.image_offset + u32(y) * comp.image_stride + u32(x)] = value;
    }
}

// A final sample as an 8-bit level: the DC shift is added by the caller, the
// result clamped to [0, 1] and rounded like a UNORM store. WGSL's round()
// rounds ties to even on every backend. Kept as a float so that it can wait in
// the levels array; it is a whole number from 0 to 255.
fn to_unorm8(v: f32) -> f32 {
    return round(clamp(v, 0.0, 1.0) * 255.0);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let comp = params.comps[wg.z];
    let out_width = i32(params.image_width);
    let out_height = i32(params.image_height);
    let tile_x = i32(wg.x * TILE);
    let tile_y = i32(wg.y * TILE);

    // Horizontal pass, straight from the bands: one invocation per segment of
    // a row of the neighbourhood, whose columns 8s..8s+16 give the filtered
    // columns 8s+4..8s+12. Each coordinate is mirrored about the level's size;
    // the low bit of a mirrored coordinate says which band the sample comes
    // from, LL, HL (odd column), LH (odd row) or HH, and mirroring keeps that
    // bit, so a row's even columns come from one band and its odd ones from
    // another. Loading the neighbourhood into workgroup memory first made the
    // transform take a quarter longer at 4K (decision D68).
    if (tid < 160u) {
        let row = tid / 4u;
        let segment = tid % 4u;
        let y = mirror(tile_y + i32(row) - APRON, out_height);
        let row_bands = u32(y & 1) << 1u;
        let band_row = u32(y >> 1u) * params.band_width;
        let even = comp.band_offset[row_bands] + band_row;
        let odd = comp.band_offset[row_bands | 1u] + band_row;
        let x0 = tile_x + i32(8u * segment) - APRON;
        var v: array<f32, 16>;
        v[0] = pyramid[even + u32(mirror(x0 + 0, out_width) >> 1u)];
        v[1] = pyramid[odd + u32(mirror(x0 + 1, out_width) >> 1u)];
        v[2] = pyramid[even + u32(mirror(x0 + 2, out_width) >> 1u)];
        v[3] = pyramid[odd + u32(mirror(x0 + 3, out_width) >> 1u)];
        v[4] = pyramid[even + u32(mirror(x0 + 4, out_width) >> 1u)];
        v[5] = pyramid[odd + u32(mirror(x0 + 5, out_width) >> 1u)];
        v[6] = pyramid[even + u32(mirror(x0 + 6, out_width) >> 1u)];
        v[7] = pyramid[odd + u32(mirror(x0 + 7, out_width) >> 1u)];
        v[8] = pyramid[even + u32(mirror(x0 + 8, out_width) >> 1u)];
        v[9] = pyramid[odd + u32(mirror(x0 + 9, out_width) >> 1u)];
        v[10] = pyramid[even + u32(mirror(x0 + 10, out_width) >> 1u)];
        v[11] = pyramid[odd + u32(mirror(x0 + 11, out_width) >> 1u)];
        v[12] = pyramid[even + u32(mirror(x0 + 12, out_width) >> 1u)];
        v[13] = pyramid[odd + u32(mirror(x0 + 13, out_width) >> 1u)];
        v[14] = pyramid[even + u32(mirror(x0 + 14, out_width) >> 1u)];
        v[15] = pyramid[odd + u32(mirror(x0 + 15, out_width) >> 1u)];
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
    let is_final = comp.is_plane != 0u;
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
            let at = 8u * segment * TILE + column;
            levels[at + 0u * TILE] = to_unorm8(v[4] + DC_SHIFT);
            levels[at + 1u * TILE] = to_unorm8(v[5] + DC_SHIFT);
            levels[at + 2u * TILE] = to_unorm8(v[6] + DC_SHIFT);
            levels[at + 3u * TILE] = to_unorm8(v[7] + DC_SHIFT);
            levels[at + 4u * TILE] = to_unorm8(v[8] + DC_SHIFT);
            levels[at + 5u * TILE] = to_unorm8(v[9] + DC_SHIFT);
            levels[at + 6u * TILE] = to_unorm8(v[10] + DC_SHIFT);
            levels[at + 7u * TILE] = to_unorm8(v[11] + DC_SHIFT);
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

    // At the last level every invocation stores four samples of a row of the
    // tile, from the levels array, in the file behind this one.
    //
    // The barrier is outside the branch on purpose: every level waits, though
    // only the last needs it. Inside `if (is_final)`, uniform as that is, it
    // crashed WARP, Direct3D's software rasteriser.
    workgroupBarrier();
    if (is_final) {
        store_plane(comp, wg.z, u32(tile_x), u32(tile_y), tid);
    }
}
