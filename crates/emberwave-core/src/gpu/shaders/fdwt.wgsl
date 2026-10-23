// Stage: forward CDF 9/7 wavelet transform, one decomposition level.
//
// Upstream counterpart: shaders/dwt.comp, the first stage of the PyroWave
// encoder. This is a reimplementation of the transform in bitstream.md
// ("Wavelet transform") as emberwave's CPU encoder does it
// (src/encode/fdwt.rs), in the shape of idwt.wgsl: a workgroup takes a 32x32
// tile of the level's image with 4 samples of whole-sample symmetric extension
// on each side (a 40x40 window), lifts its rows and then its columns in
// registers, and writes the 16x16 corner of each band the tile gives. At a
// component's finest level the image is the 8-bit plane, extended to the
// aligned size by repeating its last column and row, at the other levels the
// LL band the level before wrote. Workgroup z selects the colour component.
//
// The expressions are those of the CPU encoder, in the same order, but the
// compiler may fuse a multiply and an add, so the coefficients can differ from
// the CPU's in the last place (ARCHITECTURE.md, decision D29).

// The parameters of a level are shared with idwt.wgsl, which goes the other
// way between the same places (gpu/layout.rs, TransformLevel).
struct CompParams {
    // Offsets of the LL, HL, LH and HH bands in the pyramid, in floats.
    band_offset: vec4<u32>,
    // Where the input comes from, the level's image: the LL band of the next
    // finer level in the pyramid (offset and row length in floats), or at the
    // finest level the packed 8-bit plane (offset and row length in words).
    image_offset: u32,
    image_stride: u32,
    // Size of the plane, whose last column and row are repeated up to the
    // image size at the finest level.
    plane_width: u32,
    plane_height: u32,
    // 1: the finest level, which reads the packed plane; 0: a later level.
    is_plane: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct LevelParams {
    band_width: u32,
    band_height: u32,
    // Size of the input, twice that of the bands.
    image_width: u32,
    image_height: u32,
    comps: array<CompParams, 3>,
}

@group(0) @binding(0) var<storage, read_write> pyramid: array<f32>;
@group(0) @binding(1) var<storage, read> planes: array<u32>;
@group(0) @binding(2) var<uniform> params: LevelParams;
// The input sample of each 8-bit level, v / 255 - 0.5, as the CPU encoder
// computes it: GPUs need not divide exactly.
@group(0) @binding(3) var<storage, read> sample_levels: array<f32, 256>;

// Lifting step constants of the CDF 9/7 filter (ITU-T T.800, F.4.8.2).
const ALPHA: f32 = -1.586134342059924;
const BETA: f32 = -0.052980118572961;
const GAMMA: f32 = 0.882911075530934;
const DELTA: f32 = 0.443506852043971;
// Gain of the low-pass band; the high-pass band is scaled by its inverse.
const K: f32 = 1.230174104914001;
const INV_K: f32 = 0.8128930661159609;

// Image tile and the window of it, with its extension, that feeds a workgroup.
const TILE: u32 = 32u;
const WINDOW: u32 = 40u;
const APRON: i32 = 4;

// Workgroup memory is not zeroed for this pipeline (decision D25): the load
// fills all of `window` and the horizontal pass all of `horizontal` before
// either is read.
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

// Forward lifting of 16 samples whose first is at an even position, the
// inverse of idwt.wgsl's: the 8 samples in the middle (4..12) come out valid,
// low-pass (even) times 1/K and high-pass (odd) times K. Written out with
// constant indices so that the samples stay in registers (decision D24).
fn lift(v: ptr<function, array<f32, 16>>) {
    (*v)[1] = (*v)[1] + ALPHA * ((*v)[0] + (*v)[2]);
    (*v)[3] = (*v)[3] + ALPHA * ((*v)[2] + (*v)[4]);
    (*v)[5] = (*v)[5] + ALPHA * ((*v)[4] + (*v)[6]);
    (*v)[7] = (*v)[7] + ALPHA * ((*v)[6] + (*v)[8]);
    (*v)[9] = (*v)[9] + ALPHA * ((*v)[8] + (*v)[10]);
    (*v)[11] = (*v)[11] + ALPHA * ((*v)[10] + (*v)[12]);
    (*v)[13] = (*v)[13] + ALPHA * ((*v)[12] + (*v)[14]);
    (*v)[2] = (*v)[2] + BETA * ((*v)[1] + (*v)[3]);
    (*v)[4] = (*v)[4] + BETA * ((*v)[3] + (*v)[5]);
    (*v)[6] = (*v)[6] + BETA * ((*v)[5] + (*v)[7]);
    (*v)[8] = (*v)[8] + BETA * ((*v)[7] + (*v)[9]);
    (*v)[10] = (*v)[10] + BETA * ((*v)[9] + (*v)[11]);
    (*v)[12] = (*v)[12] + BETA * ((*v)[11] + (*v)[13]);
    (*v)[3] = (*v)[3] + GAMMA * ((*v)[2] + (*v)[4]);
    (*v)[5] = (*v)[5] + GAMMA * ((*v)[4] + (*v)[6]);
    (*v)[7] = (*v)[7] + GAMMA * ((*v)[6] + (*v)[8]);
    (*v)[9] = (*v)[9] + GAMMA * ((*v)[8] + (*v)[10]);
    (*v)[11] = (*v)[11] + GAMMA * ((*v)[10] + (*v)[12]);
    (*v)[4] = (*v)[4] + DELTA * ((*v)[3] + (*v)[5]);
    (*v)[6] = (*v)[6] + DELTA * ((*v)[5] + (*v)[7]);
    (*v)[8] = (*v)[8] + DELTA * ((*v)[7] + (*v)[9]);
    (*v)[10] = (*v)[10] + DELTA * ((*v)[9] + (*v)[11]);
    (*v)[4] = (*v)[4] * INV_K;
    (*v)[5] = (*v)[5] * K;
    (*v)[6] = (*v)[6] * INV_K;
    (*v)[7] = (*v)[7] * K;
    (*v)[8] = (*v)[8] * INV_K;
    (*v)[9] = (*v)[9] * K;
    (*v)[10] = (*v)[10] * INV_K;
    (*v)[11] = (*v)[11] * K;
}

// Sample (x, y) of the level's image, both inside it.
fn image_at(comp: CompParams, x: u32, y: u32) -> f32 {
    if (comp.is_plane != 0u) {
        let px = min(x, comp.plane_width - 1u);
        let py = min(y, comp.plane_height - 1u);
        let word = planes[comp.image_offset + py * comp.image_stride + px / 4u];
        return sample_levels[(word >> (8u * (px % 4u))) & 0xffu];
    }
    return pyramid[comp.image_offset + y * comp.image_stride + x];
}

// Stores tile sample (column, row) of the transformed tile in the band its
// parities select: LL, HL (odd column), LH (odd row) or HH.
fn store_band(comp: CompParams, tile_x: u32, tile_y: u32, column: u32, row: u32, value: f32) {
    let band = (column & 1u) | ((row & 1u) << 1u);
    let x = (tile_x + column) / 2u;
    let y = (tile_y + row) / 2u;
    if (x < params.band_width && y < params.band_height) {
        pyramid[comp.band_offset[band] + y * params.band_width + x] = value;
    }
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let comp = params.comps[wg.z];
    let image_width = i32(params.image_width);
    let image_height = i32(params.image_height);
    let tile_x = wg.x * TILE;
    let tile_y = wg.y * TILE;

    for (var i = tid; i < WINDOW * WINDOW; i = i + 256u) {
        let row = i32(i / WINDOW);
        let col = i32(i % WINDOW);
        let x = mirror(i32(tile_x) + col - APRON, image_width);
        let y = mirror(i32(tile_y) + row - APRON, image_height);
        window[i] = image_at(comp, u32(x), u32(y));
    }
    workgroupBarrier();

    // Horizontal pass: one invocation per segment of a window row. Window
    // columns 8s..8s+16 give the tile columns 8s..8s+8.
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

    // Vertical pass: one invocation per segment of a tile column. Rows
    // 8s..8s+16 of the horizontal pass give the tile rows 8s..8s+8.
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
        let row = 8u * segment;
        store_band(comp, tile_x, tile_y, column, row + 0u, v[4]);
        store_band(comp, tile_x, tile_y, column, row + 1u, v[5]);
        store_band(comp, tile_x, tile_y, column, row + 2u, v[6]);
        store_band(comp, tile_x, tile_y, column, row + 3u, v[7]);
        store_band(comp, tile_x, tile_y, column, row + 4u, v[8]);
        store_band(comp, tile_x, tile_y, column, row + 5u, v[9]);
        store_band(comp, tile_x, tile_y, column, row + 6u, v[10]);
        store_band(comp, tile_x, tile_y, column, row + 7u, v[11]);
    }
}
