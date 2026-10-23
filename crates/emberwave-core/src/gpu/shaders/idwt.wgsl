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
// Workgroup z selects the colour component.

struct CompParams {
    // Offsets of the LL, HL, LH and HH bands in the pyramid, in floats.
    band_offset: vec4<u32>,
    dst_offset: u32,
    dst_stride: u32,
    crop_width: u32,
    crop_height: u32,
    // 1: write the DC-shifted, clamped final plane; 0: write the LL band of the next level.
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
@group(0) @binding(1) var<storage, read_write> planes: array<f32>;
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
fn lift(v: ptr<function, array<f32, 16>>) {
    for (var i = 0u; i < 16u; i = i + 1u) {
        var gain = INV_K;
        if ((i & 1u) == 0u) {
            gain = K;
        }
        (*v)[i] = (*v)[i] * gain;
    }
    for (var i = 2u; i < 15u; i = i + 2u) {
        (*v)[i] = (*v)[i] - DELTA * ((*v)[i - 1u] + (*v)[i + 1u]);
    }
    for (var i = 3u; i < 14u; i = i + 2u) {
        (*v)[i] = (*v)[i] - GAMMA * ((*v)[i - 1u] + (*v)[i + 1u]);
    }
    for (var i = 4u; i < 13u; i = i + 2u) {
        (*v)[i] = (*v)[i] - BETA * ((*v)[i - 1u] + (*v)[i + 1u]);
    }
    for (var i = 5u; i < 12u; i = i + 2u) {
        (*v)[i] = (*v)[i] - ALPHA * ((*v)[i - 1u] + (*v)[i + 1u]);
    }
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
        for (var i = 0u; i < 16u; i = i + 1u) {
            v[i] = window[row * WINDOW + 8u * segment + i];
        }
        lift(&v);
        for (var i = 0u; i < 8u; i = i + 1u) {
            horizontal[row * TILE + 8u * segment + i] = v[4u + i];
        }
    }
    workgroupBarrier();

    // Vertical pass: one invocation per segment of a tile column.
    if (tid < 128u) {
        let column = tid % 32u;
        let segment = tid / 32u;
        var v: array<f32, 16>;
        for (var i = 0u; i < 16u; i = i + 1u) {
            v[i] = horizontal[(8u * segment + i) * TILE + column];
        }
        lift(&v);

        let x = tile_x + i32(column);
        for (var i = 0u; i < 8u; i = i + 1u) {
            let y = tile_y + i32(8u * segment + i);
            if (x >= out_width || y >= out_height) {
                continue;
            }
            let value = v[4u + i];
            if (comp.is_final != 0u) {
                if (u32(x) < comp.crop_width && u32(y) < comp.crop_height) {
                    planes[comp.dst_offset + u32(y) * comp.dst_stride + u32(x)] =
                        clamp(value + DC_SHIFT, 0.0, 1.0);
                }
            } else {
                pyramid[comp.dst_offset + u32(y) * comp.dst_stride + u32(x)] = value;
            }
        }
    }
}
