// Stage: the blocks of detail that a frame drawn in part lacks, after
// dequant.wgsl and before the inverse transform (decision D66). Upstream's
// decoder has no counterpart: its frames drawn in part have zeros there, as
// emberwave's have without concealment.
//
// For a frame that the decoder conceals, dequant.wgsl left those blocks as
// the last decode left them. Where the coarsest level around such a block
// changed by more than the threshold since the last decode, the picture
// moved or changed there, and the last frame's detail would be out of place:
// this pass zeroes the block. Elsewhere the picture is still, and the block
// keeps the last frame's detail. cpu::Concealer is the reference, to the
// last bit: dequantised coefficients are exact in f32 on both.
//
// One workgroup of 128 invocations per block, as in dequant.wgsl. The
// coarsest level around a block, one coefficient wider on each side, is at
// most 18x18 coefficients of each of its four bands. The invocations share it
// out and take the largest change with an atomic maximum of the changes'
// bits, which order as their values do for floats that are not negative.

struct BandInfo {
    first_block: u32,
    blocks_x: u32,
    width: u32,
    height: u32,
    dst_offset: u32,
    level: u32,
    component: u32,
    pad: u32,
}

struct DequantParams {
    band_count: u32,
    block_count: u32,
    grid_width: u32,
    conceal: u32,
    bands: array<BandInfo, 48>,
}

struct ConcealParams {
    // Per component, Y, Cb and Cr in x, y and z: where the coarsest level's
    // bands start, LL, HL, LH and HH one after the other, in the pyramid and
    // in the copy of the last decode's, in floats.
    pyramid_offset: vec4<u32>,
    last_offset: vec4<u32>,
    // The size of a band of the coarsest level.
    coarse_width: u32,
    coarse_height: u32,
    threshold: f32,
}

@group(0) @binding(0) var<storage, read> block_offsets: array<u32>;
@group(0) @binding(1) var<storage, read_write> pyramid: array<f32>;
@group(0) @binding(2) var<uniform> params: DequantParams;
@group(0) @binding(3) var<uniform> conceal: ConcealParams;
@group(0) @binding(4) var<storage, read> last: array<f32>;

// Marks a block that was not received (MISSING_BLOCK in the parser).
const MISSING_BLOCK: u32 = 0xffffffffu;

// Side of a coded block.
const BLOCK_SIZE: u32 = 32u;

// The coarsest decomposition level (DECOMPOSITION_LEVELS - 1).
const COARSEST_LEVEL: u32 = 4u;

// Workgroup memory is not zeroed for this pipeline (decision D25): each
// variable is written in full before the barrier that precedes its reads.
var<workgroup> shared_offset: u32;
var<workgroup> largest: atomic<u32>;
var<workgroup> moved: u32;

// Zeroes one coefficient, if it lies inside its band.
fn clear(band: BandInfo, x: u32, y: u32) {
    if (x < band.width && y < band.height) {
        pyramid[band.dst_offset + y * band.width + x] = 0.0;
    }
}

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let block_index = wg.y * params.grid_width + wg.x;
    if (block_index >= params.block_count || params.conceal != 1u) {
        return;
    }

    // The block's band, found as dequant.wgsl finds it.
    var band_slot = 0u;
    var step = 32u;
    loop {
        let probe = band_slot + step;
        if (probe < params.band_count && block_index >= params.bands[probe].first_block) {
            band_slot = probe;
        }
        if (step == 1u) {
            break;
        }
        step = step >> 1u;
    }
    let band = params.bands[band_slot];
    // The coarsest level's blocks are whole in a frame drawn in part.
    if (band.level >= COARSEST_LEVEL) {
        return;
    }
    if (tid == 0u) {
        shared_offset = block_offsets[block_index];
        atomicStore(&largest, 0u);
    }
    workgroupBarrier();
    if (workgroupUniformLoad(&shared_offset) != MISSING_BLOCK) {
        return;
    }
    let relative = block_index - band.first_block;
    let block_x = relative % band.blocks_x;
    let block_y = relative / band.blocks_x;

    // The coarsest level where the block lies, one coefficient around
    // included, as cpu::coarse_change takes it.
    let shift = COARSEST_LEVEL - band.level;
    let x0 = max((block_x * BLOCK_SIZE) >> shift, 1u) - 1u;
    let y0 = max((block_y * BLOCK_SIZE) >> shift, 1u) - 1u;
    let x1 = min((((block_x + 1u) * BLOCK_SIZE) >> shift) + 1u, conceal.coarse_width);
    let y1 = min((((block_y + 1u) * BLOCK_SIZE) >> shift) + 1u, conceal.coarse_height);
    // Never empty for a band of the frame's layout; were it, no region
    // rather than a subtraction that wraps.
    let width = select(0u, x1 - x0, x1 > x0);
    let per_band = width * select(0u, y1 - y0, y1 > y0);
    let band_floats = conceal.coarse_width * conceal.coarse_height;
    let current_base = conceal.pyramid_offset[band.component];
    let last_base = conceal.last_offset[band.component];
    var most = 0.0;
    for (var i = tid; i < 4u * per_band; i += 128u) {
        let at = (i / per_band) * band_floats
            + (y0 + (i % per_band) / width) * conceal.coarse_width
            + x0 + (i % per_band) % width;
        most = max(most, abs(pyramid[current_base + at] - last[last_base + at]));
    }
    atomicMax(&largest, bitcast<u32>(most));
    workgroupBarrier();
    if (tid == 0u) {
        moved = u32(bitcast<f32>(atomicLoad(&largest)) > conceal.threshold);
    }
    if (workgroupUniformLoad(&moved) == 0u) {
        return;
    }

    // Moved: zeros, as without concealment. Each invocation clears the 4x2
    // subblock that dequant.wgsl gives it.
    let block8 = tid >> 3u;
    let subblock = tid & 7u;
    let x = block_x * BLOCK_SIZE + 8u * (block8 & 3u) + 4u * (subblock >> 2u);
    let y = block_y * BLOCK_SIZE + 8u * (block8 >> 2u) + 2u * (subblock & 3u);
    for (var dx = 0u; dx < 4u; dx++) {
        clear(band, x + dx, y);
        clear(band, x + dx, y + 1u);
    }
}
