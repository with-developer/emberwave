// Stage: quantisation, and what every block costs with 0 to 15 bit-planes
// dropped. Compiled with quantise.wgsl in front of it.
//
// Upstream counterpart: shaders/wavelet_quant.comp, which quantises the
// encoder's coefficients and measures each block for the rate control. This
// is a reimplementation of emberwave's CPU encoder instead (decision D28,
// src/encode/quant.rs and pack::block_words), step for step: given the same
// coefficients it finds the same scales, magnitudes and sizes, and adds the
// distortions in the same order, so that they differ from the CPU's only
// where the compiler fuses a multiply and an add (decision D29). The rate
// control (rate.wgsl) chooses from these tables.
//
// One workgroup of 128 invocations takes one 32x32 block, in three phases:
//
// 1. Each invocation quantises one 4x2 subblock (quantise.wgsl).
// 2. The invocations measure pairs of an 8x8 block and a number of dropped
//    planes: whether the block is still coded, its bytes of bit-planes and
//    sign bits, and its weighted squared error. Only drops up to one above
//    the highest bit of the 8x8 block's magnitudes are measured: with more,
//    nothing of it is left, and its error is the same sum of the same terms.
// 3. The first 16 invocations add up the 8x8 blocks for one number of dropped
//    planes each and write the block's size in words and its distortion.

// What the rate control knows of a block (rate.rs, BlockCost).
struct BlockCost {
    // Size in words with 0 to 15 planes dropped; 0 when nothing is left.
    words: array<u32, 16>,
    // Weighted squared error of the coded 8x8 blocks with 0 to 15 planes
    // dropped.
    distortion: array<f32, 16>,
    max_drop: u32,
}

@group(0) @binding(0) var<storage, read> pyramid: array<f32>;
@group(0) @binding(1) var<storage, read_write> costs: array<BlockCost>;
@group(0) @binding(2) var<uniform> params: QuantParams;

// Reconstruction point inside a quantiser step (bitstream.md,
// "Dequantization").
const DEADZONE_OFFSET: f32 = 0.5;
// Bytes of a block header, and of the control word and the qscale byte of
// each coded 8x8 block.
const HEADER_BYTES: u32 = 8u;
const BLOCK8_TABLE_BYTES: u32 = 3u;

// Workgroup memory is not zeroed for this pipeline (decision D25): phase 1
// writes every element of the per-coefficient, per-subblock and per-8x8
// arrays before the barrier that precedes their reads. Phase 2 writes the
// pairs that it measures, the first `block8_pairs[b]` of 8x8 block b, and
// phase 3 reads no others.
// |c| and the magnitude of every coefficient, 64 per 8x8 block in
// coefficient order.
var<workgroup> absolute: array<f32, 1024>;
var<workgroup> magnitudes: array<u32, 1024>;
var<workgroup> subblock_planes: array<u32, 128>;
// Per 8x8 block: the pairs phase 2 measures, none if nothing of it is coded,
// its base planes and its scale code.
var<workgroup> block8_pairs: array<u32, 16>;
var<workgroup> block8_base: array<u32, 16>;
var<workgroup> block8_scale_code: array<u32, 16>;
// Per pair of an 8x8 block and a number of dropped planes, at 16 * block +
// drop: bytes of bit-planes, sign bits and 1 << 16 if the block is still
// coded; and the weighted squared error.
var<workgroup> pair_sizes: array<u32, 256>;
var<workgroup> pair_distortion: array<f32, 256>;

// Phase 2 for one pair: 8x8 block `block8`, which is coded, with `drop`
// planes dropped.
fn measure(block8: u32, drop: u32, to_coefficient: f32, weight: f32) {
    // Coded while some subblock keeps a plane above the base.
    let kept_base = saturating_sub(block8_base[block8], drop);
    var plane_bytes = 0u;
    for (var s = 0u; s < 8u; s = s + 1u) {
        plane_bytes = plane_bytes + saturating_sub(subblock_planes[block8 * 8u + s], drop);
    }

    // Every value against its reconstruction, in coefficient order as the
    // CPU adds them. Steps of the scale times a power of two keep the
    // reconstruction exact, and it is within a factor of two of the value,
    // so the difference is exact too; only the sum rounds.
    let step = block8_scale(block8_scale_code[block8]) * to_coefficient;
    let drop_scale = exact_pow2(i32(drop)) * step;
    var signs = 0u;
    var error = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        let at = block8 * 64u + i;
        let kept = magnitudes[at] >> drop;
        var reconstructed = 0.0;
        if (kept != 0u) {
            signs = signs + 1u;
            reconstructed = (f32(kept) + DEADZONE_OFFSET) * drop_scale;
        }
        let difference = absolute[at] - reconstructed;
        error = error + difference * difference;
    }
    var sizes = 0u;
    if (plane_bytes > 8u * kept_base) {
        sizes = plane_bytes | (signs << 8u) | (1u << 16u);
    }
    let pair = block8 * DROP_STEPS + drop;
    pair_sizes[pair] = sizes;
    pair_distortion[pair] = weight * error;
}

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    // A grid of more than 65535 workgroups has some past the last block.
    // They quantise the last block again and write nothing, rather than
    // return: a return in front of the barriers below crashed WARP,
    // Direct3D's software adapter, though no workgroup took it.
    let workgroup = wg.y * params.grid_width + wg.x;
    let block_index = min(workgroup, params.block_count - 1u);
    let band = find_band(block_index);

    // Phase 1. |c * 2^b| * 2^-b is |c| exactly, as the CPU computes it.
    let s = quantise_subblock(tid, band, block_index);
    let to_coefficient = exact_pow2(-i32(band.exponent));
    let at = tid * 8u;
    absolute[at + 0u] = s.scaled[0] * to_coefficient;
    absolute[at + 1u] = s.scaled[1] * to_coefficient;
    absolute[at + 2u] = s.scaled[2] * to_coefficient;
    absolute[at + 3u] = s.scaled[3] * to_coefficient;
    absolute[at + 4u] = s.scaled[4] * to_coefficient;
    absolute[at + 5u] = s.scaled[5] * to_coefficient;
    absolute[at + 6u] = s.scaled[6] * to_coefficient;
    absolute[at + 7u] = s.scaled[7] * to_coefficient;
    magnitudes[at + 0u] = s.magnitudes[0];
    magnitudes[at + 1u] = s.magnitudes[1];
    magnitudes[at + 2u] = s.magnitudes[2];
    magnitudes[at + 3u] = s.magnitudes[3];
    magnitudes[at + 4u] = s.magnitudes[4];
    magnitudes[at + 5u] = s.magnitudes[5];
    magnitudes[at + 6u] = s.magnitudes[6];
    magnitudes[at + 7u] = s.magnitudes[7];
    subblock_planes[tid] = s.planes;
    if ((tid & 7u) == 0u) {
        let block8 = tid >> 3u;
        // Drops 0 to top + 1: from top + 1 on every magnitude is gone.
        block8_pairs[block8] = select(0u, min(s.top + 2u, DROP_STEPS), s.coded);
        block8_base[block8] = s.base;
        block8_scale_code[block8] = s.scale_code;
    }
    workgroupBarrier();

    // Phase 2: the pairs of all 8x8 blocks one after the other, shared out
    // in turn, so that neighbouring invocations mostly share their 8x8 block
    // and read the same workgroup memory.
    var pair_count = 0u;
    for (var b = 0u; b < 16u; b = b + 1u) {
        pair_count = pair_count + block8_pairs[b];
    }
    for (var pair = tid; pair < pair_count; pair = pair + 128u) {
        var block8 = 0u;
        var first = 0u;
        loop {
            if (pair < first + block8_pairs[block8]) {
                break;
            }
            first = first + block8_pairs[block8];
            block8 = block8 + 1u;
        }
        measure(block8, pair - first, to_coefficient, band.weight);
    }
    workgroupBarrier();

    // Phase 3: the 8x8 blocks in raster order, those coded with this many
    // planes dropped for the size, all coded ones for the distortion. With
    // more planes dropped than were measured, the last pair measured holds
    // the error, and the block is not coded.
    if (workgroup < params.block_count && tid < DROP_STEPS) {
        let drop = tid;
        var coded_count = 0u;
        var plane_bytes = 0u;
        var signs = 0u;
        var distortion = 0.0;
        for (var b = 0u; b < 16u; b = b + 1u) {
            let pairs = block8_pairs[b];
            if (pairs == 0u) {
                continue;
            }
            let pair = b * DROP_STEPS + min(drop, pairs - 1u);
            if (drop < pairs) {
                let sizes = pair_sizes[pair];
                if (sizes != 0u) {
                    coded_count = coded_count + 1u;
                    plane_bytes = plane_bytes + (sizes & 0xffu);
                    signs = signs + ((sizes >> 8u) & 0xffu);
                }
            }
            distortion = distortion + pair_distortion[pair];
        }
        var words = 0u;
        if (coded_count != 0u) {
            let bytes = HEADER_BYTES + coded_count * BLOCK8_TABLE_BYTES + plane_bytes + (signs + 7u) / 8u;
            words = (bytes + 3u) / 4u;
        }
        costs[block_index].words[drop] = words;
        costs[block_index].distortion[drop] = distortion;
        if (drop == 0u) {
            costs[block_index].max_drop = band.max_drop;
        }
    }
}
