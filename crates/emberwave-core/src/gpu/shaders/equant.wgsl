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
// 1. Each invocation quantises one 4x2 subblock (quantise.wgsl) and, for
//    each number of dropped planes, adds up the squared errors of its 8
//    values and counts those that keep a magnitude, in registers. Only drops
//    up to one above the highest bit of the 8x8 block's magnitudes are
//    measured: with more, nothing of it is left, and its error is the same
//    sum of the same terms.
// 2. The invocations take the pairs of an 8x8 block and a number of dropped
//    planes and add up the 8 subblocks: whether the block is still coded,
//    its bytes of bit-planes and sign bits, and its weighted squared error.
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
// writes every element of the per-subblock and per-8x8 arrays, and of
// `subblock_error` the drops it measures, before the barrier that precedes
// their reads; phase 2 reads no others. Phase 2 writes the pairs that it
// measures, the first `block8_pairs[b]` of 8x8 block b, and phase 3 reads no
// others.
// Per subblock and number of dropped planes, at 16 * subblock + drop: the
// squared error of its 8 values.
var<workgroup> subblock_error: array<f32, 2048>;
// Per subblock, two words of eight 4-bit counts: for each number of dropped
// planes, its values that keep a magnitude and so a sign bit.
var<workgroup> subblock_signs: array<u32, 256>;
var<workgroup> subblock_planes: array<u32, 128>;
// Per 8x8 block: the pairs measured, none if nothing of it is coded, and
// its base planes.
var<workgroup> block8_pairs: array<u32, 16>;
var<workgroup> block8_base: array<u32, 16>;
// Per pair of an 8x8 block and a number of dropped planes, at 16 * block +
// drop: bytes of bit-planes, sign bits and 1 << 16 if the block is still
// coded; and the weighted squared error.
var<workgroup> pair_sizes: array<u32, 256>;
var<workgroup> pair_distortion: array<f32, 256>;

// The squared error of a value |c| reconstructed from a magnitude of which
// `kept` remains, with steps of `drop_scale`. A step of the scale times a
// power of two keeps the reconstruction exact, and it is within a factor of
// two of the value, so the difference is exact too; only the sum rounds.
fn squared_error(absolute: f32, kept: u32, drop_scale: f32) -> f32 {
    var reconstructed = 0.0;
    if (kept != 0u) {
        reconstructed = (f32(kept) + DEADZONE_OFFSET) * drop_scale;
    }
    let difference = absolute - reconstructed;
    return difference * difference;
}

// Phase 2 for one pair: 8x8 block `block8`, which is coded, with `drop`
// planes dropped. The subblocks' errors are added in their order, as the CPU
// adds them (quant.rs).
fn measure(block8: u32, drop: u32, weight: f32) {
    // Coded while some subblock keeps a plane above the base.
    let kept_base = saturating_sub(block8_base[block8], drop);
    var plane_bytes = 0u;
    var signs = 0u;
    var error = 0.0;
    for (var s = 0u; s < 8u; s = s + 1u) {
        let subblock = block8 * 8u + s;
        plane_bytes = plane_bytes + saturating_sub(subblock_planes[subblock], drop);
        signs = signs + ((subblock_signs[2u * subblock + drop / 8u] >> (4u * (drop % 8u))) & 0xfu);
        error = error + subblock_error[subblock * DROP_STEPS + drop];
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
    var absolute: array<f32, 8>;
    absolute[0] = s.scaled[0] * to_coefficient;
    absolute[1] = s.scaled[1] * to_coefficient;
    absolute[2] = s.scaled[2] * to_coefficient;
    absolute[3] = s.scaled[3] * to_coefficient;
    absolute[4] = s.scaled[4] * to_coefficient;
    absolute[5] = s.scaled[5] * to_coefficient;
    absolute[6] = s.scaled[6] * to_coefficient;
    absolute[7] = s.scaled[7] * to_coefficient;
    // Drops 0 to top + 1: from top + 1 on every magnitude is gone. The same
    // for the 8 subblocks of an 8x8 block.
    let pairs = select(0u, min(s.top + 2u, DROP_STEPS), s.coded);
    let step = block8_scale(s.scale_code) * to_coefficient;
    var signs_low = 0u;
    var signs_high = 0u;
    for (var drop = 0u; drop < pairs; drop = drop + 1u) {
        let drop_scale = exact_pow2(i32(drop)) * step;
        // The values in order, written out to stay in registers (D24).
        var error = 0.0;
        error = error + squared_error(absolute[0], s.magnitudes[0] >> drop, drop_scale);
        error = error + squared_error(absolute[1], s.magnitudes[1] >> drop, drop_scale);
        error = error + squared_error(absolute[2], s.magnitudes[2] >> drop, drop_scale);
        error = error + squared_error(absolute[3], s.magnitudes[3] >> drop, drop_scale);
        error = error + squared_error(absolute[4], s.magnitudes[4] >> drop, drop_scale);
        error = error + squared_error(absolute[5], s.magnitudes[5] >> drop, drop_scale);
        error = error + squared_error(absolute[6], s.magnitudes[6] >> drop, drop_scale);
        error = error + squared_error(absolute[7], s.magnitudes[7] >> drop, drop_scale);
        subblock_error[tid * DROP_STEPS + drop] = error;
        let kept = select(0u, 1u, (s.magnitudes[0] >> drop) != 0u) + select(0u, 1u, (s.magnitudes[1] >> drop) != 0u) +
            select(0u, 1u, (s.magnitudes[2] >> drop) != 0u) + select(0u, 1u, (s.magnitudes[3] >> drop) != 0u) +
            select(0u, 1u, (s.magnitudes[4] >> drop) != 0u) + select(0u, 1u, (s.magnitudes[5] >> drop) != 0u) +
            select(0u, 1u, (s.magnitudes[6] >> drop) != 0u) + select(0u, 1u, (s.magnitudes[7] >> drop) != 0u);
        if (drop < 8u) {
            signs_low = signs_low | (kept << (4u * drop));
        } else {
            signs_high = signs_high | (kept << (4u * (drop - 8u)));
        }
    }
    subblock_signs[2u * tid] = signs_low;
    subblock_signs[2u * tid + 1u] = signs_high;
    subblock_planes[tid] = s.planes;
    if ((tid & 7u) == 0u) {
        let block8 = tid >> 3u;
        block8_pairs[block8] = pairs;
        block8_base[block8] = s.base;
    }
    workgroupBarrier();

    // Phase 2: the pairs of all 8x8 blocks one after the other, shared out
    // in turn, so that neighbouring invocations mostly share their 8x8 block
    // and read nearby workgroup memory.
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
        measure(block8, pair - first, band.weight);
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
        // A block a partial decode needs is sent even with nothing in it:
        // its header and one 8x8 block of no bit-planes (pack.wgsl).
        if (block_index < params.pristine_blocks) {
            words = max(words, EMPTY_BLOCK_WORDS);
        }
        costs[block_index].words[drop] = words;
        costs[block_index].distortion[drop] = distortion;
        if (drop == 0u) {
            costs[block_index].max_drop = band.max_drop;
        }
    }
}
