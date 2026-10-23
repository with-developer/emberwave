// Stage: coefficient decoding and dequantisation.
//
// Upstream counterpart: shaders/wavelet_dequant.comp (the first GPU stage of the
// PyroWave decoder). This is a reimplementation of the algorithm in
// bitstream.md ("Decoding 8x8 blocks", "Dequantization") without the Vulkan
// features upstream needs (subgroup scans, 8/16-bit storage): the prefix sums
// use workgroup memory and the payload is read as u32 words (ARCHITECTURE.md,
// decision D4).
//
// One workgroup of 128 invocations decodes one 32x32 block, so the whole stage
// is a single dispatch with one workgroup per block index. Invocation
// `tid` = 8 * (8x8 block, row-major) + (4x2 subblock) decodes the 8 values of
// its subblock. Every block of every band is written, with zeros for blocks
// that were not received, so the pyramid never needs clearing.

struct BandInfo {
    first_block: u32,
    blocks_x: u32,
    width: u32,
    height: u32,
    dst_offset: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct DequantParams {
    band_count: u32,
    block_count: u32,
    grid_width: u32,
    pad: u32,
    // Transmitted bands in block index order.
    bands: array<BandInfo, 48>,
}

@group(0) @binding(0) var<storage, read> payload: array<u32>;
@group(0) @binding(1) var<storage, read> block_offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> pyramid: array<f32>;
@group(0) @binding(3) var<uniform> params: DequantParams;

// Marks a block that was not received (MISSING_BLOCK in the parser).
const MISSING_BLOCK: u32 = 0xffffffffu;

// Side of a coded block and of the 8x8 blocks inside it.
const BLOCK_SIZE: u32 = 32u;

var<workgroup> shared_offset: u32;
var<workgroup> control_words: array<u32, 16>;
var<workgroup> qscales: array<u32, 16>;
var<workgroup> plane_costs: array<u32, 16>;
var<workgroup> sign_counts: array<u32, 128>;
var<workgroup> block_sign_totals: array<u32, 16>;

fn byte_at(byte_offset: u32) -> u32 {
    return (payload[byte_offset >> 2u] >> ((byte_offset & 3u) * 8u)) & 0xffu;
}

// `byte_offset` must be even.
fn half_at(byte_offset: u32) -> u32 {
    return (payload[byte_offset >> 2u] >> ((byte_offset & 2u) * 8u)) & 0xffffu;
}

// Block32x32Scale: (8 + m) * 2^(1 - (quant_code >> 3)), built from the exponent
// bits so that it is exact, like the reference formula.
fn block_scale(quant_code: u32) -> f32 {
    let exponent_bits = (128u - (quant_code >> 3u)) << 23u;
    return f32(8u + (quant_code & 7u)) * bitcast<f32>(exponent_bits);
}

// Block8x8Scale: code / 8 + 0.25.
fn block8_scale(code: u32) -> f32 {
    return f32(code) * 0.125 + 0.25;
}

// (m + 0.5) * scale32 * scale8, in the same order as the CPU decoder, negated
// if the sign bit is set.
fn signed_value(magnitude: u32, scale: f32, negative: u32) -> f32 {
    let value = (f32(magnitude) + 0.5) * scale;
    return select(value, -value, negative != 0u);
}

// Stores one coefficient, if it lies inside its band.
fn store(band: BandInfo, x: u32, y: u32, value: f32) {
    if (x < band.width && y < band.height) {
        pyramid[band.dst_offset + y * band.width + x] = value;
    }
}

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let block_index = wg.y * params.grid_width + wg.x;
    if (block_index >= params.block_count) {
        return;
    }

    // Find the band this block belongs to. The loop bounds are uniform.
    var band_slot = 0u;
    for (var i = 1u; i < params.band_count; i = i + 1u) {
        if (block_index >= params.bands[i].first_block) {
            band_slot = i;
        }
    }
    let band = params.bands[band_slot];
    let relative = block_index - band.first_block;
    let origin_x = (relative % band.blocks_x) * BLOCK_SIZE;
    let origin_y = (relative / band.blocks_x) * BLOCK_SIZE;

    // The offset is loaded by one invocation and shared through
    // workgroupUniformLoad, which makes it uniform: the early return below
    // then leaves the later barriers legal.
    if (tid == 0u) {
        shared_offset = block_offsets[block_index];
    }
    workgroupBarrier();
    let offset_words = workgroupUniformLoad(&shared_offset);

    // This invocation's 4x2 subblock: raster index of its 8x8 block and
    // subblock number inside it.
    let block8 = tid >> 3u;
    let subblock = tid & 7u;
    let x0 = origin_x + 8u * (block8 & 3u) + 4u * (subblock >> 2u);
    let y0 = origin_y + 8u * (block8 >> 2u) + 2u * (subblock & 3u);

    var values: array<f32, 8>;

    if (offset_words != MISSING_BLOCK) {
        let ballot = payload[offset_words] & 0xffffu;
        let quant_code = payload[offset_words + 1u] & 0xffu;
        let active_count = countOneBits(ballot);
        let base = offset_words * 4u;
        let tables_end = 8u + 3u * active_count;

        // Phase A: the first 16 invocations read the tables of one 8x8 block
        // each and work out how many bytes of bit-planes it has.
        if (tid < 16u) {
            var control = 0u;
            var qscale = 0u;
            var cost = 0u;
            if (((ballot >> tid) & 1u) != 0u) {
                let index = countOneBits(ballot & ((1u << tid) - 1u));
                control = half_at(base + 8u + 2u * index);
                qscale = byte_at(base + 8u + 2u * active_count + index);
                cost = countOneBits(control & 0x5555u) + 2u * countOneBits((control & 0xaaaau) >> 1u) +
                    8u * (qscale & 0xfu);
            }
            control_words[tid] = control;
            qscales[tid] = qscale;
            plane_costs[tid] = cost;
        }
        workgroupBarrier();

        // Where the bit-planes of this 8x8 block start, and where the sign
        // bits start (right after the last bit-plane of the block).
        var plane_start = tables_end;
        var planes_total = 0u;
        for (var r = 0u; r < 16u; r = r + 1u) {
            if (r < block8) {
                plane_start = plane_start + plane_costs[r];
            }
            planes_total = planes_total + plane_costs[r];
        }
        let sign_base_bit = (tables_end + planes_total) * 8u;

        // Phase B: decode this subblock's magnitudes. The eight values are
        // handled one by one in the code rather than in loops over a variable
        // index, which keeps them in registers (ARCHITECTURE.md, decision D24).
        var magnitudes: array<u32, 8>;
        var count = 0u;
        var scale = 0.0;
        let control = control_words[block8];
        let qscale = qscales[block8];
        // An 8x8 block with an all-zero control word decodes to nothing.
        if (((ballot >> block8) & 1u) != 0u && control != 0u) {
            let base_planes = qscale & 0xfu;
            let below = control & ((1u << (2u * subblock)) - 1u);
            let extra_before = countOneBits(below & 0x5555u) + 2u * countOneBits((below & 0xaaaau) >> 1u);
            var byte_offset = base + plane_start + extra_before + base_planes * subblock;
            let planes = base_planes + ((control >> (2u * subblock)) & 3u);

            // Most significant plane first; bit v of a plane byte is value v.
            for (var plane = planes; plane > 0u; plane = plane - 1u) {
                let bits = byte_at(byte_offset);
                byte_offset = byte_offset + 1u;
                magnitudes[0] = magnitudes[0] | (((bits >> 0u) & 1u) << (plane - 1u));
                magnitudes[1] = magnitudes[1] | (((bits >> 1u) & 1u) << (plane - 1u));
                magnitudes[2] = magnitudes[2] | (((bits >> 2u) & 1u) << (plane - 1u));
                magnitudes[3] = magnitudes[3] | (((bits >> 3u) & 1u) << (plane - 1u));
                magnitudes[4] = magnitudes[4] | (((bits >> 4u) & 1u) << (plane - 1u));
                magnitudes[5] = magnitudes[5] | (((bits >> 5u) & 1u) << (plane - 1u));
                magnitudes[6] = magnitudes[6] | (((bits >> 6u) & 1u) << (plane - 1u));
                magnitudes[7] = magnitudes[7] | (((bits >> 7u) & 1u) << (plane - 1u));
            }

            count = count + select(0u, 1u, magnitudes[0] != 0u);
            count = count + select(0u, 1u, magnitudes[1] != 0u);
            count = count + select(0u, 1u, magnitudes[2] != 0u);
            count = count + select(0u, 1u, magnitudes[3] != 0u);
            count = count + select(0u, 1u, magnitudes[4] != 0u);
            count = count + select(0u, 1u, magnitudes[5] != 0u);
            count = count + select(0u, 1u, magnitudes[6] != 0u);
            count = count + select(0u, 1u, magnitudes[7] != 0u);
            scale = block_scale(quant_code) * block8_scale(qscale >> 4u);
        }

        // Every non-zero value takes one sign bit, numbered in invocation
        // order. This invocation's first sign bit follows those of all earlier
        // invocations: the totals of the earlier 8x8 blocks plus the counts of
        // the earlier subblocks of its own 8x8 block. Two barriers, no scan
        // over all 128 invocations.
        sign_counts[tid] = count;
        workgroupBarrier();
        if (tid < 16u) {
            var total = 0u;
            for (var s = 0u; s < 8u; s = s + 1u) {
                total = total + sign_counts[tid * 8u + s];
            }
            block_sign_totals[tid] = total;
        }
        workgroupBarrier();
        var exclusive = 0u;
        for (var r = 0u; r < 16u; r = r + 1u) {
            if (r < block8) {
                exclusive = exclusive + block_sign_totals[r];
            }
        }
        for (var s = 0u; s < 8u; s = s + 1u) {
            if (s < subblock) {
                exclusive = exclusive + sign_counts[block8 * 8u + s];
            }
        }

        // Read the sign bits of this invocation (at most 8, so at most two words).
        var sign_bits = 0u;
        if (count != 0u) {
            let sign_bit = sign_base_bit + exclusive;
            let word = offset_words + (sign_bit >> 5u);
            let shift = sign_bit & 31u;
            sign_bits = payload[word] >> shift;
            if (shift != 0u) {
                sign_bits = sign_bits | (payload[word + 1u] << (32u - shift));
            }
        }

        var sign_index = 0u;
        if (magnitudes[0] != 0u) {
            values[0] = signed_value(magnitudes[0], scale, (sign_bits >> sign_index) & 1u);
            sign_index = sign_index + 1u;
        }
        if (magnitudes[1] != 0u) {
            values[1] = signed_value(magnitudes[1], scale, (sign_bits >> sign_index) & 1u);
            sign_index = sign_index + 1u;
        }
        if (magnitudes[2] != 0u) {
            values[2] = signed_value(magnitudes[2], scale, (sign_bits >> sign_index) & 1u);
            sign_index = sign_index + 1u;
        }
        if (magnitudes[3] != 0u) {
            values[3] = signed_value(magnitudes[3], scale, (sign_bits >> sign_index) & 1u);
            sign_index = sign_index + 1u;
        }
        if (magnitudes[4] != 0u) {
            values[4] = signed_value(magnitudes[4], scale, (sign_bits >> sign_index) & 1u);
            sign_index = sign_index + 1u;
        }
        if (magnitudes[5] != 0u) {
            values[5] = signed_value(magnitudes[5], scale, (sign_bits >> sign_index) & 1u);
            sign_index = sign_index + 1u;
        }
        if (magnitudes[6] != 0u) {
            values[6] = signed_value(magnitudes[6], scale, (sign_bits >> sign_index) & 1u);
            sign_index = sign_index + 1u;
        }
        if (magnitudes[7] != 0u) {
            values[7] = signed_value(magnitudes[7], scale, (sign_bits >> sign_index) & 1u);
            sign_index = sign_index + 1u;
        }
    }

    // Values run down the columns of the 4x2 subblock.
    store(band, x0 + 0u, y0 + 0u, values[0]);
    store(band, x0 + 0u, y0 + 1u, values[1]);
    store(band, x0 + 1u, y0 + 0u, values[2]);
    store(band, x0 + 1u, y0 + 1u, values[3]);
    store(band, x0 + 2u, y0 + 0u, values[4]);
    store(band, x0 + 2u, y0 + 1u, values[5]);
    store(band, x0 + 3u, y0 + 0u, values[6]);
    store(band, x0 + 3u, y0 + 1u, values[7]);
}
