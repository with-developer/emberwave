// The quantisation of a 32x32 block, shared by equant.wgsl and pack.wgsl,
// which are compiled with this file in front of them: what the rate control
// measures is what the packing writes.
//
// It follows src/encode/quant.rs (analyse_block8) step for step. A workgroup
// of 128 invocations takes one block; invocation `tid` = 8 * (8x8 block,
// row-major) + (4x2 subblock) quantises the 8 coefficients of its subblock,
// and the 8 invocations of an 8x8 block agree on its scale and its base
// bit-planes through workgroup memory. Everything here is exact: the
// magnitudes do not depend on how a GPU rounds.

struct QuantBand {
    first_block: u32,
    blocks_x: u32,
    width: u32,
    height: u32,
    // Offset of the band in the pyramid, in floats.
    offset: u32,
    // The bit exponent b: coefficients are quantised as c * 2^b.
    exponent: u32,
    // What a squared coefficient error of the band counts for.
    weight: f32,
    // The most bit-planes a block of the band may drop.
    max_drop: u32,
}

struct QuantParams {
    band_count: u32,
    block_count: u32,
    grid_width: u32,
    pad: u32,
    // Transmitted bands in block index order.
    bands: array<QuantBand, 48>,
}

// One quantised 4x2 subblock and what its 8x8 block shares.
struct Subblock {
    // |c * 2^b| of the 8 values, which run down the columns of the subblock.
    scaled: array<f32, 8>,
    magnitudes: array<u32, 8>,
    // Bit v set: value v is negative.
    negative: u32,
    // The largest magnitude of the subblock.
    largest: u32,
    // Whether the 8x8 block carries anything; if not, every magnitude is 0.
    coded: bool,
    scale_code: u32,
    // The highest bit of the 8x8 block's largest magnitude, if it is coded.
    top: u32,
    // Base bit-planes of the 8x8 block, and the subblock's bit-planes.
    base: u32,
    planes: u32,
}

const BLOCK_SIZE: u32 = 32u;
// Numbers of dropped bit-planes the rate control chooses from: 0 to 15.
const DROP_STEPS: u32 = 16u;
// The largest 8x8 scale code the encoder uses: scale 1.
const MAX_SCALE_CODE: u32 = 6u;

// Workgroup memory is not zeroed (decision D25): every invocation writes its
// own element of each array before the barrier that precedes the reads.
var<workgroup> subblock_largest: array<f32, 128>;
var<workgroup> subblock_largest_magnitude: array<u32, 128>;

// 2^exponent as an exact f32, for exponents in the normal range.
fn exact_pow2(exponent: i32) -> f32 {
    return bitcast<f32>(u32(127 + exponent) << 23u);
}

// Block8x8Scale: code / 8 + 0.25, exactly.
fn block8_scale(code: u32) -> f32 {
    return f32(code + 2u) * 0.125;
}

fn saturating_sub(a: u32, b: u32) -> u32 {
    return select(0u, a - b, a > b);
}

// The band of block `block_index`: the last whose first block is not after
// it, by binary search over the at most 48 entries, as in dequant.wgsl. Every
// invocation of a workgroup takes the same steps, so the loop is uniform.
fn find_band(block_index: u32) -> QuantBand {
    var slot = 0u;
    var step = 32u;
    loop {
        let probe = slot + step;
        if (probe < params.band_count && block_index >= params.bands[probe].first_block) {
            slot = probe;
        }
        if (step == 1u) {
            break;
        }
        step = step >> 1u;
    }
    return params.bands[slot];
}

// The coefficient of the band at (x, y), 0 outside the band.
fn coefficient_at(band: QuantBand, x: u32, y: u32) -> f32 {
    if (x < band.width && y < band.height) {
        return pyramid[band.offset + y * band.width + x];
    }
    return 0.0;
}

// floor(t / scale), exactly. The product with the inverse of the scale can be
// one off next to a whole number, whatever the rounding of the inverse, and
// m * scale is exact, so two comparisons put it right.
fn magnitude(t: f32, scale: f32, inverse: f32) -> u32 {
    let estimate = u32(t * inverse);
    if (f32(estimate) * scale > t) {
        return estimate - 1u;
    }
    if (f32(estimate + 1u) * scale <= t) {
        return estimate + 1u;
    }
    return estimate;
}

fn largest_of(v: array<f32, 8>) -> f32 {
    return max(max(max(v[0], v[1]), max(v[2], v[3])), max(max(v[4], v[5]), max(v[6], v[7])));
}

fn largest_magnitude_of(v: array<u32, 8>) -> u32 {
    return max(max(max(v[0], v[1]), max(v[2], v[3])), max(max(v[4], v[5]), max(v[6], v[7])));
}

// Quantises the subblock of invocation `tid` of the workgroup that takes
// block `block_index` of `band`. Every invocation of the workgroup must call
// it: it waits at two barriers.
fn quantise_subblock(tid: u32, band: QuantBand, block_index: u32) -> Subblock {
    let relative = block_index - band.first_block;
    let block8 = tid >> 3u;
    let subblock = tid & 7u;
    let x0 = (relative % band.blocks_x) * BLOCK_SIZE + 8u * (block8 & 3u) + 4u * (subblock >> 2u);
    let y0 = (relative / band.blocks_x) * BLOCK_SIZE + 8u * (block8 >> 2u) + 2u * (subblock & 3u);
    let resolution = exact_pow2(i32(band.exponent));

    // Written out with constant indices so that the values stay in registers
    // (decision D24).
    var c: array<f32, 8>;
    c[0] = coefficient_at(band, x0 + 0u, y0 + 0u);
    c[1] = coefficient_at(band, x0 + 0u, y0 + 1u);
    c[2] = coefficient_at(band, x0 + 1u, y0 + 0u);
    c[3] = coefficient_at(band, x0 + 1u, y0 + 1u);
    c[4] = coefficient_at(band, x0 + 2u, y0 + 0u);
    c[5] = coefficient_at(band, x0 + 2u, y0 + 1u);
    c[6] = coefficient_at(band, x0 + 3u, y0 + 0u);
    c[7] = coefficient_at(band, x0 + 3u, y0 + 1u);
    var s: Subblock;
    s.scaled[0] = abs(c[0] * resolution);
    s.scaled[1] = abs(c[1] * resolution);
    s.scaled[2] = abs(c[2] * resolution);
    s.scaled[3] = abs(c[3] * resolution);
    s.scaled[4] = abs(c[4] * resolution);
    s.scaled[5] = abs(c[5] * resolution);
    s.scaled[6] = abs(c[6] * resolution);
    s.scaled[7] = abs(c[7] * resolution);
    s.negative = select(0u, 1u, c[0] < 0.0) | select(0u, 2u, c[1] < 0.0) |
        select(0u, 4u, c[2] < 0.0) | select(0u, 8u, c[3] < 0.0) |
        select(0u, 16u, c[4] < 0.0) | select(0u, 32u, c[5] < 0.0) |
        select(0u, 64u, c[6] < 0.0) | select(0u, 128u, c[7] < 0.0);

    subblock_largest[tid] = largest_of(s.scaled);
    workgroupBarrier();
    var largest = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) {
        largest = max(largest, subblock_largest[block8 * 8u + i]);
    }

    // The scale: planes at scale 1, then the smallest scale code whose scale
    // needs no more for the largest value. Below 1 nothing of the 8x8 block
    // is coded, and with scale 1 every magnitude is then zero.
    s.scale_code = MAX_SCALE_CODE;
    if (largest >= 1.0) {
        let limit = exact_pow2(i32(32u - countLeadingZeros(u32(largest))));
        for (var code = 0u; code < MAX_SCALE_CODE; code = code + 1u) {
            if (largest < limit * block8_scale(code)) {
                s.scale_code = code;
                break;
            }
        }
    }
    let scale = block8_scale(s.scale_code);
    let inverse = 1.0 / scale;
    s.magnitudes[0] = magnitude(s.scaled[0], scale, inverse);
    s.magnitudes[1] = magnitude(s.scaled[1], scale, inverse);
    s.magnitudes[2] = magnitude(s.scaled[2], scale, inverse);
    s.magnitudes[3] = magnitude(s.scaled[3], scale, inverse);
    s.magnitudes[4] = magnitude(s.scaled[4], scale, inverse);
    s.magnitudes[5] = magnitude(s.scaled[5], scale, inverse);
    s.magnitudes[6] = magnitude(s.scaled[6], scale, inverse);
    s.magnitudes[7] = magnitude(s.scaled[7], scale, inverse);
    s.largest = largest_magnitude_of(s.magnitudes);

    subblock_largest_magnitude[tid] = s.largest;
    workgroupBarrier();
    var block_largest = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        block_largest = max(block_largest, subblock_largest_magnitude[block8 * 8u + i]);
    }

    // The base bit-planes are all but the top three of the 8x8 block's
    // largest magnitude; each subblock has as many more as its own largest
    // needs.
    s.coded = largest >= 1.0 && block_largest != 0u;
    s.top = 0u;
    s.base = 0u;
    s.planes = 0u;
    if (s.coded) {
        s.top = 31u - countLeadingZeros(block_largest);
        s.base = saturating_sub(s.top, 2u);
        s.planes = s.base + 32u - countLeadingZeros(s.largest >> s.base);
    }
    return s;
}
