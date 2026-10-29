// Stage: packing, the bytes of every block that is sent, at its place in the
// payload. Compiled with quantise.wgsl in front of it.
//
// Upstream counterpart: shaders/block_packing.comp. This is a
// reimplementation of emberwave's CPU encoder instead (src/encode/pack.rs):
// given the same coefficients and the same drops it writes the same bytes,
// in the layout of bitstream.md, "Decoding 8x8 blocks".
//
// One workgroup of 128 invocations takes one 32x32 block. Each invocation
// quantises one 4x2 subblock again, as equant.wgsl did (quantise.wgsl), and
// drops the planes the rate control chose. The first 16 invocations size up
// one 8x8 block each; then every invocation knows where its bytes go and ORs
// them into the block, assembled in workgroup memory, which is finally copied
// to the block's offset in the payload. A block is at most 622 words: a
// header and 16 8x8 blocks of 3 bytes of tables, 8 subblocks of at most 18
// bit-planes and 64 sign bits.

// What the rate control knows of a block, from equant.wgsl.
struct BlockCost {
    words: array<u32, 16>,
    distortion: array<f32, 16>,
    max_drop: u32,
}

// The parameters of a frame, shared with rate.wgsl.
struct FrameParams {
    block_count: u32,
    grid_width: u32,
    budget_words: u32,
    // Sequence number of the frame, for the block headers.
    sequence: u32,
    // Words the payload buffer holds.
    capacity_words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read> pyramid: array<f32>;
@group(0) @binding(1) var<storage, read> costs: array<BlockCost>;
@group(0) @binding(2) var<storage, read> drops: array<u32>;
// Where each block starts in the payload, in words (rate.wgsl, block_words).
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> payload: array<u32>;
@group(0) @binding(5) var<uniform> params: QuantParams;
@group(0) @binding(6) var<uniform> frame: FrameParams;

// Bytes of a block header, and of the control word of an 8x8 block.
const HEADER_BYTES: u32 = 8u;
const CONTROL_WORD_BYTES: u32 = 2u;
// The bit-planes of a qscale byte are its low four bits, the scale code the
// high four (bitstream.md, "Decoding 8x8 blocks").
const QSCALE_SCALE_SHIFT: u32 = 4u;
// The block scale exponent of a quant code: 4 + b - drop (quant.rs,
// quant_code), in the bits above the mantissa.
const MAX_SCALE_EXP: u32 = 4u;
const QUANT_MANTISSA_BITS: u32 = 3u;
// Words of the assembly: 622 at most, five per invocation.
const ASSEMBLY_WORDS: u32 = 640u;

// Workgroup memory is not zeroed for this pipeline (decision D25): every
// invocation clears its five words of `assembly`, writes its element of the
// per-subblock arrays, and the first 16 their element of the per-8x8 arrays,
// each before the barrier that precedes their reads.
var<workgroup> assembly: array<atomic<u32>, 640>;
// Per subblock: planes kept, planes above the base, and sign bits.
var<workgroup> subblock_kept: array<u32, 128>;
var<workgroup> subblock_extra: array<u32, 128>;
var<workgroup> subblock_signs: array<u32, 128>;
// Per 8x8 block: 1 if it is sent, its bytes of bit-planes and its sign bits.
var<workgroup> block8_sent: array<u32, 16>;
var<workgroup> block8_plane_bytes: array<u32, 16>;
var<workgroup> block8_signs: array<u32, 16>;

fn put_byte(position: u32, byte: u32) {
    atomicOr(&assembly[position >> 2u], byte << ((position & 3u) * 8u));
}

// Bit-plane `plane` of the 8 magnitudes: bit v is value v.
fn plane_byte(m: array<u32, 8>, plane: u32) -> u32 {
    return ((m[0] >> plane) & 1u) | (((m[1] >> plane) & 1u) << 1u) |
        (((m[2] >> plane) & 1u) << 2u) | (((m[3] >> plane) & 1u) << 3u) |
        (((m[4] >> plane) & 1u) << 4u) | (((m[5] >> plane) & 1u) << 5u) |
        (((m[6] >> plane) & 1u) << 6u) | (((m[7] >> plane) & 1u) << 7u);
}

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let block_index = wg.y * params.grid_width + wg.x;
    if (block_index >= params.block_count) {
        return;
    }
    let drop = drops[block_index];
    let words = costs[block_index].words[drop];
    let offset = offsets[block_index];
    // Blocks that keep nothing are not sent. The rest fit the payload buffer
    // unless the frame is larger than it, which the CPU reports.
    if (words == 0u || offset + words > frame.capacity_words) {
        return;
    }
    let band = find_band(block_index);

    for (var i = tid; i < ASSEMBLY_WORDS; i = i + 128u) {
        atomicStore(&assembly[i], 0u);
    }
    let s = quantise_subblock(tid, band, block_index);

    // This subblock with `drop` planes dropped.
    let block8 = tid >> 3u;
    let subblock = tid & 7u;
    let kept_base = saturating_sub(s.base, drop);
    let kept = saturating_sub(s.planes, drop);
    var m: array<u32, 8>;
    m[0] = s.magnitudes[0] >> drop;
    m[1] = s.magnitudes[1] >> drop;
    m[2] = s.magnitudes[2] >> drop;
    m[3] = s.magnitudes[3] >> drop;
    m[4] = s.magnitudes[4] >> drop;
    m[5] = s.magnitudes[5] >> drop;
    m[6] = s.magnitudes[6] >> drop;
    m[7] = s.magnitudes[7] >> drop;
    let nonzero = select(0u, 1u, m[0] != 0u) | select(0u, 2u, m[1] != 0u) |
        select(0u, 4u, m[2] != 0u) | select(0u, 8u, m[3] != 0u) |
        select(0u, 16u, m[4] != 0u) | select(0u, 32u, m[5] != 0u) |
        select(0u, 64u, m[6] != 0u) | select(0u, 128u, m[7] != 0u);
    subblock_kept[tid] = kept;
    subblock_extra[tid] = kept - kept_base;
    subblock_signs[tid] = countOneBits(nonzero);
    workgroupBarrier();

    // An 8x8 block is sent while some subblock keeps a plane above the base.
    if (tid < 16u) {
        var extra = 0u;
        var plane_bytes = 0u;
        var signs = 0u;
        for (var i = 0u; i < 8u; i = i + 1u) {
            extra = extra + subblock_extra[tid * 8u + i];
            plane_bytes = plane_bytes + subblock_kept[tid * 8u + i];
            signs = signs + subblock_signs[tid * 8u + i];
        }
        block8_sent[tid] = select(0u, 1u, extra != 0u);
        block8_plane_bytes[tid] = plane_bytes;
        block8_signs[tid] = signs;
    }
    workgroupBarrier();

    // Where this 8x8 block's tables, planes and signs go: after those of the
    // 8x8 blocks before it that are sent. The planes follow the tables of
    // all of them, the signs the planes of all of them.
    var sent = 0u;
    var ballot = 0u;
    var index = 0u;
    var planes_before = 0u;
    var planes_total = 0u;
    var signs_before = 0u;
    for (var b = 0u; b < 16u; b = b + 1u) {
        if (block8_sent[b] != 0u) {
            if (b < block8) {
                index = index + 1u;
                planes_before = planes_before + block8_plane_bytes[b];
                signs_before = signs_before + block8_signs[b];
            }
            sent = sent + 1u;
            ballot = ballot | (1u << b);
            planes_total = planes_total + block8_plane_bytes[b];
        }
    }
    let tables_end = HEADER_BYTES + (CONTROL_WORD_BYTES + 1u) * sent;
    // A block sent with nothing in it, which a partial decode needs: its
    // first 8x8 block, always inside the band, with a code word and a scale
    // of zero, no bit-planes, which `assembly` holds already.
    if (sent == 0u) {
        ballot = 1u;
    }

    if (tid == 0u) {
        let quant_code = (band.exponent + MAX_SCALE_EXP - drop) << QUANT_MANTISSA_BITS;
        atomicStore(&assembly[0], ballot | ((words & 0xfffu) << 16u) | ((frame.sequence & 7u) << 28u));
        atomicStore(&assembly[1], quant_code | ((block_index & 0xffffffu) << 8u));
    }

    if (block8_sent[block8] != 0u) {
        // Within the 8x8 block, the subblocks before this one.
        var planes_within = 0u;
        var signs_within = 0u;
        var control = 0u;
        for (var i = 0u; i < 8u; i = i + 1u) {
            let other = block8 * 8u + i;
            if (i < subblock) {
                planes_within = planes_within + subblock_kept[other];
                signs_within = signs_within + subblock_signs[other];
            }
            control = control | (subblock_extra[other] << (2u * i));
        }

        if (subblock == 0u) {
            let control_at = HEADER_BYTES + CONTROL_WORD_BYTES * index;
            put_byte(control_at, control & 0xffu);
            put_byte(control_at + 1u, control >> 8u);
            put_byte(HEADER_BYTES + CONTROL_WORD_BYTES * sent + index, (s.scale_code << QSCALE_SCALE_SHIFT) | kept_base);
        }

        // Bit-planes, most significant first.
        var at = tables_end + planes_before + planes_within;
        for (var plane = kept; plane > 0u; plane = plane - 1u) {
            put_byte(at, plane_byte(m, plane - 1u));
            at = at + 1u;
        }

        // One sign bit per value that keeps a magnitude, in coefficient
        // order; 1 for negative.
        var bit = (tables_end + planes_total) * 8u + signs_before + signs_within;
        for (var v = 0u; v < 8u; v = v + 1u) {
            if (((nonzero >> v) & 1u) != 0u) {
                if (((s.negative >> v) & 1u) != 0u) {
                    atomicOr(&assembly[bit >> 5u], 1u << (bit & 31u));
                }
                bit = bit + 1u;
            }
        }
    }
    workgroupBarrier();

    for (var i = tid; i < words; i = i + 128u) {
        payload[offset + i] = atomicLoad(&assembly[i]);
    }
}
