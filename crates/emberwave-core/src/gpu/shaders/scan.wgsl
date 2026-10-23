// Exclusive prefix sums of 32-bit counts, in place, for the encoder: in block
// order, what the steps of the cutoff key of the blocks before a block save
// (rate.wgsl), and where a block's words go in the payload (pack.wgsl).
//
// Upstream's encoder sums with subgroup operations; these use workgroup
// memory only (decision D4). Three dispatches:
//
// - scan_chunks: one workgroup per chunk of 1024 values, four per
//   invocation, writes the sums within the chunk and the chunk's total;
// - scan_totals: one workgroup sums the at most 1024 chunk totals the same
//   way and appends the sum of everything;
// - add_totals: each chunk adds the sum of the chunks before it.

struct ScanParams {
    count: u32,
    // Chunks of CHUNK values, at most CHUNK of them.
    chunk_count: u32,
    // Workgroups in x of the dispatches over the chunks.
    grid_width: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read_write> values: array<u32>;
// The total of each chunk, scanned in place, then the sum of all values.
@group(0) @binding(1) var<storage, read_write> chunk_totals: array<u32>;
@group(0) @binding(2) var<uniform> params: ScanParams;

const CHUNK: u32 = 1024u;
const GROUP_SIZE: u32 = 256u;

// Workgroup memory is not zeroed (decision D25): every invocation writes its
// element before the first barrier of `exclusive_scan`.
var<workgroup> sums: array<u32, 256>;

// The sum of the values of the invocations before this one; all invocations
// of the workgroup call it, with their own `value`. Hillis and Steele's scan
// in workgroup memory: eight rounds, each reading before a barrier and
// writing after it.
fn exclusive_scan(tid: u32, value: u32) -> u32 {
    sums[tid] = value;
    workgroupBarrier();
    for (var offset = 1u; offset < GROUP_SIZE; offset = offset * 2u) {
        var addend = 0u;
        if (tid >= offset) {
            addend = sums[tid - offset];
        }
        workgroupBarrier();
        sums[tid] = sums[tid] + addend;
        workgroupBarrier();
    }
    return sums[tid] - value;
}

fn value_at(index: u32) -> u32 {
    if (index < params.count) {
        return values[index];
    }
    return 0u;
}

fn total_at(index: u32) -> u32 {
    if (index < params.chunk_count) {
        return chunk_totals[index];
    }
    return 0u;
}

@compute @workgroup_size(256)
fn scan_chunks(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    // Workgroups past the last chunk, which a grid of the chunk count never
    // has, sum the last chunk again and write nothing, rather than return:
    // a return in front of the barriers of `exclusive_scan` crashed WARP,
    // Direct3D's software adapter, though no workgroup took it.
    let workgroup = wg.y * params.grid_width + wg.x;
    let chunk = min(workgroup, params.chunk_count - 1u);
    let count = select(0u, params.count, workgroup < params.chunk_count);
    let base = chunk * CHUNK + tid * 4u;
    let v0 = value_at(base);
    let v1 = value_at(base + 1u);
    let v2 = value_at(base + 2u);
    let v3 = value_at(base + 3u);
    let sum = v0 + v1 + v2 + v3;
    let before = exclusive_scan(tid, sum);
    if (base < count) {
        values[base] = before;
    }
    if (base + 1u < count) {
        values[base + 1u] = before + v0;
    }
    if (base + 2u < count) {
        values[base + 2u] = before + v0 + v1;
    }
    if (base + 3u < count) {
        values[base + 3u] = before + v0 + v1 + v2;
    }
    if (count != 0u && tid == GROUP_SIZE - 1u) {
        chunk_totals[chunk] = before + sum;
    }
}

@compute @workgroup_size(256)
fn scan_totals(@builtin(local_invocation_index) tid: u32) {
    let base = tid * 4u;
    let v0 = total_at(base);
    let v1 = total_at(base + 1u);
    let v2 = total_at(base + 2u);
    let v3 = total_at(base + 3u);
    let sum = v0 + v1 + v2 + v3;
    let before = exclusive_scan(tid, sum);
    if (base < params.chunk_count) {
        chunk_totals[base] = before;
    }
    if (base + 1u < params.chunk_count) {
        chunk_totals[base + 1u] = before + v0;
    }
    if (base + 2u < params.chunk_count) {
        chunk_totals[base + 2u] = before + v0 + v1;
    }
    if (base + 3u < params.chunk_count) {
        chunk_totals[base + 3u] = before + v0 + v1 + v2;
    }
    if (tid == GROUP_SIZE - 1u) {
        chunk_totals[params.chunk_count] = before + sum;
    }
}

@compute @workgroup_size(256)
fn add_totals(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let chunk = wg.y * params.grid_width + wg.x;
    if (chunk >= params.chunk_count) {
        return;
    }
    let before = chunk_totals[chunk];
    let base = chunk * CHUNK + tid * 4u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        if (base + i < params.count) {
            values[base + i] = values[base + i] + before;
        }
    }
}
