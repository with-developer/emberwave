// Stage: rate control, how many bit-planes each 32x32 block drops so that
// the frame fits its budget.
//
// Upstream counterpart: shaders/analyze_rate_control.comp,
// analyze_rate_control_finalize.comp and resolve_rate_control.comp, which make
// upstream's own choices. This is a reimplementation of emberwave's rate
// control instead (decision D28, src/encode/rate.rs): given the same cost
// tables it chooses the same drops (decision D29).
//
// rate.rs applies the steps of all blocks in the order of their keys, ties by
// block index, until the frame fits. Here nothing is sorted:
//
// - analyse: each block finds its steps as rate.rs does and adds the words
//   they save to a histogram of the keys, with atomics, which is exact in any
//   order.
// - cutoff: one invocation finds the key at which the frame fits. Every step
//   below it is applied, none above it.
// - ties: each block adds up what its steps of exactly that key save, and
//   scan.wgsl sums these in block order.
// - choose: steps of the cutoff key are applied in block order while the
//   frame does not fit yet, which is where the CPU stops.

// What the rate control knows of a block, from equant.wgsl.
struct BlockCost {
    words: array<u32, 16>,
    distortion: array<f32, 16>,
    max_drop: u32,
}

// The parameters of a frame, shared with pack.wgsl.
struct FrameParams {
    block_count: u32,
    // Workgroups in x of the dispatches over the blocks, 64 blocks each.
    grid_width: u32,
    // Words the blocks may take: the budget less the start of frame header.
    budget_words: u32,
    // Sequence number of the frame.
    sequence: u32,
    // Words the payload buffer holds.
    capacity_words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// The frame-wide state, cleared before every frame.
struct RateState {
    // Words saved by the steps of each key.
    histogram: array<atomic<u32>, 512>,
    // Words of all blocks with nothing dropped, and with all they may drop.
    total: atomic<u32>,
    minimum: atomic<u32>,
    // Blocks sent: those that keep anything, and those of the coarsest level.
    sent: atomic<u32>,
    // The key at which the frame fits and what its steps must still save; 0
    // and 0 when nothing is dropped. 1 if the frame fits at all.
    cutoff_key: u32,
    remaining: u32,
    fits: u32,
}

@group(0) @binding(0) var<storage, read> costs: array<BlockCost>;
@group(0) @binding(1) var<storage, read_write> state: RateState;
// Per block STEP_SLOTS words: its first drop << 8 | the number of steps,
// then the steps, key | drop << 9 | words saved << 16.
@group(0) @binding(2) var<storage, read_write> steps: array<u32>;
// Per block, what its steps of the cutoff key save; scanned in place.
@group(0) @binding(3) var<storage, read_write> tie_savings: array<u32>;
@group(0) @binding(4) var<storage, read_write> drops: array<u32>;
// Per block, its size in words with the chosen drop; scanned in place into
// its offset in the payload.
@group(0) @binding(5) var<storage, read_write> block_words: array<u32>;
@group(0) @binding(6) var<uniform> frame: FrameParams;

// Number of keys (rate.rs, KEYS).
const KEYS: u32 = 512u;
// Numbers of dropped bit-planes: 0 to 15.
const DROP_STEPS: u32 = 16u;
// Words per block in `steps`: a header and at most 15 steps.
const STEP_SLOTS: u32 = 16u;
const BLOCKS_PER_GROUP: u32 = 64u;

// 2^exponent as an exact f32, for exponents in the normal range.
fn exact_pow2(exponent: i32) -> f32 {
    return bitcast<f32>(u32(127 + exponent) << 23u);
}

// The threshold of key `key`: a step reaches it when it adds at least
// `saving * threshold` of distortion (rate.rs, threshold). Exact.
fn threshold(key: u32) -> f32 {
    let mantissa = f32(4u + key % 4u) * 0.25;
    return mantissa * exact_pow2(i32(key / 4u) - 64);
}

// The key of a step that adds `distortion` and saves `saving` words: the
// highest threshold it reaches, 0 for none (rate.rs, key). The products are
// exact, so the comparisons are the CPU's.
fn key_of(distortion: f32, saving: u32) -> u32 {
    let words = f32(saving);
    var low = 1u;
    var high = KEYS;
    loop {
        if (low >= high) {
            break;
        }
        let middle = (low + high) / 2u;
        if (distortion >= words * threshold(middle)) {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }
    return low - 1u;
}

fn block_of(wg: vec3<u32>, lid: u32) -> u32 {
    return (wg.y * frame.grid_width + wg.x) * BLOCKS_PER_GROUP + lid;
}

// Stores the step of `block` from drop `start` to drop `end` as step number
// `count` and adds it to the histogram. Returns its key, at least
// `previous_key` so that a block's steps keep their order.
fn add_step(block: u32, count: u32, start: u32, end: u32, previous_key: u32) -> u32 {
    let saving = costs[block].words[start] - costs[block].words[end];
    let added = costs[block].distortion[end] - costs[block].distortion[start];
    let key = max(key_of(added, saving), previous_key);
    steps[block * STEP_SLOTS + 1u + count] = key | (end << 9u) | (saving << 16u);
    atomicAdd(&state.histogram[key], saving);
    return key;
}

@compute @workgroup_size(64)
fn analyse(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let block = block_of(wg, lid);
    if (block >= frame.block_count) {
        return;
    }
    let last = min(costs[block].max_drop, DROP_STEPS - 1u);
    atomicAdd(&state.total, costs[block].words[0]);
    atomicAdd(&state.minimum, costs[block].words[last]);

    // The points (rate.rs, points): for each size, the drop with the least
    // distortion, the fewest planes on a tie. `point` is the best so far at
    // the current size; when the size changes it is settled, and the step
    // from the point settled before it is known.
    var point = 0u;
    var settled = 0u;
    var has_settled = false;
    var first = 0u;
    var count = 0u;
    var key = 0u;
    for (var drop = 1u; drop <= last; drop = drop + 1u) {
        if (costs[block].words[drop] == costs[block].words[point]) {
            if (costs[block].distortion[drop] < costs[block].distortion[point]) {
                point = drop;
            }
        } else {
            if (has_settled) {
                key = add_step(block, count, settled, point, key);
                count = count + 1u;
            } else {
                first = point;
            }
            settled = point;
            has_settled = true;
            point = drop;
        }
    }
    if (has_settled) {
        key = add_step(block, count, settled, point, key);
        count = count + 1u;
    } else {
        first = point;
    }
    steps[block * STEP_SLOTS] = (first << 8u) | count;
}

@compute @workgroup_size(1)
fn cutoff() {
    let total = atomicLoad(&state.total);
    state.cutoff_key = 0u;
    state.remaining = 0u;
    state.fits = select(0u, 1u, atomicLoad(&state.minimum) <= frame.budget_words);
    if (total <= frame.budget_words) {
        return;
    }
    // The first key whose steps, with those of all keys below, save enough.
    // If none does, the frame cannot fit and `fits` is 0.
    let need = total - frame.budget_words;
    var saved = 0u;
    for (var key = 0u; key < KEYS; key = key + 1u) {
        let saving = atomicLoad(&state.histogram[key]);
        if (saved + saving >= need) {
            state.cutoff_key = key;
            state.remaining = need - saved;
            return;
        }
        saved = saved + saving;
    }
}

@compute @workgroup_size(64)
fn ties(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let block = block_of(wg, lid);
    if (block >= frame.block_count) {
        return;
    }
    var saving = 0u;
    if (state.remaining != 0u) {
        let slot = block * STEP_SLOTS;
        let count = steps[slot] & 0xffu;
        for (var i = 0u; i < count; i = i + 1u) {
            let step = steps[slot + 1u + i];
            if ((step & 0x1ffu) == state.cutoff_key) {
                saving = saving + (step >> 16u);
            }
        }
    }
    tie_savings[block] = saving;
}

@compute @workgroup_size(64)
fn choose(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let block = block_of(wg, lid);
    if (block >= frame.block_count) {
        return;
    }
    let slot = block * STEP_SLOTS;
    let header = steps[slot];
    var drop = header >> 8u;
    // What the steps of the cutoff key of the blocks before this one save.
    var saved = tie_savings[block];
    for (var i = 0u; i < (header & 0xffu); i = i + 1u) {
        let step = steps[slot + 1u + i];
        let key = step & 0x1ffu;
        if (key < state.cutoff_key) {
            drop = (step >> 9u) & 0xfu;
        } else if (key == state.cutoff_key && saved < state.remaining) {
            drop = (step >> 9u) & 0xfu;
            saved = saved + (step >> 16u);
        } else {
            break;
        }
    }
    drops[block] = drop;
    let words = costs[block].words[drop];
    block_words[block] = words;
    if (words != 0u) {
        atomicAdd(&state.sent, 1u);
    }
}
