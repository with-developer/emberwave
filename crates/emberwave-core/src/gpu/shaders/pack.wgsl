// Stage: float planes to 8-bit.
//
// Upstream counterpart: the 8-bit UNORM storage write at the end of
// shaders/idwt.comp. UNORM conversion is not part of the bitstream definition
// (bitstream.md, "Start of frame header"), so it is a stage of its own here:
// scale to 255 and round to nearest, ties to even, like a storage texture
// write would. Four pixels of a row share one u32 (first pixel in the low
// byte), rows are padded to whole words.

struct PackComp {
    src_offset: u32,
    width: u32,
    height: u32,
    dst_offset: u32,
    // Words per row of the packed plane.
    dst_stride: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct PackParams {
    grid_width: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    comps: array<PackComp, 3>,
}

@group(0) @binding(0) var<storage, read> planes: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed: array<u32>;
@group(0) @binding(2) var<uniform> params: PackParams;

const GROUP_SIZE: u32 = 256u;
const MAX_VALUE: f32 = 255.0;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let comp = params.comps[wg.z];
    let word_index = (wg.y * params.grid_width + wg.x) * GROUP_SIZE + tid;
    if (word_index >= comp.dst_stride * comp.height) {
        return;
    }

    let row = word_index / comp.dst_stride;
    let first_x = (word_index % comp.dst_stride) * 4u;

    var word = 0u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let x = first_x + k;
        if (x < comp.width) {
            let value = clamp(planes[comp.src_offset + row * comp.width + x], 0.0, 1.0);
            word = word | (u32(round(value * MAX_VALUE)) << (8u * k));
        }
    }
    packed[comp.dst_offset + word_index] = word;
}
