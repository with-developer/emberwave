// Stage: the store of the inverse transform's last level, into a buffer.
//
// Behind idwt.wgsl. Upstream's last level stores the 8-bit samples into UNORM
// storage images (shaders/idwt.comp); 8-bit storage textures are outside what
// WebGPU guarantees, so this packs them four to a word into a buffer instead,
// the first sample in the low byte. The CPU reads the planes from there, the
// colour conversion takes them, and copies take them into textures (decision
// D41).

@group(0) @binding(1) var<storage, read_write> packed: array<u32>;

// The 8-bit level of tile sample `at` collected in the levels array, or 0 for
// a padding sample past the right edge of the plane. `select` evaluates both
// arms, so the array is read for padding samples too: the vertical pass has
// to write the whole tile, not only the part inside the plane.
fn level_at(at: u32, x: u32, width: u32) -> u32 {
    return select(0u, u32(levels[at]), x < width);
}

// Every invocation writes one word of the tile: four samples of a row. Rows
// of a packed plane are padded to whole words, and the padding samples are
// zero. Words after the row's last sample are not written: a decoder pads its
// rows further, to the rows of a copy into a texture (decision D41), and
// nothing reads that padding.
fn store_plane(comp: CompParams, component: u32, tile_x: u32, tile_y: u32, tid: u32) {
    let row = tid / 8u;
    let word = tid % 8u;
    let y = tile_y + row;
    let word_x = tile_x / 4u + word;
    let x = 4u * word_x;
    if (y < comp.plane_height && x < comp.plane_width) {
        let at = row * TILE + 4u * word;
        let width = comp.plane_width;
        packed[comp.image_offset + y * comp.image_stride + word_x] = level_at(at, x, width) |
            (level_at(at + 1u, x + 1u, width) << 8u) |
            (level_at(at + 2u, x + 2u, width) << 16u) |
            (level_at(at + 3u, x + 3u, width) << 24u);
    }
}
