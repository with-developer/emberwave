// Draws the decoded picture on the canvas.
//
// Not an upstream stage: upstream's viewer samples its planes in a fragment
// shader (shaders/yuv2rgb.frag). Here the colour conversion has already run
// as a compute pass (yuv2rgb.wgsl), so this only copies its RGBA words to the
// canvas, which has the size of the picture: a fragment is a pixel.

struct Picture {
    width: u32,
    height: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> rgba: array<u32>;
@group(0) @binding(1) var<uniform> picture: Picture;

// One triangle that covers the canvas: (-1, -1), (3, -1), (-1, 3).
@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index & 1u) * 4u) - 1.0;
    let y = f32((index >> 1u) * 4u) - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let x = u32(position.x);
    let y = u32(position.y);
    if (x >= picture.width || y >= picture.height) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    // Red in the low byte, as yuv2rgb.wgsl writes it.
    return unpack4x8unorm(rgba[y * picture.width + x]);
}
