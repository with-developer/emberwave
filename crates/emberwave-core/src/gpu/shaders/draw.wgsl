// Stage: draws the decoded picture into a texture.
//
// Not an upstream stage: upstream's viewer samples its planes in a fragment
// shader (shaders/yuv2rgb.frag), and its Metal API writes the planes into the
// caller's textures from compute shaders. Here the colour conversion has run
// already as a compute pass (yuv2rgb.wgsl), so this copies its RGBA words into
// a render target of the picture's size: a fragment is a pixel. A render pass
// rather than a copy from the buffer: a copy takes rows of a multiple of 256
// bytes, and the RGBA words are packed tightly, as the read-back to the CPU
// takes them. It also fills BGRA targets. The browser player draws its canvas
// with it, and the C API the textures it hands to the caller's graphics API
// (Decoder::encode_rgb_to_texture, decision D40).

struct Picture {
    width: u32,
    height: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> rgba: array<u32>;
@group(0) @binding(1) var<uniform> picture: Picture;

// One triangle that covers the target: (-1, -1), (3, -1), (-1, 3).
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
