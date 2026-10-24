// Stage: the encoder's input from textures, before fdwt.wgsl (issue #33).
//
// Upstream counterpart: the Metal encoder's IOSurface input
// (pyrowave_encoder_encode_gpu), which reads three R8 surfaces or NV12's two
// planes as textures, binding the chroma plane twice with different swizzles.
// emberwave packs the picture into the planes buffer instead, four 8-bit
// samples to a word and rows padded to whole words with zeros: the words the
// CPU writes for a picture in CPU memory (gpu/encoder.rs, upload_planes).
// fdwt.wgsl reads them as it always does, so a picture from textures encodes
// to the bytes the same picture from the CPU does (decision D32).
//
// A texel of an 8-bit unorm texture loads as its byte divided by 255, rounded
// to a float, and times 255 rounds back to the byte: the float's error is
// orders of magnitude below the half that would change it. WGSL has no
// swizzled views, so `channel` says where a component is: green for Cr in the
// chroma texture of NV12's layout, which is bound for Cb and for Cr.
//
// The texels come four at a time, 2x2 of them, with textureGather: a load per
// texel made the texture instructions the pass's cost (issue #40).

struct GatherParams {
    // Per component, Y, Cb and Cr in x, y and z: the plane's size in samples,
    // and where its rows start in the planes buffer and how many words each
    // takes.
    width: vec4<u32>,
    height: vec4<u32>,
    offset: vec4<u32>,
    stride: vec4<u32>,
    // The channel of the component in its texture: 0 red, 1 green.
    channel: vec4<u32>,
}

@group(0) @binding(0) var<storage, read_write> planes: array<u32>;
@group(0) @binding(1) var<uniform> params: GatherParams;
@group(0) @binding(2) var luma: texture_2d<f32>;
@group(0) @binding(3) var cb: texture_2d<f32>;
@group(0) @binding(4) var cr: texture_2d<f32>;
// For textureGather, which needs a sampler; its addressing clamps to the edge.
@group(0) @binding(5) var texels: sampler;

// Where textureGather takes texels (x, y) to (x + 1, y + 1) of a texture of
// `size`: the corner they share, in normalised coordinates. Being half a texel
// from every texel centre, it is far from the rounding that would pick other
// texels.
fn corner(size: vec2<u32>, x: u32, y: u32) -> vec2<f32> {
    return vec2<f32>(f32(x + 1u), f32(y + 1u)) / vec2<f32>(size);
}

// Texels (x, y) to (x + 1, y + 1) of a component as the bytes the texture
// stores, in textureGather's order: (x, y + 1), (x + 1, y + 1), (x + 1, y),
// (x, y). Texels past the texture's edge repeat the edge.
fn gather_at(component: u32, x: u32, y: u32) -> vec4<u32> {
    let green = params.channel[component] == 1u;
    var texel: vec4<f32>;
    switch component {
        case 0u: {
            texel = textureGather(0, luma, texels, corner(textureDimensions(luma), x, y));
        }
        case 1u: {
            let at = corner(textureDimensions(cb), x, y);
            if (green) {
                texel = textureGather(1, cb, texels, at);
            } else {
                texel = textureGather(0, cb, texels, at);
            }
        }
        default: {
            let at = corner(textureDimensions(cr), x, y);
            if (green) {
                texel = textureGather(1, cr, texels, at);
            } else {
                texel = textureGather(0, cr, texels, at);
            }
        }
    }
    return vec4<u32>(round(saturate(texel) * 255.0));
}

// Samples x to x + 3 of a row as a word, the first in the low byte, with
// zeros for those at `width` and past it.
fn pack(samples: vec4<u32>, x: u32, width: u32) -> u32 {
    let inside = vec4<u32>(x) + vec4<u32>(0u, 1u, 2u, 3u) < vec4<u32>(width);
    let kept = select(vec4<u32>(0u), samples, inside);
    return kept.x | (kept.y << 8u) | (kept.z << 16u) | (kept.w << 24u);
}

// One invocation per word of a pair of rows, which it alone writes: four
// samples of each row, or at the end of a row fewer and zeros, from two
// gathers. A workgroup takes 16 words of 32 rows, 64 by 32 samples; z selects
// the component. No barriers, so the invocations past a plane may return.
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let component = id.z;
    let word = id.x;
    let row = 2u * id.y;
    let height = params.height[component];
    if (word >= params.stride[component] || row >= height) {
        return;
    }
    let width = params.width[component];
    let x = 4u * word;
    let left = gather_at(component, x, row);
    let right = gather_at(component, x + 2u, row);
    let at = params.offset[component] + row * params.stride[component] + word;
    planes[at] = pack(vec4<u32>(left.w, left.z, right.w, right.z), x, width);
    if (row + 1u < height) {
        planes[at + params.stride[component]] =
            pack(vec4<u32>(left.x, left.y, right.x, right.y), x, width);
    }
}
