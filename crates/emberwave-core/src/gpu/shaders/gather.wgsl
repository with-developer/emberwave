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

// Sample (x, y) of a component, as the byte its texture stores.
fn sample_at(component: u32, x: u32, y: u32) -> u32 {
    let at = vec2<u32>(x, y);
    var texel: vec4<f32>;
    switch component {
        case 0u: {
            texel = textureLoad(luma, at, 0);
        }
        case 1u: {
            texel = textureLoad(cb, at, 0);
        }
        default: {
            texel = textureLoad(cr, at, 0);
        }
    }
    let value = select(texel.r, texel.g, params.channel[component] == 1u);
    return u32(round(saturate(value) * 255.0));
}

// One invocation per word of a row, which it alone writes: four samples, or
// at the end of a row fewer and zeros. A workgroup takes 16 words of 16 rows,
// 64 by 16 samples, close enough to square for textures, which GPUs store in
// tiles; z selects the component. No barriers, so the invocations past a
// plane may return.
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let component = id.z;
    let word = id.x;
    let row = id.y;
    if (word >= params.stride[component] || row >= params.height[component]) {
        return;
    }
    let width = params.width[component];
    var packed = 0u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let x = word * 4u + i;
        if (x < width) {
            packed = packed | (sample_at(component, x, row) << (8u * i));
        }
    }
    planes[params.offset[component] + row * params.stride[component] + word] = packed;
}
