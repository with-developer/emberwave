// Stage: the encoder's input from an RGB texture, before fdwt.wgsl
// (decision D65).
//
// Upstream has no counterpart: its encoders take Y'CbCr planes, and hosts
// convert their captures before. emberwave converts a picture of R', G' and
// B' as it packs it into the planes buffer, as gather.wgsl packs planes in
// textures: into the words the CPU writes for the planes that color.rs's
// rgb_to_ycbcr8 makes of the same picture, with the same integers
// (ForwardMatrix: 16-bit fixed point, sums of up to 8 pixels well within
// 32 bits). So a picture from an RGB texture encodes to the bytes of those
// planes.
//
// A texel of an 8-bit unorm texture loads as its byte divided by 255, and
// times 255 rounds back to the byte, as in gather.wgsl. A Bgra8Unorm texture
// loads in the order of an Rgba8Unorm one, so the shader takes both.

struct RgbParams {
    // Per component, Y, Cb and Cr in x, y and z: the plane's size in samples,
    // and where its rows start in the planes buffer and how many words each
    // takes.
    width: vec4<u32>,
    height: vec4<u32>,
    offset: vec4<u32>,
    stride: vec4<u32>,
    // ForwardMatrix's rows for Y, Cb and Cr: the factors of R', G' and B' in
    // x, y and z, and the offset of one pixel in w.
    y_row: vec4<i32>,
    cb_row: vec4<i32>,
    cr_row: vec4<i32>,
    // 1 where 4:2:0's chroma sits on the left of the two columns it covers,
    // which weighs three columns 1, 2 and 1; 0 where it sits between them,
    // which weighs two alike.
    left: u32,
}

@group(0) @binding(0) var<storage, read_write> planes: array<u32>;
@group(0) @binding(1) var<uniform> params: RgbParams;
@group(0) @binding(2) var picture: texture_2d<f32>;

// color.rs's FORWARD_BITS.
const FORWARD_BITS: u32 = 16u;

// R', G' and B' of the pixel at (x, y) as bytes; past the picture's edge, the
// edge's.
fn pixel(x: i32, y: u32) -> vec3<i32> {
    let size = textureDimensions(picture);
    let at = vec2<u32>(u32(clamp(x, 0, i32(size.x) - 1)), min(y, size.y - 1u));
    return vec3<i32>(round(saturate(textureLoad(picture, at, 0).rgb) * 255.0));
}

// ForwardMatrix::apply: `row` of the matrix for pixels whose R', G' and B',
// each weighed, add up to `sum`, the weights to 1 << shift.
fn apply(row: vec4<i32>, sum: vec3<i32>, shift: u32) -> u32 {
    let total = dot(row.xyz, sum) + (row.w << shift) + (1i << (FORWARD_BITS + shift - 1u));
    return u32(clamp(total >> (FORWARD_BITS + shift), 0i, 255i));
}

// Samples x to x + 3 of a row as a word, the first in the low byte, with
// zeros for those at `width` and past it.
fn pack(samples: vec4<u32>, x: u32, width: u32) -> u32 {
    let inside = vec4<u32>(x) + vec4<u32>(0u, 1u, 2u, 3u) < vec4<u32>(width);
    let kept = select(vec4<u32>(0u), samples, inside);
    return kept.x | (kept.y << 8u) | (kept.z << 16u) | (kept.w << 24u);
}

// 4:4:4: one invocation per word of a row, of all three planes, which have
// one size: four pixels, and their Y, Cb and Cr.
@compute @workgroup_size(16, 16)
fn yuv444(@builtin(global_invocation_id) id: vec3<u32>) {
    let word = id.x;
    let row = id.y;
    if (word >= params.stride.x || row >= params.height.x) {
        return;
    }
    let x = 4u * word;
    var luma: vec4<u32>;
    var cb: vec4<u32>;
    var cr: vec4<u32>;
    for (var i = 0u; i < 4u; i++) {
        let rgb = pixel(i32(x + i), row);
        luma[i] = apply(params.y_row, rgb, 0u);
        cb[i] = apply(params.cb_row, rgb, 0u);
        cr[i] = apply(params.cr_row, rgb, 0u);
    }
    let width = params.width.x;
    planes[params.offset.x + row * params.stride.x + word] = pack(luma, x, width);
    planes[params.offset.y + row * params.stride.y + word] = pack(cb, x, width);
    planes[params.offset.z + row * params.stride.z + word] = pack(cr, x, width);
}

// 4:2:0: one invocation per word of a chroma row: four samples of Cb and of
// Cr, and the luma of the 8 columns and 2 rows they cover, two words of each
// row, or one at the end of a row of an odd count of words. The luma rows
// take at most twice the chroma rows' words, so every word is written.
@compute @workgroup_size(16, 16)
fn yuv420(@builtin(global_invocation_id) id: vec3<u32>) {
    let word = id.x;
    let chroma_row = id.y;
    if (word >= params.stride.y || chroma_row >= params.height.y) {
        return;
    }
    let x = 8u * word;
    let left = params.left == 1u;
    // Luma by row and word: row 0's two words, then row 1's.
    var luma: array<vec4<u32>, 4>;
    var sums: array<vec3<i32>, 4>;
    for (var r = 0u; r < 2u; r++) {
        let y = 2u * chroma_row + r;
        // The column before the chroma sample's two, for the left siting.
        var before = pixel(i32(x) - 1, y);
        for (var c = 0u; c < 4u; c++) {
            let even = pixel(i32(x + 2u * c), y);
            let odd = pixel(i32(x + 2u * c + 1u), y);
            let at = 2u * r + c / 2u;
            luma[at][(2u * c) % 4u] = apply(params.y_row, even, 0u);
            luma[at][(2u * c + 1u) % 4u] = apply(params.y_row, odd, 0u);
            if (left) {
                sums[c] += before + 2i * even + odd;
            } else {
                sums[c] += even + odd;
            }
            before = odd;
        }
    }
    let shift = select(2u, 3u, left);
    var cb: vec4<u32>;
    var cr: vec4<u32>;
    for (var c = 0u; c < 4u; c++) {
        cb[c] = apply(params.cb_row, sums[c], shift);
        cr[c] = apply(params.cr_row, sums[c], shift);
    }
    for (var r = 0u; r < 2u; r++) {
        let y = 2u * chroma_row + r;
        for (var w = 0u; w < 2u; w++) {
            let luma_word = 2u * word + w;
            if (y < params.height.x && luma_word < params.stride.x) {
                planes[params.offset.x + y * params.stride.x + luma_word] =
                    pack(luma[2u * r + w], x + 4u * w, params.width.x);
            }
        }
    }
    planes[params.offset.y + chroma_row * params.stride.y + word] = pack(cb, 4u * word, params.width.y);
    planes[params.offset.z + chroma_row * params.stride.z + word] = pack(cr, 4u * word, params.width.z);
}
