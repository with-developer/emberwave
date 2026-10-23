// Stage: YCbCr to RGB.
//
// Upstream counterpart: shaders/yuv2rgb.frag, the conversion of upstream's
// viewer (it is not part of the codec). Like it, this undoes the video range,
// applies the matrix of the stream and upsamples 4:2:0 chroma bilinearly at the
// position the siting says. Unlike it, the weights are exact integers (4:2:0
// chroma sits a quarter chroma sample away from a pixel), and the matrix is
// passed in so that BT.601, BT.709 and BT.2020 share one shader. The CPU twin
// is `yuv_to_rgb8` in color.rs; both evaluate the same expressions in the same
// order.
//
// Reads the packed 8-bit planes written by the last level of idwt.wgsl and
// writes one RGBA word
// per pixel (red in the low byte, alpha 255).

struct YuvParams {
    width: u32,
    height: u32,
    chroma_420: u32,
    siting_left: u32,

    // Word offset and words per row of each packed plane.
    y_offset: u32,
    y_stride: u32,
    cb_offset: u32,
    cb_stride: u32,

    cr_offset: u32,
    cr_stride: u32,
    chroma_width: u32,
    chroma_height: u32,

    limited: u32,
    grid_width: u32,
    pad0: u32,
    pad1: u32,

    // R from Cr, G from Cb, G from Cr, B from Cb.
    coefficients: vec4<f32>,
    // Video black level, luma scale, chroma scale and the zero of chroma,
    // all computed on the CPU so that both sides use the same f32 values.
    levels: vec4<f32>,
    // 1 / 255 in x.
    scale: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> planes: array<u32>;
@group(0) @binding(1) var<storage, read_write> rgba: array<u32>;
@group(0) @binding(2) var<uniform> params: YuvParams;

const GROUP_SIZE: u32 = 256u;

fn sample8(offset: u32, stride: u32, x: u32, y: u32) -> u32 {
    let word = planes[offset + y * stride + (x >> 2u)];
    return (word >> ((x & 3u) * 8u)) & 0xffu;
}

// The two chroma samples around luma position `p` along one axis, as
// (index of the first, weight of the second in quarters). The index can be -1,
// which the caller clamps. Same as chroma_taps in color.rs.
fn taps(p: u32, left: bool) -> vec2<i32> {
    let half_p = i32(p >> 1u);
    if (!left) {
        if ((p & 1u) == 0u) {
            return vec2<i32>(half_p - 1, 3);
        }
        return vec2<i32>(half_p, 1);
    }
    if ((p & 1u) == 0u) {
        return vec2<i32>(half_p, 0);
    }
    return vec2<i32>(half_p, 2);
}

// Bilinear chroma value in 8-bit units, exact: weights are multiples of 1/16.
fn chroma_sample(offset: u32, stride: u32, tx: vec2<i32>, ty: vec2<i32>) -> f32 {
    let max_x = i32(params.chroma_width) - 1;
    let max_y = i32(params.chroma_height) - 1;
    let x0 = u32(clamp(tx.x, 0, max_x));
    let x1 = u32(clamp(tx.x + 1, 0, max_x));
    let y0 = u32(clamp(ty.x, 0, max_y));
    let y1 = u32(clamp(ty.x + 1, 0, max_y));
    let fx = u32(tx.y);
    let fy = u32(ty.y);
    let sum = (4u - fx) * (4u - fy) * sample8(offset, stride, x0, y0)
        + fx * (4u - fy) * sample8(offset, stride, x1, y0)
        + (4u - fx) * fy * sample8(offset, stride, x0, y1)
        + fx * fy * sample8(offset, stride, x1, y1);
    return f32(sum) * 0.0625;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let index = (wg.y * params.grid_width + wg.x) * GROUP_SIZE + tid;
    if (index >= params.width * params.height) {
        return;
    }
    let x = index % params.width;
    let y = index / params.width;

    var cb8: f32;
    var cr8: f32;
    if (params.chroma_420 != 0u) {
        let tx = taps(x, params.siting_left != 0u);
        let ty = taps(y, false);
        cb8 = chroma_sample(params.cb_offset, params.cb_stride, tx, ty);
        cr8 = chroma_sample(params.cr_offset, params.cr_stride, tx, ty);
    } else {
        cb8 = f32(sample8(params.cb_offset, params.cb_stride, x, y));
        cr8 = f32(sample8(params.cr_offset, params.cr_stride, x, y));
    }

    let inv_255 = params.scale.x;
    var luma = f32(sample8(params.y_offset, params.y_stride, x, y)) * inv_255;
    var cb = cb8 * inv_255 - params.levels.w;
    var cr = cr8 * inv_255 - params.levels.w;
    if (params.limited != 0u) {
        luma = (luma - params.levels.x) * params.levels.y;
        cb = cb * params.levels.z;
        cr = cr * params.levels.z;
        luma = clamp(luma, 0.0, 1.0);
        cb = clamp(cb, -0.5, 0.5);
        cr = clamp(cr, -0.5, 0.5);
    }

    let r = luma + params.coefficients.x * cr;
    let g = luma - params.coefficients.y * cb - params.coefficients.z * cr;
    let b = luma + params.coefficients.w * cb;

    // Round to nearest, ties to even, like a UNORM store.
    let rgb = vec3<u32>(round(clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0));
    rgba[index] = rgb.x | (rgb.y << 8u) | (rgb.z << 16u) | (0xffu << 24u);
}
