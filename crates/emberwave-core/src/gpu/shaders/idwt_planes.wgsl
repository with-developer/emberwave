// Stage: the store of the inverse transform's last level, into textures.
//
// Behind idwt.wgsl, in place of idwt_packed.wgsl, for decodes into the Y, Cb
// and Cr textures of a caller, on adapters that can write 8-bit storage
// textures (decision D52). As upstream's last level does (shaders/idwt.comp),
// it stores the samples straight into a UNORM texture per plane, which saves
// writing the packed planes and copying them out (decision D41). It is not
// one of the portable shaders: WebGPU does not guarantee `r8unorm` storage.

@group(0) @binding(3) var plane_y: texture_storage_2d<r8unorm, write>;
@group(0) @binding(4) var plane_cb: texture_storage_2d<r8unorm, write>;
@group(0) @binding(5) var plane_cr: texture_storage_2d<r8unorm, write>;

// Stores an 8-bit level into the texture of `component`. The level is a whole
// number from 0 to 255, and Metal turns a float into a UNORM texel by rounding
// to the nearest level, so level / 255 comes back as the level exactly; the
// tests compare the stores with the packed planes byte for byte.
fn store_level(component: u32, at: vec2<u32>, level: f32) {
    let texel = vec4<f32>(level / 255.0, 0.0, 0.0, 1.0);
    switch (component) {
        case 0u: {
            textureStore(plane_y, at, texel);
        }
        case 1u: {
            textureStore(plane_cb, at, texel);
        }
        default: {
            textureStore(plane_cr, at, texel);
        }
    }
}

// Every invocation stores four samples of a row of the tile, those inside the
// plane, whose texture is the plane's size.
fn store_plane(comp: CompParams, component: u32, tile_x: u32, tile_y: u32, tid: u32) {
    let row = tid / 8u;
    let column = 4u * (tid % 8u);
    let y = tile_y + row;
    let x = tile_x + column;
    let at = row * TILE + column;
    if (y < comp.plane_height) {
        if (x < comp.plane_width) {
            store_level(component, vec2<u32>(x, y), window[at]);
        }
        if (x + 1u < comp.plane_width) {
            store_level(component, vec2<u32>(x + 1u, y), window[at + 1u]);
        }
        if (x + 2u < comp.plane_width) {
            store_level(component, vec2<u32>(x + 2u, y), window[at + 2u]);
        }
        if (x + 3u < comp.plane_width) {
            store_level(component, vec2<u32>(x + 3u, y), window[at + 3u]);
        }
    }
}
