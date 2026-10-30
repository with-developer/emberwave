//! The submission hooks of the calls that take them (decisions D61 to D63):
//! what runs, and in which order, on whatever backend runs the tests. The
//! hooks of the C API add Vulkan semaphores or Direct3D 12 fences and
//! transitions in these places.
//!
//! They need a GPU or a software Vulkan driver (lavapipe), and skip without
//! one unless `EMBERWAVE_REQUIRE_GPU` is set.

#![cfg(feature = "gpu")]

mod common;

use common::{shared_context, vector_frames};
use emberwave_core::bitstream::Component;
use emberwave_core::encode::Signalling;
use emberwave_core::gpu::{Decoder, Encoder, SubmissionHooks, TexturePicture};

/// Notes what the call ran, in order.
#[derive(Default)]
struct Recorder {
    events: Vec<String>,
}

impl SubmissionHooks for Recorder {
    fn record(&mut self, encoder: &mut wgpu::CommandEncoder) {
        // Something to record, as the Direct3D hooks record transitions.
        encoder.push_debug_group("hooks");
        encoder.pop_debug_group();
        self.events.push("record".into());
    }

    fn before_submit(&mut self, _queue: &wgpu::Queue) {
        self.events.push("before".into());
    }

    fn after_submit(&mut self, _queue: &wgpu::Queue, submitted: bool) {
        self.events.push(format!("after {submitted}"));
    }
}

#[test]
fn hooks_run_around_the_submission_of_a_decode_and_of_an_encode() {
    let Some(context) = shared_context() else {
        return;
    };
    let (layout, frames) = vector_frames("noise_192x136_420");
    let mut decoder =
        Decoder::new(context, layout.width(), layout.height(), layout.chroma()).expect("a decoder");
    let textures = Component::ALL.map(|component| {
        decoder
            .create_plane_texture(component)
            .expect("a plane texture")
    });
    let views = textures
        .each_ref()
        .map(|texture| texture.create_view(&Default::default()));

    // A decode: the commands, then the submission between the other two.
    decoder.push_packet(&frames[0]).expect("a packet");
    let mut hooks = Recorder::default();
    decoder
        .submit_assembled_to_plane_textures_with(views.each_ref(), &mut hooks)
        .expect("a decode");
    assert_eq!(hooks.events, ["record", "before", "after true"]);

    // A call that fails before it records anything runs none of them.
    let mut hooks = Recorder::default();
    let wrong = [&views[1], &views[0], &views[2]];
    decoder.push_packet(&frames[1]).expect("a packet");
    assert!(
        decoder
            .submit_assembled_to_plane_textures_with(wrong, &mut hooks)
            .is_err()
    );
    assert!(hooks.events.is_empty(), "{:?}", hooks.events);

    // An encode from the planes the decode left in the textures.
    let mut encoder = Encoder::new(
        context,
        layout.width(),
        layout.height(),
        layout.chroma(),
        Signalling::default(),
    )
    .expect("an encoder");
    let mut hooks = Recorder::default();
    let picture = TexturePicture::Planes(views.each_ref());
    encoder
        .encode_textures_with(picture, 20_000, &mut hooks)
        .expect("an encode");
    assert_eq!(hooks.events, ["record", "before", "after true"]);

    // A budget too small shows once the GPU has run: the hooks ran too.
    let mut hooks = Recorder::default();
    assert!(
        encoder
            .encode_textures_with(TexturePicture::Planes(views.each_ref()), 13, &mut hooks)
            .is_err()
    );
    assert_eq!(hooks.events, ["record", "before", "after true"]);
}
