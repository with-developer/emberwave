//! Tests against the committed vectors, which were produced by the original
//! PyroWave encoder (see `testdata/README.md`).

mod common;

use common::{Reference, max_abs_diff, psnr, read_reference, read_vector, vector_names};
use emberwave_core::bitstream::block::BlockStructure;
use emberwave_core::bitstream::{FrameAssembler, FrameLayout, Pwv1File};
use emberwave_core::cpu;

/// Assembles every frame of a vector, calling `check` with the assembler
/// after each frame and the index of the frame.
fn for_each_frame(name: &str, mut check: impl FnMut(&FrameAssembler, usize)) {
    let bytes = read_vector(name);
    let file = Pwv1File::parse(&bytes).expect("PWV1 header");
    let stream = file.header().stream;
    let layout = FrameLayout::new(stream.width, stream.height, stream.chroma).expect("layout");
    let mut assembler = FrameAssembler::new(layout);

    let mut frames = 0;
    for (index, frame) in file.frames().enumerate() {
        let frame = frame.expect("frame data");
        assembler.push_packet(frame).expect("packet parses");
        check(&assembler, index);
        frames += 1;
    }
    assert!(frames > 0, "{name} has no frames, so nothing was checked");
}

#[test]
fn headers_match_the_reference_decode() {
    for name in vector_names() {
        let bytes = read_vector(&name);
        let file = Pwv1File::parse(&bytes).unwrap();
        let reference = read_reference(&name, Reference::Default);

        let stream = file.header().stream;
        assert_eq!(stream.width as usize, reference.width, "{name}");
        assert_eq!(stream.height as usize, reference.height, "{name}");
        assert_eq!(
            stream.chroma == emberwave_core::bitstream::ChromaSubsampling::Yuv444,
            reference.chroma_444,
            "{name}"
        );
        assert_eq!(file.frames().count(), reference.frames.len(), "{name}");
    }
}

#[test]
fn every_frame_assembles_completely() {
    for name in vector_names() {
        for_each_frame(&name, |assembler, index| {
            let sof = assembler
                .start_of_frame()
                .unwrap_or_else(|| panic!("{name} frame {index}: no start of frame header"));
            assert_eq!(
                (sof.width, sof.height),
                (assembler.layout().width(), assembler.layout().height()),
                "{name} frame {index}"
            );
            assert_eq!(
                assembler.received_blocks(),
                sof.total_blocks,
                "{name} frame {index}: announced vs received blocks"
            );
            assert!(assembler.is_ready(false), "{name} frame {index}");

            // The sequence counter advances by one per frame, starting at 1.
            let expected_sequence = ((index + 1) % 8) as u8;
            assert_eq!(sof.sequence, expected_sequence, "{name} frame {index}");
            assert_eq!(
                assembler.sequence(),
                Some(expected_sequence),
                "{name} frame {index}"
            );
        });
    }
}

#[test]
fn block_sizes_follow_the_documented_layout_exactly() {
    // payload_words == ceil((header + tables + bit-planes + ceil(signs / 8)) / 4)
    // for every block the original encoder wrote. This pins down the layout
    // of the planes and the sign area, including their sizes.
    let mut blocks_checked = 0;
    for name in vector_names() {
        for_each_frame(&name, |assembler, index| {
            let frame = assembler.frame();
            let layout = frame.layout();
            for block_index in 0..layout.block_count() {
                let Some(words) = frame.block(block_index) else {
                    continue;
                };
                let location = layout.locate(block_index).unwrap();
                let band = layout
                    .band(location.component, location.level, location.band)
                    .unwrap();

                let structure = BlockStructure::parse(words).unwrap();
                assert_eq!(structure.header().block_index, block_index);
                assert_ne!(
                    structure.header().ballot,
                    0,
                    "{name}: empty blocks are not transmitted"
                );

                let mut scratch = vec![0.0f32; (band.width * band.height) as usize];
                let sign_bits = cpu::dequant::decode_block(
                    words,
                    band,
                    location.block_x,
                    location.block_y,
                    &mut scratch,
                )
                .unwrap();

                let used_bytes = structure.sign_offset() + sign_bits.div_ceil(8);
                assert_eq!(
                    words.len(),
                    used_bytes.div_ceil(4),
                    "{name} frame {index} block {block_index} ({location:?}): {sign_bits} sign bits"
                );
                blocks_checked += 1;
            }
        });
    }
    assert!(
        blocks_checked > 100,
        "only {blocks_checked} blocks were checked"
    );
}

/// Decodes every frame of every vector on the CPU and compares it to the
/// decodes made by the original decoder.
#[test]
fn cpu_decode_matches_the_reference_decoder() {
    for which in Reference::ALL {
        let mut worst = f64::INFINITY;
        for name in vector_names() {
            let reference = read_reference(&name, which);
            for_each_frame(&name, |assembler, index| {
                let decoded = cpu::decode_frame(&assembler.frame()).expect("decode");
                for (plane_index, (plane, expected)) in decoded
                    .planes
                    .iter()
                    .zip(&reference.frames[index])
                    .enumerate()
                {
                    let actual = plane.to_unorm8();
                    let db = psnr(expected, &actual);
                    let max_diff = max_abs_diff(expected, &actual);
                    println!(
                        "{which:?} {name:28} frame {index} plane {plane_index}: PSNR {db:8.2} dB, max |diff| {max_diff}"
                    );
                    worst = worst.min(db);
                }
            });
        }
        println!(
            "{which:?}: worst PSNR {worst:.2} dB (needs {})",
            which.min_psnr()
        );
        assert!(
            worst >= which.min_psnr(),
            "CPU decode disagrees with the {which:?} reference: {worst:.2} dB"
        );
    }
}
