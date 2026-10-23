//! Malformed input must never panic, and must never leave the assembler in an
//! inconsistent state.

mod common;

use std::sync::LazyLock;

use common::{read_vector, vector_names};
use emberwave_core::bitstream::block::BlockStructure;
use emberwave_core::bitstream::{
    ChromaSubsampling, FrameAssembler, FrameLayout, Header, MISSING_BLOCK, Pwv1File, Pwv1Header,
};
use emberwave_core::cpu;
use proptest::prelude::*;
use proptest::sample::Index;

/// The first frame of every committed vector, with the layout it belongs to.
static FRAMES: LazyLock<Vec<(FrameLayout, Vec<u8>)>> = LazyLock::new(|| {
    vector_names()
        .iter()
        .map(|name| {
            let bytes = read_vector(name);
            let file = Pwv1File::parse(&bytes).unwrap();
            let stream = file.header().stream;
            let layout = FrameLayout::new(stream.width, stream.height, stream.chroma).unwrap();
            let frame = file.frames().next().unwrap().unwrap().to_vec();
            (layout, frame)
        })
        .collect()
});

/// The smallest vector's frame: a solid colour picture of a few hundred bytes.
fn solid() -> &'static (FrameLayout, Vec<u8>) {
    FRAMES
        .iter()
        .min_by_key(|(_, frame)| frame.len())
        .expect("vectors")
}

/// Checks that what the assembler hands out is self-consistent.
fn check_invariants(assembler: &FrameAssembler) {
    let frame = assembler.frame();
    let offsets = frame.block_offsets();
    assert_eq!(offsets.len(), assembler.layout().block_count() as usize);

    let mut received = 0;
    let mut words_in_blocks = 0;
    for (index, &offset) in offsets.iter().enumerate() {
        if offset == MISSING_BLOCK {
            assert!(frame.block(index as u32).is_none());
            continue;
        }
        received += 1;

        let words = frame
            .block(index as u32)
            .expect("a recorded block has its words");
        words_in_blocks += words.len();

        let structure = BlockStructure::parse(words).expect("a stored block is well formed");
        assert_eq!(structure.header().block_index as usize, index);
        assert_eq!(structure.header().payload_words as usize, words.len());
    }

    assert_eq!(received, assembler.received_blocks());
    assert_eq!(words_in_blocks, frame.payload().len());
}

/// Pushes `data` into a fresh assembler, decodes whatever arrived and checks
/// the invariants. Errors are fine; panics are not.
fn push_and_decode(layout: &FrameLayout, data: &[u8]) {
    let mut assembler = FrameAssembler::new(layout.clone());
    let _ = assembler.push_packet(data);
    check_invariants(&assembler);
    let _ = cpu::decode_frame(&assembler.frame());
}

fn mutate(mut data: Vec<u8>, mutations: &[(Index, u8)], truncate: Option<Index>) -> Vec<u8> {
    if data.is_empty() {
        return data;
    }
    for (index, value) in mutations {
        let position = index.index(data.len());
        data[position] = *value;
    }
    if let Some(index) = truncate {
        data.truncate(index.index(data.len() + 1));
    }
    data
}

#[test]
fn every_prefix_of_a_real_frame() {
    let (layout, frame) = solid();
    for length in 0..=frame.len() {
        push_and_decode(layout, &frame[..length]);
    }
}

#[test]
fn every_single_bit_flip_of_a_real_frame() {
    let (layout, frame) = solid();
    let mut flipped = frame.clone();
    for bit in 0..frame.len() * 8 {
        flipped[bit / 8] ^= 1 << (bit % 8);

        // Parsing is cheap, so every flip is parsed; decoding is slower, so
        // only some are decoded as well.
        let mut assembler = FrameAssembler::new(layout.clone());
        let _ = assembler.push_packet(&flipped);
        check_invariants(&assembler);
        if bit % 13 == 0 {
            let _ = cpu::decode_frame(&assembler.frame());
        }

        flipped[bit / 8] ^= 1 << (bit % 8);
    }
}

// 256 cases per property by default; PROPTEST_CASES=100000 runs a deeper pass.
proptest! {
    #[test]
    fn random_bytes_never_panic(data in prop::collection::vec(any::<u8>(), 0..4096)) {
        let (layout, _) = solid();
        push_and_decode(layout, &data);
    }

    #[test]
    fn mutated_real_frames_never_panic(
        which in any::<Index>(),
        mutations in prop::collection::vec((any::<Index>(), any::<u8>()), 0..=8),
        truncate in prop::option::of(any::<Index>()),
    ) {
        let (layout, frame) = &FRAMES[which.index(FRAMES.len())];
        push_and_decode(layout, &mutate(frame.clone(), &mutations, truncate));
    }

    #[test]
    fn packet_sequences_keep_the_assembler_consistent(
        which in any::<Index>(),
        packets in prop::collection::vec(
            (prop::collection::vec((any::<Index>(), any::<u8>()), 0..=4), prop::option::of(any::<Index>())),
            1..=6,
        ),
    ) {
        let (layout, frame) = &FRAMES[which.index(FRAMES.len())];
        let mut assembler = FrameAssembler::new(layout.clone());
        for (mutations, truncate) in &packets {
            let _ = assembler.push_packet(&mutate(frame.clone(), mutations, *truncate));
            check_invariants(&assembler);
            let _ = assembler.is_ready(true);
        }
    }

    #[test]
    fn header_parsing_is_total(data in prop::collection::vec(any::<u8>(), 0..64)) {
        let _ = Header::parse(&data);
        let _ = Pwv1Header::parse(&data);
    }

    #[test]
    fn block_structure_parsing_is_total_and_consistent(
        words in prop::collection::vec(any::<u32>(), 0..64),
    ) {
        if let Ok(structure) = BlockStructure::parse(&words) {
            let size = words.len() * 4;
            prop_assert!(structure.sign_offset() <= size);
            let mut cursor = 8 + 3 * structure.active_count();
            for raster in 0..16 {
                if let Some(block8) = structure.block8(raster) {
                    prop_assert_eq!(block8.plane_offset, cursor);
                    cursor += block8.plane_bytes;
                }
            }
            prop_assert_eq!(cursor, structure.sign_offset());
        }
    }

    #[test]
    fn pwv1_files_never_panic(
        header in prop::collection::vec(any::<u8>(), 0..64),
        frames in prop::collection::vec(any::<u8>(), 0..256),
    ) {
        // Random bytes behind a valid magic and version reach the frame iterator.
        let mut bytes = b"PWV1".to_vec();
        bytes.extend_from_slice(&header);
        bytes.resize(21, 0);
        bytes.push(1);
        bytes.extend_from_slice(&[0, 0]);
        bytes.extend_from_slice(&frames);

        if let Ok(file) = Pwv1File::parse(&bytes) {
            for frame in file.frames() {
                let _ = frame;
            }
        }
    }

    #[test]
    fn layout_is_total_and_locates_every_block(
        width in 0u32..20_000,
        height in 0u32..20_000,
        chroma_444 in any::<bool>(),
        probe in any::<u32>(),
    ) {
        let chroma = if chroma_444 { ChromaSubsampling::Yuv444 } else { ChromaSubsampling::Yuv420 };
        if let Ok(layout) = FrameLayout::new(width, height, chroma) {
            let count = layout.block_count();
            prop_assert!(count > 0);
            for index in [0, count - 1, probe % count] {
                let location = layout.locate(index).expect("every index below the count is located");
                let band = layout.band(location.component, location.level, location.band).unwrap();
                prop_assert_eq!(
                    band.first_block + location.block_y * band.blocks_x + location.block_x,
                    index
                );
            }
            prop_assert!(layout.locate(count).is_none());
        }
    }
}
