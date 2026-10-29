//! Collecting the blocks of a frame from packets.
//!
//! This is the part of decoding that runs on the CPU in the original
//! implementation. It turns a stream of packets into the two flat arrays the
//! dequantisation stage consumes: a table with the position of every 32x32
//! block in the payload, and the payload words themselves. The behaviour
//! follows upstream's `Decoder::push_packet` (sequence handling, first copy of
//! a duplicate block wins) with one deliberate difference: upstream starts the
//! new frame as soon as a header of a newer frame arrives and validates it
//! afterwards, so a corrupt packet can discard the frame being assembled.
//! Here a header or block is validated first, and one that is rejected changes
//! nothing.

use super::block::BlockStructure;
use super::consts::{
    DECOMPOSITION_LEVELS, HEADER_BYTES, SEQUENCE_COUNT_MASK, SEQUENCE_MAX_FORWARD_DISTANCE,
};
use super::error::Error;
use super::header::{Header, SequenceHeader, peek_sequence};
use super::layout::{Band, Component, FrameLayout};

/// Value in the block offset table for a block that has not been received.
pub const MISSING_BLOCK: u32 = u32::MAX;

/// Default number of coarse bands that must be complete before a partial
/// frame may be decoded (the next-to-final LL band).
///
/// Source: upstream `Decoder::decode_is_ready`.
pub const DEFAULT_PRISTINE_BANDS: usize = 2;

/// Default fraction of the announced blocks that must have arrived before a
/// partial frame may be decoded.
///
/// Source: upstream `Decoder::decode_is_ready`.
pub const DEFAULT_MINIMUM_PACKET_RATIO: f32 = 0.9;

/// How a sequence number relates to the frame being assembled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Freshness {
    /// Belongs to an older frame: ignore.
    Stale,
    /// Belongs to the frame being assembled.
    Same,
    /// Starts a new frame (or is the very first packet).
    Newer,
}

/// Assembles the blocks of one frame at a time from packets.
#[derive(Debug, Clone)]
pub struct FrameAssembler {
    layout: FrameLayout,
    block_offsets: Vec<u32>,
    payload: Vec<u32>,
    received_blocks: u32,
    total_blocks: u32,
    last_sequence: Option<u8>,
    frame_decoded: bool,
    start_of_frame: Option<SequenceHeader>,
}

impl FrameAssembler {
    /// Creates an assembler for frames with the given layout.
    pub fn new(layout: FrameLayout) -> Self {
        let block_count = layout.block_count();
        let mut assembler = Self {
            block_offsets: vec![MISSING_BLOCK; block_count as usize],
            payload: Vec::new(),
            received_blocks: 0,
            total_blocks: block_count,
            last_sequence: None,
            frame_decoded: false,
            start_of_frame: None,
            layout,
        };
        assembler.clear();
        assembler
    }

    /// The layout frames are assembled for.
    pub fn layout(&self) -> &FrameLayout {
        &self.layout
    }

    /// Drops everything received so far. Allocated memory is kept.
    pub fn clear(&mut self) {
        self.block_offsets.fill(MISSING_BLOCK);
        self.payload.clear();
        self.received_blocks = 0;
        self.total_blocks = self.layout.block_count();
        self.last_sequence = None;
        self.frame_decoded = false;
        self.start_of_frame = None;
    }

    /// Number of distinct blocks received for the current frame.
    pub fn received_blocks(&self) -> u32 {
        self.received_blocks
    }

    /// Number of blocks the current frame is announced to have.
    ///
    /// Before a start of frame header arrives this is the number of blocks
    /// the layout can hold.
    pub fn total_blocks(&self) -> u32 {
        self.total_blocks
    }

    /// Sequence number (modulo 8) of the frame being assembled.
    pub fn sequence(&self) -> Option<u8> {
        self.last_sequence
    }

    /// The most recent start of frame header of the current frame.
    pub fn start_of_frame(&self) -> Option<&SequenceHeader> {
        self.start_of_frame.as_ref()
    }

    /// Feeds one packet: a start of frame header and/or any number of whole
    /// blocks back to back.
    ///
    /// Packets of an older frame are ignored. A packet of a newer frame
    /// discards what was collected so far. Blocks that were already received
    /// are ignored.
    ///
    /// A header or block that is rejected changes nothing, even if it belongs
    /// to a newer frame; blocks of the same packet that came before it are
    /// kept.
    pub fn push_packet(&mut self, mut data: &[u8]) -> Result<(), Error> {
        while data.len() >= HEADER_BYTES {
            let header = match Header::parse(data) {
                Ok(header) => header,
                // Like any other header of an old frame, one whose kind this
                // decoder does not know is ignored together with the rest of
                // the packet.
                Err(Error::UnknownExtendedCode(code)) => {
                    let stale = peek_sequence(data)
                        .is_some_and(|sequence| self.freshness(sequence) == Freshness::Stale);
                    return if stale {
                        Ok(())
                    } else {
                        Err(Error::UnknownExtendedCode(code))
                    };
                }
                Err(error) => return Err(error),
            };

            match header {
                Header::StartOfFrame(header) => {
                    if header.chroma != self.layout.chroma() {
                        return Err(Error::ChromaMismatch {
                            stream: header.chroma,
                            expected: self.layout.chroma(),
                        });
                    }

                    let freshness = self.freshness(header.sequence);
                    if freshness == Freshness::Stale {
                        return Ok(());
                    }

                    let stream = (header.width, header.height);
                    let expected = (self.layout.width(), self.layout.height());
                    if stream != expected {
                        return Err(Error::DimensionMismatch { stream, expected });
                    }

                    if freshness == Freshness::Newer {
                        self.restart(header.sequence);
                    }
                    self.total_blocks = header.total_blocks;
                    self.start_of_frame = Some(header);
                    data = &data[HEADER_BYTES..];
                }
                Header::Block(header) => {
                    let size = header.payload_bytes();
                    if size > data.len() {
                        return Err(Error::TruncatedBlock {
                            claimed: size,
                            available: data.len(),
                        });
                    }
                    // A block of an older frame is ignored, whatever else is
                    // wrong with it, as upstream ignores it.
                    let freshness = self.freshness(header.sequence);
                    if freshness == Freshness::Stale {
                        return Ok(());
                    }
                    // Before the data moves on: a block shorter than its
                    // header would not move it.
                    if size < HEADER_BYTES {
                        return Err(Error::BlockTooSmall {
                            words: usize::from(header.payload_words),
                        });
                    }

                    let count = self.layout.block_count();
                    if header.block_index >= count {
                        return Err(Error::BlockIndexOutOfRange {
                            index: header.block_index,
                            count,
                        });
                    }

                    // Validate before anything is changed, so a rejected block
                    // never discards the frame being assembled.
                    BlockStructure::check_bytes(&data[..size])?;

                    if freshness == Freshness::Newer {
                        self.restart(header.sequence);
                    }

                    self.insert_block(header.block_index, &data[..size])?;
                    data = &data[size..];
                }
            }
        }

        if data.is_empty() {
            Ok(())
        } else {
            Err(Error::TrailingBytes(data.len()))
        }
    }

    /// Marks the current frame as decoded so it is not decoded twice.
    pub fn mark_frame_decoded(&mut self) {
        self.frame_decoded = true;
    }

    /// Whether the current frame can be decoded, using the default
    /// thresholds for partial frames.
    ///
    /// A complete frame is always ready. If `allow_partial_frame` is set, a
    /// frame that lost packets is ready once the coarsest bands are complete
    /// and more than 90 percent of the blocks have arrived.
    pub fn is_ready(&self, allow_partial_frame: bool) -> bool {
        self.is_ready_with(
            allow_partial_frame,
            DEFAULT_PRISTINE_BANDS,
            DEFAULT_MINIMUM_PACKET_RATIO,
            None,
        )
    }

    /// Like [`is_ready`](Self::is_ready) with explicit thresholds.
    ///
    /// `pristine_bands` is how many of the coarsest bands must be complete
    /// (1 is the lowest resolution LL band, 2 adds the detail bands needed
    /// for the next resolution, and so on; values above 4 count as 4).
    /// `active_block_mask` has one bit per block index; a missing block whose
    /// bit is clear is not expected and does not count against the frame.
    /// Without a mask every block is expected.
    pub fn is_ready_with(
        &self,
        allow_partial_frame: bool,
        pristine_bands: usize,
        minimum_packet_ratio: f32,
        active_block_mask: Option<&[u32]>,
    ) -> bool {
        if self.frame_decoded || self.last_sequence.is_none() {
            return false;
        }

        if self.received_blocks < self.total_blocks {
            if !allow_partial_frame || !self.has_pristine_bands(pristine_bands, active_block_mask) {
                return false;
            }
            if self.received_blocks as f32 <= self.total_blocks as f32 * minimum_packet_ratio {
                return false;
            }
        }

        true
    }

    /// The blocks collected for the current frame.
    pub fn frame(&self) -> FrameView<'_> {
        FrameView {
            layout: &self.layout,
            block_offsets: &self.block_offsets,
            payload: &self.payload,
            start_of_frame: self.start_of_frame.as_ref(),
        }
    }

    fn freshness(&self, sequence: u8) -> Freshness {
        match self.last_sequence {
            None => Freshness::Newer,
            Some(last) => match sequence.wrapping_sub(last) & SEQUENCE_COUNT_MASK {
                0 => Freshness::Same,
                distance if distance > SEQUENCE_MAX_FORWARD_DISTANCE => Freshness::Stale,
                _ => Freshness::Newer,
            },
        }
    }

    fn restart(&mut self, sequence: u8) {
        self.clear();
        self.last_sequence = Some(sequence);
    }

    /// Appends a block that has been validated, unless its index was seen before.
    fn insert_block(&mut self, block_index: u32, block: &[u8]) -> Result<(), Error> {
        let Some(slot) = self.block_offsets.get_mut(block_index as usize) else {
            return Ok(());
        };
        if *slot != MISSING_BLOCK {
            return Ok(());
        }

        let offset = u32::try_from(self.payload.len()).map_err(|_| Error::FrameTooLarge)?;
        self.payload.extend(
            block
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        );
        *slot = offset;
        self.received_blocks += 1;
        Ok(())
    }

    fn block_is_missing(&self, block_index: u32, active_block_mask: Option<&[u32]>) -> bool {
        if self.block_offsets.get(block_index as usize) != Some(&MISSING_BLOCK) {
            return false;
        }

        let word = active_block_mask.and_then(|mask| mask.get(block_index as usize / 32));
        match word {
            // Without usable mask information every block is expected.
            None => true,
            Some(word) => (word >> (block_index % 32)) & 1 != 0,
        }
    }

    fn has_pristine_bands(&self, bands: usize, active_block_mask: Option<&[u32]>) -> bool {
        // Upstream asserts bands < DECOMPOSITION_LEVELS; clamp instead of panicking.
        let bands = bands.min(DECOMPOSITION_LEVELS - 1);

        let all_present = |component: Component, level: usize, band: Band| {
            self.layout
                .band(component, level, band)
                .is_none_or(|layout| {
                    (layout.first_block..layout.first_block + layout.block_count())
                        .all(|index| !self.block_is_missing(index, active_block_mask))
                })
        };

        (0..bands).all(|band| {
            Component::ALL.into_iter().all(|component| {
                if band == 0 {
                    all_present(component, DECOMPOSITION_LEVELS - 1, Band::Ll)
                } else {
                    // The three detail bands of this level are enough to
                    // rebuild the LL band of the next finer level.
                    [Band::Hl, Band::Lh, Band::Hh]
                        .into_iter()
                        .all(|detail| all_present(component, DECOMPOSITION_LEVELS - band, detail))
                }
            })
        })
    }
}

/// A read-only view of the blocks collected for a frame.
#[derive(Debug, Clone, Copy)]
pub struct FrameView<'a> {
    layout: &'a FrameLayout,
    block_offsets: &'a [u32],
    payload: &'a [u32],
    start_of_frame: Option<&'a SequenceHeader>,
}

impl<'a> FrameView<'a> {
    /// The frame layout.
    pub fn layout(&self) -> &'a FrameLayout {
        self.layout
    }

    /// Word offset of every block in [`payload`](Self::payload), indexed by
    /// block index; [`MISSING_BLOCK`] for blocks that were not received.
    pub fn block_offsets(&self) -> &'a [u32] {
        self.block_offsets
    }

    /// The received blocks back to back, headers included.
    pub fn payload(&self) -> &'a [u32] {
        self.payload
    }

    /// The start of frame header that came with this frame, if any.
    pub fn start_of_frame(&self) -> Option<&'a SequenceHeader> {
        self.start_of_frame
    }

    /// The words of one block, header included, or `None` if it was not
    /// received.
    pub fn block(&self, block_index: u32) -> Option<&'a [u32]> {
        let offset = *self.block_offsets.get(block_index as usize)?;
        if offset == MISSING_BLOCK {
            return None;
        }

        let offset = offset as usize;
        let words = ((self.payload.get(offset)? >> 16) & 0xfff) as usize;
        self.payload.get(offset..offset.checked_add(words)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::header::ChromaSubsampling;

    fn layout() -> FrameLayout {
        // 160x96 4:2:0 has 66 blocks; indices 0..12 are the level 4 blocks.
        FrameLayout::new(160, 96, ChromaSubsampling::Yuv420).unwrap()
    }

    fn start_of_frame(sequence: u8, total_blocks: u32) -> Vec<u8> {
        let word0 = (159u32) | (95 << 14) | (u32::from(sequence) << 28) | (1 << 31);
        let word1 = total_blocks; // 4:2:0, code 0
        [word0.to_le_bytes(), word1.to_le_bytes()].concat()
    }

    /// A block with an empty ballot: just the 2 header words plus `extra` words.
    fn block(index: u32, sequence: u8, extra_words: u16) -> Vec<u8> {
        let words = 2 + u32::from(extra_words);
        let word0 = (words << 16) | (u32::from(sequence) << 28);
        let word1 = index << 8;
        let mut bytes = [word0.to_le_bytes(), word1.to_le_bytes()].concat();
        bytes.resize(words as usize * 4, 0xAA);
        bytes
    }

    fn assembler() -> FrameAssembler {
        FrameAssembler::new(layout())
    }

    #[test]
    fn complete_frame_is_ready_and_incomplete_is_not() {
        let mut a = assembler();
        assert!(!a.is_ready(false));

        a.push_packet(&start_of_frame(1, 3)).unwrap();
        assert_eq!((a.sequence(), a.total_blocks()), (Some(1), 3));
        assert!(!a.is_ready(false));

        a.push_packet(&[block(0, 1, 0), block(5, 1, 3)].concat())
            .unwrap();
        assert_eq!(a.received_blocks(), 2);
        assert!(!a.is_ready(false));

        a.push_packet(&block(40, 1, 1)).unwrap();
        assert!(a.is_ready(false));

        let frame = a.frame();
        assert_eq!(frame.block(0).map(<[u32]>::len), Some(2));
        assert_eq!(frame.block(5).map(<[u32]>::len), Some(5));
        assert_eq!(frame.block(40).map(<[u32]>::len), Some(3));
        assert_eq!(frame.block(1), None);
        assert_eq!(frame.block(10_000), None);
        assert_eq!(frame.payload().len(), 2 + 5 + 3);
        assert_eq!(frame.start_of_frame().map(|h| h.total_blocks), Some(3));

        a.mark_frame_decoded();
        assert!(!a.is_ready(false));
    }

    #[test]
    fn blocks_may_arrive_before_the_start_of_frame_header() {
        let mut a = assembler();
        a.push_packet(&block(2, 4, 0)).unwrap();
        assert_eq!(a.total_blocks(), a.layout().block_count());
        a.push_packet(&start_of_frame(4, 1)).unwrap();
        assert_eq!(a.received_blocks(), 1);
        assert!(a.is_ready(false));
    }

    #[test]
    fn first_copy_of_a_duplicate_block_wins() {
        let mut a = assembler();
        let mut first = block(7, 2, 1);
        first[8] = 0x11;
        let mut second = block(7, 2, 1);
        second[8] = 0x22;
        a.push_packet(&first).unwrap();
        a.push_packet(&second).unwrap();

        assert_eq!(a.received_blocks(), 1);
        let words = a.frame().block(7).unwrap();
        assert_eq!(words[2] & 0xff, 0x11);
        assert_eq!(a.frame().payload().len(), 3);
    }

    #[test]
    fn a_block_of_an_older_frame_is_ignored_even_when_malformed() {
        // As upstream: whether a block is stale comes before its own checks.
        let mut a = assembler();
        a.push_packet(&block(0, 3, 0)).unwrap();
        // One word, shorter than its own header.
        let tiny = |sequence: u8| {
            let word0 = (1u32 << 16) | (u32::from(sequence) << 28);
            [word0.to_le_bytes(), 0u32.to_le_bytes()].concat()
        };
        assert!(a.push_packet(&tiny(0)).is_ok(), "frame 0 is older than 3");
        assert_eq!((a.sequence(), a.received_blocks()), (Some(3), 1));
        assert!(matches!(
            a.push_packet(&tiny(3)),
            Err(Error::BlockTooSmall { words: 1 })
        ));
        assert_eq!((a.sequence(), a.received_blocks()), (Some(3), 1));
    }

    #[test]
    fn newer_frame_discards_and_older_frame_is_ignored() {
        let mut a = assembler();
        a.push_packet(&[block(0, 1, 0), block(1, 1, 0)].concat())
            .unwrap();
        assert_eq!(a.received_blocks(), 2);

        // Sequence 2 is newer: starts over.
        a.push_packet(&block(3, 2, 0)).unwrap();
        assert_eq!((a.sequence(), a.received_blocks()), (Some(2), 1));
        assert!(a.frame().block(0).is_none());

        // Sequence 7 is 5 behind 2 (modulo 8): stale, ignored, rest of packet dropped.
        a.push_packet(&[block(9, 7, 0), block(10, 2, 0)].concat())
            .unwrap();
        assert_eq!((a.sequence(), a.received_blocks()), (Some(2), 1));
    }

    #[test]
    fn sequence_counter_wraps() {
        let mut a = assembler();
        a.push_packet(&block(0, 7, 0)).unwrap();
        a.push_packet(&block(1, 0, 0)).unwrap();
        assert_eq!((a.sequence(), a.received_blocks()), (Some(0), 1));
    }

    #[test]
    fn malformed_blocks_are_rejected_without_side_effects() {
        let mut a = assembler();
        a.push_packet(&block(0, 1, 0)).unwrap();

        // Index out of range.
        assert_eq!(
            a.push_packet(&block(66, 1, 0)),
            Err(Error::BlockIndexOutOfRange {
                index: 66,
                count: 66
            })
        );

        // Claims 6 words but only 4 follow.
        let mut truncated = block(1, 1, 4);
        truncated.truncate(16);
        assert_eq!(
            a.push_packet(&truncated),
            Err(Error::TruncatedBlock {
                claimed: 24,
                available: 16
            })
        );

        // Smaller than its own header.
        let mut tiny = block(1, 1, 0);
        tiny[2] = 1; // payload_words = 1
        tiny[3] = 0x10;
        assert_eq!(a.push_packet(&tiny), Err(Error::BlockTooSmall { words: 1 }));

        // Ballot promises data that is not there.
        let mut hollow = block(1, 1, 0);
        hollow[0] = 0xFF;
        hollow[1] = 0xFF;
        assert!(matches!(
            a.push_packet(&hollow),
            Err(Error::BlockContentOverrun { .. })
        ));

        // Trailing bytes after a block.
        assert_eq!(
            a.push_packet(&[block(2, 1, 0), vec![0; 5]].concat()),
            Err(Error::TrailingBytes(5))
        );

        // The good blocks before the failure stay, the bad ones left no trace.
        assert_eq!(a.received_blocks(), 2);
        assert_eq!(a.frame().payload().len(), 4);
        assert!(a.frame().block(1).is_none());
    }

    #[test]
    fn a_rejected_block_of_a_newer_frame_leaves_the_frame_in_place() {
        let mut a = assembler();
        a.push_packet(&[start_of_frame(1, 2), block(0, 1, 0), block(5, 1, 0)].concat())
            .unwrap();
        assert!(a.is_ready(false));

        // Sequence 2 is newer, but the block promises data it does not have.
        let mut hollow = block(1, 2, 0);
        hollow[0] = 0xFF;
        hollow[1] = 0xFF;
        assert!(matches!(
            a.push_packet(&hollow),
            Err(Error::BlockContentOverrun { .. })
        ));
        // Sequence 3, index out of range.
        assert!(matches!(
            a.push_packet(&block(66, 3, 0)),
            Err(Error::BlockIndexOutOfRange { .. })
        ));
        // Sequence 4, smaller than its header.
        let mut tiny = block(1, 4, 0);
        tiny[2] = 1;
        tiny[3] = 0x40;
        assert!(matches!(
            a.push_packet(&tiny),
            Err(Error::BlockTooSmall { .. })
        ));
        // A start of frame of a newer frame with the wrong size.
        let mut wrong = start_of_frame(2, 9);
        wrong[0] = 0;
        assert!(matches!(
            a.push_packet(&wrong),
            Err(Error::DimensionMismatch { .. })
        ));

        // The first frame survived all of that.
        assert_eq!((a.sequence(), a.received_blocks()), (Some(1), 2));
        assert!(a.is_ready(false));
        assert!(a.frame().block(0).is_some() && a.frame().block(5).is_some());
    }

    #[test]
    fn an_old_header_of_an_unknown_kind_is_ignored_but_a_current_one_is_an_error() {
        let mut a = assembler();
        a.push_packet(&[block(0, 3, 0)].concat()).unwrap();

        // Extended header with code 2 and sequence 3 + 5 = 0 (mod 8): stale,
        // so it is dropped together with the rest of the packet.
        let unknown = |sequence: u8| {
            let word0 = (u32::from(sequence) << 28) | (1 << 31);
            let word1 = 2u32 << 24;
            [word0.to_le_bytes(), word1.to_le_bytes()].concat()
        };
        assert_eq!(
            a.push_packet(&[unknown(0), block(7, 3, 0)].concat()),
            Ok(())
        );
        assert_eq!(a.received_blocks(), 1);

        // The same kind of header for the current or a newer frame is an error
        // and changes nothing.
        for sequence in [3, 4] {
            assert_eq!(
                a.push_packet(&unknown(sequence)),
                Err(Error::UnknownExtendedCode(2))
            );
        }
        assert_eq!((a.sequence(), a.received_blocks()), (Some(3), 1));
    }

    #[test]
    fn start_of_frame_must_match_the_layout() {
        let mut a = assembler();

        let mut wrong_size = start_of_frame(1, 5);
        wrong_size[0] = 0x00; // width 1 instead of 160 (low bits)
        assert!(matches!(
            a.push_packet(&wrong_size),
            Err(Error::DimensionMismatch { .. })
        ));

        let mut wrong_chroma = start_of_frame(1, 5);
        wrong_chroma[7] |= 0x04; // chroma_resolution = 4:4:4
        assert_eq!(
            a.push_packet(&wrong_chroma),
            Err(Error::ChromaMismatch {
                stream: ChromaSubsampling::Yuv444,
                expected: ChromaSubsampling::Yuv420,
            })
        );

        let mut unknown = start_of_frame(1, 5);
        unknown[7] |= 0x02; // code = 2
        assert_eq!(a.push_packet(&unknown), Err(Error::UnknownExtendedCode(2)));
    }

    #[test]
    fn partial_frames_need_pristine_coarse_bands_and_enough_blocks() {
        let mut a = assembler();
        a.push_packet(&start_of_frame(1, 20)).unwrap();

        // All 12 level 4 blocks (the two coarsest bands) are present, but that is
        // 60 percent of the 20 announced blocks.
        let level4: Vec<u8> = (0..12).flat_map(|i| block(i, 1, 0)).collect();
        a.push_packet(&level4).unwrap();

        assert!(!a.is_ready(false));
        assert!(!a.is_ready(true));
        assert!(a.is_ready_with(true, 2, 0.5, None));
        assert!(!a.is_ready_with(false, 2, 0.5, None));
        assert!(!a.is_ready_with(true, 2, 0.7, None));

        // Pristine bands beyond what has arrived (level 3 detail is missing).
        assert!(!a.is_ready_with(true, 3, 0.5, None));
        // A mask that does not expect the missing blocks lets that pass.
        let mut mask = [0u32; 3];
        for i in 0..12 {
            mask[i / 32] |= 1 << (i % 32);
        }
        assert!(a.is_ready_with(true, 3, 0.5, Some(&mask)));
        // Out-of-range band counts are clamped rather than panicking.
        assert!(!a.is_ready_with(true, 99, 0.5, None));
    }

    #[test]
    fn a_partial_frame_needs_more_than_the_share() {
        let mut a = assembler();
        a.push_packet(&start_of_frame(1, 24)).unwrap();
        // The 12 blocks of level 4 are half of the 24 announced: not more.
        let level4: Vec<u8> = (0..12).flat_map(|i| block(i, 1, 0)).collect();
        a.push_packet(&level4).unwrap();
        assert!(!a.is_ready_with(true, 2, 0.5, None));
        a.push_packet(&block(12, 1, 0)).unwrap();
        assert!(a.is_ready_with(true, 2, 0.5, None));
    }

    #[test]
    fn missing_coarse_block_blocks_partial_decode() {
        let mut a = assembler();
        a.push_packet(&start_of_frame(1, 20)).unwrap();
        let blocks: Vec<u8> = (0..12)
            .filter(|&i| i != 5)
            .flat_map(|i| block(i, 1, 0))
            .collect();
        a.push_packet(&blocks).unwrap();
        assert!(!a.is_ready_with(true, 2, 0.1, None));
        // One band only needs the LL blocks (0, 4 and 8).
        assert!(a.is_ready_with(true, 1, 0.1, None));
    }

    #[test]
    fn clear_resets_everything() {
        let mut a = assembler();
        a.push_packet(&[start_of_frame(3, 1), block(0, 3, 2)].concat())
            .unwrap();
        assert!(a.is_ready(false));
        a.clear();
        assert_eq!(a.received_blocks(), 0);
        assert_eq!(a.sequence(), None);
        assert!(a.start_of_frame().is_none());
        assert!(a.frame().payload().is_empty());
        assert!(!a.is_ready(false));
    }
}
