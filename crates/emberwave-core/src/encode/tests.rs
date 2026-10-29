//! The encoder against the reference decoder: what goes in comes out, within
//! the budget, in frames the decoder's assembler accepts.

use super::*;
use crate::bitstream::FrameAssembler;
use crate::cpu::decode_frame;

/// Test pictures: a smooth gradient with a sharp edge and some texture, and
/// noise, which no budget compresses well.
fn picture(width: u32, height: u32, noise: bool, seed: u32) -> [Vec<u8>; 3] {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut random = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let plane = |w: u32, h: u32, offset: f32, random: &mut dyn FnMut() -> u32| -> Vec<u8> {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                if noise {
                    (random() >> 24) as u8
                } else {
                    let smooth = 60.0 + 120.0 * (x / w as f32) + 40.0 * (y * 0.07 + offset).sin();
                    let edge = if x > w as f32 * 0.6 { 30.0 } else { 0.0 };
                    let texture = ((x as u32 * 7 + y as u32 * 3) % 5) as f32 * 2.0;
                    (smooth + edge + texture).clamp(0.0, 255.0) as u8
                }
            })
            .collect()
    };
    let layout = FrameLayout::new(width, height, ChromaSubsampling::Yuv420)
        .or_else(|_| FrameLayout::new(width, height, ChromaSubsampling::Yuv444))
        .expect("a valid size");
    let (cw, ch) = layout.plane_size(Component::Cb);
    [
        plane(width, height, 0.0, &mut random),
        plane(cw, ch, 1.0, &mut random),
        plane(cw, ch, 2.0, &mut random),
    ]
}

fn chroma_planes(chroma: ChromaSubsampling, width: u32, height: u32, noise: bool) -> [Vec<u8>; 3] {
    let layout = FrameLayout::new(width, height, chroma).unwrap();
    let [y, ..] = picture(width, height, noise, 7);
    let (cw, ch) = layout.plane_size(Component::Cb);
    let [_, cb, cr] = picture(cw.max(1), ch.max(1), noise, 9);
    let fit = |plane: Vec<u8>, w: u32, h: u32| -> Vec<u8> {
        let source_width = cw.max(1);
        (0..w * h)
            .map(|i| plane[((i / w) * source_width + i % w) as usize % plane.len()])
            .collect()
    };
    [y, fit(cb, cw, ch), fit(cr, cw, ch)]
}

fn psnr(reference: &[u8], test: &[u8]) -> f64 {
    let squared: f64 = reference
        .iter()
        .zip(test)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    let mse = squared / reference.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

/// Assembles a frame as a decoder receives it and decodes it on the CPU.
fn decode(layout: &FrameLayout, frame: &[u8]) -> [Vec<u8>; 3] {
    let mut assembler = FrameAssembler::new(layout.clone());
    assembler
        .push_packet(frame)
        .expect("the decoder takes the frame");
    assert!(
        assembler.is_ready(false),
        "every block of the frame arrived"
    );
    let decoded = decode_frame(&assembler.frame()).expect("the frame decodes");
    decoded.planes.map(|plane| plane.to_unorm8())
}

fn worst_psnr(source: &[Vec<u8>; 3], decoded: &[Vec<u8>; 3]) -> f64 {
    source
        .iter()
        .zip(decoded)
        .map(|(a, b)| psnr(a, b))
        .fold(f64::INFINITY, f64::min)
}

fn encode(
    chroma: ChromaSubsampling,
    width: u32,
    height: u32,
    noise: bool,
    budget: usize,
) -> (FrameLayout, [Vec<u8>; 3], Vec<u8>) {
    let planes = chroma_planes(chroma, width, height, noise);
    let mut encoder = Encoder::new(width, height, chroma, Signalling::default()).unwrap();
    let frame = encoder
        .encode([&planes[0], &planes[1], &planes[2]], budget)
        .expect("the frame fits");
    (encoder.layout().clone(), planes, frame)
}

#[test]
fn frames_decode_close_to_their_pictures_within_the_budget() {
    for (chroma, width, height, budget, bar) in [
        // Plenty of bytes: near what 8 bits can tell apart (75.6 and 58.3 dB
        // when written).
        (ChromaSubsampling::Yuv420, 192, 136, 60_000, 70.0),
        (ChromaSubsampling::Yuv444, 131, 67, 60_000, 55.0),
        // Tight: the rate control has to drop planes.
        (ChromaSubsampling::Yuv420, 200, 120, 4_000, 30.0),
        (ChromaSubsampling::Yuv444, 160, 96, 3_000, 28.0),
    ] {
        let (layout, planes, frame) = encode(chroma, width, height, false, budget);
        assert!(
            frame.len() <= budget,
            "{width}x{height}: {} > {budget}",
            frame.len()
        );
        let quality = worst_psnr(&planes, &decode(&layout, &frame));
        assert!(
            quality >= bar,
            "{width}x{height} {chroma:?} at {budget} B: {quality:.2} dB"
        );
    }
}

#[test]
fn more_bytes_never_make_a_worse_picture() {
    let mut previous = 0.0;
    for budget in [1_500, 3_000, 6_000, 12_000, 24_000] {
        let (layout, planes, frame) = encode(ChromaSubsampling::Yuv420, 192, 136, false, budget);
        let quality = worst_psnr(&planes, &decode(&layout, &frame));
        assert!(
            quality >= previous - 0.01,
            "{budget} B: {quality:.2} dB after {previous:.2} dB"
        );
        previous = quality;
    }
}

#[test]
fn a_tight_budget_is_filled_and_not_exceeded() {
    // Noise needs far more than this: the frame should come close to the
    // budget without passing it.
    for budget in [2_000, 5_000, 20_000] {
        let (_, _, frame) = encode(ChromaSubsampling::Yuv420, 192, 136, true, budget);
        assert!(frame.len() <= budget, "{} > {budget}", frame.len());
        assert!(
            frame.len() >= budget * 9 / 10,
            "{} of {budget}",
            frame.len()
        );
    }
}

#[test]
fn the_start_of_frame_header_counts_the_blocks_and_signals_the_stream() {
    let planes = chroma_planes(ChromaSubsampling::Yuv420, 200, 120, false);
    let signalling = Signalling {
        ycbcr_range: YcbcrRange::Full,
        chroma_siting: ChromaSiting::Left,
        ..Signalling::default()
    };
    let mut encoder = Encoder::new(200, 120, ChromaSubsampling::Yuv420, signalling).unwrap();
    for expected_sequence in [1, 2, 3, 4, 5, 6, 7, 0, 1] {
        let frame = encoder
            .encode([&planes[0], &planes[1], &planes[2]], 8_000)
            .unwrap();
        let Ok(Header::StartOfFrame(header)) = Header::parse(&frame) else {
            panic!("a frame starts with a start of frame header");
        };
        assert_eq!(header.sequence, expected_sequence);
        assert_eq!((header.width, header.height), (200, 120));
        assert_eq!(header.ycbcr_range, YcbcrRange::Full);
        assert_eq!(header.chroma_siting, ChromaSiting::Left);

        // total_blocks is the number of blocks that follow, all of this frame.
        let blocks = packets(&frame, HEADER_BYTES)
            .into_iter()
            .skip(1)
            .map(|block| block_header(block).expect("a block header"))
            .collect::<Vec<_>>();
        assert_eq!(blocks.len() as u32, header.total_blocks);
        assert!(blocks.iter().all(|b| b.sequence == expected_sequence));
        assert!(
            blocks
                .windows(2)
                .all(|w| w[0].block_index < w[1].block_index)
        );
    }
}

#[test]
fn packets_reassemble_the_frame_in_any_order() {
    let (layout, _, frame) = encode(ChromaSubsampling::Yuv420, 192, 136, true, 20_000);
    let mut split = packets(&frame, 1_400);
    assert!(split.len() > 5);
    assert!(split.iter().all(|packet| packet.len() <= 1_400));
    assert_eq!(split.concat(), frame);

    split.reverse();
    let mut assembler = FrameAssembler::new(layout.clone());
    for packet in split {
        assembler.push_packet(packet).unwrap();
    }
    assert!(assembler.is_ready(false));
    let mut whole = FrameAssembler::new(layout);
    whole.push_packet(&frame).unwrap();
    assert_eq!(
        decode_frame(&assembler.frame())
            .unwrap()
            .planes
            .map(|p| p.to_unorm8()),
        decode_frame(&whole.frame())
            .unwrap()
            .planes
            .map(|p| p.to_unorm8())
    );
}

#[test]
fn bad_input_is_an_error() {
    let planes = chroma_planes(ChromaSubsampling::Yuv420, 192, 136, false);
    let mut encoder =
        Encoder::new(192, 136, ChromaSubsampling::Yuv420, Signalling::default()).unwrap();
    assert!(matches!(
        encoder.encode([&planes[0][1..], &planes[1], &planes[2]], 10_000),
        Err(EncodeError::PlaneSize {
            component: Component::Y,
            ..
        })
    ));
    assert!(matches!(
        encoder.encode([&planes[0], &planes[1], &planes[2]], 7),
        Err(EncodeError::BudgetTooSmall { budget: 7, .. })
    ));
    // Eight bytes always fit: a frame of nothing but its header.
    let frame = encoder
        .encode([&planes[0], &planes[1], &planes[2]], 8)
        .unwrap();
    assert_eq!(frame.len(), 8);
    assert!(Encoder::new(191, 136, ChromaSubsampling::Yuv420, Signalling::default()).is_err());
}

#[test]
fn a_flat_picture_costs_almost_nothing() {
    let layout = FrameLayout::new(160, 96, ChromaSubsampling::Yuv420).unwrap();
    let (cw, ch) = layout.plane_size(Component::Cb);
    let planes = [
        vec![77u8; 160 * 96],
        vec![128u8; (cw * ch) as usize],
        vec![90u8; (cw * ch) as usize],
    ];
    let mut encoder =
        Encoder::new(160, 96, ChromaSubsampling::Yuv420, Signalling::default()).unwrap();
    let frame = encoder
        .encode([&planes[0], &planes[1], &planes[2]], 100_000)
        .unwrap();
    // Only the LL band of each component carries anything: three blocks,
    // whose subblocks all pay for the base bit-planes of the largest value.
    // Upstream's encoder makes frames of this vector that size too.
    assert!(frame.len() < 300, "{} bytes", frame.len());
    assert_eq!(packets(&frame, HEADER_BYTES).len(), 1 + 3);
    assert_eq!(decode(&layout, &frame), planes);
}

#[test]
fn every_budget_holds_down_to_the_byte() {
    // Budgets that are not whole words included: the frame never passes
    // them, and it decodes.
    let planes = chroma_planes(ChromaSubsampling::Yuv420, 64, 64, true);
    let mut encoder =
        Encoder::new(64, 64, ChromaSubsampling::Yuv420, Signalling::default()).unwrap();
    for budget in (8..=260).chain(1501..=1507) {
        let frame = encoder
            .encode([&planes[0], &planes[1], &planes[2]], budget)
            .unwrap();
        assert!(frame.len() <= budget, "{} > {budget}", frame.len());
        decode(encoder.layout(), &frame);
    }
}

#[test]
fn packets_take_any_bytes_without_losing_one() {
    let (_, _, frame) = encode(ChromaSubsampling::Yuv420, 192, 136, true, 8_000);
    // Cut short, corrupted, and empty: every byte comes out again, in order.
    let mut damaged = frame.clone();
    damaged[HEADER_BYTES + 2] ^= 0xff;
    for bytes in [
        &frame[..frame.len() - 3],
        &frame[..5],
        &[][..],
        &damaged[..],
    ] {
        for max_bytes in [0, 1, 8, 100, 1_400, usize::MAX] {
            assert_eq!(packets(bytes, max_bytes).concat(), bytes);
        }
    }
}

#[test]
fn chroma_planes_of_the_wrong_size_are_an_error() {
    let planes = chroma_planes(ChromaSubsampling::Yuv420, 192, 136, false);
    let mut encoder =
        Encoder::new(192, 136, ChromaSubsampling::Yuv420, Signalling::default()).unwrap();
    for (component, planes) in [
        (
            Component::Cb,
            [&planes[0][..], &planes[1][1..], &planes[2][..]],
        ),
        (
            Component::Cr,
            [&planes[0][..], &planes[1][..], &planes[0][..]],
        ),
    ] {
        assert!(matches!(
            encoder.encode(planes, 10_000),
            Err(EncodeError::PlaneSize { component: c, .. }) if c == component
        ));
    }
}

/// The coefficient error of `frame` against `planes`, assembled as a decoder
/// receives it.
fn error_of(layout: &FrameLayout, planes: &[Vec<u8>; 3], frame: &[u8]) -> CoefficientError {
    let mut assembler = FrameAssembler::new(layout.clone());
    assembler.push_packet(frame).unwrap();
    coefficient_error([&planes[0], &planes[1], &planes[2]], &assembler.frame()).unwrap()
}

#[test]
fn the_coefficient_error_is_what_the_rate_control_counted() {
    // The rate control counts each 32x32 block's distortion at the drop it
    // chose; an 8x8 block too small ever to be coded counts for its whole
    // weighted energy, which no choice changes. Measured from the frame
    // instead, the sum is the same, up to the rounding of f32 sums.
    for (width, height, chroma, budget) in [
        (200, 120, ChromaSubsampling::Yuv420, 6000),
        (200, 120, ChromaSubsampling::Yuv420, 300),
        (131, 67, ChromaSubsampling::Yuv444, 4000),
    ] {
        let planes = chroma_planes(chroma, width, height, false);
        let sources = [&planes[0][..], &planes[1][..], &planes[2][..]];
        let mut encoder = Encoder::new(width, height, chroma, Signalling::default()).unwrap();
        let frame = encoder.encode(sources, budget).unwrap();

        let layout = encoder.layout().clone();
        let bands = encoder.transform(sources);
        let blocks = encoder.quantise(&bands);
        let costs: Vec<BlockCost> = blocks.iter().map(|block| block.cost.clone()).collect();
        let drops = rate::choose(&costs, ((budget - HEADER_BYTES) / 4) as u64).unwrap();
        let mut counted: f64 = blocks
            .iter()
            .zip(&drops)
            .map(|(block, &drop)| f64::from(block.cost.distortion[usize::from(drop)]))
            .sum();
        let mut blocks = blocks.iter();
        for (component, level, band, band_layout) in layout.bands() {
            let coefficients = &bands[component.index()][level][band.index()];
            for block_y in 0..band_layout.blocks_y {
                for block_x in 0..band_layout.blocks_x {
                    let block = blocks.next().unwrap();
                    for (raster, block8) in block.blocks8.iter().enumerate() {
                        if block8.is_none() {
                            let values = quant::gather_block8(
                                coefficients,
                                band_layout,
                                block_x,
                                block_y,
                                raster,
                            );
                            let energy: f32 = values.iter().map(|v| v * v).sum();
                            counted += f64::from(block.quant.weight * energy);
                        }
                    }
                }
            }
        }

        let measured = error_of(&layout, &planes, &frame).weighted_total();
        assert!(
            (measured - counted).abs() <= 1e-5 * counted,
            "{width}x{height} {chroma:?} at {budget} bytes: measured {measured}, counted {counted}"
        );
    }
}

#[test]
fn the_coefficient_error_falls_as_the_budget_grows() {
    let (width, height) = (200, 120);
    let planes = chroma_planes(ChromaSubsampling::Yuv420, width, height, false);
    let mut encoder = Encoder::new(
        width,
        height,
        ChromaSubsampling::Yuv420,
        Signalling::default(),
    )
    .unwrap();
    let layout = encoder.layout().clone();
    let errors: Vec<CoefficientError> = [500, 2000, 8000, 30000]
        .into_iter()
        .map(|budget| {
            let frame = encoder
                .encode([&planes[0], &planes[1], &planes[2]], budget)
                .unwrap();
            error_of(&layout, &planes, &frame)
        })
        .collect();
    for pair in errors.windows(2) {
        assert!(
            pair[1].weighted_total() < pair[0].weighted_total(),
            "{errors:?}"
        );
    }
    // Weighted and not, every component has some error at the lowest budget.
    assert!(errors[0].weighted.iter().all(|&e| e > 0.0), "{errors:?}");
    assert!(errors[0].squared.iter().all(|&e| e > 0.0), "{errors:?}");
}

#[test]
fn the_coefficient_error_needs_the_frames_picture_size() {
    let (width, height) = (64, 64);
    let planes = chroma_planes(ChromaSubsampling::Yuv420, width, height, false);
    let mut encoder = Encoder::new(
        width,
        height,
        ChromaSubsampling::Yuv420,
        Signalling::default(),
    )
    .unwrap();
    let frame = encoder
        .encode([&planes[0], &planes[1], &planes[2]], 4000)
        .unwrap();
    let mut assembler = FrameAssembler::new(encoder.layout().clone());
    assembler.push_packet(&frame).unwrap();
    let short = &planes[1][1..];
    assert!(matches!(
        coefficient_error([&planes[0], short, &planes[2]], &assembler.frame()),
        Err(EncodeError::PlaneSize {
            component: Component::Cb,
            ..
        })
    ));
}
