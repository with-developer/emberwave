//! The GPU encoder's stages against the CPU encoder's (decision D29).

use super::*;
use crate::bitstream::{Band, FrameAssembler};
use crate::cpu::decode_frame;
use crate::encode::rate::{self, BlockCost};
use crate::encode::{Bands, Encoder as CpuEncoder};
use crate::gpu::context::test_context;
use crate::gpu::resources::{entry, read_back, uniform_buffer};

use ChromaSubsampling::{Yuv420, Yuv444};

/// Sizes and chroma formats: aligned and not, odd 4:4:4, 4:2:0 whose chroma
/// rows end inside a word and whose chroma height is odd, the minimum frame,
/// and more than one tile at every level.
const SHAPES: [(u32, u32, ChromaSubsampling); 6] = [
    (200, 120, Yuv420),
    (131, 67, Yuv444),
    (196, 70, Yuv420),
    (640, 360, Yuv420),
    (2, 2, Yuv420),
    (1056, 544, Yuv444),
];

/// A picture with something of everything: flat areas, a gradient, sharp
/// edges, fine texture and noise, different in each plane.
fn picture(layout: &FrameLayout, seed: u32) -> [Vec<u8>; 3] {
    let mut state = seed.wrapping_mul(2_654_435_761) | 1;
    let mut random = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    Component::ALL.map(|component| {
        let (width, height) = layout.plane_size(component);
        let phase = component.index() as f32;
        (0..width * height)
            .map(|i| {
                let (x, y) = (i % width, i / width);
                let (fx, fy) = (x as f32 / width as f32, y as f32 / height as f32);
                let value = match (4 * x / width, 2 * y / height) {
                    (0, 0) => 90.0 + 20.0 * phase,
                    (1, 0) => 30.0 + 200.0 * fx * fy,
                    (2, 0) => {
                        if (x / 3 + y / 5) % 2 == 0 {
                            40.0
                        } else {
                            220.0
                        }
                    }
                    (3, 0) => {
                        128.0 + 60.0 * ((x as f32 * 0.9 + phase).sin() * (y as f32 * 0.7).cos())
                    }
                    (0 | 1, _) => 128.0 + 100.0 * (fx * 7.0 + fy * 3.0 + phase).sin(),
                    _ => (random() >> 24) as f32,
                };
                value.clamp(0.0, 255.0) as u8
            })
            .collect()
    })
}

fn as_slices(planes: &[Vec<u8>; 3]) -> [&[u8]; 3] {
    [&planes[0], &planes[1], &planes[2]]
}

/// The bands of a picture that the CPU encoder keeps, those that are sent,
/// with their places: the GPU's other LL bands are inputs of the next level.
fn sent_bands(bands: &Bands) -> impl Iterator<Item = (Component, usize, Band, &[f32])> {
    Component::ALL.into_iter().flat_map(move |component| {
        bands[component.index()]
            .iter()
            .enumerate()
            .flat_map(move |(level, of_level)| {
                Band::ALL.into_iter().filter_map(move |band| {
                    let values = &of_level[band.index()];
                    (!values.is_empty()).then_some((component, level, band, values.as_slice()))
                })
            })
    })
}

impl Encoder {
    /// Records what `record` records and waits for the GPU to do it.
    fn run(&self, record: impl FnOnce(&Self, &mut wgpu::CommandEncoder)) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        record(self, &mut encoder);
        self.queue.submit([encoder.finish()]);
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the GPU finishes");
    }

    /// Runs the forward transform, unless `transform` is false, and the
    /// quantisation, and waits for them.
    fn run_stages(&self, transform: bool) {
        self.run(|this, encoder| {
            if transform {
                this.encode_upload(encoder);
                this.encode_transform(encoder, None, None);
            }
            this.encode_quantisation(encoder, None);
        });
    }

    fn write_frame_params(&self, budget_words: u32, sequence: u8) {
        let params = FrameParams {
            block_count: self.layout.block_count(),
            grid_width: self.rate_grid.0,
            budget_words,
            sequence: u32::from(sequence),
            capacity_words: self.payload_capacity_words as u32,
            pad: [0; 3],
        };
        self.queue
            .write_buffer(&self.frame_params, 0, bytemuck::bytes_of(&params));
    }

    fn write_costs(&self, costs: &[BlockCost]) {
        let records: Vec<CostRecord> = costs.iter().map(CostRecord::from_cost).collect();
        self.queue
            .write_buffer(&self.costs, 0, bytemuck::cast_slice(&records));
    }

    /// Writes the bands of a picture's coefficients into the pyramid.
    fn write_bands(&self, bands: &Bands) {
        for (component, level, band, values) in sent_bands(bands) {
            self.write_band(component, level, band, values);
        }
    }

    fn read_words(&self, source: &wgpu::Buffer, offset: u64, count: u64) -> Vec<u32> {
        bytemuck::pod_collect_to_vec(&self.read_buffer(source, offset, count * 4))
    }

    /// `bytes` bytes of `source` from `offset` on.
    fn read_buffer(&self, source: &wgpu::Buffer, offset: u64, bytes: u64) -> Vec<u8> {
        let staging = buffer(
            &self.device,
            "test readback",
            bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(source, offset, &staging, 0, bytes);
        self.queue.submit([encoder.finish()]);
        read_back(&self.device, &staging, bytes).expect("read back")
    }

    fn band_place(&self, component: Component, level: usize, band: Band) -> (u64, u64) {
        let offset = self
            .buffers
            .band_offset(component, level, band)
            .expect("the band exists");
        let (width, height) = self.layout.level_size(level);
        (
            u64::from(offset) * 4,
            u64::from(width) * u64::from(height) * 4,
        )
    }

    fn read_band(&self, component: Component, level: usize, band: Band) -> Vec<f32> {
        let (offset, bytes) = self.band_place(component, level, band);
        bytemuck::pod_collect_to_vec(&self.read_buffer(&self.pyramid, offset, bytes))
    }

    fn write_band(&self, component: Component, level: usize, band: Band, values: &[f32]) {
        let (offset, bytes) = self.band_place(component, level, band);
        assert_eq!(values.len() as u64 * 4, bytes);
        self.queue
            .write_buffer(&self.pyramid, offset, bytemuck::cast_slice(values));
    }

    fn read_costs(&self) -> Vec<CostRecord> {
        let bytes = u64::from(self.layout.block_count()) * std::mem::size_of::<CostRecord>() as u64;
        bytemuck::pod_collect_to_vec(&self.read_buffer(&self.costs, 0, bytes))
    }
}

#[test]
fn the_forward_transform_matches_the_cpu() {
    let Some(context) = test_context() else {
        return;
    };
    // The GPU evaluates the CPU's expressions in the CPU's order, but its
    // compiler may fuse a multiply and an add, which moves a result by a unit
    // in the last place; ten passes of lifting add up a few of those.
    const TOLERANCE: f32 = 1e-5;
    let mut worst = 0.0f32;
    let mut compared = 0usize;

    for (width, height, chroma) in SHAPES {
        let mut gpu = Encoder::new(context, width, height, chroma, Signalling::default()).unwrap();
        let cpu = CpuEncoder::new(width, height, chroma, Signalling::default()).unwrap();
        let planes = picture(gpu.layout(), width ^ height);
        gpu.upload_planes(as_slices(&planes), None).unwrap();
        gpu.run_stages(true);
        let bands = cpu.transform(as_slices(&planes));

        for (component, level, band, expected) in sent_bands(&bands) {
            let actual = gpu.read_band(component, level, band);
            assert_eq!(actual.len(), expected.len());
            for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
                let difference = (a - e).abs() / e.abs().max(1.0);
                worst = worst.max(difference);
                assert!(
                    difference <= TOLERANCE,
                    "{width}x{height} {chroma:?} {component:?} level {level} {band:?} \
                     coefficient {i}: GPU {a}, CPU {e}"
                );
            }
            compared += actual.len();
        }
    }
    assert!(
        compared > 1_000_000,
        "only {compared} coefficients compared"
    );
    println!("largest difference to the CPU in any coefficient: {worst:e}");
}

#[test]
fn quantising_the_cpus_coefficients_gives_the_cpus_tables() {
    let Some(context) = test_context() else {
        return;
    };
    // Relative to each distortion: the sums are the CPU's, in its order, but
    // a fused multiply-add rounds a square once where the CPU rounds twice.
    // A sum of 64 positive terms then moves by 64 units in the last place at
    // most, 4e-6 of it.
    const TOLERANCE: f32 = 1e-5;
    let mut worst = 0.0f32;
    let mut entries = 0usize;

    for (width, height, chroma) in SHAPES {
        let gpu = Encoder::new(context, width, height, chroma, Signalling::default()).unwrap();
        let cpu = CpuEncoder::new(width, height, chroma, Signalling::default()).unwrap();
        for seed in [3, 4] {
            let planes = picture(gpu.layout(), seed);
            let bands = cpu.transform(as_slices(&planes));
            gpu.write_bands(&bands);
            gpu.run_stages(false);

            let records = gpu.read_costs();
            let blocks = cpu.quantise(&bands);
            assert_eq!(records.len(), blocks.len());
            for (record, block) in records.iter().zip(&blocks) {
                let index = block.index;
                let cost = record.to_cost().expect("numbers in range");
                assert_eq!(cost.words, block.cost.words, "block {index}: words");
                assert_eq!(cost.max_drop, block.cost.max_drop, "block {index}");
                for (drop, (g, c)) in cost
                    .distortion
                    .iter()
                    .zip(&block.cost.distortion)
                    .enumerate()
                {
                    // A distortion of zero, nothing lost, must be zero too.
                    let difference = (g - c).abs();
                    assert!(
                        difference <= TOLERANCE * c.abs(),
                        "{width}x{height} {chroma:?} block {index} drop {drop}: GPU {g}, CPU {c}"
                    );
                    if *c != 0.0 {
                        worst = worst.max(difference / c.abs());
                    }
                }
                entries += 1;
            }
        }
    }
    assert!(entries > 1000, "only {entries} blocks compared");
    println!("largest relative difference in any distortion: {worst:e}");
}

/// A cost table of `blocks` blocks of random sizes and distortions, drawn
/// from few values so that keys tie often, across blocks and within them.
fn random_costs(blocks: usize, seed: u32) -> Vec<BlockCost> {
    let mut state = seed.wrapping_mul(2_654_435_761) | 1;
    let mut random = move |below: u32| {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state % below
    };
    (0..blocks)
        .map(|_| {
            let mut cost = BlockCost {
                words: [0; 16],
                distortion: [0.0; 16],
                max_drop: random(16) as u8,
            };
            let mut words = 8 + random(300) as u16;
            let mut distortion = random(4) as f32;
            for drop in 0..16 {
                cost.words[drop] = words;
                cost.distortion[drop] = distortion;
                // Sizes never grow; some planes cost no word at all, and
                // dropping one sometimes lands nearer.
                words = words.saturating_sub([0, 0, 1, 3, 8, 30][random(6) as usize]);
                distortion =
                    (distortion + [-0.5, 0.0, 1.0, 2.0, 4.0, 64.0][random(6) as usize]).max(0.0);
            }
            cost
        })
        .collect()
}

#[test]
fn the_rate_control_chooses_the_cpus_drops() {
    let Some(context) = test_context() else {
        return;
    };
    // More than 1024 blocks: the prefix sums take both of their passes.
    let (width, height, chroma) = (1056, 544, Yuv444);
    let gpu = Encoder::new(context, width, height, chroma, Signalling::default()).unwrap();
    let cpu = CpuEncoder::new(width, height, chroma, Signalling::default()).unwrap();
    let blocks = gpu.layout().block_count() as usize;
    assert!(blocks > 1024, "{blocks} blocks");

    let mut tables: Vec<(&str, Vec<BlockCost>)> = Vec::new();
    for seed in [5, 6] {
        let bands = cpu.transform(as_slices(&picture(gpu.layout(), seed)));
        let costs = cpu.quantise(&bands).into_iter().map(|b| b.cost).collect();
        tables.push(("picture", costs));
    }
    // Every block alike: all steps of a key tie, and block order decides.
    tables.push(("alike", vec![random_costs(1, 9)[0].clone(); blocks]));
    tables.push(("random", random_costs(blocks, 10)));
    tables.push(("random", random_costs(blocks, 11)));

    let mut checked = 0;
    for (name, costs) in &tables {
        gpu.write_costs(costs);
        let total: u32 = costs.iter().map(|c| u32::from(c.words[0])).sum();
        let minimum: u32 = costs
            .iter()
            .map(|c| u32::from(c.words[usize::from(c.max_drop.min(15))]))
            .sum();
        let mut budgets = vec![
            total + 10,
            total,
            total - 1,
            total - total / 10,
            (total + minimum) / 2,
            minimum + 1,
            minimum,
            minimum.saturating_sub(1),
            0,
        ];
        budgets.dedup();
        for budget_words in budgets {
            gpu.write_frame_params(budget_words, 1);
            gpu.run(|this, encoder| {
                encoder.clear_buffer(&this.rate_state, 0, None);
                this.encode_rate_control(encoder, None);
            });
            let summary: RateSummary = bytemuck::pod_read_unaligned(&gpu.read_buffer(
                &gpu.rate_state,
                RATE_KEYS * 4,
                std::mem::size_of::<RateSummary>() as u64,
            ));
            assert_eq!((summary.total, summary.minimum), (total, minimum), "{name}");

            match rate::choose(costs, u64::from(budget_words)) {
                Some(expected) => {
                    assert_eq!(summary.fits, 1, "{name} at {budget_words} words");
                    let drops = gpu.read_words(&gpu.drops, 0, blocks as u64);
                    for (block, (&drop, &want)) in drops.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            drop,
                            u32::from(want),
                            "{name} at {budget_words} words: block {block}"
                        );
                    }
                    let words: u32 = costs
                        .iter()
                        .zip(&expected)
                        .map(|(c, &d)| u32::from(c.words[usize::from(d)]))
                        .sum();
                    let chunks = u64::from(gpu.scan_params.chunk_count);
                    let scanned = gpu.read_words(&gpu.chunk_totals, chunks * 4, 1)[0];
                    assert_eq!(scanned, words, "{name}: the payload's size");
                    let sent = costs
                        .iter()
                        .zip(&expected)
                        .filter(|&(c, &d)| c.words[usize::from(d)] != 0)
                        .count();
                    assert_eq!(summary.sent as usize, sent, "{name}");
                }
                None => assert_eq!(summary.fits, 0, "{name} at {budget_words} words"),
            }
            checked += 1;
        }
    }
    assert!(checked > 30);
}

#[test]
fn packing_the_cpus_choices_gives_the_cpus_bytes() {
    let Some(context) = test_context() else {
        return;
    };
    for (width, height, chroma) in SHAPES {
        let mut gpu = Encoder::new(context, width, height, chroma, Signalling::default()).unwrap();
        for (seed, share) in [(7, 1.2), (8, 0.5), (9, 0.1)] {
            let planes = picture(gpu.layout(), seed);
            let mut cpu = CpuEncoder::new(width, height, chroma, Signalling::default()).unwrap();
            let bands = cpu.transform(as_slices(&planes));
            let costs: Vec<BlockCost> = cpu.quantise(&bands).into_iter().map(|b| b.cost).collect();
            let total: usize = costs.iter().map(|c| usize::from(c.words[0])).sum();
            // Even a tenth of the words fits: it drops planes, not blocks.
            let budget = HEADER_BYTES + (total as f64 * 4.0 * share) as usize;
            let frame = cpu
                .encode(as_slices(&planes), budget)
                .expect("the frame fits");
            let drops = rate::choose(&costs, ((budget - HEADER_BYTES) / 4) as u64).unwrap();
            let words: Vec<u32> = costs
                .iter()
                .zip(&drops)
                .map(|(c, &d)| u32::from(c.words[usize::from(d)]))
                .collect();
            let offsets: Vec<u32> = words
                .iter()
                .scan(0, |offset, &w| {
                    let start = *offset;
                    *offset += w;
                    Some(start)
                })
                .collect();
            let payload_words = words.iter().map(|&w| u64::from(w)).sum::<u64>();

            gpu.ensure_payload_capacity(payload_words).unwrap();
            gpu.write_bands(&bands);
            gpu.write_costs(&costs);
            let drops: Vec<u32> = drops.into_iter().map(u32::from).collect();
            gpu.queue
                .write_buffer(&gpu.drops, 0, bytemuck::cast_slice(&drops));
            gpu.queue
                .write_buffer(&gpu.block_words, 0, bytemuck::cast_slice(&offsets));
            // The CPU encoder's first frame is number 1.
            gpu.write_frame_params(u32::MAX, 1);
            gpu.run(|this, encoder| this.encode_packing(encoder, None));

            let payload: Vec<u8> = gpu.read_buffer(&gpu.payload, 0, payload_words * 4);
            assert_eq!(frame.len(), HEADER_BYTES + payload.len());
            if let Some(at) = (0..payload.len()).find(|&i| payload[i] != frame[HEADER_BYTES + i]) {
                panic!(
                    "{width}x{height} {chroma:?} seed {seed}: byte {at} of the payload is {:#04x}, \
                     the CPU's {:#04x}",
                    payload[at],
                    frame[HEADER_BYTES + at]
                );
            }
        }
    }
}

fn psnr(reference: &[u8], test: &[u8]) -> f64 {
    let squared: f64 = reference
        .iter()
        .zip(test)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    let mse = squared / reference.len() as f64;
    10.0 * (255.0 * 255.0 / mse.max(1e-12)).log10()
}

/// The PSNR of the decode of `frame` against `planes`, of all samples.
fn quality(layout: &FrameLayout, planes: &[Vec<u8>; 3], frame: &[u8]) -> f64 {
    let mut assembler = FrameAssembler::new(layout.clone());
    assembler
        .push_packet(frame)
        .expect("the decoder takes the frame");
    assert!(
        assembler.is_ready(false),
        "every block of the frame arrived"
    );
    let decoded = decode_frame(&assembler.frame()).expect("the frame decodes");
    let decoded: Vec<u8> = decoded
        .planes
        .iter()
        .flat_map(|plane| plane.to_unorm8())
        .collect();
    psnr(&planes.concat(), &decoded)
}

#[test]
fn frames_are_the_cpu_encoders_or_as_good() {
    let Some(context) = test_context() else {
        return;
    };
    let (mut frames, mut identical, mut worst) = (0, 0, 0.0f64);
    for (width, height, chroma) in SHAPES {
        let mut gpu = Encoder::new(context, width, height, chroma, Signalling::default()).unwrap();
        let mut cpu = CpuEncoder::new(width, height, chroma, Signalling::default()).unwrap();
        let samples = gpu.layout().plane_size(Component::Y).0 as usize
            * gpu.layout().plane_size(Component::Y).1 as usize;
        for (seed, bytes_per_sample) in [(11, 2.0), (12, 0.4), (13, 0.1), (14, 0.02)] {
            let planes = picture(gpu.layout(), seed);
            // Every budget fits; refusals are tests/gpu_encoder.rs's.
            let budget = HEADER_BYTES + 64 + (samples as f64 * bytes_per_sample) as usize;
            let expected = cpu
                .encode(as_slices(&planes), budget)
                .expect("the frame fits");
            let actual = gpu
                .encode(as_slices(&planes), budget)
                .expect("the frame fits");
            assert!(actual.len() <= budget);
            frames += 1;
            if actual == expected {
                identical += 1;
                continue;
            }
            let (cpu_db, gpu_db) = (
                quality(gpu.layout(), &planes, &expected),
                quality(gpu.layout(), &planes, &actual),
            );
            println!(
                "{width}x{height} {chroma:?} at {budget} B: CPU {} B, {cpu_db:.3} dB; GPU {} B, {gpu_db:.3} dB",
                expected.len(),
                actual.len()
            );
            worst = worst.max(cpu_db - gpu_db);
        }
    }
    println!("{identical} of {frames} frames identical; worst loss {worst:.4} dB");
    assert_eq!(frames, SHAPES.len() * 4);
    assert!(worst < 0.05, "the GPU's frames are {worst:.3} dB worse");
}

#[test]
fn prefix_sums_are_exclusive_across_chunks() {
    let Some(context) = test_context() else {
        return;
    };
    // 1995 blocks: a full chunk of 1024 values and a partial one.
    let gpu = Encoder::new(context, 1056, 544, Yuv444, Signalling::default()).unwrap();
    let count = gpu.layout().block_count() as usize;
    assert_eq!(gpu.scan_params.chunk_count, 2);
    let values: Vec<u32> = (0..count as u32).map(|i| (i * 7919) % 1000).collect();
    gpu.queue
        .write_buffer(&gpu.tie_savings, 0, bytemuck::cast_slice(&values));
    gpu.run(|this, encoder| this.encode_scan(encoder, &this.scan_ties, None));

    let scanned = gpu.read_words(&gpu.tie_savings, 0, count as u64);
    let mut sum = 0u32;
    for (i, (&value, &prefix)) in values.iter().zip(&scanned).enumerate() {
        assert_eq!(prefix, sum, "value {i}");
        sum += value;
    }
    let total = gpu.read_words(&gpu.chunk_totals, 2 * 4, 1)[0];
    assert_eq!(total, sum, "the sum of everything");
}

#[test]
fn workgroups_past_the_end_change_nothing() {
    // A grid of more than 65535 workgroups has some past the end, which
    // must neither return early (a return before a barrier crashed WARP)
    // nor write. A narrower grid than the encoder's makes some here.
    let Some(context) = test_context() else {
        return;
    };
    let mut gpu = Encoder::new(context, 1056, 544, Yuv444, Signalling::default()).unwrap();
    let blocks = gpu.layout().block_count();
    let planes = picture(gpu.layout(), 21);
    gpu.upload_planes(as_slices(&planes), None).unwrap();
    gpu.run_stages(true);
    let device = &gpu.device;
    let expected = gpu.read_costs();

    // 1995 blocks in rows of 1000: five workgroups past the last block.
    let mut params = QuantParams::new(&gpu.layout, &gpu.buffers);
    params.grid_width = 1000;
    let params = uniform_buffer(device, &gpu.queue, "test params", &params);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test equant"),
        layout: &gpu.pipelines.equant.layout,
        entries: &[
            entry(0, &gpu.pyramid),
            entry(1, &gpu.costs),
            entry(2, &params),
        ],
    });
    gpu.queue.write_buffer(
        &gpu.costs,
        0,
        &vec![0xA5; expected.len() * std::mem::size_of::<CostRecord>()],
    );
    gpu.run(|this, encoder| {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&this.pipelines.equant.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1000, blocks.div_ceil(1000), 1);
    });
    assert!(gpu.read_costs() == expected, "the cost tables");

    // Two chunks of the prefix sums, and three workgroups past them.
    let values: Vec<u32> = (0..blocks).map(|i| i % 97).collect();
    gpu.queue
        .write_buffer(&gpu.tie_savings, 0, bytemuck::cast_slice(&values));
    let mut scan = gpu.scan_params;
    scan.grid_width = scan.chunk_count + 3;
    let scan_params = uniform_buffer(device, &gpu.queue, "test scan params", &scan);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test scan"),
        layout: &gpu.pipelines.scan.chunks.layout,
        entries: &[
            entry(0, &gpu.tie_savings),
            entry(1, &gpu.chunk_totals),
            entry(2, &scan_params),
        ],
    });
    gpu.run(|this, encoder| {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_pipeline(&this.pipelines.scan.chunks.pipeline);
        pass.dispatch_workgroups(scan.grid_width, 1, 1);
        pass.set_pipeline(&this.pipelines.scan.totals.pipeline);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&this.pipelines.scan.add.pipeline);
        pass.dispatch_workgroups(scan.grid_width, 1, 1);
    });
    let scanned = gpu.read_words(&gpu.tie_savings, 0, u64::from(blocks));
    let mut sum = 0;
    for (i, (&value, &prefix)) in values.iter().zip(&scanned).enumerate() {
        assert_eq!(prefix, sum, "value {i}");
        sum += value;
    }
    let total = gpu.read_words(&gpu.chunk_totals, u64::from(scan.chunk_count) * 4, 1)[0];
    assert_eq!(total, sum);
}

/// A texture of `format` on the encoder's device holding `data`, row-major
/// without padding: a picture's plane, or Cb and Cr interleaved.
fn texture_of(
    gpu: &Encoder,
    format: wgpu::TextureFormat,
    (width, height): (u32, u32),
    data: &[u8],
) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test picture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let texel_bytes = format
        .block_copy_size(None)
        .expect("an uncompressed format");
    gpu.queue.write_texture(
        texture.as_image_copy(),
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * texel_bytes),
            rows_per_image: Some(height),
        },
        size,
    );
    texture
}

#[test]
fn gathering_textures_writes_the_words_of_the_upload() {
    // The packed planes, padding included, from textures as from the CPU,
    // for every byte value: so a picture in textures encodes to the bytes
    // of the same picture in CPU memory (decision D32).
    let Some(context) = test_context() else {
        return;
    };
    for (width, height, chroma) in SHAPES {
        let mut gpu = Encoder::new(context, width, height, chroma, Signalling::default()).unwrap();
        let layout = gpu.layout.clone();
        // Every value in each plane that has room for them, after a picture.
        let mut planes = picture(&layout, width ^ height);
        for (component, plane) in Component::ALL.into_iter().zip(&mut planes) {
            let shift = 85 * component.index();
            for (i, sample) in plane.iter_mut().enumerate().take(256) {
                *sample = ((i + shift) % 256) as u8;
            }
        }
        let bytes = gpu.buffers.packed_words * 4;
        gpu.upload_planes(as_slices(&planes), None).unwrap();
        gpu.run(|this, encoder| this.encode_upload(encoder));
        let uploaded = gpu.read_buffer(&gpu.planes, 0, bytes);

        let sizes = Component::ALL.map(|component| layout.plane_size(component));
        let textures: Vec<wgpu::Texture> = (0..3)
            .map(|i| texture_of(&gpu, wgpu::TextureFormat::R8Unorm, sizes[i], &planes[i]))
            .collect();
        let views: Vec<wgpu::TextureView> = textures
            .iter()
            .map(|texture| texture.create_view(&Default::default()))
            .collect();
        let interleaved: Vec<u8> = planes[1]
            .iter()
            .zip(&planes[2])
            .flat_map(|(&cb, &cr)| [cb, cr])
            .collect();
        let chroma_texture =
            texture_of(&gpu, wgpu::TextureFormat::Rg8Unorm, sizes[1], &interleaved);
        let chroma_view = chroma_texture.create_view(&Default::default());

        let pictures = [
            (
                "planes",
                TexturePicture::Planes([&views[0], &views[1], &views[2]]),
            ),
            (
                "luma and chroma",
                TexturePicture::LumaChroma {
                    luma: &views[0],
                    chroma: &chroma_view,
                },
            ),
        ];
        for (name, picture) in pictures {
            // A pattern where the gather must write, padding included.
            gpu.queue
                .write_buffer(&gpu.planes, 0, &vec![0xab; bytes as usize]);
            let bind_group = gpu.gather_bind_group(picture).unwrap();
            gpu.run(|this, encoder| this.encode_transform(encoder, Some(&bind_group), None));
            let gathered = gpu.read_buffer(&gpu.planes, 0, bytes);
            assert!(
                gathered == uploaded,
                "{width}x{height} {chroma:?} {name}: {} of {bytes} bytes differ",
                gathered
                    .iter()
                    .zip(&uploaded)
                    .filter(|(a, b)| a != b)
                    .count()
            );
        }
    }
}
