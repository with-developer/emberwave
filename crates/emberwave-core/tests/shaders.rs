//! Offline validation of the WGSL shaders with naga: no GPU needed.
//!
//! Validating with an empty set of capabilities shows that the shaders use no
//! optional feature: no subgroup operations, no 16-bit floats or integers, no
//! atomics on floats and so on.

#![cfg(feature = "gpu")]

use emberwave_core::gpu::shader_sources;
use naga::valid::{Capabilities, ValidationFlags, Validator};

fn validate(name: &str, source: &str) -> naga::Module {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("{name} does not parse:\n{}", error.emit_to_string(source)));
    Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|error| panic!("{name} does not validate: {error:?}"));
    module
}

#[test]
fn every_shader_validates_without_optional_capabilities() {
    for (name, source, entry_points) in shader_sources::ALL {
        let module = validate(name, source);

        // The compute entry points the pipelines are built from, no others.
        let found: Vec<&str> = module
            .entry_points
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();
        assert_eq!(found, entry_points, "{name}");
        for entry in &module.entry_points {
            assert_eq!(entry.stage, naga::ShaderStage::Compute, "{name}");
        }
    }
}

#[test]
fn workgroup_sizes_and_memory_stay_within_the_portable_limits() {
    // WebGPU guarantees 256 invocations and 16 KiB of workgroup memory.
    const MAX_INVOCATIONS: u32 = 256;
    const MAX_WORKGROUP_BYTES: u32 = 16 * 1024;

    for (name, source, _) in shader_sources::ALL {
        let module = validate(name, source);
        for entry in &module.entry_points {
            let [x, y, z] = entry.workgroup_size;
            assert!(x * y * z <= MAX_INVOCATIONS, "{name}: {x}x{y}x{z}");
        }
        let [x, y, z] = module.entry_points[0].workgroup_size;

        // Sizes of the workgroup variables, in bytes.
        let layouter = {
            let mut layouter = naga::proc::Layouter::default();
            layouter.update(module.to_ctx()).expect("layout");
            layouter
        };
        let total: u32 = module
            .global_variables
            .iter()
            .filter(|(_, variable)| variable.space == naga::AddressSpace::WorkGroup)
            .map(|(_, variable)| layouter[variable.ty].size)
            .sum();
        assert!(
            total <= MAX_WORKGROUP_BYTES,
            "{name}: {total} bytes of workgroup memory"
        );
        println!("{name}: workgroup {x}x{y}x{z}, {total} bytes of workgroup memory");
    }
}

#[test]
fn the_workgroup_variables_are_the_ones_checked_for_writes_before_reads() {
    // The pipelines do not zero workgroup memory (ARCHITECTURE.md, decision
    // D25). That is only right while every workgroup variable is written
    // before it is read, which was checked for exactly these variables and
    // sizes. A shader that gains, loses or resizes one fails here: check its
    // writes and reads again, then update the list.
    type Variable = (&'static str, u32); // name, size in bytes
    let dequant: &[Variable] = &[
        ("shared_offset", 4),
        ("control_words", 64),
        ("qscales", 64),
        ("plane_costs", 64),
        ("sign_counts", 512),
        ("block_sign_totals", 64),
    ];
    let idwt: &[Variable] = &[("window", 6400), ("horizontal", 5120)];
    // quantise.wgsl's, in front of equant.wgsl's and pack.wgsl's.
    let quantise: &[Variable] = &[
        ("subblock_largest", 512),
        ("subblock_largest_magnitude", 512),
    ];
    let equant: &[Variable] = &[
        ("subblock_error", 8192),
        ("subblock_signs", 1024),
        ("subblock_planes", 512),
        ("block8_pairs", 64),
        ("block8_base", 64),
        ("pair_sizes", 1024),
        ("pair_distortion", 1024),
    ];
    let pack: &[Variable] = &[
        ("assembly", 2560),
        ("subblock_kept", 512),
        ("subblock_extra", 512),
        ("subblock_signs", 512),
        ("block8_sent", 64),
        ("block8_plane_bytes", 64),
        ("block8_signs", 64),
    ];
    let lists: [(&str, Vec<Variable>); 8] = [
        ("dequant", dequant.to_vec()),
        ("idwt", idwt.to_vec()),
        ("yuv2rgb", Vec::new()),
        ("fdwt", idwt.to_vec()),
        ("equant", [quantise, equant].concat()),
        ("rate", Vec::new()),
        ("scan", vec![("sums", 1024)]),
        ("pack", [quantise, pack].concat()),
    ];
    for ((name, source, _), (listed, variables)) in shader_sources::ALL.into_iter().zip(lists) {
        assert_eq!(name, listed, "the lists follow shader_sources::ALL");
        let module = validate(name, source);
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).expect("layout");
        let found: Vec<(String, u32)> = module
            .global_variables
            .iter()
            .filter(|(_, variable)| variable.space == naga::AddressSpace::WorkGroup)
            .map(|(_, variable)| {
                let name = variable.name.clone().unwrap_or_default();
                (name, layouter[variable.ty].size)
            })
            .collect();
        let variables: Vec<(String, u32)> = variables
            .iter()
            .map(|&(name, size)| (name.to_owned(), size))
            .collect();
        assert_eq!(found, variables, "{name}");
    }
}
