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
    for (name, source) in [
        ("dequant", shader_sources::DEQUANT),
        ("idwt", shader_sources::IDWT),
        ("pack", shader_sources::PACK),
    ] {
        let module = validate(name, source);

        // One compute entry point named `main` each.
        assert_eq!(module.entry_points.len(), 1, "{name}");
        let entry = &module.entry_points[0];
        assert_eq!(entry.name, "main", "{name}");
        assert_eq!(entry.stage, naga::ShaderStage::Compute, "{name}");
    }
}

#[test]
fn workgroup_sizes_and_memory_stay_within_the_portable_limits() {
    // WebGPU guarantees 256 invocations and 16 KiB of workgroup memory.
    const MAX_INVOCATIONS: u32 = 256;
    const MAX_WORKGROUP_BYTES: u32 = 16 * 1024;

    for (name, source) in [
        ("dequant", shader_sources::DEQUANT),
        ("idwt", shader_sources::IDWT),
        ("pack", shader_sources::PACK),
    ] {
        let module = validate(name, source);
        let [x, y, z] = module.entry_points[0].workgroup_size;
        assert!(x * y * z <= MAX_INVOCATIONS, "{name}: {x}x{y}x{z}");

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
