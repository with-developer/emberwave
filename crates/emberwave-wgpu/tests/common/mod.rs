//! Helpers shared by the GPU tests: those of `emberwave-core`'s tests
//! (locating the committed test vectors, reading reference decodes, PSNR),
//! included from there, and the GPU's own.

#![allow(dead_code)] // each test binary uses a different subset

#[path = "../../../emberwave-core/tests/common/mod.rs"]
mod core_common;
pub use core_common::*;

/// Whether a missing GPU is a failure rather than a reason to skip. An empty
/// value counts as unset so a CI matrix can leave it blank.
pub fn gpu_required() -> bool {
    std::env::var_os("EMBERWAVE_REQUIRE_GPU").is_some_and(|value| !value.is_empty())
}

/// Creates a context, or returns `None` if there is no GPU and none is required.
pub fn create_context(options: &emberwave_wgpu::GpuOptions) -> Option<emberwave_wgpu::GpuContext> {
    match emberwave_wgpu::GpuContext::new(options) {
        Ok(context) => {
            eprintln!("GPU tests run on {:?}", context.adapter_info());
            Some(context)
        }
        Err(error) => {
            assert!(
                !gpu_required(),
                "a GPU is required but none is available: {error}"
            );
            eprintln!("no GPU available, skipping: {error}");
            None
        }
    }
}

/// Every count of live GPU objects and memory that wgpu keeps for a context's
/// device (the tests build wgpu with its `counters` feature), as text: buffers,
/// textures and their views, bind groups and their layouts, pipelines,
/// samplers, command encoders, shader modules, query sets, fences, memory.
/// wgpu 30 keeps the object counts on Vulkan, Metal and DX12; Metal leaves the
/// memory at zero, and no backend counts memory allocations.
pub fn live_resources(context: &emberwave_wgpu::GpuContext) -> String {
    format!("{:?}", context.device().get_internal_counters().hal)
}

/// The options the tests start from: the defaults, with the backends that
/// `EMBERWAVE_BACKEND` names, so that a run can force a backend, as the
/// command line tool does. A value that names none is ignored.
pub fn options() -> emberwave_wgpu::GpuOptions {
    emberwave_wgpu::GpuOptions::from_env().unwrap_or_default()
}

/// The GPU context shared by the tests of one test binary, or `None` if
/// there is no GPU and none is required.
pub fn shared_context() -> Option<&'static emberwave_wgpu::GpuContext> {
    static CONTEXT: std::sync::OnceLock<Option<emberwave_wgpu::GpuContext>> =
        std::sync::OnceLock::new();
    CONTEXT.get_or_init(|| create_context(&options())).as_ref()
}
