//! Creating the wgpu device.

use std::sync::{Arc, Mutex, PoisonError};

use super::GpuError;
use super::pipelines::Pipelines;

/// Environment variable that restricts the backends considered, for example
/// `vulkan`, `metal`, `dx12` or a comma separated list. Unset or `auto` means
/// all of them.
pub const BACKEND_ENV: &str = "EMBERWAVE_BACKEND";

/// How the adapter is chosen.
#[derive(Debug, Clone)]
pub struct GpuOptions {
    /// Backends to consider.
    pub backends: wgpu::Backends,
    /// Preference between an integrated and a discrete GPU.
    pub power_preference: wgpu::PowerPreference,
    /// Insist on a software adapter (lavapipe, WARP).
    pub force_fallback_adapter: bool,
    /// Restrict the device to what the WebGPU specification guarantees: the
    /// default limits and no optional features. Used by tests to show that the
    /// shaders are portable.
    pub strict_webgpu: bool,
    /// Ask for GPU timestamp queries if the adapter has them, so that
    /// [`Decoder::time_frame`](super::Decoder::time_frame) can report the time
    /// of each stage. Ignored with `strict_webgpu`.
    pub timestamps: bool,
}

impl GpuOptions {
    /// The options without any influence of the environment.
    fn base() -> Self {
        Self {
            backends: wgpu::Backends::PRIMARY,
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            strict_webgpu: false,
            timestamps: false,
        }
    }

    /// The default options with the backends taken from [`BACKEND_ENV`] if it
    /// is set. A value that names no backend is an error here, where
    /// [`Default`] would ignore it.
    pub fn from_env() -> Result<Self, String> {
        let mut options = Self::base();
        if let Some(value) = std::env::var_os(BACKEND_ENV) {
            let value = value.to_string_lossy();
            options.backends = parse_backends(&value).ok_or_else(|| {
                format!(
                    "{BACKEND_ENV}={value} names no backend: use vulkan, metal, dx12, a comma separated list of them, or auto"
                )
            })?;
        }
        Ok(options)
    }
}

impl Default for GpuOptions {
    /// All native backends unless [`BACKEND_ENV`] names others, preferring the
    /// high-performance GPU. An unusable value of the variable is ignored; use
    /// [`GpuOptions::from_env`] to be told about it.
    fn default() -> Self {
        Self::from_env().unwrap_or_else(|_| Self::base())
    }
}

/// Parses a backend name (`vulkan`, `metal` or `dx12`), a comma separated list
/// of them, or `auto` (or `all`) for all primary backends. Anything else,
/// including `gl`, which is not supported, gives `None`.
pub fn parse_backends(text: &str) -> Option<wgpu::Backends> {
    let mut backends = wgpu::Backends::empty();
    for name in text.split(',') {
        backends |= match name.trim().to_ascii_lowercase().as_str() {
            "" | "auto" | "all" => wgpu::Backends::PRIMARY,
            "vulkan" | "vk" => wgpu::Backends::VULKAN,
            "metal" | "mtl" => wgpu::Backends::METAL,
            "dx12" | "d3d12" => wgpu::Backends::DX12,
            _ => return None,
        };
    }
    (!backends.is_empty()).then_some(backends)
}

/// An adapter, a device and its queue.
#[derive(Debug, Clone)]
pub struct GpuContext {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    timestamps: bool,
    /// Built on first use, shared by every clone of the context.
    pipelines: Arc<Mutex<Option<Arc<Pipelines>>>>,
    /// Why the device was lost, once it has been; shared with the decoders.
    lost: Arc<Mutex<Option<String>>>,
}

impl GpuContext {
    /// Creates a context, blocking until the device is ready.
    pub fn new(options: &GpuOptions) -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async(options))
    }

    /// Creates a context.
    pub async fn new_async(options: &GpuOptions) -> Result<Self, GpuError> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = options.backends;
        if options.strict_webgpu {
            descriptor.flags |= wgpu::InstanceFlags::STRICT_WEBGPU_COMPLIANCE;
        }
        let instance = wgpu::Instance::new(descriptor);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: options.power_preference,
                force_fallback_adapter: options.force_fallback_adapter,
                compatible_surface: None,
                ..Default::default()
            })
            .await?;

        // The portable default limits, except that buffers may be as large as
        // the adapter allows: large frames need large buffers.
        let mut limits = wgpu::Limits::default();
        if !options.strict_webgpu {
            let supported = adapter.limits();
            limits.max_storage_buffer_binding_size = supported.max_storage_buffer_binding_size;
            limits.max_buffer_size = supported.max_buffer_size;
        }

        let timestamps = options.timestamps
            && !options.strict_webgpu
            && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let required_features = if timestamps {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("emberwave"),
                required_features,
                required_limits: limits,
                ..Default::default()
            })
            .await?;

        // A lost device stays lost. Errors of a lost device do not reach the
        // error scopes, so the decoders check this record before and after
        // their work instead of returning whatever is in their buffers.
        let lost: Arc<Mutex<Option<String>>> = Arc::default();
        let record = Arc::clone(&lost);
        device.set_device_lost_callback(move |reason, message| {
            if let Ok(mut lost) = record.lock() {
                lost.get_or_insert_with(|| format!("{reason:?}: {message}"));
            }
        });

        Ok(Self {
            adapter,
            device,
            queue,
            timestamps,
            pipelines: Arc::default(),
            lost,
        })
    }

    /// Why the device was lost, if it was: a driver reset, a GPU that went
    /// away, or [`wgpu::Device::destroy`]. A lost device does not come back;
    /// create a new context.
    pub fn lost(&self) -> Option<String> {
        self.lost.lock().ok().and_then(|lost| lost.clone())
    }

    /// The record of a lost device, shared with the decoders of this context.
    pub(crate) fn lost_record(&self) -> Arc<Mutex<Option<String>>> {
        Arc::clone(&self.lost)
    }

    /// Name, backend and driver of the adapter in use.
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    /// The wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Whether the device can time the GPU stages (`timestamps` was asked for
    /// and the adapter supports it).
    pub fn supports_timestamps(&self) -> bool {
        self.timestamps
    }

    /// The compute pipelines, compiled the first time they are asked for.
    pub(crate) fn pipelines(&self) -> Result<Arc<Pipelines>, GpuError> {
        let mut cache = self
            .pipelines
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(pipelines) = cache.as_ref() {
            return Ok(Arc::clone(pipelines));
        }
        let pipelines = Arc::new(Pipelines::new(&self.device)?);
        *cache = Some(Arc::clone(&pipelines));
        Ok(pipelines)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_names() {
        assert_eq!(parse_backends("vulkan"), Some(wgpu::Backends::VULKAN));
        assert_eq!(parse_backends("Metal"), Some(wgpu::Backends::METAL));
        assert_eq!(parse_backends(" dx12 "), Some(wgpu::Backends::DX12));
        assert_eq!(
            parse_backends("vulkan, metal"),
            Some(wgpu::Backends::VULKAN | wgpu::Backends::METAL)
        );
        assert_eq!(parse_backends("auto"), Some(wgpu::Backends::PRIMARY));
        assert_eq!(parse_backends("opengl"), None);
        assert_eq!(parse_backends("gl"), None);
        assert_eq!(parse_backends("vulkan,nonsense"), None);
    }
}
