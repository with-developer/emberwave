//! Creating the wgpu device.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use super::GpuError;
use super::decoder::PLANE_FORMAT;
use super::pipelines::{EncoderPipelines, Pipelines};

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
    /// Let the decoders store the Y, Cb and Cr planes straight into textures
    /// that allow it, where the adapter can write 8-bit storage textures: on
    /// Metal, through the adapter's own format features (decision D52).
    /// Without it, and elsewhere, the decoders copy the planes into the
    /// textures (decision D41). Ignored with `strict_webgpu`.
    pub plane_stores: bool,
    /// Let the decoders write a frame, and the encoders a picture's planes,
    /// where their shaders read them, on GPUs that share their memory with
    /// the CPU: on Metal on Apple silicon, through mappable storage buffers
    /// (decisions D53, D54). Without it, and elsewhere, they go into an
    /// upload buffer that the GPU copies from (decisions D30, D34). Ignored
    /// with `strict_webgpu`.
    pub shared_uploads: bool,
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
            plane_stores: true,
            shared_uploads: true,
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

/// The limits a context asks for: the portable default limits, except that
/// buffers and textures may be as large as the adapter allows, and that no
/// limit is beyond what the adapter has. Large frames need large buffers, and
/// the textures of pictures to encode are as large as the pictures, up to
/// 16384 where WebGPU guarantees 8192. Some adapters fall short of a default
/// that emberwave does not need: Metal gives the iOS Simulator and Apple GPUs
/// before the A11 15 inter-stage variables, one fewer than WebGPU's default,
/// and the compute shaders use none. A limit the shaders do need fails where
/// the pipelines are made, naming it. With `strict_webgpu`, the defaults
/// exactly.
fn required_limits(supported: &wgpu::Limits, strict_webgpu: bool) -> wgpu::Limits {
    if strict_webgpu {
        return wgpu::Limits::default();
    }
    let mut limits = wgpu::Limits::default().or_worse_values_from(supported);
    limits.max_storage_buffer_binding_size = supported.max_storage_buffer_binding_size;
    limits.max_buffer_size = supported.max_buffer_size;
    limits.max_texture_dimension_2d = supported.max_texture_dimension_2d;
    limits
}

/// Held while [`GpuContext::new`] makes a context, so that two never overlap
/// (decision D31).
#[cfg(not(target_arch = "wasm32"))]
static CREATING: Mutex<()> = Mutex::new(());

/// The id of the next context made (see [`GpuContext::id`]).
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// An adapter, a device and its queue.
#[derive(Debug, Clone)]
pub struct GpuContext {
    /// Tells this context's device from those of other contexts.
    id: u64,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    timestamps: bool,
    /// Whether the decoders store planes into textures (decision D52).
    plane_stores: bool,
    /// Whether the decoders write frames, and the encoders planes, where
    /// their shaders read them (decisions D53, D54).
    shared_uploads: bool,
    /// Built on first use, shared by every clone of the context: the
    /// decoder's and the encoder's, so that neither compiles the other's.
    pipelines: Arc<Mutex<Option<Arc<Pipelines>>>>,
    encoder_pipelines: Arc<Mutex<Option<Arc<EncoderPipelines>>>>,
    /// Why the device was lost, once it has been; shared with the decoders
    /// and the encoders.
    lost: Arc<Mutex<Option<String>>>,
    /// Serialises the recording and submission of the work of every decoder
    /// and encoder of the context (#126). WIP: held for the whole call, the
    /// wait for the GPU included, as the review's prototype holds it.
    pub(crate) record_lock: Arc<Mutex<()>>,
}

impl GpuContext {
    /// Creates a context, blocking until the device is ready. Not in a
    /// browser, where nothing may block: use [`new_async`](Self::new_async).
    ///
    /// The contexts of a process are made here one at a time. On Linux,
    /// Vulkan loaders 1.3.284 to 1.4.344 can crash the process when one
    /// thread creates a device while another lists the adapters of a new
    /// instance (decision D31; loader 1.4.345 fixed it). That covers the
    /// contexts made here and nothing else: not those of
    /// [`new_async`](Self::new_async) or [`from_adapter`](Self::from_adapter),
    /// not other users of Vulkan in the process, and not, in debug builds,
    /// the naming of every object wgpu creates, which can crash in the same
    /// way while another thread makes a context.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(options: &GpuOptions) -> Result<Self, GpuError> {
        let _creating = CREATING.lock().unwrap_or_else(PoisonError::into_inner);
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
        Self::from_adapter(adapter, options).await
    }

    /// Creates a context on an adapter the caller has chosen, for example one
    /// that can draw to the caller's window or canvas. The options say which
    /// features to ask for; the backend and adapter options are not used.
    pub async fn from_adapter(
        adapter: wgpu::Adapter,
        options: &GpuOptions,
    ) -> Result<Self, GpuError> {
        let (device, queue) = adapter
            .request_device(&Self::device_descriptor(&adapter, options))
            .await?;
        Ok(Self::from_device(adapter, device, queue))
    }

    /// What a context asks of `adapter` for `options`: the features and the
    /// limits that [`from_adapter`](Self::from_adapter) requests. For a device
    /// made another way, such as on objects of the caller's graphics API with
    /// `wgpu::Adapter::create_device_from_hal`, which
    /// [`from_device`](Self::from_device) then takes.
    pub fn device_descriptor(
        adapter: &wgpu::Adapter,
        options: &GpuOptions,
    ) -> wgpu::DeviceDescriptor<'static> {
        let timestamps = options.timestamps
            && !options.strict_webgpu
            && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let plane_stores = options.plane_stores && !options.strict_webgpu && stores_planes(adapter);
        let shared_uploads =
            options.shared_uploads && !options.strict_webgpu && shares_memory(adapter);
        let mut required_features = wgpu::Features::empty();
        required_features.set(wgpu::Features::TIMESTAMP_QUERY, timestamps);
        required_features.set(
            wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
            plane_stores,
        );
        required_features.set(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS, shared_uploads);
        wgpu::DeviceDescriptor {
            label: Some("emberwave"),
            required_features,
            required_limits: required_limits(&adapter.limits(), options.strict_webgpu),
            ..Default::default()
        }
    }

    /// Creates a context on a device of `adapter` made with
    /// [`device_descriptor`](Self::device_descriptor), and its queue. The
    /// context takes over the device's lost callback. Without the limits of
    /// the descriptor, making the pipelines fails, naming the limit.
    pub fn from_device(adapter: wgpu::Adapter, device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let timestamps = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let plane_stores = device
            .features()
            .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
            && stores_planes(&adapter);
        let shared_uploads = device
            .features()
            .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS)
            && shares_memory(&adapter);

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

        Self {
            adapter,
            device,
            queue,
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            timestamps,
            plane_stores,
            shared_uploads,
            pipelines: Arc::default(),
            encoder_pipelines: Arc::default(),
            lost,
            record_lock: Arc::default(),
        }
    }

    /// Why the device was lost, if it was: a driver reset, a GPU that went
    /// away, or [`wgpu::Device::destroy`]. A lost device does not come back;
    /// create a new context.
    pub fn lost(&self) -> Option<String> {
        self.lost.lock().ok().and_then(|lost| lost.clone())
    }

    /// A number that no other context of the process has and that clones of
    /// this one share, which tells whether two objects are on the same
    /// device. Comparing `wgpu::Device`s cannot: every context makes its own
    /// `wgpu::Instance`, devices of different instances can compare equal,
    /// and an object of one instance used on another names whatever has its
    /// id there.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The error of a call that failed on this context's device: the loss of
    /// the device, if it was lost, and `error` otherwise.
    ///
    /// A device that the driver lost has called back already. A destroyed
    /// one calls back only when it is polled with no work left on it, so
    /// this polls once, without waiting, while no loss is recorded, and not
    /// once one is: wgpu panics when a poll finds the driver lost the device
    /// (decision D22).
    pub(crate) fn explain(&self, error: GpuError) -> GpuError {
        if matches!(error, GpuError::DeviceLost(_)) {
            return error;
        }
        notice_loss(&self.device, &self.lost);
        match self.lost() {
            Some(reason) => GpuError::DeviceLost(reason),
            None => error,
        }
    }

    /// The record of a lost device, shared with the decoders and the encoders
    /// of this context.
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

    /// Whether the decoders store the planes straight into the textures of
    /// decodes into plane textures, those that allow it, rather than copy
    /// them in: on Metal, unless [`GpuOptions::plane_stores`] was off
    /// (decision D52). Should the device fail to build the shader that
    /// stores, the decoders copy all the same, which
    /// [`Decoder::stores_planes`](super::Decoder::stores_planes) tells.
    pub fn stores_planes(&self) -> bool {
        self.plane_stores
    }

    /// Whether the decoders write a frame's data, and the encoders a
    /// picture's planes, where their shaders read them, without an upload
    /// buffer and the copies from it: on Metal on Apple silicon, unless
    /// [`GpuOptions::shared_uploads`] was off (decisions D53, D54).
    pub fn shares_uploads(&self) -> bool {
        self.shared_uploads
    }

    /// The decoder's compute pipelines, compiled the first time they are
    /// asked for.
    pub(crate) fn pipelines(&self) -> Result<Arc<Pipelines>, GpuError> {
        cached(&self.pipelines, || {
            Pipelines::new(&self.device, self.plane_stores)
        })
    }

    /// The encoder's compute pipelines, compiled the first time they are
    /// asked for.
    pub(crate) fn encoder_pipelines(&self) -> Result<Arc<EncoderPipelines>, GpuError> {
        cached(&self.encoder_pipelines, || {
            EncoderPipelines::new(&self.device)
        })
    }
}

/// Whether the decoders can store the planes into textures on `adapter`, once
/// its device has the adapter's own format features: on Metal, whose GPUs
/// write 8-bit storage textures (decision D52). Other adapters may write
/// them too, but only Metal's gain was measured, and WARP, which runs the
/// Windows tests, has crashed on shaders that the others ran: they keep the
/// copies.
fn stores_planes(adapter: &wgpu::Adapter) -> bool {
    let format = adapter.get_texture_format_features(PLANE_FORMAT);
    adapter.get_info().backend == wgpu::Backend::Metal
        && adapter
            .features()
            .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
        && format
            .allowed_usages
            .contains(wgpu::TextureUsages::STORAGE_BINDING)
        && format
            .flags
            .contains(wgpu::TextureFormatFeatureFlags::STORAGE_WRITE_ONLY)
}

/// Whether the decoders can write a frame where their shaders read it on
/// `adapter`, once its device has mappable storage buffers: on Metal with
/// memory that the CPU and the GPU share, as Apple silicon has (decision D53).
/// wgpu warns that such buffers cost a GPU of its own memory dearly.
fn shares_memory(adapter: &wgpu::Adapter) -> bool {
    let info = adapter.get_info();
    info.backend == wgpu::Backend::Metal
        && info.device_type == wgpu::DeviceType::IntegratedGpu
        && adapter
            .features()
            .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS)
}

/// After a call failed on `device`: polls it once, without waiting, while
/// `lost` records no loss, so that wgpu reports a destroyed device, which it
/// does only when the device is polled with no work left on it. Not once a
/// loss is recorded: a device that the driver lost has called back already,
/// and wgpu panics when a poll finds it (decision D22).
pub(crate) fn notice_loss(device: &wgpu::Device, lost: &Mutex<Option<String>>) {
    let recorded = lost.lock().map_or(true, |lost| lost.is_some());
    if !recorded {
        let _ = device.poll(wgpu::PollType::Poll);
    }
}

/// The value in `cache`, which `build` makes if there is none yet.
fn cached<T>(
    cache: &Mutex<Option<Arc<T>>>,
    build: impl FnOnce() -> Result<T, GpuError>,
) -> Result<Arc<T>, GpuError> {
    let mut cache = cache.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(value) = cache.as_ref() {
        return Ok(Arc::clone(value));
    }
    let value = Arc::new(build()?);
    *cache = Some(Arc::clone(&value));
    Ok(value)
}

/// The context the unit tests share, or `None` if there is no GPU and none is
/// required: as for the integration tests, `EMBERWAVE_REQUIRE_GPU` turns a
/// missing adapter into a failure.
#[cfg(test)]
pub(crate) fn test_context() -> Option<&'static GpuContext> {
    static CONTEXT: std::sync::OnceLock<Option<GpuContext>> = std::sync::OnceLock::new();
    CONTEXT
        .get_or_init(|| match GpuContext::new(&GpuOptions::default()) {
            Ok(context) => Some(context),
            Err(error) => {
                let required =
                    std::env::var_os("EMBERWAVE_REQUIRE_GPU").is_some_and(|v| !v.is_empty());
                assert!(
                    !required,
                    "a GPU is required but none is available: {error}"
                );
                eprintln!("no GPU available, skipping: {error}");
                None
            }
        })
        .as_ref()
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

    #[test]
    fn a_context_asks_for_no_limit_beyond_the_adapter() {
        // Metal's limits for the iOS Simulator and Apple GPUs before the A11.
        let supported = wgpu::Limits {
            max_inter_stage_shader_variables: 15,
            ..wgpu::Limits::default()
        };
        let limits = required_limits(&supported, false);
        assert_eq!(limits.max_inter_stage_shader_variables, 15);
        assert!(limits.check_limits(&supported));

        // An alignment the adapter needs coarser than the default, a minimum
        // rather than a maximum, is kept at the adapter's.
        let supported = wgpu::Limits {
            min_storage_buffer_offset_alignment: 1024,
            ..wgpu::Limits::default()
        };
        let limits = required_limits(&supported, false);
        assert_eq!(limits.min_storage_buffer_offset_alignment, 1024);
        assert!(limits.check_limits(&supported));
    }

    #[test]
    fn a_context_asks_for_the_largest_buffers_and_textures_and_otherwise_the_defaults() {
        let supported = wgpu::Limits {
            max_storage_buffer_binding_size: 1 << 31,
            max_buffer_size: 1 << 34,
            max_texture_dimension_2d: 16384,
            max_inter_stage_shader_variables: 31,
            max_compute_workgroup_storage_size: 32768,
            ..wgpu::Limits::default()
        };
        let limits = required_limits(&supported, false);
        assert_eq!(limits.max_storage_buffer_binding_size, 1 << 31);
        assert_eq!(limits.max_buffer_size, 1 << 34);
        assert_eq!(limits.max_texture_dimension_2d, 16384);
        // Better than the default elsewhere, the adapter's values are not
        // asked for: a device with the defaults behaves alike everywhere.
        assert_eq!(limits.max_inter_stage_shader_variables, 16);
        assert_eq!(limits.max_compute_workgroup_storage_size, 16384);

        // Strict WebGPU: the defaults, however much the adapter has.
        assert_eq!(required_limits(&supported, true), wgpu::Limits::default());
    }
}
