//! The C API on a Vulkan device of the caller's, made here as a client makes
//! its own: an instance, then a device with what
//! `emberwave_vulkan_requirements_create` lists, then contexts on its queue
//! (decision D61). Linux CI runs it on lavapipe; elsewhere it skips without a
//! Vulkan device.

#![cfg(any(
    target_os = "linux",
    target_os = "android",
    all(windows, target_pointer_width = "64")
))]

use std::ffi::{CStr, c_char, c_void};
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

use ash::vk::{self, Handle};
use emberwave_core::bitstream::Pwv1File;
use emberwave_ffi::*;

fn testdata(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn last_error() -> String {
    // SAFETY: emberwave_last_error always returns a C string.
    unsafe { CStr::from_ptr(emberwave_last_error()) }
        .to_string_lossy()
        .into_owned()
}

/// The loader's vkGetInstanceProcAddr, as the C API's type: Vulkan's
/// calling convention is C's on the 64-bit systems these tests run on.
fn proc_addr(entry: &ash::Entry) -> EmberwaveVkGetInstanceProcAddr {
    // SAFETY: as above.
    unsafe {
        std::mem::transmute::<vk::PFN_vkGetInstanceProcAddr, EmberwaveVkGetInstanceProcAddr>(
            entry.static_fn().get_instance_proc_addr,
        )
    }
}

/// The Vulkan tests run one at a time: Vulkan loaders 1.3.284 to 1.4.344
/// can crash when one thread makes a device while another lists adapters
/// (decision D31).
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// The validation layer's errors, where the instance has the layer.
static VALIDATION_ERRORS: Mutex<Vec<String>> = Mutex::new(Vec::new());

unsafe extern "system" fn on_validation(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _kinds: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: the layer's callback data and its message ID's name, while
    // this runs.
    let id = unsafe { data.as_ref() }
        .filter(|data| !data.p_message_id_name.is_null())
        .map(|data| unsafe { CStr::from_ptr(data.p_message_id_name) });
    // naga 30 decorates arrays in workgroup memory with ArrayStride, which
    // the validation layer's spirv-val of 2025 rejects; drivers ignore it.
    // That is wgpu's, and the one complaint left out.
    if id == Some(c"VUID-StandaloneSpirv-None-10684") {
        return vk::FALSE;
    }
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        // SAFETY: the layer's callback data and its message, while this runs.
        let message = unsafe { data.as_ref() }
            .filter(|data| !data.p_message.is_null())
            .map(|data| {
                // SAFETY: as above.
                unsafe { CStr::from_ptr(data.p_message) }
                    .to_string_lossy()
                    .into_owned()
            })
            .unwrap_or_default();
        VALIDATION_ERRORS.lock().unwrap().push(message);
    }
    vk::FALSE
}

/// The caller's instance and the physical device it chose.
struct Client {
    entry: ash::Entry,
    instance: ash::Instance,
    /// The instance extensions it was made with, emberwave's among them.
    extensions: Vec<*const c_char>,
    /// The validation layer's messenger, where the layer is installed.
    validation: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    physical: vk::PhysicalDevice,
    /// A queue family with graphics and compute.
    family: u32,
    /// How many queues it has.
    queues: u32,
    _one_at_a_time: std::sync::MutexGuard<'static, ()>,
}

impl Client {
    /// An instance of Vulkan 1.3 with the extensions emberwave asks for, and
    /// the validation layer where it is installed, and the first physical
    /// device emberwave takes; or `None` without them where the GPU tests may
    /// skip.
    fn new() -> Option<Self> {
        let one_at_a_time = ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        VALIDATION_ERRORS.lock().unwrap().clear();
        let required = cfg!(target_os = "linux")
            && std::env::var_os("EMBERWAVE_REQUIRE_GPU").is_some_and(|value| !value.is_empty());
        let skip = |what: String| {
            assert!(!required, "{what}");
            eprintln!("{what}: skipped");
        };
        // SAFETY: loads the system's Vulkan loader.
        let entry = match unsafe { ash::Entry::load() } {
            Ok(entry) => entry,
            Err(error) => {
                skip(format!("no Vulkan loader: {error}"));
                return None;
            }
        };
        let get_instance_proc_addr = proc_addr(&entry);
        let mut count = 0;
        // SAFETY: the loader's vkGetInstanceProcAddr, and room for `count`.
        let mut extensions = unsafe {
            assert_eq!(
                emberwave_vulkan_instance_extensions(
                    get_instance_proc_addr,
                    ptr::null_mut(),
                    &mut count
                ),
                EmberwaveStatus::Ok,
                "{}",
                last_error()
            );
            let mut names = vec![ptr::null(); count as usize];
            assert_eq!(
                emberwave_vulkan_instance_extensions(
                    get_instance_proc_addr,
                    names.as_mut_ptr(),
                    &mut count
                ),
                EmberwaveStatus::Ok,
                "{}",
                last_error()
            );
            names
        };
        // The validation layer, where it is installed, with its messages.
        let layer = c"VK_LAYER_KHRONOS_validation";
        // SAFETY: the loader's functions.
        let validated = unsafe {
            entry
                .enumerate_instance_layer_properties()
                .unwrap_or_default()
                .iter()
                .any(|found| found.layer_name_as_c_str() == Ok(layer))
                && entry
                    .enumerate_instance_extension_properties(None)
                    .unwrap_or_default()
                    .iter()
                    .any(|found| found.extension_name_as_c_str() == Ok(ash::ext::debug_utils::NAME))
        };
        // SAFETY: names that live as long as the process.
        let named =
            |&name: &*const c_char| unsafe { CStr::from_ptr(name) } == ash::ext::debug_utils::NAME;
        if validated && !extensions.iter().any(named) {
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }
        let layers = if validated {
            vec![layer.as_ptr()]
        } else {
            Vec::new()
        };
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let info = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layers);
        // SAFETY: a valid description of an instance.
        let instance = match unsafe { entry.create_instance(&info, None) } {
            Ok(instance) => instance,
            Err(error) => {
                skip(format!("no Vulkan instance: {error}"));
                return None;
            }
        };
        let validation = validated.then(|| {
            let utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let messages = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(on_validation));
            // SAFETY: a valid description of a messenger of the instance.
            let messenger = unsafe { utils.create_debug_utils_messenger(&messages, None) }
                .expect("a validation messenger");
            (utils, messenger)
        });
        // The first physical device that emberwave takes, and a family of it
        // with graphics and compute.
        // SAFETY: a live instance and its physical devices.
        let chosen = unsafe { instance.enumerate_physical_devices() }
            .unwrap_or_default()
            .into_iter()
            .find_map(|physical| {
                // SAFETY: a physical device of the instance.
                let families =
                    unsafe { instance.get_physical_device_queue_family_properties(physical) };
                let wanted = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
                let (family, properties) = families
                    .iter()
                    .enumerate()
                    .find(|(_, family)| family.queue_flags.contains(wanted))?;
                let described = EmberwaveVulkanInstance {
                    get_instance_proc_addr,
                    instance: instance.handle().as_raw() as usize as *mut c_void,
                    api_version: vk::API_VERSION_1_3,
                    extensions: extensions.as_ptr(),
                    extension_count: extensions.len() as u32,
                };
                let mut requirements = ptr::null_mut();
                // SAFETY: a live instance and physical device of it, and a
                // pointer to store the requirements in.
                let status = unsafe {
                    emberwave_vulkan_requirements_create(
                        &described,
                        physical.as_raw() as usize as *mut c_void,
                        &mut requirements,
                    )
                };
                // SAFETY: requirements made just now, or NULL.
                unsafe { emberwave_vulkan_requirements_destroy(requirements) };
                (status == EmberwaveStatus::Ok).then_some((
                    physical,
                    family as u32,
                    properties.queue_count,
                ))
            });
        let Some((physical, family, queues)) = chosen else {
            // SAFETY: nothing else made on it.
            unsafe {
                if let Some((utils, messenger)) = &validation {
                    utils.destroy_debug_utils_messenger(*messenger, None);
                }
                instance.destroy_instance(None);
            }
            skip("no Vulkan device that emberwave takes".into());
            return None;
        };
        Some(Self {
            entry,
            instance,
            extensions,
            validation,
            physical,
            family,
            queues,
            _one_at_a_time: one_at_a_time,
        })
    }

    fn described(&self) -> EmberwaveVulkanInstance {
        self.described_through(proc_addr(&self.entry))
    }

    /// The instance, its functions found through `get_instance_proc_addr`.
    fn described_through(
        &self,
        get_instance_proc_addr: EmberwaveVkGetInstanceProcAddr,
    ) -> EmberwaveVulkanInstance {
        EmberwaveVulkanInstance {
            get_instance_proc_addr,
            instance: self.instance.handle().as_raw() as usize as *mut c_void,
            api_version: vk::API_VERSION_1_3,
            extensions: self.extensions.as_ptr(),
            extension_count: self.extensions.len() as u32,
        }
    }

    fn physical_device(&self) -> *mut c_void {
        self.physical.as_raw() as usize as *mut c_void
    }

    /// A device with what emberwave lists, and nothing more, as a client
    /// needs nothing more here: its extensions' names, and the device.
    fn device(&self) -> (Vec<*const c_char>, ash::Device) {
        let mut requirements = ptr::null_mut();
        // SAFETY: a live instance and physical device, and a pointer to
        // store the requirements in.
        let status = unsafe {
            emberwave_vulkan_requirements_create(
                &self.described(),
                self.physical_device(),
                &mut requirements,
            )
        };
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
        let mut count = 0;
        // SAFETY: live requirements; the names live as long as the process.
        let names = unsafe {
            let at = emberwave_vulkan_requirements_extensions(requirements, &mut count);
            if count == 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(at, count as usize).to_vec()
            }
        };
        // SAFETY: live requirements.
        let features = unsafe { emberwave_vulkan_requirements_features(requirements) };
        assert!(!features.is_null());
        // SAFETY: a VkPhysicalDeviceFeatures2, as the C API says.
        let s_type = unsafe { (*features.cast::<vk::PhysicalDeviceFeatures2>()).s_type };
        assert_eq!(s_type, vk::StructureType::PHYSICAL_DEVICE_FEATURES_2);
        let priorities = vec![1.0; self.queues.min(2) as usize];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(self.family)
            .queue_priorities(&priorities)];
        let mut info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&names);
        // The requirements' features as they are.
        info.p_next = features;
        // SAFETY: a valid description of a device of that physical device.
        let device = unsafe { self.instance.create_device(self.physical, &info, None) }
            .expect("a device with emberwave's requirements");
        // SAFETY: the device is made; the names outlive the requirements.
        unsafe { emberwave_vulkan_requirements_destroy(requirements) };
        (names, device)
    }

    /// A context on `device`'s queue `index`, under `lock` if given.
    fn context(
        &self,
        device: &ash::Device,
        names: &[*const c_char],
        index: u32,
        lock: Option<&QueueLock>,
    ) -> *mut EmberwaveContext {
        self.try_context(device, names, index, lock, proc_addr(&self.entry))
            .unwrap_or_else(|status| panic!("{status:?}: {}", last_error()))
    }

    /// A context on `device`'s queue `index`, under `lock` if given, with its
    /// functions found through `get_instance_proc_addr`.
    fn try_context(
        &self,
        device: &ash::Device,
        names: &[*const c_char],
        index: u32,
        lock: Option<&QueueLock>,
        get_instance_proc_addr: EmberwaveVkGetInstanceProcAddr,
    ) -> Result<*mut EmberwaveContext, EmberwaveStatus> {
        let described = EmberwaveVulkanDevice {
            instance: self.described_through(get_instance_proc_addr),
            physical_device: self.physical_device(),
            device: device.handle().as_raw() as usize as *mut c_void,
            extensions: names.as_ptr(),
            extension_count: names.len() as u32,
            queue_family_index: self.family,
            queue_index: index,
            lock_queue: lock.map(|_| QueueLock::lock as unsafe extern "C" fn(*mut c_void)),
            unlock_queue: lock.map(|_| QueueLock::unlock as unsafe extern "C" fn(*mut c_void)),
            user: lock.map_or(ptr::null_mut(), |lock| {
                ptr::from_ref(lock).cast_mut().cast()
            }),
        };
        let mut context = ptr::null_mut();
        // SAFETY: a live device made with emberwave's requirements, and a
        // pointer to store the context in.
        let status = unsafe { emberwave_context_create_vulkan(&described, &mut context) };
        if status == EmberwaveStatus::Ok {
            Ok(context)
        } else {
            assert!(context.is_null());
            Err(status)
        }
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        // SAFETY: everything made on it is gone by now.
        unsafe {
            if let Some((utils, messenger)) = &self.validation {
                utils.destroy_debug_utils_messenger(*messenger, None);
            }
            self.instance.destroy_instance(None);
        }
        let errors = std::mem::take(&mut *VALIDATION_ERRORS.lock().unwrap());
        if !std::thread::panicking() {
            assert!(errors.is_empty(), "the validation layer: {errors:#?}");
        }
    }
}

/// A client's lock of a queue, which counts its uses.
#[derive(Default)]
struct QueueLock {
    state: Mutex<(bool, usize)>,
    freed: Condvar,
}

impl QueueLock {
    /// # Safety
    ///
    /// `user` is a live `QueueLock`.
    unsafe extern "C" fn lock(user: *mut c_void) {
        // SAFETY: as the caller promises.
        let this = unsafe { &*user.cast::<QueueLock>() };
        let mut state = this.state.lock().unwrap();
        while state.0 {
            state = this.freed.wait(state).unwrap();
        }
        *state = (true, state.1 + 1);
        *LOCK_HOLDER.lock().unwrap() = Some(std::thread::current().id());
    }

    /// # Safety
    ///
    /// `user` is a live `QueueLock`, which `lock` took.
    unsafe extern "C" fn unlock(user: *mut c_void) {
        // SAFETY: as the caller promises.
        let this = unsafe { &*user.cast::<QueueLock>() };
        let mut state = this.state.lock().unwrap();
        assert!(state.0, "unlocked, not locked");
        *LOCK_HOLDER.lock().unwrap() = None;
        state.0 = false;
        this.freed.notify_one();
    }

    /// How many times it was taken, once it is free.
    fn taken(&self) -> usize {
        let state = self.state.lock().unwrap();
        assert!(!state.0, "still locked");
        state.1
    }
}

/// The thread that holds a `QueueLock`, if one does.
static LOCK_HOLDER: Mutex<Option<std::thread::ThreadId>> = Mutex::new(None);

/// Whether the calling thread holds a `QueueLock`.
fn held_here() -> bool {
    *LOCK_HOLDER.lock().unwrap() == Some(std::thread::current().id())
}

/// The loader's functions behind the watched ones, as addresses.
static REAL_GET_INSTANCE_PROC_ADDR: AtomicUsize = AtomicUsize::new(0);
static REAL_GET_DEVICE_PROC_ADDR: AtomicUsize = AtomicUsize::new(0);
static REAL_QUEUE_SUBMIT: AtomicUsize = AtomicUsize::new(0);
static REAL_QUEUE_WAIT_IDLE: AtomicUsize = AtomicUsize::new(0);

/// What the watched functions saw: submissions, waits for an idle queue,
/// and those of either without a `QueueLock` held on the calling thread.
static SUBMITS: AtomicUsize = AtomicUsize::new(0);
static WAITS: AtomicUsize = AtomicUsize::new(0);
static UNLOCKED: AtomicUsize = AtomicUsize::new(0);

/// A vkGetInstanceProcAddr for emberwave that hands out the loader's
/// functions, but for vkQueueSubmit and vkQueueWaitIdle, found through
/// vkGetDeviceProcAddr, watched ones, which count their calls and whether
/// the client's lock was held: what Vulkan's external synchronisation of
/// the queue requires.
fn watched_proc_addr(entry: &ash::Entry) -> EmberwaveVkGetInstanceProcAddr {
    REAL_GET_INSTANCE_PROC_ADDR.store(
        entry.static_fn().get_instance_proc_addr as usize,
        Ordering::SeqCst,
    );
    for count in [&SUBMITS, &WAITS, &UNLOCKED] {
        count.store(0, Ordering::SeqCst);
    }
    // SAFETY: a vkGetInstanceProcAddr, as the C API's type: Vulkan's calling
    // convention is C's on the 64-bit systems these tests run on.
    unsafe {
        std::mem::transmute::<vk::PFN_vkGetInstanceProcAddr, EmberwaveVkGetInstanceProcAddr>(
            watched_get_instance_proc_addr,
        )
    }
}

unsafe extern "system" fn watched_get_instance_proc_addr(
    instance: vk::Instance,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    // SAFETY: the loader's vkGetInstanceProcAddr, stored before.
    let real: vk::PFN_vkGetInstanceProcAddr =
        unsafe { std::mem::transmute(REAL_GET_INSTANCE_PROC_ADDR.load(Ordering::SeqCst)) };
    // SAFETY: as the caller passes them.
    let found = unsafe { real(instance, name) };
    // SAFETY: a function's name, from the caller.
    if unsafe { CStr::from_ptr(name) } == c"vkGetDeviceProcAddr" {
        REAL_GET_DEVICE_PROC_ADDR.store(found.map_or(0, |found| found as usize), Ordering::SeqCst);
        // SAFETY: a function of vkGetDeviceProcAddr's type, as a void function.
        return Some(unsafe {
            std::mem::transmute::<vk::PFN_vkGetDeviceProcAddr, unsafe extern "system" fn()>(
                watched_get_device_proc_addr,
            )
        });
    }
    found
}

unsafe extern "system" fn watched_get_device_proc_addr(
    device: vk::Device,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    // SAFETY: the loader's vkGetDeviceProcAddr, stored before.
    let real: vk::PFN_vkGetDeviceProcAddr =
        unsafe { std::mem::transmute(REAL_GET_DEVICE_PROC_ADDR.load(Ordering::SeqCst)) };
    // SAFETY: as the caller passes them.
    let found = unsafe { real(device, name) };
    let address = found.map_or(0, |found| found as usize);
    // SAFETY: a function's name, from the caller.
    let name = unsafe { CStr::from_ptr(name) };
    // SAFETY: functions of those types, as void functions.
    unsafe {
        if name == c"vkQueueSubmit" {
            REAL_QUEUE_SUBMIT.store(address, Ordering::SeqCst);
            return Some(std::mem::transmute::<
                vk::PFN_vkQueueSubmit,
                unsafe extern "system" fn(),
            >(watched_queue_submit));
        }
        if name == c"vkQueueWaitIdle" {
            REAL_QUEUE_WAIT_IDLE.store(address, Ordering::SeqCst);
            return Some(std::mem::transmute::<
                vk::PFN_vkQueueWaitIdle,
                unsafe extern "system" fn(),
            >(watched_queue_wait_idle));
        }
    }
    found
}

unsafe extern "system" fn watched_queue_submit(
    queue: vk::Queue,
    count: u32,
    submits: *const vk::SubmitInfo<'_>,
    fence: vk::Fence,
) -> vk::Result {
    SUBMITS.fetch_add(1, Ordering::SeqCst);
    if !held_here() {
        UNLOCKED.fetch_add(1, Ordering::SeqCst);
    }
    // SAFETY: the device's vkQueueSubmit, stored before, with the caller's
    // arguments.
    unsafe {
        let real: vk::PFN_vkQueueSubmit =
            std::mem::transmute(REAL_QUEUE_SUBMIT.load(Ordering::SeqCst));
        real(queue, count, submits, fence)
    }
}

unsafe extern "system" fn watched_queue_wait_idle(queue: vk::Queue) -> vk::Result {
    WAITS.fetch_add(1, Ordering::SeqCst);
    if !held_here() {
        UNLOCKED.fetch_add(1, Ordering::SeqCst);
    }
    // SAFETY: the device's vkQueueWaitIdle, stored before, with the
    // caller's queue.
    unsafe {
        let real: vk::PFN_vkQueueWaitIdle =
            std::mem::transmute(REAL_QUEUE_WAIT_IDLE.load(Ordering::SeqCst));
        real(queue)
    }
}

/// The frames of a test vector of 192x136 4:2:0.
fn frames() -> Vec<Vec<u8>> {
    let bytes = testdata("noise_192x136_420.pwv1");
    Pwv1File::parse(&bytes)
        .unwrap()
        .frames()
        .map(|frame| frame.unwrap().to_vec())
        .collect()
}

/// The Y, Cb and Cr planes of each of `frames` decoded on `context`.
fn decoded(context: *mut EmberwaveContext, frames: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let (width, height) = (192usize, 136usize);
    // SAFETY: a live context, frames of the lengths passed, planes of the
    // sizes and strides passed, and pointers to store results in.
    unsafe {
        let mut decoder = ptr::null_mut();
        assert_eq!(
            emberwave_decoder_create(context, 192, 136, EMBERWAVE_CHROMA_420, &mut decoder),
            EmberwaveStatus::Ok,
            "{}",
            last_error()
        );
        let pictures = frames
            .iter()
            .map(|frame| {
                emberwave_decoder_clear(decoder);
                assert_eq!(
                    emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
                    EmberwaveStatus::Ok
                );
                let mut planes = vec![0u8; width * height * 3 / 2];
                let (y, chroma) = planes.split_at_mut(width * height);
                let (cb, cr) = chroma.split_at_mut(width * height / 4);
                let plane = |data: &mut [u8], stride| EmberwavePlane {
                    data: data.as_mut_ptr(),
                    stride,
                };
                assert_eq!(
                    emberwave_decoder_decode_yuv(
                        decoder,
                        plane(y, width),
                        plane(cb, width / 2),
                        plane(cr, width / 2)
                    ),
                    EmberwaveStatus::Ok,
                    "{}",
                    last_error()
                );
                planes
            })
            .collect();
        emberwave_decoder_destroy(decoder);
        pictures
    }
}

/// A 192x136 picture encoded on `context`.
fn encoded(context: *mut EmberwaveContext) -> Vec<u8> {
    let (width, height) = (192usize, 136usize);
    let y: Vec<u8> = (0..width * height).map(|i| (i * 7 % 251) as u8).collect();
    let chroma = vec![128u8; width * height / 4];
    let mut frame = vec![0u8; 100_000];
    // SAFETY: a live context, planes of the sizes and strides passed, a
    // frame buffer of `budget` bytes, and pointers to store results in.
    unsafe {
        let mut encoder = ptr::null_mut();
        assert_eq!(
            emberwave_encoder_create(
                context,
                width as u32,
                height as u32,
                EMBERWAVE_CHROMA_420,
                ptr::null(),
                &mut encoder
            ),
            EmberwaveStatus::Ok,
            "{}",
            last_error()
        );
        let plane = |data: &[u8], stride| EmberwaveSourcePlane {
            data: data.as_ptr(),
            stride,
        };
        let mut size = 0;
        assert_eq!(
            emberwave_encoder_encode(
                encoder,
                plane(&y, width),
                plane(&chroma, width / 2),
                plane(&chroma, width / 2),
                frame.as_mut_ptr(),
                frame.len(),
                &mut size
            ),
            EmberwaveStatus::Ok,
            "{}",
            last_error()
        );
        emberwave_encoder_destroy(encoder);
        frame.truncate(size);
        frame
    }
}

#[test]
fn a_context_on_the_callers_device_works_as_one_of_emberwaves() {
    let Some(client) = Client::new() else {
        return;
    };
    let (names, device) = client.device();
    let context = client.context(&device, &names, 0, None);
    let mut own = ptr::null_mut();
    // On Vulkan too, for the same bytes.
    let options = EmberwaveContextOptions {
        backends: EMBERWAVE_BACKEND_VULKAN,
        flags: 0,
    };
    // SAFETY: options and a pointer to store the context in.
    let status = unsafe { emberwave_context_create(&options, &mut own) };
    assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());

    let frames = frames();
    assert_eq!(decoded(context, &frames), decoded(own, &frames));
    assert_eq!(encoded(context), encoded(own));
    // SAFETY: live contexts, which nothing uses any more.
    unsafe {
        emberwave_context_destroy(context);
        emberwave_context_destroy(own);
    }
    // SAFETY: emberwave let go of the device.
    unsafe { device.destroy_device(None) };
}

#[test]
fn contexts_on_one_queue_hold_the_callers_lock_around_their_submissions() {
    let Some(client) = Client::new() else {
        return;
    };
    let (names, device) = client.device();
    let lock = QueueLock::default();
    let watched = watched_proc_addr(&client.entry);
    let contexts = [0, 1].map(|_| {
        let made = client.try_context(&device, &names, 0, Some(&lock), watched);
        Shared(made.unwrap_or_else(|status| panic!("{status:?}: {}", last_error())))
    });
    // Another context on the queue with another lock, or none, is refused:
    // the contexts of a queue share one.
    let other = QueueLock::default();
    for refused in [Some(&other), None] {
        assert_eq!(
            client.try_context(&device, &names, 0, refused, watched),
            Err(EmberwaveStatus::InvalidArgument)
        );
        assert!(last_error().contains("same"), "{}", last_error());
    }
    let frames = frames();
    // A thread on each context: they take turns, and the caller's lock
    // around each submission.
    std::thread::scope(|scope| {
        for context in &contexts {
            let frames = &frames;
            scope.spawn(move || {
                let context = context;
                decoded(context.0, frames);
                encoded(context.0);
            });
        }
    });
    for context in contexts {
        // SAFETY: a live context, which nothing uses any more.
        unsafe { emberwave_context_destroy(context.0) };
    }
    // Each decode and each encode submits, besides the making of contexts,
    // decoders and encoders, and the last destroy waits for the queue to be
    // idle: all with the client's lock held on the thread.
    let (submits, waits) = (SUBMITS.load(Ordering::SeqCst), WAITS.load(Ordering::SeqCst));
    assert!(submits >= 2 * (frames.len() + 1), "{submits} submissions");
    assert!(waits >= 1, "no wait for the queue to be idle");
    assert_eq!(
        UNLOCKED.load(Ordering::SeqCst),
        0,
        "uses of the queue without the lock"
    );
    assert!(lock.taken() >= submits + waits);
    assert_eq!(other.taken(), 0);
    // SAFETY: emberwave let go of the device.
    unsafe { device.destroy_device(None) };
}

#[test]
fn a_device_without_the_required_extensions_is_refused() {
    let Some(client) = Client::new() else {
        return;
    };
    let (names, device) = client.device();
    let mut context = ptr::null_mut();
    let described = |extension_count: u32, lock: bool, unlock: bool| EmberwaveVulkanDevice {
        instance: client.described(),
        physical_device: client.physical_device(),
        device: device.handle().as_raw() as usize as *mut c_void,
        extensions: names.as_ptr(),
        extension_count,
        queue_family_index: client.family,
        queue_index: 0,
        lock_queue: lock.then_some(QueueLock::lock as unsafe extern "C" fn(*mut c_void)),
        unlock_queue: unlock.then_some(QueueLock::unlock as unsafe extern "C" fn(*mut c_void)),
        user: ptr::null_mut(),
    };
    // SAFETY: descriptions of a live device, and a pointer to store a
    // context in, which none of them makes.
    unsafe {
        if !names.is_empty() {
            let status = emberwave_context_create_vulkan(&described(0, false, false), &mut context);
            assert_eq!(status, EmberwaveStatus::InvalidArgument);
            assert!(last_error().contains("lacks"), "{}", last_error());
            assert!(context.is_null());
        }
        let status = emberwave_context_create_vulkan(
            &described(names.len() as u32, true, false),
            &mut context,
        );
        assert_eq!(status, EmberwaveStatus::InvalidArgument);
        assert!(context.is_null());
        assert_eq!(
            emberwave_context_create_vulkan(ptr::null(), &mut context),
            EmberwaveStatus::InvalidArgument
        );
        let mut requirements = ptr::null_mut();
        assert_eq!(
            emberwave_vulkan_requirements_create(
                &client.described(),
                ptr::null_mut(),
                &mut requirements
            ),
            EmberwaveStatus::InvalidArgument
        );
        assert!(requirements.is_null());
        let mut count = 7;
        assert!(emberwave_vulkan_requirements_extensions(ptr::null(), &mut count).is_null());
        assert_eq!(count, 0);
        assert!(emberwave_vulkan_requirements_features(ptr::null()).is_null());
        emberwave_vulkan_requirements_destroy(ptr::null_mut());
    }
    // SAFETY: nothing of emberwave's is on the device.
    unsafe { device.destroy_device(None) };
}

/// A context to hand to another thread: the C API lets several use one.
struct Shared(*mut EmberwaveContext);

// SAFETY: a context may be used by several threads at once.
unsafe impl Send for Shared {}
// SAFETY: as above.
unsafe impl Sync for Shared {}

/// How long a wait for the GPU may take before the test fails, rather than
/// hangs, in nanoseconds.
const TEN_SECONDS: u64 = 10_000_000_000;

/// An image of the client's, with its memory.
struct Image {
    image: vk::Image,
    memory: vk::DeviceMemory,
    width: u32,
    height: u32,
    /// Bytes a texel.
    texel: u32,
}

impl Client {
    /// A memory type of `bits` with `flags`.
    fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> u32 {
        // SAFETY: a physical device of the instance.
        let properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical)
        };
        (0..properties.memory_type_count)
            .find(|&index| {
                bits & (1 << index) != 0
                    && properties.memory_types[index as usize]
                        .property_flags
                        .contains(flags)
            })
            .expect("a memory type")
    }

    /// A 2D image of `format` for `usage`, as a client makes one.
    fn image(
        &self,
        device: &ash::Device,
        format: vk::Format,
        (width, height): (u32, u32),
        usage: vk::ImageUsageFlags,
    ) -> Image {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: a valid description of an image, and memory of a type it
        // takes, bound once.
        unsafe {
            let image = device.create_image(&info, None).expect("an image");
            let needs = device.get_image_memory_requirements(image);
            let allocation = vk::MemoryAllocateInfo::default()
                .allocation_size(needs.size)
                .memory_type_index(self.memory_type(
                    needs.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                ));
            let memory = device.allocate_memory(&allocation, None).expect("memory");
            device.bind_image_memory(image, memory, 0).expect("bound");
            Image {
                image,
                memory,
                width,
                height,
                texel: if format == vk::Format::R8_UNORM { 1 } else { 4 },
            }
        }
    }

    /// The texels of `image`, in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL as
    /// emberwave leaves it, copied out on the GPU and left so again, after
    /// `wait`, a timeline semaphore and its value, if given.
    fn read(
        &self,
        device: &ash::Device,
        image: vk::Image,
        (width, height, texel): (u32, u32, u32),
        wait: Option<(vk::Semaphore, u64)>,
    ) -> Vec<u8> {
        let bytes = u64::from(width * height * texel);
        // SAFETY: valid descriptions of a buffer, its memory, a command pool
        // and buffer of the queue's family; the image is in the layout the
        // barriers name; everything made is destroyed once the GPU is done.
        unsafe {
            let buffer = device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(bytes)
                        .usage(vk::BufferUsageFlags::TRANSFER_DST),
                    None,
                )
                .expect("a buffer");
            let needs = device.get_buffer_memory_requirements(buffer);
            let memory = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(needs.size)
                        .memory_type_index(self.memory_type(
                            needs.memory_type_bits,
                            vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                        )),
                    None,
                )
                .expect("memory");
            device.bind_buffer_memory(buffer, memory, 0).expect("bound");
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(self.family),
                    None,
                )
                .expect("a command pool");
            let commands = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .command_buffer_count(1),
                )
                .expect("a command buffer")[0];
            device
                .begin_command_buffer(commands, &vk::CommandBufferBeginInfo::default())
                .expect("begun");
            let whole = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            let barrier = |from, to, src_access, dst_access| {
                vk::ImageMemoryBarrier::default()
                    .old_layout(from)
                    .new_layout(to)
                    .src_access_mask(src_access)
                    .dst_access_mask(dst_access)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(whole)
            };
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier(
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::AccessFlags::MEMORY_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                )],
            );
            device.cmd_copy_image_to_buffer(
                commands,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer,
                &[vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })],
            );
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier(
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::AccessFlags::TRANSFER_READ,
                    vk::AccessFlags::MEMORY_READ,
                )],
            );
            device.end_command_buffer(commands).expect("ended");
            let fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .expect("a fence");
            let buffers = [commands];
            let (semaphores, values) = match wait {
                Some((semaphore, value)) => (vec![semaphore], vec![value]),
                None => (Vec::new(), Vec::new()),
            };
            let stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; semaphores.len()];
            let mut timeline =
                vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&values);
            let submit = vk::SubmitInfo::default()
                .command_buffers(&buffers)
                .wait_semaphores(&semaphores)
                .wait_dst_stage_mask(&stages)
                .push_next(&mut timeline);
            let queue = device.get_device_queue(self.family, 0);
            device
                .queue_submit(queue, &[submit], fence)
                .expect("submitted");
            device
                .wait_for_fences(&[fence], true, TEN_SECONDS)
                .expect("done");
            let mapped = device
                .map_memory(memory, 0, bytes, vk::MemoryMapFlags::empty())
                .expect("mapped");
            let texels = std::slice::from_raw_parts(mapped.cast::<u8>(), bytes as usize).to_vec();
            device.unmap_memory(memory);
            device.destroy_fence(fence, None);
            device.destroy_command_pool(pool, None);
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
            texels
        }
    }

    /// A timeline semaphore at 0.
    fn timeline(&self, device: &ash::Device) -> vk::Semaphore {
        let mut kind =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        // SAFETY: a valid description of a semaphore.
        unsafe {
            device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
                None,
            )
        }
        .expect("a timeline semaphore")
    }
}

/// Wraps `image` for `plane`, or for the picture, of `decoder`.
fn wrapped(
    decoder: *mut EmberwaveDecoder,
    image: &Image,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    plane: Option<u32>,
) -> Result<*mut EmberwaveTexture, (EmberwaveStatus, String)> {
    wrap(
        decoder,
        &EmberwaveVulkanImage {
            image: image.image.as_raw(),
            format: format.as_raw() as u32,
            usage: usage.as_raw(),
            layout: vk::ImageLayout::UNDEFINED.as_raw() as u32,
            release: None,
            user: ptr::null_mut(),
        },
        plane,
    )
}

/// Wraps the image `described`, for `plane` or for the picture, of
/// `decoder`.
fn wrap(
    decoder: *mut EmberwaveDecoder,
    described: &EmberwaveVulkanImage,
    plane: Option<u32>,
) -> Result<*mut EmberwaveTexture, (EmberwaveStatus, String)> {
    let mut texture = ptr::null_mut();
    // SAFETY: a live decoder, an image made as described, and a pointer to
    // store the texture in.
    let status = unsafe {
        match plane {
            Some(plane) => {
                emberwave_texture_from_vulkan_plane(decoder, plane, described, &mut texture)
            }
            None => emberwave_texture_from_vulkan(decoder, described, &mut texture),
        }
    };
    if status == EmberwaveStatus::Ok {
        Ok(texture)
    } else {
        Err((status, last_error()))
    }
}

/// The decoder's frame, pushed again, decoded the way that waits: BGRA, and
/// the Y, Cb and Cr planes.
fn expected(decoder: *mut EmberwaveDecoder, frame: &[u8]) -> (Vec<u8>, [Vec<u8>; 3]) {
    let (width, height) = (192usize, 136usize);
    // SAFETY: a live decoder, a frame of the length passed, and planes of
    // the sizes and strides passed.
    unsafe {
        emberwave_decoder_clear(decoder);
        assert_eq!(
            emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
            EmberwaveStatus::Ok
        );
        let mut rgb = vec![0u8; width * height * 3];
        let target = EmberwavePlane {
            data: rgb.as_mut_ptr(),
            stride: width * 3,
        };
        assert_eq!(
            emberwave_decoder_decode_rgb(decoder, ptr::null(), target),
            EmberwaveStatus::Ok
        );
        emberwave_decoder_clear(decoder);
        assert_eq!(
            emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
            EmberwaveStatus::Ok
        );
        let mut planes =
            [width * height, width * height / 4, width * height / 4].map(|size| vec![0u8; size]);
        let [y, cb, cr] = planes.each_mut().map(|plane| plane.as_mut_ptr());
        let plane = |data, stride| EmberwavePlane { data, stride };
        assert_eq!(
            emberwave_decoder_decode_yuv(
                decoder,
                plane(y, width),
                plane(cb, width / 2),
                plane(cr, width / 2)
            ),
            EmberwaveStatus::Ok
        );
        let bgra = rgb
            .chunks_exact(3)
            .flat_map(|p| [p[2], p[1], p[0], 255])
            .collect();
        (bgra, planes)
    }
}

#[test]
fn the_callers_images_take_the_picture_and_the_planes_without_a_wait() {
    let Some(client) = Client::new() else {
        return;
    };
    let (names, device) = client.device();
    let context = client.context(&device, &names, 0, None);
    let mut decoder = ptr::null_mut();
    // SAFETY: a live context, and a pointer to store the decoder in.
    let status =
        unsafe { emberwave_decoder_create(context, 192, 136, EMBERWAVE_CHROMA_420, &mut decoder) };
    assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());

    let usage = vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::TRANSFER_DST
        | vk::ImageUsageFlags::TRANSFER_SRC;
    let picture = client.image(&device, vk::Format::B8G8R8A8_UNORM, (192, 136), usage);
    let planes = [(192, 136), (96, 68), (96, 68)]
        .map(|size| client.image(&device, vk::Format::R8_UNORM, size, usage));
    let picture_texture =
        wrapped(decoder, &picture, vk::Format::B8G8R8A8_UNORM, usage, None).unwrap();
    let plane_textures: Vec<*mut EmberwaveTexture> = planes
        .iter()
        .zip([EMBERWAVE_PLANE_Y, EMBERWAVE_PLANE_CB, EMBERWAVE_PLANE_CR])
        .map(|(image, plane)| {
            wrapped(decoder, image, vk::Format::R8_UNORM, usage, Some(plane)).unwrap()
        })
        .collect();
    // SAFETY: a live texture of the caller's image.
    let image = unsafe { emberwave_texture_vulkan(picture_texture) };
    assert_eq!(image, picture.image.as_raw());
    let semaphore = client.timeline(&device);

    for (index, frame) in frames().iter().enumerate() {
        let (bgra, yuv) = expected(decoder, frame);
        let value = 2 * index as u64 + 1;
        let signal = EmberwaveVulkanSync {
            signal_semaphore: semaphore.as_raw(),
            signal_value: value,
            flags: EMBERWAVE_SYNC_SIGNAL_TIMELINE,
            ..Default::default()
        };
        // SAFETY: a live decoder, a frame of the length passed, live
        // textures, and a semaphore of the device.
        unsafe {
            emberwave_decoder_clear(decoder);
            assert_eq!(
                emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
                EmberwaveStatus::Ok
            );
            assert_eq!(
                emberwave_decoder_submit_texture_vulkan(
                    decoder,
                    ptr::null(),
                    picture_texture,
                    &signal
                ),
                EmberwaveStatus::Ok,
                "{}",
                last_error()
            );
        }
        let read = client.read(
            &device,
            picture.image,
            (192, 136, 4),
            Some((semaphore, value)),
        );
        assert!(read == bgra, "frame {index}: the picture");

        let signal = EmberwaveVulkanSync {
            signal_value: value + 1,
            ..signal
        };
        // SAFETY: as above.
        unsafe {
            emberwave_decoder_clear(decoder);
            assert_eq!(
                emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
                EmberwaveStatus::Ok
            );
            assert_eq!(
                emberwave_decoder_submit_plane_textures_vulkan(
                    decoder,
                    plane_textures[0],
                    plane_textures[1],
                    plane_textures[2],
                    &signal
                ),
                EmberwaveStatus::Ok,
                "{}",
                last_error()
            );
        }
        for (component, (image, plane)) in planes.iter().zip(&yuv).enumerate() {
            let read = client.read(
                &device,
                image.image,
                (image.width, image.height, image.texel),
                Some((semaphore, value + 1)),
            );
            assert!(read == *plane, "frame {index}: plane {component}");
        }
    }

    // SAFETY: live objects, which nothing uses any more; the GPU is done
    // with the images and the semaphore once the context is destroyed.
    unsafe {
        emberwave_texture_destroy(picture_texture);
        for texture in plane_textures {
            emberwave_texture_destroy(texture);
        }
        emberwave_decoder_destroy(decoder);
        emberwave_context_destroy(context);
        device.destroy_semaphore(semaphore, None);
        for image in planes.iter().chain([&picture]) {
            device.destroy_image(image.image, None);
            device.free_memory(image.memory, None);
        }
        device.destroy_device(None);
    }
}

#[test]
fn a_texture_of_a_vulkan_context_gives_its_image_after_a_decode_that_waits() {
    let Some(client) = Client::new() else {
        return;
    };
    let (names, device) = client.device();
    let context = client.context(&device, &names, 0, None);
    let frames = frames();
    // SAFETY: a live context and decoder, frames of the lengths passed, and
    // pointers to store results in.
    unsafe {
        let mut decoder = ptr::null_mut();
        assert_eq!(
            emberwave_decoder_create(context, 192, 136, EMBERWAVE_CHROMA_420, &mut decoder),
            EmberwaveStatus::Ok
        );
        let mut texture = ptr::null_mut();
        assert_eq!(
            emberwave_texture_create(decoder, EMBERWAVE_TEXTURE_BGRA8, &mut texture),
            EmberwaveStatus::Ok
        );
        let image = emberwave_texture_vulkan(texture);
        assert_ne!(image, 0);
        let (bgra, _) = expected(decoder, &frames[0]);
        emberwave_decoder_clear(decoder);
        assert_eq!(
            emberwave_decoder_push_packet(decoder, frames[0].as_ptr(), frames[0].len()),
            EmberwaveStatus::Ok
        );
        assert_eq!(
            emberwave_decoder_decode_texture(decoder, ptr::null(), texture),
            EmberwaveStatus::Ok,
            "{}",
            last_error()
        );
        let read = client.read(&device, vk::Image::from_raw(image), (192, 136, 4), None);
        assert!(read == bgra, "the picture");

        // A context of emberwave's own gives no image: its device is not the
        // caller's.
        let mut own = ptr::null_mut();
        assert_eq!(
            emberwave_context_create(ptr::null(), &mut own),
            EmberwaveStatus::Ok
        );
        let mut own_decoder = ptr::null_mut();
        assert_eq!(
            emberwave_decoder_create(own, 192, 136, EMBERWAVE_CHROMA_420, &mut own_decoder),
            EmberwaveStatus::Ok
        );
        let mut own_texture = ptr::null_mut();
        assert_eq!(
            emberwave_texture_create(own_decoder, EMBERWAVE_TEXTURE_BGRA8, &mut own_texture),
            EmberwaveStatus::Ok
        );
        assert_eq!(emberwave_texture_vulkan(own_texture), 0);
        assert_eq!(emberwave_texture_vulkan(ptr::null()), 0);
        // Nor do its decoders decode into textures without waiting this way.
        assert_eq!(
            emberwave_decoder_submit_texture_vulkan(
                own_decoder,
                ptr::null(),
                own_texture,
                ptr::null()
            ),
            EmberwaveStatus::InvalidArgument
        );

        emberwave_texture_destroy(own_texture);
        emberwave_decoder_destroy(own_decoder);
        emberwave_context_destroy(own);
        emberwave_texture_destroy(texture);
        emberwave_decoder_destroy(decoder);
        emberwave_context_destroy(context);
        device.destroy_device(None);
    }
}

#[test]
fn images_emberwave_cannot_fill_or_leave_readable_are_refused() {
    let Some(client) = Client::new() else {
        return;
    };
    let (names, device) = client.device();
    let context = client.context(&device, &names, 0, None);
    let mut decoder = ptr::null_mut();
    // SAFETY: a live context, and a pointer to store the decoder in.
    let status =
        unsafe { emberwave_decoder_create(context, 192, 136, EMBERWAVE_CHROMA_420, &mut decoder) };
    assert_eq!(status, EmberwaveStatus::Ok);
    let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
    let picture = client.image(&device, vk::Format::B8G8R8A8_UNORM, (192, 136), usage);
    let refused = |format, usage, plane| {
        let (status, message) =
            wrapped(decoder, &picture, format, usage, plane).expect_err("refused");
        assert_eq!(status, EmberwaveStatus::InvalidArgument);
        message
    };
    assert!(refused(vk::Format::R8_UNORM, usage, None).contains("format"));
    assert!(refused(vk::Format::B8G8R8A8_UNORM, usage, Some(EMBERWAVE_PLANE_Y)).contains("format"));
    assert!(
        refused(
            vk::Format::B8G8R8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST,
            None
        )
        .contains("SAMPLED")
    );
    assert!(
        refused(
            vk::Format::B8G8R8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED,
            None
        )
        .contains("neither")
    );
    let described = EmberwaveVulkanImage {
        image: picture.image.as_raw(),
        format: vk::Format::B8G8R8A8_UNORM.as_raw() as u32,
        usage: usage.as_raw(),
        layout: vk::ImageLayout::GENERAL.as_raw() as u32,
        release: None,
        user: ptr::null_mut(),
    };
    let mut texture = ptr::null_mut();
    // SAFETY: a live decoder, and descriptions that are refused.
    unsafe {
        assert_eq!(
            emberwave_texture_from_vulkan(decoder, &described, &mut texture),
            EmberwaveStatus::InvalidArgument
        );
        assert!(last_error().contains("layout"), "{}", last_error());
        assert!(texture.is_null());
        assert_eq!(
            emberwave_texture_from_vulkan(decoder, ptr::null(), &mut texture),
            EmberwaveStatus::InvalidArgument
        );
        emberwave_decoder_destroy(decoder);
        emberwave_context_destroy(context);
        device.destroy_image(picture.image, None);
        device.free_memory(picture.memory, None);
        device.destroy_device(None);
    }
}

impl Client {
    /// Moves `image`, in `VK_IMAGE_LAYOUT_UNDEFINED`, to
    /// `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` on the GPU, and waits.
    fn make_readable(&self, device: &ash::Device, image: vk::Image) {
        // SAFETY: a command pool and buffer of the queue's family, a valid
        // barrier for a live image, and everything made destroyed once done.
        unsafe {
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(self.family),
                    None,
                )
                .expect("a command pool");
            let commands = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .command_buffer_count(1),
                )
                .expect("a command buffer")[0];
            device
                .begin_command_buffer(commands, &vk::CommandBufferBeginInfo::default())
                .expect("begun");
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
            device.end_command_buffer(commands).expect("ended");
            let fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .expect("a fence");
            let buffers = [commands];
            let queue = device.get_device_queue(self.family, 0);
            device
                .queue_submit(
                    queue,
                    &[vk::SubmitInfo::default().command_buffers(&buffers)],
                    fence,
                )
                .expect("submitted");
            device
                .wait_for_fences(&[fence], true, TEN_SECONDS)
                .expect("done within ten seconds");
            device.destroy_fence(fence, None);
            device.destroy_command_pool(pool, None);
        }
    }

    /// A binary semaphore.
    fn binary(&self, device: &ash::Device) -> vk::Semaphore {
        // SAFETY: a valid description of a semaphore.
        unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
            .expect("a binary semaphore")
    }
}

/// The picture of `frame`, pushed again into `decoder` of `width` by
/// `height`, decoded the way that waits, as RGBA or BGRA.
fn expected_picture(
    decoder: *mut EmberwaveDecoder,
    frame: &[u8],
    (width, height): (usize, usize),
    bgra: bool,
) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    // SAFETY: a live decoder, a frame of the length passed, and a buffer of
    // the size and stride passed.
    unsafe {
        emberwave_decoder_clear(decoder);
        assert_eq!(
            emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
            EmberwaveStatus::Ok
        );
        let target = EmberwavePlane {
            data: rgb.as_mut_ptr(),
            stride: width * 3,
        };
        assert_eq!(
            emberwave_decoder_decode_rgb(decoder, ptr::null(), target),
            EmberwaveStatus::Ok
        );
        emberwave_decoder_clear(decoder);
        assert_eq!(
            emberwave_decoder_push_packet(decoder, frame.as_ptr(), frame.len()),
            EmberwaveStatus::Ok
        );
    }
    rgb.chunks_exact(3)
        .flat_map(|p| {
            if bgra {
                [p[2], p[1], p[0], 255]
            } else {
                [p[0], p[1], p[2], 255]
            }
        })
        .collect()
}

#[test]
fn semaphores_hold_the_decode_back_and_order_the_reads() {
    let Some(client) = Client::new() else {
        return;
    };
    let (names, device) = client.device();
    let context = client.context(&device, &names, 0, None);
    let mut decoder = ptr::null_mut();
    // SAFETY: a live context, and a pointer to store the decoder in.
    let status =
        unsafe { emberwave_decoder_create(context, 192, 136, EMBERWAVE_CHROMA_420, &mut decoder) };
    assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    let usage = vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::TRANSFER_DST
        | vk::ImageUsageFlags::TRANSFER_SRC;
    let picture = client.image(&device, vk::Format::B8G8R8A8_UNORM, (192, 136), usage);
    let texture = wrapped(decoder, &picture, vk::Format::B8G8R8A8_UNORM, usage, None).unwrap();
    let frames = frames();
    let submit = |sync: Option<&EmberwaveVulkanSync>| {
        // SAFETY: a live decoder and texture, and a sync that is NULL or
        // holds semaphores of the device.
        unsafe {
            emberwave_decoder_submit_texture_vulkan(
                decoder,
                ptr::null(),
                texture,
                sync.map_or(ptr::null(), ptr::from_ref),
            )
        }
    };

    // A timeline wait holds the decode back until the client signals it.
    let gate = client.timeline(&device);
    let done = client.timeline(&device);
    let bgra = expected_picture(decoder, &frames[0], (192, 136), true);
    let sync = EmberwaveVulkanSync {
        wait_semaphore: gate.as_raw(),
        wait_value: 1,
        signal_semaphore: done.as_raw(),
        signal_value: 1,
        flags: EMBERWAVE_SYNC_WAIT_TIMELINE | EMBERWAVE_SYNC_SIGNAL_TIMELINE,
    };
    assert_eq!(submit(Some(&sync)), EmberwaveStatus::Ok, "{}", last_error());
    std::thread::sleep(std::time::Duration::from_millis(300));
    // SAFETY: timeline semaphores of the device.
    unsafe {
        assert_eq!(
            device.get_semaphore_counter_value(done).unwrap(),
            0,
            "the decode ran before its wait was signalled"
        );
        device
            .signal_semaphore(&vk::SemaphoreSignalInfo::default().semaphore(gate).value(1))
            .unwrap();
    }
    let read = client.read(&device, picture.image, (192, 136, 4), Some((done, 1)));
    assert!(read == bgra, "the picture after the timeline wait");

    // Binary semaphores: the client's own work signals the one the decode
    // waits for, and its read waits for the one the decode signals.
    let ready = client.binary(&device);
    let decoded = client.binary(&device);
    let bgra = expected_picture(decoder, &frames[1], (192, 136), true);
    // SAFETY: a queue of the device, and a binary semaphore to signal.
    unsafe {
        let queue = device.get_device_queue(client.family, 0);
        let signals = [ready];
        device
            .queue_submit(
                queue,
                &[vk::SubmitInfo::default().signal_semaphores(&signals)],
                vk::Fence::null(),
            )
            .unwrap();
    }
    let sync = EmberwaveVulkanSync {
        wait_semaphore: ready.as_raw(),
        signal_semaphore: decoded.as_raw(),
        ..Default::default()
    };
    assert_eq!(submit(Some(&sync)), EmberwaveStatus::Ok, "{}", last_error());
    let read = client.read(&device, picture.image, (192, 136, 4), Some((decoded, 0)));
    assert!(read == bgra, "the picture after the binary semaphores");

    // No semaphores: the client's later work on the queue reads the
    // picture after a barrier from all commands, as its read has.
    let bgra = expected_picture(decoder, &frames[2], (192, 136), true);
    assert_eq!(submit(None), EmberwaveStatus::Ok, "{}", last_error());
    let read = client.read(&device, picture.image, (192, 136, 4), None);
    assert!(read == bgra, "the picture without semaphores");

    // Flags it does not know.
    let sync = EmberwaveVulkanSync {
        flags: 1 << 7,
        ..Default::default()
    };
    assert_eq!(submit(Some(&sync)), EmberwaveStatus::InvalidArgument);

    // SAFETY: live objects, which nothing uses any more; the GPU is done
    // with them once the context is destroyed.
    unsafe {
        emberwave_texture_destroy(texture);
        emberwave_decoder_destroy(decoder);
        emberwave_context_destroy(context);
        for semaphore in [gate, done, ready, decoded] {
            device.destroy_semaphore(semaphore, None);
        }
        device.destroy_image(picture.image, None);
        device.free_memory(picture.memory, None);
        device.destroy_device(None);
    }
}

/// The calls of the release callback of `a_wrapped_image_is_drawn_into_and_released`.
static RELEASED: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn count_release(_user: *mut c_void) {
    RELEASED.fetch_add(1, Ordering::SeqCst);
}

#[test]
fn a_wrapped_image_is_drawn_into_and_released() {
    let Some(client) = Client::new() else {
        return;
    };
    let (names, device) = client.device();
    let context = client.context(&device, &names, 0, None);
    let mut decoder = ptr::null_mut();
    // 200 pixels wide, the picture is drawn rather than copied.
    // SAFETY: a live context, and a pointer to store the decoder in.
    let status =
        unsafe { emberwave_decoder_create(context, 200, 120, EMBERWAVE_CHROMA_420, &mut decoder) };
    assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
    let bytes = testdata("gradient_200x120_420.pwv1");
    let frames: Vec<Vec<u8>> = Pwv1File::parse(&bytes)
        .unwrap()
        .frames()
        .map(|frame| frame.unwrap().to_vec())
        .collect();
    let usage = vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::COLOR_ATTACHMENT
        | vk::ImageUsageFlags::TRANSFER_SRC;
    let picture = client.image(&device, vk::Format::R8G8B8A8_UNORM, (200, 120), usage);
    // Wrapped in the layout a decode leaves, which the client gave it.
    client.make_readable(&device, picture.image);
    RELEASED.store(0, Ordering::SeqCst);
    let texture = wrap(
        decoder,
        &EmberwaveVulkanImage {
            image: picture.image.as_raw(),
            format: vk::Format::R8G8B8A8_UNORM.as_raw() as u32,
            usage: usage.as_raw(),
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL.as_raw() as u32,
            release: Some(count_release),
            user: ptr::null_mut(),
        },
        None,
    )
    .unwrap();
    let done = client.timeline(&device);
    for (index, frame) in frames.iter().enumerate() {
        let rgba = expected_picture(decoder, frame, (200, 120), false);
        let value = index as u64 + 1;
        let sync = EmberwaveVulkanSync {
            signal_semaphore: done.as_raw(),
            signal_value: value,
            flags: EMBERWAVE_SYNC_SIGNAL_TIMELINE,
            ..Default::default()
        };
        // SAFETY: a live decoder and texture, and a semaphore of the device.
        let status = unsafe {
            emberwave_decoder_submit_texture_vulkan(decoder, ptr::null(), texture, &sync)
        };
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
        let read = client.read(&device, picture.image, (200, 120, 4), Some((done, value)));
        assert!(read == rgba, "frame {index}");
    }
    // The image is released once emberwave and the GPU are done with it:
    // by the time the context is gone at the latest.
    // SAFETY: live objects, which nothing uses any more.
    unsafe {
        emberwave_texture_destroy(texture);
        emberwave_decoder_destroy(decoder);
        emberwave_context_destroy(context);
    }
    assert_eq!(RELEASED.load(Ordering::SeqCst), 1, "released once");
    // SAFETY: released by emberwave, and the semaphore's work is done.
    unsafe {
        device.destroy_semaphore(done, None);
        device.destroy_image(picture.image, None);
        device.free_memory(picture.memory, None);
        device.destroy_device(None);
    }
}
