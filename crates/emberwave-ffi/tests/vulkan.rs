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
