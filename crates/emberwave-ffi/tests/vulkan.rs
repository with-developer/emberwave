//! The C API on a Vulkan device of the caller's, made here as a client makes
//! its own: an instance, then a device with what
//! `emberwave_vulkan_requirements_create` lists, then contexts on its queue
//! (decision D61). Linux CI runs it on lavapipe; elsewhere it skips without a
//! Vulkan device.

#![cfg(any(target_os = "linux", target_os = "android", windows))]

use std::ffi::{CStr, c_char, c_void};
use std::path::PathBuf;
use std::ptr;
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

/// The caller's instance and the physical device it chose.
struct Client {
    entry: ash::Entry,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    /// A queue family with graphics and compute.
    family: u32,
    /// How many queues it has.
    queues: u32,
}

impl Client {
    /// An instance of Vulkan 1.3 and its first physical device, or `None`
    /// without either where the GPU tests may skip.
    fn new() -> Option<Self> {
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
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let info = vk::InstanceCreateInfo::default().application_info(&app);
        // SAFETY: a valid description of an instance.
        let instance = match unsafe { entry.create_instance(&info, None) } {
            Ok(instance) => instance,
            Err(error) => {
                skip(format!("no Vulkan instance: {error}"));
                return None;
            }
        };
        // SAFETY: a live instance.
        let physical = unsafe { instance.enumerate_physical_devices() }
            .ok()
            .and_then(|all| all.into_iter().next());
        let Some(physical) = physical else {
            // SAFETY: nothing made on it.
            unsafe { instance.destroy_instance(None) };
            skip("no Vulkan device".into());
            return None;
        };
        // SAFETY: a physical device of the instance.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let wanted = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
        let (family, properties) = families
            .iter()
            .enumerate()
            .find(|(_, family)| family.queue_flags.contains(wanted))
            .expect("a family with graphics and compute");
        Some(Self {
            entry,
            instance,
            physical,
            family: family as u32,
            queues: properties.queue_count,
        })
    }

    fn described(&self) -> EmberwaveVulkanInstance {
        EmberwaveVulkanInstance {
            // SAFETY: the loader's vkGetInstanceProcAddr, as the C API's type:
            // Vulkan's calling convention is C's on 64-bit systems.
            get_instance_proc_addr: unsafe {
                std::mem::transmute::<vk::PFN_vkGetInstanceProcAddr, EmberwaveVkGetInstanceProcAddr>(
                    self.entry.static_fn().get_instance_proc_addr,
                )
            },
            instance: self.instance.handle().as_raw() as usize as *mut c_void,
            api_version: vk::API_VERSION_1_3,
            extensions: ptr::null(),
            extension_count: 0,
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
        let described = EmberwaveVulkanDevice {
            instance: self.described(),
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
        assert_eq!(status, EmberwaveStatus::Ok, "{}", last_error());
        context
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        // SAFETY: everything made on it is gone by now.
        unsafe { self.instance.destroy_instance(None) };
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
    }

    /// # Safety
    ///
    /// `user` is a live `QueueLock`, which `lock` took.
    unsafe extern "C" fn unlock(user: *mut c_void) {
        // SAFETY: as the caller promises.
        let this = unsafe { &*user.cast::<QueueLock>() };
        let mut state = this.state.lock().unwrap();
        assert!(state.0, "unlocked, not locked");
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
    // SAFETY: NULL options and a pointer to store the context in.
    let status = unsafe { emberwave_context_create(ptr::null(), &mut own) };
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
    let contexts = [0, 1].map(|_| Shared(client.context(&device, &names, 0, Some(&lock))));
    let before = lock.taken();
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
    // Each decode and each encode submits, besides the making of the
    // decoders and encoders.
    let during = lock.taken() - before;
    assert!(during >= 2 * (frames.len() + 1), "taken {during} times");
    for context in contexts {
        // SAFETY: a live context, which nothing uses any more.
        unsafe { emberwave_context_destroy(context.0) };
    }
    lock.taken();
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
