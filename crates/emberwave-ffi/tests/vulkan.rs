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
                .wait_for_fences(&[fence], true, u64::MAX)
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
    let described = EmberwaveVulkanImage {
        image: image.image.as_raw(),
        format: format.as_raw() as u32,
        usage: usage.as_raw(),
        layout: vk::ImageLayout::UNDEFINED.as_raw() as u32,
    };
    let mut texture = ptr::null_mut();
    // SAFETY: a live decoder, an image made as described, and a pointer to
    // store the texture in.
    let status = unsafe {
        match plane {
            Some(plane) => {
                emberwave_texture_from_vulkan_plane(decoder, plane, &described, &mut texture)
            }
            None => emberwave_texture_from_vulkan(decoder, &described, &mut texture),
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
