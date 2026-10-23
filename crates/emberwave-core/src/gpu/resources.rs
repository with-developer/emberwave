//! Buffers, bind groups and read-backs: what the decoder and the encoder share.

use std::sync::mpsc;

use wgpu::util::DeviceExt;

use super::error::GpuError;
use super::pipelines::with_error_scope;

/// The most bytes one buffer can have and still be bound as a storage buffer.
pub(crate) fn buffer_limit(device: &wgpu::Device) -> u64 {
    let limits = device.limits();
    limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size)
}

/// An error unless a buffer of `bytes` bytes fits the device's limit;
/// `what` names it in the error.
pub(crate) fn check_size(
    device: &wgpu::Device,
    what: &'static str,
    bytes: u64,
) -> Result<(), GpuError> {
    let limit = buffer_limit(device);
    if bytes > limit {
        Err(GpuError::LimitExceeded {
            what,
            needed: bytes,
            limit,
        })
    } else {
        Ok(())
    }
}

/// A buffer of at least 16 bytes, uninitialised.
pub(crate) fn buffer(
    device: &wgpu::Device,
    label: &str,
    bytes: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16),
        usage,
        mapped_at_creation: false,
    })
}

/// A bind group entry for a whole buffer.
pub(crate) fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// A uniform buffer holding `value`.
pub(crate) fn uniform_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
    value: &T,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(value),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

/// Waits for the GPU and copies a mappable buffer's first `bytes` bytes out.
pub(crate) fn read_back(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    bytes: u64,
) -> Result<Vec<u8>, GpuError> {
    // Mapping and unmapping report errors through the error scopes like the
    // other calls. Waiting cannot: wgpu treats a device error while polling as
    // fatal (decision D22).
    with_error_scope(device, || {
        let slice = buffer.slice(..bytes);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            // The receiver is only gone if this function already returned.
            let _ = sender.send(result);
        });

        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| GpuError::Readback(error.to_string()))?;
        receiver
            .recv()
            .map_err(|error| GpuError::Readback(error.to_string()))?
            .map_err(|error| GpuError::Readback(error.to_string()))?;

        let data = slice
            .get_mapped_range()
            .map_err(|error| GpuError::Readback(error.to_string()))?
            .to_vec();
        buffer.unmap();
        Ok(data)
    })?
}
