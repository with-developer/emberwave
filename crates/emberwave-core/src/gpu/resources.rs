//! Buffers, bind groups and read-backs: what the decoder and the encoder share.

use std::sync::mpsc;
use std::time::Duration;

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

/// A uniform buffer holding `value`, written through the queue. Not
/// `DeviceExt::create_buffer_init`, which maps the buffer at its creation
/// and panics when the device is lost (issue #35).
pub(crate) fn uniform_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    value: &T,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<T>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::bytes_of(value));
    buffer
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

/// Timestamp queries at the boundaries of compute passes: a beginning and an
/// end for each of a number of stages, a stage being one pass or several in
/// a row. Only on a device created with timestamps.
pub(crate) struct StageQueries {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// Nanoseconds per timestamp tick.
    period_ns: f64,
    stages: u32,
}

impl StageQueries {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue, stages: u32) -> Self {
        let bytes = u64::from(2 * stages) * 8;
        Self {
            query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("stage timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2 * stages,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamps"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp readback"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            period_ns: f64::from(queue.get_timestamp_period()),
            stages,
        }
    }

    /// The timestamps of a pass that begins stage `begins`, ends stage
    /// `ends`, or both.
    pub(crate) fn writes(
        &self,
        begins: Option<u32>,
        ends: Option<u32>,
    ) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: begins.map(|stage| 2 * stage),
            end_of_pass_write_index: ends.map(|stage| 2 * stage + 1),
        }
    }

    /// The time of each stage, once the work with the timestamps is done:
    /// `None` for a stage whose end came before its beginning, and `None`
    /// altogether if the device wrote no timestamps. A stage shorter than one
    /// tick of the timer is a valid measurement of zero, not a missing one.
    ///
    /// The timestamps are resolved here, in a submission of their own. With
    /// the work, in its command buffer, Metal now and then resolved the last
    /// stages before their timestamps were written, and gave the previous
    /// frame's or an end before the beginning.
    pub(crate) fn read(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Option<Vec<Option<Duration>>>, GpuError> {
        let bytes = u64::from(2 * self.stages) * 8;
        with_error_scope(device, || {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("timestamps"),
            });
            encoder.resolve_query_set(&self.query_set, 0..2 * self.stages, &self.resolve, 0);
            encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, bytes);
            queue.submit([encoder.finish()]);
        })?;
        let bytes = read_back(device, &self.readback, bytes)?;
        let ticks: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();
        // A device that wrote nothing leaves zeros.
        if ticks.iter().all(|&tick| tick == 0) {
            return Ok(None);
        }
        Ok(Some(
            ticks
                .chunks_exact(2)
                .map(|pair| {
                    (pair[1] >= pair[0]).then(|| {
                        Duration::from_secs_f64((pair[1] - pair[0]) as f64 * self.period_ns * 1e-9)
                    })
                })
                .collect(),
        ))
    }
}
