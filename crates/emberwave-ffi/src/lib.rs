//! The C API of emberwave: PyroWave decoding and encoding on the GPU for C,
//! and for any language that can call C functions.
//!
//! `include/emberwave.h` is generated from this file by cbindgen, and
//! `tests/header.rs` fails when it is out of date. `examples/c/decode.c` and
//! `examples/c/encode.c` show the API in use, and `README.md` how to build
//! and link it.
//!
//! The conventions, which the header repeats:
//!
//! - Functions that can fail return an [`EmberwaveStatus`]. After a failure,
//!   [`emberwave_last_error`] describes it, on the thread that made the call.
//! - Objects are opaque. `*_create` makes one, `*_destroy` frees it, and
//!   destroying `NULL` does nothing.
//! - A context may be used by several threads at once. A decoder or an
//!   encoder may move between threads, but only one thread may use it at a
//!   time.
//! - A panic inside emberwave is caught and reported as
//!   `EMBERWAVE_STATUS_INTERNAL`. Destroy the object it happened in.

use std::cell::RefCell;
use std::ffi::{CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

use emberwave_core::bitstream::{
    ChromaSiting, ChromaSubsampling, ColorPrimaries, Component, Error as BitstreamError, Header,
    Pwv1Header, SequenceHeader, TransferFunction, YcbcrRange, YcbcrTransform,
};
use emberwave_core::color::{ColorMatrix, ColorParams};
use emberwave_core::encode::{self, EncodeError, Signalling};
use emberwave_core::gpu::{Decoder, Encoder, GpuContext, GpuError, GpuOptions};

/// The result of a call.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmberwaveStatus {
    /// The call succeeded.
    Ok = 0,
    /// A pointer was `NULL`, a value out of range, or an output buffer too
    /// small.
    InvalidArgument = 1,
    /// The data is not a PyroWave stream this decoder can take: malformed, of
    /// another picture size or chroma format, or not a start of frame header
    /// where one was expected.
    Bitstream = 2,
    /// No GPU adapter matched the options, or it refused to create a device.
    NoGpu = 3,
    /// The picture needs more GPU memory than the device allows.
    LimitExceeded = 4,
    /// The GPU reported an error, or reading the picture back failed.
    Gpu = 5,
    /// The GPU device was lost (a driver reset, or a GPU that went away). The
    /// context and its decoders and encoders cannot be used any more; create
    /// new ones.
    DeviceLost = 6,
    /// A bug in emberwave. Destroy the object the call was made on.
    Internal = 7,
}

/// 4:2:0: the chroma planes have half the width and height of the luma plane,
/// whose width and height are even.
pub const EMBERWAVE_CHROMA_420: u32 = 0;
/// 4:4:4: the chroma planes have the size of the luma plane.
pub const EMBERWAVE_CHROMA_444: u32 = 1;

/// Any backend this build supports: Vulkan, Metal or Direct3D 12.
pub const EMBERWAVE_BACKEND_ANY: u32 = 0;
/// Vulkan.
pub const EMBERWAVE_BACKEND_VULKAN: u32 = 1;
/// Metal.
pub const EMBERWAVE_BACKEND_METAL: u32 = 2;
/// Direct3D 12.
pub const EMBERWAVE_BACKEND_DX12: u32 = 4;

/// Only a software adapter will do: Mesa's lavapipe, or WARP on Windows.
pub const EMBERWAVE_CONTEXT_SOFTWARE: u32 = 1;

/// The ITU-R BT.601 matrix.
pub const EMBERWAVE_MATRIX_BT601: u32 = 0;
/// The ITU-R BT.709 matrix.
pub const EMBERWAVE_MATRIX_BT709: u32 = 1;
/// The ITU-R BT.2020 matrix, non-constant luminance.
pub const EMBERWAVE_MATRIX_BT2020: u32 = 2;

/// Video range: luma from 16 to 235, chroma from 16 to 240.
pub const EMBERWAVE_RANGE_LIMITED: u32 = 0;
/// Full range: 0 to 255.
pub const EMBERWAVE_RANGE_FULL: u32 = 1;

/// 4:2:0 chroma sits between the two luma samples it covers.
pub const EMBERWAVE_SITING_CENTER: u32 = 0;
/// 4:2:0 chroma sits on the left one of the two luma samples it covers.
pub const EMBERWAVE_SITING_LEFT: u32 = 1;
/// The primaries of sRGB and BT.709.
pub const EMBERWAVE_PRIMARIES_SRGB: u32 = 0;
/// The primaries of BT.2020.
pub const EMBERWAVE_PRIMARIES_BT2020: u32 = 1;
/// The sRGB transfer function, for standard dynamic range.
pub const EMBERWAVE_TRANSFER_SRGB: u32 = 0;
/// The PQ transfer function of SMPTE ST 2084, for high dynamic range.
pub const EMBERWAVE_TRANSFER_PQ: u32 = 1;

/// How `emberwave_context_create` chooses the GPU.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EmberwaveContextOptions {
    /// `EMBERWAVE_BACKEND_*` values combined with `|`, or
    /// `EMBERWAVE_BACKEND_ANY`.
    pub backends: u32,
    /// `EMBERWAVE_CONTEXT_*` flags combined with `|`, or 0.
    pub flags: u32,
}

/// What a stream says about its pictures.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EmberwaveStreamInfo {
    /// Width of the pictures in pixels.
    pub width: u32,
    /// Height of the pictures in pixels.
    pub height: u32,
    /// `EMBERWAVE_CHROMA_420` or `EMBERWAVE_CHROMA_444`.
    pub chroma: u32,
    /// The colour matrix the stream signals: `EMBERWAVE_MATRIX_BT709` or
    /// `EMBERWAVE_MATRIX_BT2020`.
    pub matrix: u32,
    /// `EMBERWAVE_RANGE_LIMITED` or `EMBERWAVE_RANGE_FULL`.
    pub range: u32,
    /// `EMBERWAVE_SITING_CENTER` or `EMBERWAVE_SITING_LEFT`.
    pub chroma_siting: u32,
    /// The frame rate as a fraction, from a PWV1 file header; 0/0 when it is
    /// not known.
    pub frame_rate_num: u32,
    /// See `frame_rate_num`.
    pub frame_rate_den: u32,
}

/// The sizes of the planes a decoder writes or an encoder reads.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EmberwaveFrameInfo {
    /// Width of the luma plane, and of the picture.
    pub width: u32,
    /// Height of the luma plane, and of the picture.
    pub height: u32,
    /// `EMBERWAVE_CHROMA_420` or `EMBERWAVE_CHROMA_444`.
    pub chroma: u32,
    /// Width of each chroma plane.
    pub chroma_width: u32,
    /// Height of each chroma plane.
    pub chroma_height: u32,
}

/// Where a decoder writes one plane or picture.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EmberwavePlane {
    /// The first byte of the first row.
    pub data: *mut u8,
    /// Bytes from the start of one row to the start of the next: at least the
    /// width of the plane, or three times the width for RGB.
    pub stride: usize,
}

/// The colour conversion of `emberwave_decoder_decode_rgb`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EmberwaveColor {
    /// An `EMBERWAVE_MATRIX_*` value.
    pub matrix: u32,
    /// An `EMBERWAVE_RANGE_*` value.
    pub range: u32,
    /// An `EMBERWAVE_SITING_*` value.
    pub chroma_siting: u32,
}

/// What the frames of an encoder say about their colours, in every start of
/// frame header. A decoder converts to RGB with the matrix, the range and the
/// chroma siting; the primaries and the transfer function are for whatever
/// shows the picture.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EmberwaveSignalling {
    /// `EMBERWAVE_MATRIX_BT709` or `EMBERWAVE_MATRIX_BT2020`: the format
    /// cannot signal BT.601.
    pub matrix: u32,
    /// An `EMBERWAVE_RANGE_*` value.
    pub range: u32,
    /// An `EMBERWAVE_SITING_*` value.
    pub chroma_siting: u32,
    /// An `EMBERWAVE_PRIMARIES_*` value.
    pub primaries: u32,
    /// An `EMBERWAVE_TRANSFER_*` value.
    pub transfer: u32,
    /// Zeros: room for what later versions may add.
    pub reserved: [u32; 3],
}

/// One plane of a picture to encode, in the caller's memory.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EmberwaveSourcePlane {
    /// The first byte of the first row.
    pub data: *const u8,
    /// Bytes from the start of one row to the start of the next: at least the
    /// width of the plane.
    pub stride: usize,
}

/// One packet of a frame: where it starts in the frame and how many bytes it
/// has.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EmberwavePacket {
    /// Bytes before the packet in the frame.
    pub offset: usize,
    /// Bytes of the packet.
    pub size: usize,
}

/// A GPU device, shared by the decoders and encoders created on it. Creating
/// one compiles the shaders, so make one and keep it.
pub struct EmberwaveContext {
    context: GpuContext,
    adapter: CString,
}

/// Decodes the frames of one stream: one picture size and chroma format.
pub struct EmberwaveDecoder {
    decoder: Decoder,
}

/// Encodes the pictures of one stream on the GPU: one picture size and
/// chroma format.
pub struct EmberwaveEncoder {
    encoder: Encoder,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

/// A failed call: the status to return, and the message for
/// [`emberwave_last_error`].
struct Failure(EmberwaveStatus, String);

impl Failure {
    fn argument(message: impl Into<String>) -> Self {
        Self(EmberwaveStatus::InvalidArgument, message.into())
    }
}

impl From<GpuError> for Failure {
    fn from(error: GpuError) -> Self {
        let status = match &error {
            GpuError::NoAdapter(_) | GpuError::RequestDevice(_) => EmberwaveStatus::NoGpu,
            GpuError::LimitExceeded { .. } | GpuError::TooLarge => EmberwaveStatus::LimitExceeded,
            GpuError::Device(_) | GpuError::Readback(_) => EmberwaveStatus::Gpu,
            GpuError::DeviceLost(_) => EmberwaveStatus::DeviceLost,
            GpuError::IncompleteFrame { .. } | GpuError::Bitstream(_) => EmberwaveStatus::Bitstream,
            // Planes or a budget the encoder cannot take, from
            // emberwave_encoder_encode; textures, which the C API has no way
            // to pass yet.
            GpuError::Encode(_) | GpuError::Texture(_) => EmberwaveStatus::InvalidArgument,
        };
        Self(status, error.to_string())
    }
}

impl From<BitstreamError> for Failure {
    fn from(error: BitstreamError) -> Self {
        Self(EmberwaveStatus::Bitstream, error.to_string())
    }
}

/// Runs the body of an API function: keeps the message of a failure for
/// [`emberwave_last_error`] and turns a panic into
/// [`EmberwaveStatus::Internal`].
fn run(body: impl FnOnce() -> Result<(), Failure>) -> EmberwaveStatus {
    run_or_lost(body, || None)
}

/// Like [`run`], but a panic is reported as a lost device if `lost` says the
/// device is lost. wgpu panics when the device fails while it waits for the
/// GPU (decision D22), after it has recorded the loss.
fn run_or_lost(
    body: impl FnOnce() -> Result<(), Failure>,
    lost: impl FnOnce() -> Option<String>,
) -> EmberwaveStatus {
    let Failure(status, message) = match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => return EmberwaveStatus::Ok,
        Ok(Err(failure)) => failure,
        Err(panic) => match lost() {
            Some(reason) => GpuError::DeviceLost(reason).into(),
            None => {
                let detail = panic
                    .downcast_ref::<&str>()
                    .map(|text| (*text).to_owned())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_default();
                Failure(
                    EmberwaveStatus::Internal,
                    format!("internal error: {detail}"),
                )
            }
        },
    };
    // C strings end at the first NUL, so a message is cut there.
    let bytes: Vec<u8> = message
        .into_bytes()
        .into_iter()
        .take_while(|&b| b != 0)
        .collect();
    let message = CString::new(bytes).unwrap_or_default();
    LAST_ERROR.with(|last| *last.borrow_mut() = message);
    status
}

/// The bytes `data` points to; C may pass `NULL` when `size` is 0.
///
/// # Safety
///
/// `data` is `NULL`, or points to `size` readable bytes that stay unchanged
/// while the returned slice is used.
unsafe fn input<'a>(data: *const u8, size: usize) -> Result<&'a [u8], Failure> {
    if size == 0 {
        return Ok(&[]);
    }
    if data.is_null() {
        return Err(Failure::argument("data is NULL"));
    }
    if size > isize::MAX as usize {
        return Err(Failure::argument("size is too large"));
    }
    // SAFETY: not NULL, not too large, and the caller promises `size` bytes.
    Ok(unsafe { std::slice::from_raw_parts(data, size) })
}

/// Stores `value` where `out` points.
///
/// # Safety
///
/// `out` is `NULL` or points to memory where a `T` may be written.
unsafe fn output<T>(out: *mut T, what: &str, value: T) -> Result<(), Failure> {
    if out.is_null() {
        return Err(Failure::argument(format!("{what} is NULL")));
    }
    // SAFETY: not NULL, and the caller promises that it may be written.
    unsafe { out.write(value) };
    Ok(())
}

/// The decoder `decoder` points to.
///
/// # Safety
///
/// `decoder` is `NULL` or a live decoder.
unsafe fn live<'a>(decoder: *const EmberwaveDecoder) -> Result<&'a EmberwaveDecoder, Failure> {
    // SAFETY: as the caller promises.
    unsafe { decoder.as_ref() }.ok_or_else(|| Failure::argument("decoder is NULL"))
}

/// The decoder `decoder` points to, for a call that changes it.
///
/// # Safety
///
/// `decoder` is `NULL` or a live decoder that no other thread is using.
unsafe fn live_mut<'a>(
    decoder: *mut EmberwaveDecoder,
) -> Result<&'a mut EmberwaveDecoder, Failure> {
    // SAFETY: as the caller promises.
    unsafe { decoder.as_mut() }.ok_or_else(|| Failure::argument("decoder is NULL"))
}

/// Why the device of the decoder `decoder` points to was lost, if it was.
///
/// # Safety
///
/// `decoder` is `NULL` or a live decoder.
unsafe fn lost(decoder: *const EmberwaveDecoder) -> Option<String> {
    // SAFETY: as the caller promises.
    unsafe { decoder.as_ref() }.and_then(|decoder| decoder.decoder.lost())
}

/// The encoder `encoder` points to, for a call that changes it.
///
/// # Safety
///
/// `encoder` is `NULL` or a live encoder that no other thread is using.
unsafe fn live_encoder<'a>(
    encoder: *mut EmberwaveEncoder,
) -> Result<&'a mut EmberwaveEncoder, Failure> {
    // SAFETY: as the caller promises.
    unsafe { encoder.as_mut() }.ok_or_else(|| Failure::argument("encoder is NULL"))
}

/// Why the device of the encoder `encoder` points to was lost, if it was.
///
/// # Safety
///
/// `encoder` is `NULL` or a live encoder.
unsafe fn encoder_lost(encoder: *const EmberwaveEncoder) -> Option<String> {
    // SAFETY: as the caller promises.
    unsafe { encoder.as_ref() }.and_then(|encoder| encoder.encoder.lost())
}

/// The chroma format an `EMBERWAVE_CHROMA_*` value names.
fn chroma_subsampling(chroma: u32) -> Result<ChromaSubsampling, Failure> {
    match chroma {
        EMBERWAVE_CHROMA_420 => Ok(ChromaSubsampling::Yuv420),
        EMBERWAVE_CHROMA_444 => Ok(ChromaSubsampling::Yuv444),
        other => Err(Failure::argument(format!("unknown chroma format {other}"))),
    }
}

/// The error of creating a decoder or an encoder of the caller's size: an
/// invalid size is the caller's numbers, not a stream's.
fn creation_failure(error: GpuError) -> Failure {
    match error {
        GpuError::Bitstream(BitstreamError::InvalidDimensions { .. }) => {
            Failure::argument(error.to_string())
        }
        other => other.into(),
    }
}

fn signalling(value: &EmberwaveSignalling) -> Result<Signalling, Failure> {
    if value.reserved != [0; 3] {
        return Err(Failure::argument("signalling.reserved is not zero"));
    }
    Ok(Signalling {
        color_primaries: match value.primaries {
            EMBERWAVE_PRIMARIES_SRGB => ColorPrimaries::Srgb,
            EMBERWAVE_PRIMARIES_BT2020 => ColorPrimaries::Bt2020,
            other => return Err(Failure::argument(format!("unknown primaries {other}"))),
        },
        transfer_function: match value.transfer {
            EMBERWAVE_TRANSFER_SRGB => TransferFunction::Srgb,
            EMBERWAVE_TRANSFER_PQ => TransferFunction::Pq,
            other => {
                return Err(Failure::argument(format!(
                    "unknown transfer function {other}"
                )));
            }
        },
        ycbcr_transform: match value.matrix {
            EMBERWAVE_MATRIX_BT709 => YcbcrTransform::Bt709,
            EMBERWAVE_MATRIX_BT2020 => YcbcrTransform::Bt2020,
            EMBERWAVE_MATRIX_BT601 => {
                return Err(Failure::argument(
                    "the format cannot signal the BT.601 matrix",
                ));
            }
            other => return Err(Failure::argument(format!("unknown matrix {other}"))),
        },
        ycbcr_range: match value.range {
            EMBERWAVE_RANGE_LIMITED => YcbcrRange::Limited,
            EMBERWAVE_RANGE_FULL => YcbcrRange::Full,
            other => return Err(Failure::argument(format!("unknown range {other}"))),
        },
        chroma_siting: match value.chroma_siting {
            EMBERWAVE_SITING_CENTER => ChromaSiting::Center,
            EMBERWAVE_SITING_LEFT => ChromaSiting::Left,
            other => return Err(Failure::argument(format!("unknown chroma siting {other}"))),
        },
    })
}

/// The bytes of a source plane of `rows` rows of `row_bytes` bytes, from the
/// first byte of the first row to the last byte of the last.
///
/// # Safety
///
/// `source.data` is `NULL` or points to `stride * (rows - 1) + row_bytes`
/// readable bytes that stay unchanged while the slice is used.
unsafe fn source_plane<'a>(
    source: &EmberwaveSourcePlane,
    name: &str,
    row_bytes: usize,
    rows: usize,
) -> Result<&'a [u8], Failure> {
    if source.data.is_null() {
        return Err(Failure::argument(format!("{name}.data is NULL")));
    }
    if source.stride < row_bytes {
        return Err(Failure::argument(format!(
            "{name}.stride is {}, less than the {row_bytes} bytes of a row",
            source.stride
        )));
    }
    let span = source
        .stride
        .checked_mul(rows.saturating_sub(1))
        .and_then(|bytes| bytes.checked_add(row_bytes))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| {
            Failure::argument(format!(
                "{name} spans more memory than a pointer can address"
            ))
        })?;
    // SAFETY: not NULL, not too large, and the caller promises the bytes.
    Ok(unsafe { std::slice::from_raw_parts(source.data, span) })
}

fn chroma_code(chroma: ChromaSubsampling) -> u32 {
    match chroma {
        ChromaSubsampling::Yuv420 => EMBERWAVE_CHROMA_420,
        ChromaSubsampling::Yuv444 => EMBERWAVE_CHROMA_444,
    }
}

fn stream_info(stream: &SequenceHeader) -> EmberwaveStreamInfo {
    let color = ColorParams::from_stream(stream);
    EmberwaveStreamInfo {
        width: stream.width,
        height: stream.height,
        chroma: chroma_code(stream.chroma),
        matrix: match color.matrix {
            ColorMatrix::Bt601 => EMBERWAVE_MATRIX_BT601,
            ColorMatrix::Bt709 => EMBERWAVE_MATRIX_BT709,
            ColorMatrix::Bt2020 => EMBERWAVE_MATRIX_BT2020,
        },
        range: match color.range {
            YcbcrRange::Limited => EMBERWAVE_RANGE_LIMITED,
            YcbcrRange::Full => EMBERWAVE_RANGE_FULL,
        },
        chroma_siting: match color.chroma_siting {
            ChromaSiting::Center => EMBERWAVE_SITING_CENTER,
            ChromaSiting::Left => EMBERWAVE_SITING_LEFT,
        },
        frame_rate_num: 0,
        frame_rate_den: 0,
    }
}

fn gpu_options(options: &EmberwaveContextOptions) -> Result<GpuOptions, Failure> {
    const BACKENDS: [(u32, wgpu::Backends); 3] = [
        (EMBERWAVE_BACKEND_VULKAN, wgpu::Backends::VULKAN),
        (EMBERWAVE_BACKEND_METAL, wgpu::Backends::METAL),
        (EMBERWAVE_BACKEND_DX12, wgpu::Backends::DX12),
    ];
    let known = BACKENDS.iter().fold(0, |bits, (bit, _)| bits | bit);
    if options.backends & !known != 0 {
        return Err(Failure::argument(format!(
            "unknown backend bits {:#x}",
            options.backends & !known
        )));
    }
    if options.flags & !EMBERWAVE_CONTEXT_SOFTWARE != 0 {
        return Err(Failure::argument(format!(
            "unknown flags {:#x}",
            options.flags & !EMBERWAVE_CONTEXT_SOFTWARE
        )));
    }
    let mut backends = wgpu::Backends::empty();
    for (bit, backend) in BACKENDS {
        if options.backends & bit != 0 {
            backends |= backend;
        }
    }
    if backends.is_empty() {
        backends = wgpu::Backends::PRIMARY;
    }
    Ok(GpuOptions {
        backends,
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: options.flags & EMBERWAVE_CONTEXT_SOFTWARE != 0,
        strict_webgpu: false,
        timestamps: false,
    })
}

fn color_params(color: &EmberwaveColor) -> Result<ColorParams, Failure> {
    Ok(ColorParams {
        matrix: match color.matrix {
            EMBERWAVE_MATRIX_BT601 => ColorMatrix::Bt601,
            EMBERWAVE_MATRIX_BT709 => ColorMatrix::Bt709,
            EMBERWAVE_MATRIX_BT2020 => ColorMatrix::Bt2020,
            other => return Err(Failure::argument(format!("unknown matrix {other}"))),
        },
        range: match color.range {
            EMBERWAVE_RANGE_LIMITED => YcbcrRange::Limited,
            EMBERWAVE_RANGE_FULL => YcbcrRange::Full,
            other => return Err(Failure::argument(format!("unknown range {other}"))),
        },
        chroma_siting: match color.chroma_siting {
            EMBERWAVE_SITING_CENTER => ChromaSiting::Center,
            EMBERWAVE_SITING_LEFT => ChromaSiting::Left,
            other => return Err(Failure::argument(format!("unknown chroma siting {other}"))),
        },
    })
}

/// Checks that `target` takes `rows` rows of `row_bytes` bytes: its pointer is
/// not `NULL`, its stride is at least a row, and the last byte is
/// addressable.
fn check_target(
    target: &EmberwavePlane,
    name: &str,
    row_bytes: usize,
    rows: usize,
) -> Result<(), Failure> {
    if target.data.is_null() {
        return Err(Failure::argument(format!("{name}.data is NULL")));
    }
    if target.stride < row_bytes {
        return Err(Failure::argument(format!(
            "{name}.stride is {}, less than the {row_bytes} bytes of a row",
            target.stride
        )));
    }
    let span = target
        .stride
        .checked_mul(rows.saturating_sub(1))
        .and_then(|bytes| bytes.checked_add(row_bytes))
        .filter(|&bytes| bytes <= isize::MAX as usize);
    if span.is_none() {
        return Err(Failure::argument(format!(
            "{name} spans more memory than a pointer can address"
        )));
    }
    Ok(())
}

/// Copies tightly packed rows of `row_bytes` bytes into `target`.
///
/// # Safety
///
/// `target` passed [`check_target`] for `source.len() / row_bytes` rows of
/// `row_bytes` bytes, and the memory it describes may be written and does not
/// overlap `source`.
unsafe fn copy_rows(source: &[u8], row_bytes: usize, target: &EmberwavePlane) {
    if row_bytes == 0 {
        return;
    }
    for (row, line) in source.chunks_exact(row_bytes).enumerate() {
        // SAFETY: the row lies within the span check_target accepted, which
        // the caller promises may be written.
        unsafe {
            std::ptr::copy_nonoverlapping(
                line.as_ptr(),
                target.data.add(row * target.stride),
                row_bytes,
            );
        }
    }
}

/// The version of emberwave, such as `"0.3.0"`. The string is static.
#[unsafe(no_mangle)]
pub extern "C" fn emberwave_version() -> *const c_char {
    const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    VERSION.as_ptr().cast()
}

/// Describes the last failure on the calling thread, or is empty if no call on
/// it has failed. The string stays valid until the next failing call on the
/// same thread.
#[unsafe(no_mangle)]
pub extern "C" fn emberwave_last_error() -> *const c_char {
    LAST_ERROR.with(|last| last.borrow().as_ptr())
}

/// Describes the stream of a PWV1 file from its header, the first 24 bytes:
/// picture size, chroma format, colour signalling and frame rate. Frames
/// follow the header, each a little-endian 32-bit size and that many bytes to
/// give to `emberwave_decoder_push_packet`.
///
/// # Safety
///
/// `data` points to `size` readable bytes, and `info` to an
/// `EmberwaveStreamInfo` to fill in.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_probe_pwv1(
    data: *const u8,
    size: usize,
    info: *mut EmberwaveStreamInfo,
) -> EmberwaveStatus {
    run(|| {
        // SAFETY: as the caller promises.
        let bytes = unsafe { input(data, size) }?;
        let header = Pwv1Header::parse(bytes)?;
        let described = EmberwaveStreamInfo {
            frame_rate_num: header.frame_rate_num,
            frame_rate_den: header.frame_rate_den,
            ..stream_info(&header.stream)
        };
        // SAFETY: as the caller promises.
        unsafe { output(info, "info", described) }
    })
}

/// Describes the stream from a packet that starts with a start of frame
/// header, as the first packet of every frame does: picture size, chroma
/// format and colour signalling. A packet that starts with a block gives
/// `EMBERWAVE_STATUS_BITSTREAM`.
///
/// # Safety
///
/// `data` points to `size` readable bytes, and `info` to an
/// `EmberwaveStreamInfo` to fill in.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_probe_packet(
    data: *const u8,
    size: usize,
    info: *mut EmberwaveStreamInfo,
) -> EmberwaveStatus {
    run(|| {
        // SAFETY: as the caller promises.
        let bytes = unsafe { input(data, size) }?;
        match Header::parse(bytes)? {
            // SAFETY: as the caller promises.
            Header::StartOfFrame(stream) => unsafe { output(info, "info", stream_info(&stream)) },
            Header::Block(_) => Err(Failure(
                EmberwaveStatus::Bitstream,
                "the packet starts with a block, not with a start of frame header".into(),
            )),
        }
    })
}

/// Creates a context on a GPU that `options` allow, or with `NULL` options on
/// any. With `NULL` options the environment variable `EMBERWAVE_BACKEND` can
/// narrow the choice, as for the command line tool: `vulkan`, `metal`, `dx12`,
/// a comma separated list of them, or `auto`; a value that names no backend is
/// ignored. A discrete GPU is preferred to an integrated one.
///
/// Contexts are created one at a time: calls from several threads wait for
/// each other. On Linux, Vulkan loaders 1.3.284 to 1.4.344 can crash the
/// process when one thread creates a device while another lists the
/// adapters of a new instance (loader 1.4.345 fixed it). The waiting covers
/// the contexts created here, not other users of Vulkan in the process.
///
/// # Safety
///
/// `options` is `NULL` or points to an `EmberwaveContextOptions`. `context`
/// points to where the new context is stored; it is set to `NULL` when the
/// call fails.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_context_create(
    options: *const EmberwaveContextOptions,
    context: *mut *mut EmberwaveContext,
) -> EmberwaveStatus {
    run(|| {
        // SAFETY: as the caller promises.
        unsafe { output(context, "context", std::ptr::null_mut()) }?;
        // SAFETY: as the caller promises.
        let gpu_options = match unsafe { options.as_ref() } {
            Some(options) => gpu_options(options)?,
            None => GpuOptions::default(),
        };
        let created = GpuContext::new(&gpu_options)?;
        let info = created.adapter_info();
        let adapter = format!("{} ({:?})", info.name, info.backend).replace('\0', "");
        let boxed = Box::new(EmberwaveContext {
            context: created,
            adapter: CString::new(adapter).unwrap_or_default(),
        });
        // SAFETY: checked by the first output() above.
        unsafe { context.write(Box::into_raw(boxed)) };
        Ok(())
    })
}

/// Destroys a context. Decoders and encoders created on it keep working, and
/// keep the GPU device until they are destroyed as well.
///
/// # Safety
///
/// `context` is `NULL`, or a context from `emberwave_context_create` that
/// has not been destroyed and that no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_context_destroy(context: *mut EmberwaveContext) {
    if !context.is_null() {
        // SAFETY: it came from Box::into_raw, as the caller promises.
        let boxed = unsafe { Box::from_raw(context) };
        let _ = catch_unwind(AssertUnwindSafe(move || drop(boxed)));
    }
}

/// The name and backend of the context's GPU, such as
/// `"Apple M3 Max (Metal)"`. The string belongs to the context. It is empty
/// for `NULL`.
///
/// # Safety
///
/// `context` is `NULL` or a live context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_context_adapter(
    context: *const EmberwaveContext,
) -> *const c_char {
    // SAFETY: as the caller promises.
    match unsafe { context.as_ref() } {
        Some(context) => context.adapter.as_ptr(),
        None => c"".as_ptr(),
    }
}

/// Creates a decoder on `context` for pictures of `width` x `height` pixels
/// in the chroma format `chroma`, an `EMBERWAVE_CHROMA_*` value. Width and
/// height are from 1 to 16384, and even for 4:2:0; other sizes give
/// `EMBERWAVE_STATUS_INVALID_ARGUMENT`. All the GPU memory the decoder needs is
/// allocated here.
///
/// # Safety
///
/// `context` is a live context. `decoder` points to where the new decoder is
/// stored; it is set to `NULL` when the call fails.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_decoder_create(
    context: *const EmberwaveContext,
    width: u32,
    height: u32,
    chroma: u32,
    decoder: *mut *mut EmberwaveDecoder,
) -> EmberwaveStatus {
    run(|| {
        // SAFETY: as the caller promises.
        unsafe { output(decoder, "decoder", std::ptr::null_mut()) }?;
        // SAFETY: as the caller promises.
        let Some(context) = (unsafe { context.as_ref() }) else {
            return Err(Failure::argument("context is NULL"));
        };
        let chroma = chroma_subsampling(chroma)?;
        let created =
            Decoder::new(&context.context, width, height, chroma).map_err(creation_failure)?;
        let boxed = Box::new(EmberwaveDecoder { decoder: created });
        // SAFETY: checked by the first output() above.
        unsafe { decoder.write(Box::into_raw(boxed)) };
        Ok(())
    })
}

/// Destroys a decoder and frees its GPU memory.
///
/// # Safety
///
/// `decoder` is `NULL`, or a decoder from `emberwave_decoder_create` that
/// has not been destroyed and that no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_decoder_destroy(decoder: *mut EmberwaveDecoder) {
    if !decoder.is_null() {
        // SAFETY: it came from Box::into_raw, as the caller promises.
        let boxed = unsafe { Box::from_raw(decoder) };
        let _ = catch_unwind(AssertUnwindSafe(move || drop(boxed)));
    }
}

/// The sizes of the planes the decoder writes.
///
/// # Safety
///
/// `decoder` is a live decoder, and `info` points to an `EmberwaveFrameInfo`
/// to fill in.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_decoder_info(
    decoder: *const EmberwaveDecoder,
    info: *mut EmberwaveFrameInfo,
) -> EmberwaveStatus {
    run(|| {
        // SAFETY: as the caller promises.
        let layout = unsafe { live(decoder) }?.decoder.layout();
        let (chroma_width, chroma_height) = layout.plane_size(Component::Cb);
        let described = EmberwaveFrameInfo {
            width: layout.width(),
            height: layout.height(),
            chroma: chroma_code(layout.chroma()),
            chroma_width,
            chroma_height,
        };
        // SAFETY: as the caller promises.
        unsafe { output(info, "info", described) }
    })
}

/// Gives received data to the decoder: one packet, or several back to back,
/// such as a whole frame of a PWV1 file. Packets may come in any order.
/// Packets of a newer frame discard what was received of an older one, and
/// packets of older frames are ignored. A malformed header or block is
/// rejected without changing anything, but blocks before it in the same data
/// are kept.
///
/// # Safety
///
/// `decoder` is a live decoder that no other thread is using, and `data`
/// points to `size` readable bytes (or is `NULL` with `size` 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_decoder_push_packet(
    decoder: *mut EmberwaveDecoder,
    data: *const u8,
    size: usize,
) -> EmberwaveStatus {
    run(|| {
        // SAFETY: as the caller promises.
        let decoder = unsafe { live_mut(decoder) }?;
        // SAFETY: as the caller promises.
        let bytes = unsafe { input(data, size) }?;
        Ok(decoder.decoder.push_packet(bytes)?)
    })
}

/// Whether the frame being received can be decoded: all its blocks arrived,
/// or, with `allow_partial_frame`, enough of them for a usable picture. False
/// once the frame has been decoded, and for `NULL`.
///
/// # Safety
///
/// `decoder` is `NULL` or a live decoder that no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_decoder_is_ready(
    decoder: *const EmberwaveDecoder,
    allow_partial_frame: bool,
) -> bool {
    // SAFETY: as the caller promises.
    match unsafe { decoder.as_ref() } {
        Some(decoder) => catch_unwind(AssertUnwindSafe(|| {
            decoder.decoder.is_ready(allow_partial_frame)
        }))
        .unwrap_or(false),
        None => false,
    }
}

/// Discards what was received, and forgets the order of the frames: the next
/// packet starts a frame, whatever its sequence number.
///
/// # Safety
///
/// `decoder` is `NULL` or a live decoder that no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_decoder_clear(decoder: *mut EmberwaveDecoder) {
    // SAFETY: as the caller promises.
    if let Some(decoder) = unsafe { decoder.as_mut() } {
        let _ = catch_unwind(AssertUnwindSafe(|| decoder.decoder.clear()));
    }
}

/// Decodes the frame received so far and writes its Y, Cb and Cr planes, 8
/// bits per sample in the range the stream uses. Blocks that did not arrive
/// decode as zeros, so call it when `emberwave_decoder_is_ready` says so.
/// Waits for the GPU.
///
/// # Safety
///
/// `decoder` is a live decoder that no other thread is using. Each plane's
/// `data` points to `stride * (height - 1) + width` bytes that may be
/// written, with the sizes of `emberwave_decoder_info`, and the planes do
/// not overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_decoder_decode_yuv(
    decoder: *mut EmberwaveDecoder,
    y: EmberwavePlane,
    cb: EmberwavePlane,
    cr: EmberwavePlane,
) -> EmberwaveStatus {
    run_or_lost(
        || {
            // SAFETY: as the caller promises.
            let decoder = unsafe { live_mut(decoder) }?;
            let targets = [
                (y, Component::Y, "y"),
                (cb, Component::Cb, "cb"),
                (cr, Component::Cr, "cr"),
            ];
            // Checked before decoding, so that a bad argument does not use up the
            // frame.
            let layout = decoder.decoder.layout();
            for (target, component, name) in &targets {
                let (width, height) = layout.plane_size(*component);
                check_target(target, name, width as usize, height as usize)?;
            }
            let frame = decoder.decoder.decode_assembled()?;
            for (plane, (target, _, _)) in frame.planes.iter().zip(&targets) {
                // SAFETY: checked above for the plane's size, and the caller
                // promises the memory.
                unsafe { copy_rows(&plane.data, plane.width as usize, target) };
            }
            Ok(())
        },
        // SAFETY: as the caller promises; the body's borrow has ended.
        || unsafe { lost(decoder) },
    )
}

/// Decodes the frame received so far, converts it to RGB on the GPU and writes
/// it, three bytes per pixel with red first. `color` chooses the conversion;
/// with `NULL` it is the one the frame's start of frame header signals
/// (BT.709, video range and centred chroma while that header has not
/// arrived). Upstream's Metal encoder signals BT.709, full range and centred
/// chroma in every frame, whatever the pictures are, so a caller that knows
/// better, such as from the header of a PWV1 file, passes that. Waits for the
/// GPU.
///
/// # Safety
///
/// `decoder` is a live decoder that no other thread is using, `color` is
/// `NULL` or points to an `EmberwaveColor`, and `rgb.data` points to
/// `stride * (height - 1) + 3 * width` bytes that may be written.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_decoder_decode_rgb(
    decoder: *mut EmberwaveDecoder,
    color: *const EmberwaveColor,
    rgb: EmberwavePlane,
) -> EmberwaveStatus {
    run_or_lost(
        || {
            // SAFETY: as the caller promises.
            let decoder = unsafe { live_mut(decoder) }?;
            // SAFETY: as the caller promises.
            let params = match unsafe { color.as_ref() } {
                Some(color) => color_params(color)?,
                None => decoder
                    .decoder
                    .start_of_frame()
                    .map(ColorParams::from_stream)
                    .unwrap_or_default(),
            };
            let layout = decoder.decoder.layout();
            let row_bytes = 3 * layout.width() as usize;
            check_target(&rgb, "rgb", row_bytes, layout.height() as usize)?;
            let frame = decoder.decoder.decode_assembled_rgb(&params)?;
            // SAFETY: checked above, and the caller promises the memory.
            unsafe { copy_rows(&frame.data, row_bytes, &rgb) };
            Ok(())
        },
        // SAFETY: as the caller promises; the body's borrow has ended.
        || unsafe { lost(decoder) },
    )
}

/// Creates an encoder on `context` for pictures of `width` x `height` pixels
/// in the chroma format `chroma`, an `EMBERWAVE_CHROMA_*` value. Width and
/// height are from 1 to 16384, and even for 4:2:0; other sizes give
/// `EMBERWAVE_STATUS_INVALID_ARGUMENT`. `signalling` is what the frames say
/// about their colours; with `NULL`, BT.709, video range, centred chroma and
/// the primaries and transfer function of sRGB. The GPU memory the encoder
/// needs is allocated here, but for the frame's, which grows to the largest
/// budget it is given.
///
/// # Safety
///
/// `context` is a live context, and `signalling` is `NULL` or points to an
/// `EmberwaveSignalling`. `encoder` points to where the new encoder is
/// stored; it is set to `NULL` when the call fails.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_encoder_create(
    context: *const EmberwaveContext,
    width: u32,
    height: u32,
    chroma: u32,
    signalling: *const EmberwaveSignalling,
    encoder: *mut *mut EmberwaveEncoder,
) -> EmberwaveStatus {
    run(|| {
        // SAFETY: as the caller promises.
        unsafe { output(encoder, "encoder", std::ptr::null_mut()) }?;
        // SAFETY: as the caller promises.
        let Some(context) = (unsafe { context.as_ref() }) else {
            return Err(Failure::argument("context is NULL"));
        };
        let chroma = chroma_subsampling(chroma)?;
        // SAFETY: as the caller promises.
        let signalling = match unsafe { signalling.as_ref() } {
            Some(value) => self::signalling(value)?,
            None => Signalling::default(),
        };
        let created = Encoder::new(&context.context, width, height, chroma, signalling)
            .map_err(creation_failure)?;
        let boxed = Box::new(EmberwaveEncoder { encoder: created });
        // SAFETY: checked by the first output() above.
        unsafe { encoder.write(Box::into_raw(boxed)) };
        Ok(())
    })
}

/// Destroys an encoder and frees its GPU memory.
///
/// # Safety
///
/// `encoder` is `NULL`, or an encoder from `emberwave_encoder_create` that
/// has not been destroyed and that no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_encoder_destroy(encoder: *mut EmberwaveEncoder) {
    if !encoder.is_null() {
        // SAFETY: it came from Box::into_raw, as the caller promises.
        let boxed = unsafe { Box::from_raw(encoder) };
        let _ = catch_unwind(AssertUnwindSafe(move || drop(boxed)));
    }
}

/// The sizes of the planes the encoder takes.
///
/// # Safety
///
/// `encoder` is a live encoder that no other thread is using, and `info`
/// points to an `EmberwaveFrameInfo` to fill in.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_encoder_info(
    encoder: *mut EmberwaveEncoder,
    info: *mut EmberwaveFrameInfo,
) -> EmberwaveStatus {
    run(|| {
        // SAFETY: as the caller promises.
        let layout = unsafe { live_encoder(encoder) }?.encoder.layout();
        let (chroma_width, chroma_height) = layout.plane_size(Component::Cb);
        let described = EmberwaveFrameInfo {
            width: layout.width(),
            height: layout.height(),
            chroma: chroma_code(layout.chroma()),
            chroma_width,
            chroma_height,
        };
        // SAFETY: as the caller promises.
        unsafe { output(info, "info", described) }
    })
}

/// Encodes one picture into a frame of at most `budget` bytes, writes it to
/// `frame` and stores its size in `*size`: a start of frame header and the
/// coded blocks, which `emberwave_frame_packets` splits into packets for a
/// decoder. The planes have the sizes of `emberwave_encoder_info`, each row
/// `stride` bytes from the next, and the bytes between the rows are not
/// read. A budget too small for even the smallest frame is
/// `EMBERWAVE_STATUS_INVALID_ARGUMENT` with the smallest frame's size in
/// `*size`; nothing is encoded then, and the next frame takes the sequence
/// number this one would have had. Waits for the GPU.
///
/// # Safety
///
/// `encoder` is a live encoder that no other thread is using. Each plane's
/// `data` points to `stride * (height - 1) + width` readable bytes. `frame`
/// points to `budget` bytes that may be written, and `size` to a `size_t`;
/// neither overlaps the planes or the other.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_encoder_encode(
    encoder: *mut EmberwaveEncoder,
    y: EmberwaveSourcePlane,
    cb: EmberwaveSourcePlane,
    cr: EmberwaveSourcePlane,
    frame: *mut u8,
    budget: usize,
    size: *mut usize,
) -> EmberwaveStatus {
    run_or_lost(
        || {
            // SAFETY: as the caller promises.
            unsafe { output(size, "size", 0) }?;
            // SAFETY: as the caller promises.
            let encoder = unsafe { live_encoder(encoder) }?;
            if frame.is_null() && budget > 0 {
                return Err(Failure::argument("frame is NULL"));
            }
            if budget > isize::MAX as usize {
                return Err(Failure::argument("budget is too large"));
            }
            let layout = encoder.encoder.layout();
            let sources = [
                (y, Component::Y, "y"),
                (cb, Component::Cb, "cb"),
                (cr, Component::Cr, "cr"),
            ];
            let mut planes: [&[u8]; 3] = [&[]; 3];
            let mut strides = [0; 3];
            for (i, (source, component, name)) in sources.iter().enumerate() {
                let (width, height) = layout.plane_size(*component);
                // SAFETY: as the caller promises.
                planes[i] = unsafe { source_plane(source, name, width as usize, height as usize) }?;
                strides[i] = source.stride;
            }
            match encoder.encoder.encode_strided(planes, strides, budget) {
                Ok(bytes) => {
                    // SAFETY: a frame is at most `budget` bytes, which the
                    // caller promises may be written.
                    unsafe {
                        std::ptr::copy_nonoverlapping(bytes.as_ptr(), frame, bytes.len());
                        size.write(bytes.len());
                    }
                    Ok(())
                }
                Err(GpuError::Encode(error @ EncodeError::BudgetTooSmall { minimum, .. })) => {
                    // SAFETY: as the caller promises.
                    unsafe { size.write(minimum) };
                    Err(GpuError::Encode(error).into())
                }
                Err(error) => Err(error.into()),
            }
        },
        // SAFETY: as the caller promises; the body's borrow has ended.
        || unsafe { encoder_lost(encoder) },
    )
}

/// Splits a frame into packets of at most `max_packet_bytes` bytes, at the
/// boundaries of its blocks, and stores how many in `*count`. With `packets`
/// not `NULL`, also where each starts in the frame and its size, in order;
/// with fewer than `*count` places there (`capacity`), nothing is written to
/// `packets` and the result is `EMBERWAVE_STATUS_INVALID_ARGUMENT`. The first
/// packet starts with the start of frame header. A block larger than
/// `max_packet_bytes` gets a packet of its own, as does the rest of a frame
/// from a malformed block on, so a packet can be larger. A decoder takes the
/// packets in any order.
///
/// # Safety
///
/// `frame` points to `size` readable bytes (or is `NULL` with `size` 0),
/// `packets` is `NULL` or points to `capacity` `EmberwavePacket`s that may be
/// written, and `count` points to a `size_t`; neither `packets` nor `count`
/// overlaps the frame.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_frame_packets(
    frame: *const u8,
    size: usize,
    max_packet_bytes: usize,
    packets: *mut EmberwavePacket,
    capacity: usize,
    count: *mut usize,
) -> EmberwaveStatus {
    run(|| {
        // SAFETY: as the caller promises.
        unsafe { output(count, "count", 0) }?;
        if max_packet_bytes == 0 {
            return Err(Failure::argument("max_packet_bytes is 0"));
        }
        // Worked out before anything is written, so that no borrow of the
        // frame is alive while the results are.
        let described: Vec<EmberwavePacket> = {
            // SAFETY: as the caller promises.
            let bytes = unsafe { input(frame, size) }?;
            encode::packets(bytes, max_packet_bytes)
                .iter()
                .map(|packet| EmberwavePacket {
                    offset: packet.as_ptr() as usize - bytes.as_ptr() as usize,
                    size: packet.len(),
                })
                .collect()
        };
        // SAFETY: as the caller promises.
        unsafe { count.write(described.len()) };
        if packets.is_null() {
            return Ok(());
        }
        if capacity < described.len() {
            return Err(Failure::argument(format!(
                "capacity is {capacity}, less than the {} packets of the frame",
                described.len()
            )));
        }
        for (i, packet) in described.into_iter().enumerate() {
            // SAFETY: i is below capacity, which the caller promises.
            unsafe { packets.add(i).write(packet) };
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn handles_may_cross_threads_as_the_header_says() {
        assert_send::<EmberwaveContext>();
        assert_sync::<EmberwaveContext>();
        assert_send::<EmberwaveDecoder>();
        assert_send::<EmberwaveEncoder>();
    }

    #[test]
    fn a_panic_on_a_lost_device_is_reported_as_the_loss() {
        let mut context = std::ptr::null_mut();
        // SAFETY: NULL options and a pointer to store the context.
        if unsafe { emberwave_context_create(std::ptr::null(), &mut context) }
            != EmberwaveStatus::Ok
        {
            assert!(
                std::env::var_os("EMBERWAVE_REQUIRE_GPU").is_none_or(|value| value.is_empty()),
                "no GPU"
            );
            return;
        }
        let mut decoder = std::ptr::null_mut();
        // SAFETY: a live context, and pointers to store results; the device
        // is destroyed as wgpu does when a GPU goes away.
        unsafe {
            let status =
                emberwave_decoder_create(context, 64, 64, EMBERWAVE_CHROMA_420, &mut decoder);
            assert_eq!(status, EmberwaveStatus::Ok);
            assert_eq!(lost(decoder), None);
            (*context).context.device().destroy();
        }

        // Decoding finds the device lost.
        let mut y = vec![0u8; 64 * 64];
        let mut cb = vec![0u8; 32 * 32];
        let mut cr = vec![0u8; 32 * 32];
        let plane = |bytes: &mut Vec<u8>, stride| EmberwavePlane {
            data: bytes.as_mut_ptr(),
            stride,
        };
        // SAFETY: a live decoder, and planes as large as their strides say.
        let status = unsafe {
            emberwave_decoder_decode_yuv(
                decoder,
                plane(&mut y, 64),
                plane(&mut cb, 32),
                plane(&mut cr, 32),
            )
        };
        assert_eq!(status, EmberwaveStatus::DeviceLost);
        // SAFETY: a live decoder.
        assert!(unsafe { lost(decoder) }.is_some());

        // So does making another decoder on the context, which panicked
        // (issue #35).
        let mut other = std::ptr::null_mut();
        // SAFETY: a live context, and a pointer to store the decoder.
        let status =
            unsafe { emberwave_decoder_create(context, 64, 64, EMBERWAVE_CHROMA_420, &mut other) };
        assert_eq!(status, EmberwaveStatus::DeviceLost);
        assert!(other.is_null());

        // What wgpu does when the device fails while it is waited for.
        let status = run_or_lost(
            || panic!("the device failed while it was waited for"),
            // SAFETY: a live decoder.
            || unsafe { lost(decoder) },
        );
        assert_eq!(status, EmberwaveStatus::DeviceLost);
        // SAFETY: a static C string.
        let message = unsafe { std::ffi::CStr::from_ptr(emberwave_last_error()) };
        assert!(message.to_string_lossy().contains("lost"), "{message:?}");

        // A panic with a working device stays an internal error.
        assert_eq!(
            run_or_lost(|| panic!("a bug"), || None),
            EmberwaveStatus::Internal
        );

        // SAFETY: both live, destroyed once.
        unsafe {
            emberwave_decoder_destroy(decoder);
            emberwave_context_destroy(context);
        }
    }

    #[test]
    fn an_encoder_reports_a_lost_device_as_the_loss() {
        let mut context = std::ptr::null_mut();
        // SAFETY: NULL options and a pointer to store the context.
        if unsafe { emberwave_context_create(std::ptr::null(), &mut context) }
            != EmberwaveStatus::Ok
        {
            assert!(
                std::env::var_os("EMBERWAVE_REQUIRE_GPU").is_none_or(|value| value.is_empty()),
                "no GPU"
            );
            return;
        }
        let mut encoder = std::ptr::null_mut();
        let create = |encoder: &mut *mut EmberwaveEncoder| {
            // SAFETY: a live context, NULL signalling and a pointer to store
            // the encoder.
            unsafe {
                emberwave_encoder_create(
                    context,
                    64,
                    32,
                    EMBERWAVE_CHROMA_420,
                    std::ptr::null(),
                    encoder,
                )
            }
        };
        assert_eq!(create(&mut encoder), EmberwaveStatus::Ok);
        // SAFETY: a live context; the device is destroyed as wgpu does when
        // a GPU goes away.
        unsafe { (*context).context.device().destroy() };

        let planes = [vec![128u8; 64 * 32], vec![128; 32 * 16], vec![128; 32 * 16]];
        let source = |plane: &Vec<u8>, stride| EmberwaveSourcePlane {
            data: plane.as_ptr(),
            stride,
        };
        let mut frame = vec![0u8; 10_000];
        let mut size = 0;
        // SAFETY: a live encoder, planes of its sizes and a frame of 10_000
        // bytes.
        let status = unsafe {
            emberwave_encoder_encode(
                encoder,
                source(&planes[0], 64),
                source(&planes[1], 32),
                source(&planes[2], 32),
                frame.as_mut_ptr(),
                frame.len(),
                &mut size,
            )
        };
        assert_eq!(status, EmberwaveStatus::DeviceLost);
        // SAFETY: a live encoder.
        assert!(unsafe { encoder_lost(encoder) }.is_some());
        // What wgpu does when the device fails while it is waited for: the
        // panic of an encode is the loss.
        let status = run_or_lost(
            || panic!("the device failed while it was waited for"),
            // SAFETY: a live encoder.
            || unsafe { encoder_lost(encoder) },
        );
        assert_eq!(status, EmberwaveStatus::DeviceLost);
        // So does making another encoder on the context.
        let mut other = std::ptr::null_mut();
        assert_eq!(create(&mut other), EmberwaveStatus::DeviceLost);
        assert!(other.is_null());

        // SAFETY: both live, destroyed once.
        unsafe {
            emberwave_encoder_destroy(encoder);
            emberwave_context_destroy(context);
        }
    }

    #[test]
    fn the_version_ends_in_a_nul() {
        let version = emberwave_version();
        // SAFETY: a static string with a NUL at its end.
        let text = unsafe { std::ffi::CStr::from_ptr(version) };
        assert_eq!(text.to_str(), Ok(env!("CARGO_PKG_VERSION")));
    }
}
