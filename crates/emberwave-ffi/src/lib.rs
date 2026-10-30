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
//!   time. A context keeps a few hundred bytes for every thread that has
//!   called it or its decoders and encoders, until they are all destroyed,
//!   as wgpu keeps its error scopes by thread: call from a pool of threads,
//!   not from new ones without end.
//! - A panic inside emberwave is caught and reported as
//!   `EMBERWAVE_STATUS_INTERNAL`. Destroy the object it happened in.

use std::cell::RefCell;
use std::ffi::{CString, c_char, c_void};
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
    /// A pointer was `NULL`, a value out of range, an output buffer too
    /// small, or an object that does not fit the call, such as a texture of
    /// another device, size or kind.
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

/// A texture of four unsigned normalised 8-bit channels, red first and alpha
/// 255: `MTLPixelFormatRGBA8Unorm` in Metal.
pub const EMBERWAVE_TEXTURE_RGBA8: u32 = 0;
/// A texture of four unsigned normalised 8-bit channels, blue first and alpha
/// 255: `MTLPixelFormatBGRA8Unorm` in Metal, the format of most display
/// surfaces, onto which a copy needs the same order.
pub const EMBERWAVE_TEXTURE_BGRA8: u32 = 1;

/// The luma plane, for `emberwave_texture_create_plane` and
/// `emberwave_texture_from_metal_plane`.
pub const EMBERWAVE_PLANE_Y: u32 = 0;
/// The blue-difference chroma plane, for `emberwave_texture_create_plane` and
/// `emberwave_texture_from_metal_plane`.
pub const EMBERWAVE_PLANE_CB: u32 = 1;
/// The red-difference chroma plane, for `emberwave_texture_create_plane` and
/// `emberwave_texture_from_metal_plane`.
pub const EMBERWAVE_PLANE_CR: u32 = 2;

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
    /// Whether the context runs on the caller's own Metal queue
    /// (`emberwave_context_create_metal`), where decodes need not wait.
    shared_queue: bool,
}

impl EmberwaveContext {
    /// A new context of `created`'s, on the caller's queue with
    /// `shared_queue`.
    fn boxed(created: GpuContext, shared_queue: bool) -> *mut Self {
        let info = created.adapter_info();
        let adapter = format!("{} ({:?})", info.name, info.backend).replace('\0', "");
        Box::into_raw(Box::new(Self {
            context: created,
            adapter: CString::new(adapter).unwrap_or_default(),
            shared_queue,
        }))
    }
}

/// Decodes the frames of one stream: one picture size and chroma format.
pub struct EmberwaveDecoder {
    decoder: Decoder,
    /// Whether the decoder's context runs on the caller's own queue, which
    /// the decoder keeps when the context is destroyed.
    shared_queue: bool,
}

/// A texture on a decoder's GPU device that the decodes into textures fill
/// with the picture, `emberwave_decoder_decode_texture` and
/// `emberwave_decoder_submit_texture`, or with one of its planes,
/// `emberwave_decoder_decode_plane_textures` and
/// `emberwave_decoder_submit_plane_textures`, for the caller's graphics API
/// to read on the GPU. emberwave makes it (`emberwave_texture_create`,
/// `emberwave_texture_create_plane`), or wraps a Metal texture of the
/// caller's (`emberwave_texture_from_metal`,
/// `emberwave_texture_from_metal_plane`).
///
/// How the decodes and the caller's reads of a texture are ordered:
/// - On a context of `emberwave_context_create`, emberwave works on a queue
///   of its own. A decode that waits returns once the GPU has written the
///   texture, so the caller's work committed afterwards, on any queue of the
///   device, reads the picture. Nothing orders the next decode after the
///   caller's work that may still read the texture: keep a texture for each
///   command buffer of the caller's that may still read one, and one to
///   decode into, three with a display's synchronisation and two frames in
///   flight, and decode only into one whose readers have completed.
/// - On a context of `emberwave_context_create_metal`, emberwave works on the
///   caller's queue, and for a texture whose hazards Metal tracks the queue
///   orders both ways, for the decodes that wait and those that do not: the
///   caller's command buffers committed to it after the call read the
///   picture, and a decode waits for those committed before it. One texture
///   takes every frame (decision D43).
/// - A texture whose hazards Metal does not track, as a caller's from a heap
///   usually is, is ordered by neither: only the decodes that wait take it,
///   and with textures in turn, as on a context of emberwave's own queue.
/// - The queue orders only command buffers that use the texture: one that
///   does not may run beside the decode and finish first. Whatever else reads
///   the memory, such as an `AVSampleBufferDisplayLayer` given the pixel
///   buffer of a texture, or the CPU, waits until the GPU has written it:
///   until a decode that waits returns, or until a command buffer that uses
///   the texture, committed to the queue after a decode that does not wait,
///   has completed.
pub struct EmberwaveTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// The context whose device it is on, which decodes into it must use.
    context_id: u64,
    /// The plane the texture is for, or `None` for the RGB picture. A plane
    /// of 4:4:4 has the size of the others, so only this tells them apart.
    plane: Option<Component>,
    /// Whether Metal tracks the texture's hazards, which the decodes that do
    /// not wait need (decision D43). Those emberwave makes are tracked, and
    /// a caller's texture from a heap usually is not.
    tracked: bool,
}

impl EmberwaveTexture {
    /// A new texture of `decoder`'s, for `plane` or for the RGB picture.
    fn boxed(
        decoder: &EmberwaveDecoder,
        texture: wgpu::Texture,
        plane: Option<Component>,
    ) -> *mut Self {
        Box::into_raw(Box::new(Self::wrap(decoder, texture, plane, true)))
    }

    /// `texture`, of `decoder`'s device, for `plane` or for the RGB picture.
    fn wrap(
        decoder: &EmberwaveDecoder,
        texture: wgpu::Texture,
        plane: Option<Component>,
        tracked: bool,
    ) -> Self {
        let view = texture.create_view(&Default::default());
        Self {
            texture,
            view,
            context_id: decoder.decoder.context_id(),
            plane,
            tracked,
        }
    }
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
            // emberwave_encoder_encode; a texture a decode cannot fill.
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
    // Not `with`: from a thread-local destructor that runs after this one's,
    // it would panic outside any `catch_unwind`, and abort the process. Then
    // the message is lost, and the status still returned.
    let _ = LAST_ERROR.try_with(|last| *last.borrow_mut() = message);
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

/// The conversion to RGB that `color` asks for, or with `NULL` the one the
/// frame's start of frame header signals (BT.709, video range and centred
/// chroma while that header has not arrived).
///
/// # Safety
///
/// `color` is `NULL` or points to an `EmberwaveColor`.
unsafe fn conversion(
    decoder: &EmberwaveDecoder,
    color: *const EmberwaveColor,
) -> Result<ColorParams, Failure> {
    // SAFETY: as the caller promises.
    match unsafe { color.as_ref() } {
        Some(color) => color_params(color),
        None => Ok(decoder
            .decoder
            .start_of_frame()
            .map(ColorParams::from_stream)
            .unwrap_or_default()),
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
        plane_stores: true,
        shared_uploads: true,
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

/// Copies `line` into row `row` of `target`, through a pointer rather than a
/// slice: the caller's memory may be uninitialised, which a `&mut [u8]` must
/// not refer to.
///
/// # Safety
///
/// `target` passed [`check_target`] for rows of `line.len()` bytes and more
/// than `row` rows, and the memory it describes may be written and does not
/// overlap `line`.
unsafe fn copy_row(line: &[u8], row: usize, target: &EmberwavePlane) {
    // SAFETY: the row lies within the span check_target accepted, which the
    // caller promises may be written.
    unsafe {
        std::ptr::copy_nonoverlapping(
            line.as_ptr(),
            target.data.add(row * target.stride),
            line.len(),
        );
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
/// same thread, and no longer than the thread: copy it to keep it, or to
/// hand it to another thread. Empty, too, while the thread is ending.
#[unsafe(no_mangle)]
pub extern "C" fn emberwave_last_error() -> *const c_char {
    // Not `with`, for the reason in `run_or_lost`.
    LAST_ERROR
        .try_with(|last| last.borrow().as_ptr())
        .unwrap_or(c"".as_ptr())
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
        // SAFETY: checked by the first output() above.
        unsafe { context.write(EmberwaveContext::boxed(created, false)) };
        Ok(())
    })
}

/// Creates a context on the caller's own Metal device and command queue:
/// `device` is an `id<MTLDevice>` and `queue` an `id<MTLCommandQueue>` of it,
/// which emberwave retains while the context, or anything made on it, lives.
/// emberwave's GPU work then goes onto the caller's queue, and
/// `emberwave_decoder_submit_texture` and
/// `emberwave_decoder_submit_plane_textures` need not wait for it: Metal runs
/// the command buffers the caller commits to the queue after such a call
/// after the decode, and a decode after the command buffers committed before
/// it, for the textures they share (decision D43). Only on iPhones, iPads and
/// Macs; elsewhere `EMBERWAVE_STATUS_NO_GPU` for arguments that are not
/// `NULL`.
///
/// The queue is shared:
/// - Commit the command buffers that read the textures after the call that
///   decoded into them has returned, from whichever thread.
/// - Use the textures in them directly, not through a heap or through an
///   argument buffer without `useResource`: Metal does not track those.
/// - A call makes all its command buffers before it commits them: on an
///   Apple M3 Max, 9 for a decode to a texture or to RGB, 7 to planes, 6 to
///   plane textures and 18 for an encode, as wgpu makes one for each pass
///   and transition. The decoders and encoders of every context made on
///   the queue take turns at this. A full queue makes a call wait until the
///   GPU frees one; a queue without room for 20 besides those the caller
///   holds uncommitted can wait for good. The default of 64 is plenty.
/// - Do not call emberwave inside a completion handler of a command buffer
///   of this queue. A call that waits for the GPU never returns there: the
///   wait ends in a handler of emberwave's that Metal runs after that one.
///   The others can wait there too, for another thread's call or for room
///   on the queue, and hold up the handlers behind them.
///
/// # Safety
///
/// `device` is `NULL` or an `id<MTLDevice>`, and `queue` is `NULL` or an
/// `id<MTLCommandQueue>`. `context` points to where the new context is
/// stored; it is set to `NULL` when the call fails.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_context_create_metal(
    device: *mut c_void,
    queue: *mut c_void,
    context: *mut *mut EmberwaveContext,
) -> EmberwaveStatus {
    run(|| {
        // SAFETY: as the caller promises.
        unsafe { output(context, "context", std::ptr::null_mut()) }?;
        if device.is_null() {
            return Err(Failure::argument("device is NULL"));
        }
        if queue.is_null() {
            return Err(Failure::argument("queue is NULL"));
        }
        // SAFETY: as the caller promises.
        let created = unsafe { metal_context(device, queue) }?;
        // SAFETY: checked by the first output() above.
        unsafe { context.write(EmberwaveContext::boxed(created, true)) };
        Ok(())
    })
}

/// A context on the `id<MTLDevice>` `device` and its `id<MTLCommandQueue>`
/// `queue`, through wgpu's Metal objects made from them.
///
/// # Safety
///
/// `device` and `queue` are Metal objects of those kinds.
#[cfg(any(target_os = "macos", target_os = "ios"))]
unsafe fn metal_context(device: *mut c_void, queue: *mut c_void) -> Result<GpuContext, Failure> {
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{MTLCommandQueue, MTLDevice};

    // SAFETY: Metal objects, as the caller promises, retained here for wgpu,
    // which keeps them as long as the device it makes of them.
    let device: Retained<ProtocolObject<dyn MTLDevice>> =
        unsafe { Retained::retain(device.cast()) }.ok_or(Failure::argument("device is NULL"))?;
    // SAFETY: as above.
    let queue: Retained<ProtocolObject<dyn MTLCommandQueue>> =
        unsafe { Retained::retain(queue.cast()) }.ok_or(Failure::argument("queue is NULL"))?;
    if !std::ptr::eq(&*queue.device(), &*device) {
        return Err(Failure::argument("queue is a queue of another device"));
    }
    let queue_address = Retained::as_ptr(&queue) as *const c_void as usize;

    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::METAL;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::METAL))
        .into_iter()
        .find(|adapter| {
            // SAFETY: only the adapter's device is looked at, to compare it.
            unsafe { adapter.as_hal::<wgpu::hal::api::Metal>() }
                .is_some_and(|hal| std::ptr::eq(hal.raw_device(), &*device))
        })
        .ok_or(Failure(
            EmberwaveStatus::NoGpu,
            "wgpu offers no Metal adapter on that device".into(),
        ))?;

    let descriptor = GpuContext::device_descriptor(&adapter, &GpuOptions::default());
    // SAFETY: the device is the adapter's, and the queue the device's, as
    // checked above. The timestamp period does not matter: such a context
    // does not ask for timestamps.
    let open: wgpu::hal::OpenDevice<wgpu::hal::api::Metal> = unsafe {
        wgpu::hal::OpenDevice {
            device: wgpu::hal::metal::Device::device_from_raw(
                device,
                descriptor.required_features,
                &descriptor.required_limits,
            ),
            queue: wgpu::hal::metal::Queue::queue_from_raw(queue, 1.0),
        }
    };
    // SAFETY: the device and the queue are the adapter's and have the
    // features and the limits of the descriptor, as above.
    let (device, queue) = unsafe { adapter.create_device_from_hal(open, &descriptor) }
        .map_err(|error| Failure(EmberwaveStatus::NoGpu, error.to_string()))?;
    // Contexts made on one queue share its room for command buffers: their
    // decoders and encoders take turns at it (#126, decision D60).
    Ok(
        GpuContext::from_device(adapter, device, queue)
            .share_record_lock(queue_lock(queue_address)),
    )
}

/// The lock of the contexts made on the Metal queue at `address`: one for
/// each queue, while a context, decoder or encoder made on it lives. Those
/// keep the queue, so that no other queue takes its address meanwhile.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn queue_lock(address: usize) -> std::sync::Arc<std::sync::Mutex<()>> {
    use std::sync::{Arc, Mutex, PoisonError, Weak};
    static LOCKS: Mutex<Vec<(usize, Weak<Mutex<()>>)>> = Mutex::new(Vec::new());
    let mut locks = LOCKS.lock().unwrap_or_else(PoisonError::into_inner);
    locks.retain(|(_, lock)| lock.strong_count() > 0);
    if let Some(lock) = locks
        .iter()
        .find(|&&(at, _)| at == address)
        .and_then(|(_, lock)| lock.upgrade())
    {
        return lock;
    }
    let lock = Arc::new(Mutex::new(()));
    locks.push((address, Arc::downgrade(&lock)));
    lock
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
unsafe fn metal_context(_device: *mut c_void, _queue: *mut c_void) -> Result<GpuContext, Failure> {
    Err(Failure(
        EmberwaveStatus::NoGpu,
        "Metal contexts are made on iPhones, iPads and Macs only".into(),
    ))
}

/// Destroys a context. Decoders and encoders created on it keep working, and
/// keep the GPU device until they are destroyed as well.
///
/// # Safety
///
/// `context` is `NULL`, or a context from `emberwave_context_create` or
/// `emberwave_context_create_metal` that has not been destroyed and that no
/// other thread is using.
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
        let boxed = Box::new(EmberwaveDecoder {
            decoder: created,
            shared_queue: context.shared_queue,
        });
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
/// `decoder` is a live decoder that no other thread is using, and `info`
/// points to an `EmberwaveFrameInfo`
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
/// such as a whole frame of a PWV1 file. Packets may come in any order, and a
/// block that arrived already is ignored. A malformed header or block is
/// rejected without changing anything, but blocks before it in the same data
/// are kept.
///
/// Frames are told apart by the bitstream's sequence number, which has three
/// bits: a packet of a frame one to three frames newer discards what was
/// received of the older one and starts its own; one of a frame four to seven
/// frames newer, or one to four frames older, is ignored; one five to seven
/// frames older passes for a newer frame, and one eight frames off for the
/// same frame. So a transport keeps frame numbers of its own: it drops
/// packets of frames older than the one being received, and before it gives
/// the first packet of a newer frame it decodes the current one if it is
/// ready, then calls `emberwave_decoder_clear`, not after, which would
/// discard that packet too.
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
/// or, with `allow_partial_frame`, enough of them for a usable picture: every
/// block of the coarsest level, and more than 90 percent of the blocks its
/// start of frame header announced, or the share
/// `emberwave_decoder_set_partial_share` set. Until that header arrives the count is of
/// all the picture's blocks, so a frame without it is ready only if its
/// encoder left out few blocks. emberwave's encoder sends every block of the
/// coarsest level for partial decodes; upstream's leaves out the empty ones,
/// so that a frame of it that lost any packet may never be ready in part.
/// False once the frame has been decoded, and for `NULL`. Any
/// `allow_partial_frame` but 0 allows a partial frame: an integer, not a
/// `bool`, so that no value a caller passes is an invalid one.
///
/// # Safety
///
/// `decoder` is `NULL` or a live decoder that no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_decoder_is_ready(
    decoder: *const EmberwaveDecoder,
    allow_partial_frame: u32,
) -> bool {
    // SAFETY: as the caller promises.
    match unsafe { decoder.as_ref() } {
        Some(decoder) => catch_unwind(AssertUnwindSafe(|| {
            decoder.decoder.is_ready(allow_partial_frame != 0)
        }))
        .unwrap_or(false),
        None => false,
    }
}

/// Sets the share of a frame's announced blocks that must have arrived, with
/// every block of the coarsest level, for `emberwave_decoder_is_ready` with
/// `allow_partial_frame` to accept a frame that lost packets: 0.9 by
/// default, as upstream's decoder has it. A lower share accepts frames that
/// lost more, with more of their detail blank, where a client would skip
/// them; 1 accepts whole frames only. The share stays until it is set again.
/// A share outside 0 to 1, NaN included, gives
/// `EMBERWAVE_STATUS_INVALID_ARGUMENT` and changes nothing.
///
/// # Safety
///
/// `decoder` is `NULL` or a live decoder that no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_decoder_set_partial_share(
    decoder: *mut EmberwaveDecoder,
    share: f32,
) -> EmberwaveStatus {
    run(|| {
        // SAFETY: as the caller promises.
        let decoder = unsafe { live_mut(decoder) }?;
        if !(0.0..=1.0).contains(&share) {
            return Err(Failure::argument(format!(
                "a share of {share} is not from 0 to 1"
            )));
        }
        decoder.decoder.set_partial_share(share);
        Ok(())
    })
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
/// Waits for the GPU. The rows go from the GPU's memory straight to the
/// planes, so a call that fails may have written some of them.
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
            decoder
                .decoder
                .decode_assembled_rows(|component, row, samples| {
                    let target = &targets[component.index()].0;
                    // SAFETY: the row is one of the plane's, as wide as the
                    // plane, so it lies within the span check_target accepted
                    // above, which the caller promises may be written.
                    unsafe { copy_row(samples, row as usize, target) };
                })?;
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
/// GPU. The rows go from the GPU's memory straight to `rgb`, so a call that
/// fails may have written some of them.
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
            let params = unsafe { conversion(decoder, color) }?;
            let layout = decoder.decoder.layout();
            let row_bytes = 3 * layout.width() as usize;
            check_target(&rgb, "rgb", row_bytes, layout.height() as usize)?;
            decoder
                .decoder
                .decode_assembled_rgb_rows(&params, |row, pixels| {
                    // SAFETY: the row is one of the picture's, of row_bytes
                    // bytes, so it lies within the span check_target accepted
                    // above, which the caller promises may be written.
                    unsafe { copy_row(pixels, row as usize, &rgb) };
                })?;
            Ok(())
        },
        // SAFETY: as the caller promises; the body's borrow has ended.
        || unsafe { lost(decoder) },
    )
}

/// Creates a texture that `emberwave_decoder_decode_texture` draws the
/// pictures of `decoder` into, on the decoder's GPU device: the pictures'
/// size, in `format`, `EMBERWAVE_TEXTURE_RGBA8` or `EMBERWAVE_TEXTURE_BGRA8`,
/// which the GPU can sample and copy from but the CPU cannot read (Metal's
/// private storage). `emberwave_texture_metal` gives its Metal texture.
/// Decoders of the same picture size on the same context may share textures.
///
/// # Safety
///
/// `decoder` is a live decoder that no other thread is using. `texture`
/// points to where the new texture is stored; it is set to `NULL` when the
/// call fails.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_texture_create(
    decoder: *const EmberwaveDecoder,
    format: u32,
    texture: *mut *mut EmberwaveTexture,
) -> EmberwaveStatus {
    run_or_lost(
        || {
            // SAFETY: as the caller promises.
            unsafe { output(texture, "texture", std::ptr::null_mut()) }?;
            // SAFETY: as the caller promises.
            let decoder = unsafe { live(decoder) }?;
            let format = match format {
                EMBERWAVE_TEXTURE_RGBA8 => wgpu::TextureFormat::Rgba8Unorm,
                EMBERWAVE_TEXTURE_BGRA8 => wgpu::TextureFormat::Bgra8Unorm,
                other => {
                    return Err(Failure::argument(format!(
                        "format is {other}, not EMBERWAVE_TEXTURE_RGBA8 or EMBERWAVE_TEXTURE_BGRA8"
                    )));
                }
            };
            let created = decoder.decoder.create_texture(format)?;
            // SAFETY: checked by the first output() above.
            unsafe { texture.write(EmberwaveTexture::boxed(decoder, created, None)) };
            Ok(())
        },
        // SAFETY: as the caller promises.
        || unsafe { lost(decoder) },
    )
}

/// Creates a texture that `emberwave_decoder_decode_plane_textures` puts one
/// plane of the pictures of `decoder` into, on the decoder's GPU device:
/// `plane` is `EMBERWAVE_PLANE_Y`, `EMBERWAVE_PLANE_CB` or
/// `EMBERWAVE_PLANE_CR`, and the texture is for that plane only, of its size
/// as `emberwave_decoder_info` gives it, in one unsigned normalised 8-bit
/// channel (`MTLPixelFormatR8Unorm` in Metal), which the GPU can sample and
/// copy from but the CPU cannot read. On Metal the decodes store the planes
/// into it from their shaders rather than copy them (decision D52), so its
/// usage includes `MTLTextureUsageShaderWrite`. `emberwave_texture_metal`
/// gives its Metal texture. Decoders of the same picture size and chroma
/// format on the same context may share textures.
///
/// # Safety
///
/// `decoder` is a live decoder that no other thread is using. `texture`
/// points to where the new texture is stored; it is set to `NULL` when the
/// call fails.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_texture_create_plane(
    decoder: *const EmberwaveDecoder,
    plane: u32,
    texture: *mut *mut EmberwaveTexture,
) -> EmberwaveStatus {
    run_or_lost(
        || {
            // SAFETY: as the caller promises.
            unsafe { output(texture, "texture", std::ptr::null_mut()) }?;
            // SAFETY: as the caller promises.
            let decoder = unsafe { live(decoder) }?;
            let component = plane_component(plane)?;
            let created = decoder.decoder.create_plane_texture(component)?;
            // SAFETY: checked by the first output() above.
            unsafe { texture.write(EmberwaveTexture::boxed(decoder, created, Some(component))) };
            Ok(())
        },
        // SAFETY: as the caller promises.
        || unsafe { lost(decoder) },
    )
}

/// Wraps the caller's own Metal texture, `metal`, an `id<MTLTexture>`, for
/// the decodes of `decoder` to put the picture into, as into one of
/// `emberwave_texture_create`'s: a texture of a `CVPixelBuffer` from a
/// `CVMetalTextureCache`, to hand to an `AVSampleBufferDisplayLayer`, one the
/// caller made, or a drawable (decision D44). Only on iPhones, iPads and
/// Macs; elsewhere `EMBERWAVE_STATUS_NO_GPU` for arguments that are not
/// `NULL`.
///
/// The texture is on the device of the decoder's context
/// (`emberwave_context_metal_device`), 2D, of one level, layer and sample,
/// not a view of another texture nor memoryless, of the picture's size, and
/// `MTLPixelFormatBGRA8Unorm` or `MTLPixelFormatRGBA8Unorm`. A decode copies
/// the picture into it when its width is a multiple of 64 pixels and it is
/// not framebuffer-only, as a texture of a `CVPixelBuffer` is not; otherwise
/// it draws it, which needs `MTLTextureUsageRenderTarget`. A texture that
/// neither can fill gives `EMBERWAVE_STATUS_INVALID_ARGUMENT`, and so does one
/// that is not right; `texture` is then `NULL`. On a context that does not
/// run on Metal, a Vulkan context on a Mac included, no texture is on the
/// context's Metal device.
///
/// emberwave keeps a reference to `metal` until the texture is destroyed.
/// Keep what the texture comes from, such as the `CVMetalTextureRef`, until
/// the GPU has written the picture, and hand it to other readers, such as an
/// `AVSampleBufferDisplayLayer`, only then: after a decode that waits has
/// returned, or once a command buffer that uses the texture, committed to
/// the context's queue after a decode that does not wait, has completed. A texture whose hazards
/// Metal does not track, as a heap's usually is not, takes only the decodes
/// that wait, with textures in turn (see `EmberwaveTexture`).
///
/// # Safety
///
/// `decoder` is a live decoder that no other thread is using, and `metal` is
/// `NULL` or an `id<MTLTexture>`. `texture` points to where the new texture is
/// stored.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_texture_from_metal(
    decoder: *const EmberwaveDecoder,
    metal: *mut c_void,
    texture: *mut *mut EmberwaveTexture,
) -> EmberwaveStatus {
    run_or_lost(
        || {
            // SAFETY: as the caller promises.
            unsafe { output(texture, "texture", std::ptr::null_mut()) }?;
            // SAFETY: as the caller promises.
            let decoder = unsafe { live(decoder) }?;
            // SAFETY: as the caller promises.
            let wrapped = unsafe { metal_import(decoder, metal, None) }?;
            // SAFETY: checked by the first output() above.
            unsafe { texture.write(Box::into_raw(Box::new(wrapped))) };
            Ok(())
        },
        // SAFETY: as the caller promises.
        || unsafe { lost(decoder) },
    )
}

/// Wraps the caller's own Metal texture, `metal`, for the Y, Cb or Cr plane
/// of the pictures of `decoder`, as `emberwave_texture_create_plane` makes
/// one: `plane` is `EMBERWAVE_PLANE_Y`, `EMBERWAVE_PLANE_CB` or
/// `EMBERWAVE_PLANE_CR`. As for `emberwave_texture_from_metal`, but the
/// texture is `MTLPixelFormatR8Unorm` of the plane's size, and as a decode
/// copies a plane, not framebuffer-only. When all three textures of a decode
/// have `MTLTextureUsageShaderWrite`, the decode stores the planes into them
/// from its shaders instead, which takes less time (decision D52). A Metal
/// texture can be wrapped for more than one plane, but a decode refuses it in
/// two of them.
///
/// # Safety
///
/// As for `emberwave_texture_from_metal`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_texture_from_metal_plane(
    decoder: *const EmberwaveDecoder,
    plane: u32,
    metal: *mut c_void,
    texture: *mut *mut EmberwaveTexture,
) -> EmberwaveStatus {
    run_or_lost(
        || {
            // SAFETY: as the caller promises.
            unsafe { output(texture, "texture", std::ptr::null_mut()) }?;
            // SAFETY: as the caller promises.
            let decoder = unsafe { live(decoder) }?;
            let component = plane_component(plane)?;
            // SAFETY: as the caller promises.
            let wrapped = unsafe { metal_import(decoder, metal, Some(component)) }?;
            // SAFETY: checked by the first output() above.
            unsafe { texture.write(Box::into_raw(Box::new(wrapped))) };
            Ok(())
        },
        // SAFETY: as the caller promises.
        || unsafe { lost(decoder) },
    )
}

/// The plane an `EMBERWAVE_PLANE_*` value names.
fn plane_component(plane: u32) -> Result<Component, Failure> {
    match plane {
        EMBERWAVE_PLANE_Y => Ok(Component::Y),
        EMBERWAVE_PLANE_CB => Ok(Component::Cb),
        EMBERWAVE_PLANE_CR => Ok(Component::Cr),
        other => Err(Failure::argument(format!(
            "plane is {other}, not EMBERWAVE_PLANE_Y, EMBERWAVE_PLANE_CB or EMBERWAVE_PLANE_CR"
        ))),
    }
}

/// The caller's `id<MTLTexture>` `metal` as a texture of `decoder`'s, for
/// `plane` or for the picture, once it has passed the checks of
/// `emberwave_texture_from_metal`.
///
/// # Safety
///
/// `metal` is `NULL` or an `id<MTLTexture>`.
#[cfg(any(target_os = "macos", target_os = "ios"))]
unsafe fn metal_import(
    decoder: &EmberwaveDecoder,
    metal: *mut c_void,
    plane: Option<Component>,
) -> Result<EmberwaveTexture, Failure> {
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{
        MTLHazardTrackingMode, MTLPixelFormat, MTLResource, MTLStorageMode, MTLTexture,
        MTLTextureType, MTLTextureUsage,
    };

    // SAFETY: an id<MTLTexture>, as the caller promises, retained here for
    // wgpu's texture, which keeps it until it is dropped.
    let metal: Retained<ProtocolObject<dyn MTLTexture>> =
        unsafe { Retained::retain(metal.cast()) }.ok_or(Failure::argument("metal is NULL"))?;
    let device = decoder.decoder.device();
    if !std::ptr::eq(
        Retained::as_ptr(&metal.device()).cast::<c_void>(),
        metal_device(device).cast_const(),
    ) {
        return Err(Failure::argument(
            "the texture is not on the Metal device of the decoder's context",
        ));
    }

    let layout = decoder.decoder.layout();
    let (width, height) = match plane {
        Some(component) => layout.plane_size(component),
        None => (layout.width(), layout.height()),
    };
    let format = metal.pixelFormat();
    let format = match plane {
        Some(_) if format == MTLPixelFormat::R8Unorm => emberwave_core::gpu::PLANE_FORMAT,
        None if format == MTLPixelFormat::RGBA8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        None if format == MTLPixelFormat::BGRA8Unorm => wgpu::TextureFormat::Bgra8Unorm,
        Some(_) => {
            return Err(Failure::argument(format!(
                "the texture's pixel format is {}, not MTLPixelFormatR8Unorm",
                format.0
            )));
        }
        None => {
            return Err(Failure::argument(format!(
                "the texture's pixel format is {}, not MTLPixelFormatBGRA8Unorm or \
                 MTLPixelFormatRGBA8Unorm",
                format.0
            )));
        }
    };
    let whose = if plane.is_some() {
        "plane's"
    } else {
        "picture's"
    };
    let problem = if metal.textureType() != MTLTextureType::Type2D {
        Some("is not 2D".to_owned())
    } else if (
        metal.mipmapLevelCount(),
        metal.arrayLength(),
        metal.sampleCount(),
    ) != (1, 1, 1)
    {
        Some("has more than one level, layer or sample".to_owned())
    } else if (metal.width(), metal.height()) != (width as usize, height as usize) {
        Some(format!(
            "is {}x{} rather than the {whose} {width}x{height}",
            metal.width(),
            metal.height()
        ))
    } else if metal.parentTexture().is_some() {
        Some("is a view of another texture".to_owned())
    } else if metal.storageMode() == MTLStorageMode::Memoryless {
        Some("is memoryless".to_owned())
    } else {
        None
    };
    if let Some(problem) = problem {
        return Err(Failure::argument(format!("the texture {problem}")));
    }

    // What wgpu may do with it: copies unless it is framebuffer-only, as a
    // drawable is by default, draws with RenderTarget, sampling with
    // ShaderRead, and for a plane, where the decoder stores planes, stores
    // with ShaderWrite (decision D52).
    let copies = !metal.isFramebufferOnly();
    let mut usage = wgpu::TextureUsages::empty();
    if copies {
        usage |= wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC;
    }
    if metal.usage().contains(MTLTextureUsage::RenderTarget) {
        usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }
    if metal.usage().contains(MTLTextureUsage::ShaderRead) {
        usage |= wgpu::TextureUsages::TEXTURE_BINDING;
    }
    if plane.is_some()
        && decoder.decoder.stores_planes()
        && metal.usage().contains(MTLTextureUsage::ShaderWrite)
    {
        usage |= wgpu::TextureUsages::STORAGE_BINDING;
    }
    let fillable = match plane {
        Some(_) => copies,
        None => {
            (copies && (width * 4).is_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT))
                || usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
        }
    };
    if !fillable {
        return Err(Failure::argument(
            "neither a copy nor a draw can fill the texture: a copy needs a texture that is \
             not framebuffer-only (and for the picture a width of a multiple of 64 pixels), \
             a draw of the picture MTLTextureUsageRenderTarget",
        ));
    }
    let tracked = metal.hazardTrackingMode() == MTLHazardTrackingMode::Tracked;

    // SAFETY: the texture is on the device, 2D, of one level and layer, in
    // the format and of the size given, as checked above.
    let hal = unsafe {
        wgpu::hal::metal::Device::texture_from_raw(
            metal,
            format,
            MTLTextureType::Type2D,
            1,
            1,
            wgpu::hal::CopyExtent {
                width,
                height,
                depth: 1,
            },
            None,
        )
    };
    // SAFETY: as above, and the usage is what the texture allows; whatever it
    // holds, a decode writes all of it.
    let texture = unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Metal>(
            hal,
            &wgpu::TextureDescriptor {
                label: Some("caller's texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            },
            wgpu::TextureUses::UNINITIALIZED,
        )
    };
    Ok(EmberwaveTexture::wrap(decoder, texture, plane, tracked))
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
unsafe fn metal_import(
    _decoder: &EmberwaveDecoder,
    _metal: *mut c_void,
    _plane: Option<Component>,
) -> Result<EmberwaveTexture, Failure> {
    Err(Failure(
        EmberwaveStatus::NoGpu,
        "Metal textures exist on iPhones, iPads and Macs only".into(),
    ))
}

/// Destroys a texture.
///
/// # Safety
///
/// `texture` is `NULL`, or a texture from `emberwave_texture_create`,
/// `emberwave_texture_create_plane`, `emberwave_texture_from_metal` or
/// `emberwave_texture_from_metal_plane` that has not been destroyed and that
/// no other thread is using. A Metal texture of one emberwave made goes with
/// it unless the caller retained it; of one that wraps a texture of the
/// caller's, only emberwave's reference goes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_texture_destroy(texture: *mut EmberwaveTexture) {
    if !texture.is_null() {
        // SAFETY: it came from Box::into_raw, as the caller promises.
        let boxed = unsafe { Box::from_raw(texture) };
        let _ = catch_unwind(AssertUnwindSafe(move || drop(boxed)));
    }
}

/// Decodes the frame received so far, converts it to RGB and draws it into
/// `texture` on the GPU, then waits for the GPU: when this returns, the
/// texture holds the picture, for the caller's graphics API to read on any
/// queue of the device. The picture never goes through CPU memory. Blocks
/// that did not arrive decode as zeros.
///
/// `color` chooses the conversion as for `emberwave_decoder_decode_rgb`, and
/// with `NULL` follows the frame. Upstream's Metal encoder signals BT.709,
/// full range and centred chroma whatever the pictures are, so a caller that
/// knows better passes that, as the converted picture cannot be taken back.
///
/// The call does not wait for the caller's GPU work that still reads the
/// texture. Whether something orders the decode after that work, and so how
/// many textures the caller keeps in turn, depends on the context and the
/// texture: see `EmberwaveTexture`. In short, on a context of
/// `emberwave_context_create` keep three, and on one of
/// `emberwave_context_create_metal` one tracked texture does.
///
/// `texture` is a texture of the picture from `emberwave_texture_create` or
/// `emberwave_texture_from_metal`, on a decoder of the same context and
/// picture size. `NULL`, a texture of a plane, of another context's device or
/// of another picture size gives `EMBERWAVE_STATUS_INVALID_ARGUMENT` and
/// leaves the frame to decode.
///
/// # Safety
///
/// `decoder` is a live decoder that no other thread is using, `texture` is a
/// live texture, and `color` is `NULL` or points to an `EmberwaveColor`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_decoder_decode_texture(
    decoder: *mut EmberwaveDecoder,
    color: *const EmberwaveColor,
    texture: *const EmberwaveTexture,
) -> EmberwaveStatus {
    run_or_lost(
        || {
            // SAFETY: as the caller promises.
            let decoder = unsafe { live_mut(decoder) }?;
            // SAFETY: as the caller promises.
            let texture = unsafe { picture_texture(texture, decoder.decoder.context_id()) }?;
            // SAFETY: as the caller promises.
            let params = unsafe { conversion(decoder, color) }?;
            decoder
                .decoder
                .decode_assembled_to_texture(&params, &texture.view)?;
            Ok(())
        },
        // SAFETY: as the caller promises; the body's borrow has ended.
        || unsafe { lost(decoder) },
    )
}

/// `texture`, unless it is `NULL`, on the device of another context than
/// `context_id`'s, or for a plane.
///
/// # Safety
///
/// `texture` is `NULL` or a live texture, which outlives the borrow.
unsafe fn picture_texture<'a>(
    texture: *const EmberwaveTexture,
    context_id: u64,
) -> Result<&'a EmberwaveTexture, Failure> {
    // SAFETY: as the caller promises.
    let Some(texture) = (unsafe { texture.as_ref() }) else {
        return Err(Failure::argument("texture is NULL"));
    };
    // Not wgpu's device equality, which devices of two contexts can pass
    // (GpuContext::id).
    if texture.context_id != context_id {
        return Err(Failure::argument(
            "the texture is on the GPU device of another context",
        ));
    }
    if let Some(plane) = texture.plane {
        return Err(Failure::argument(format!(
            "the texture is for the {plane:?} plane, which \
             emberwave_decoder_decode_plane_textures fills"
        )));
    }
    Ok(texture)
}

/// An error unless Metal tracks the hazards of `texture`, the argument
/// `name`, which a decode that does not wait needs (decision D43).
fn require_tracked(texture: &EmberwaveTexture, name: &str) -> Result<(), Failure> {
    if texture.tracked {
        Ok(())
    } else {
        Err(Failure::argument(format!(
            "Metal does not track the hazards of {name}, as it does not for a heap's \
             textures, so the queue cannot order the decode: use the decode that waits, \
             with textures in turn, or a texture made with MTLHazardTrackingModeTracked"
        )))
    }
}

/// Decodes the frame received so far into `texture`, as
/// `emberwave_decoder_decode_texture` does, but returns once the work is on
/// the queue, without waiting for it. Only for a decoder whose context was
/// made by `emberwave_context_create_metal`, on the caller's own queue, and a
/// texture whose hazards Metal tracks: the queue orders the work then. The
/// caller's command buffers committed to it after this call returns see the
/// picture, and the decode waits for those committed before it that still
/// read the texture, so one texture takes every frame (see
/// `EmberwaveTexture`). A decoder of a context of `emberwave_context_create`,
/// or a texture Metal does not track, gives
/// `EMBERWAVE_STATUS_INVALID_ARGUMENT`, as nothing there would order the
/// caller's reads after the decode.
///
/// The queue orders only command buffers that use the texture. Hand the
/// picture to anything else that reads its memory, such as an
/// `AVSampleBufferDisplayLayer` given the pixel buffer of the texture, or the
/// CPU, only once a command buffer that uses the texture, committed to the
/// queue after this call, has completed.
///
/// The frame goes to the GPU through memory wgpu allocates for it, which a
/// waiting decode does without. A failure that the GPU reports later, such
/// as a lost device, is that of a later call.
///
/// # Safety
///
/// As for `emberwave_decoder_decode_texture`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_decoder_submit_texture(
    decoder: *mut EmberwaveDecoder,
    color: *const EmberwaveColor,
    texture: *const EmberwaveTexture,
) -> EmberwaveStatus {
    run_or_lost(
        || {
            // SAFETY: as the caller promises.
            let decoder = unsafe { live_mut(decoder) }?;
            require_shared_queue(decoder)?;
            // SAFETY: as the caller promises.
            let texture = unsafe { picture_texture(texture, decoder.decoder.context_id()) }?;
            require_tracked(texture, "texture")?;
            // SAFETY: as the caller promises.
            let params = unsafe { conversion(decoder, color) }?;
            decoder
                .decoder
                .submit_assembled_to_texture(&params, &texture.view)?;
            Ok(())
        },
        // SAFETY: as the caller promises; the body's borrow has ended.
        || unsafe { lost(decoder) },
    )
}

/// Decodes the frame received so far into the plane textures `y`, `cb` and
/// `cr`, as `emberwave_decoder_decode_plane_textures` does, but returns once
/// the work is on the queue, as `emberwave_decoder_submit_texture` does: only
/// for a decoder whose context was made by `emberwave_context_create_metal`,
/// and textures whose hazards Metal tracks.
///
/// # Safety
///
/// As for `emberwave_decoder_decode_plane_textures`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_decoder_submit_plane_textures(
    decoder: *mut EmberwaveDecoder,
    y: *const EmberwaveTexture,
    cb: *const EmberwaveTexture,
    cr: *const EmberwaveTexture,
) -> EmberwaveStatus {
    run_or_lost(
        || {
            // SAFETY: as the caller promises.
            let decoder = unsafe { live_mut(decoder) }?;
            require_shared_queue(decoder)?;
            let context_id = decoder.decoder.context_id();
            // SAFETY: as the caller promises.
            let textures = unsafe {
                [
                    plane_texture(y, Component::Y, "y", context_id)?,
                    plane_texture(cb, Component::Cb, "cb", context_id)?,
                    plane_texture(cr, Component::Cr, "cr", context_id)?,
                ]
            };
            distinct_metal(&textures)?;
            for (texture, name) in textures.iter().zip(["y", "cb", "cr"]) {
                require_tracked(texture, name)?;
            }
            let targets = textures.map(|texture| &texture.view);
            decoder
                .decoder
                .submit_assembled_to_plane_textures(targets)?;
            Ok(())
        },
        // SAFETY: as the caller promises; the body's borrow has ended.
        || unsafe { lost(decoder) },
    )
}

/// An error unless `decoder`'s context runs on the caller's own queue, which
/// the decodes that do not wait need.
fn require_shared_queue(decoder: &EmberwaveDecoder) -> Result<(), Failure> {
    if decoder.shared_queue {
        Ok(())
    } else {
        Err(Failure::argument(
            "the decoder's context is not on the caller's queue \
             (emberwave_context_create_metal), so nothing would order the \
             caller's reads after the decode: use the decode that waits",
        ))
    }
}

/// Decodes the frame received so far into its Y, Cb and Cr planes in `y`,
/// `cb` and `cr` on the GPU, then waits for the GPU: when this returns,
/// the textures hold the planes, for the caller's graphics API to read on any
/// queue of the device. They hold the samples of the stream, as
/// `emberwave_decoder_decode_yuv` writes them, which the caller converts to
/// colours itself, with the matrix, range and chroma siting the stream
/// signals (`emberwave_probe_packet`) or that it knows better. In 4:2:0 a
/// chroma sample sits vertically halfway between two rows of luma, and
/// horizontally halfway between two luma samples with
/// `EMBERWAVE_SITING_CENTER`, on the left one with `EMBERWAVE_SITING_LEFT`.
/// The planes never go through CPU memory: on Metal the decode stores them
/// into the textures from its shaders when all three allow it, and copies
/// them in otherwise (decision D52). Blocks that did not arrive decode as
/// zeros.
///
/// Each texture is one for that plane, from `emberwave_texture_create_plane`
/// or `emberwave_texture_from_metal_plane`, on a decoder of the same context,
/// picture size and chroma format, and no Metal texture serves two planes.
/// Another, a texture of the RGB picture included, gives
/// `EMBERWAVE_STATUS_INVALID_ARGUMENT` and leaves the frame to decode. As
/// with `emberwave_decoder_decode_texture`, the call does not wait for the
/// caller's GPU work that still reads the textures, and how many sets of
/// three to keep in turn depends on the context and the textures: see
/// `EmberwaveTexture`.
///
/// # Safety
///
/// `decoder` is a live decoder that no other thread is using, and `y`, `cb`
/// and `cr` are live textures.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_decoder_decode_plane_textures(
    decoder: *mut EmberwaveDecoder,
    y: *const EmberwaveTexture,
    cb: *const EmberwaveTexture,
    cr: *const EmberwaveTexture,
) -> EmberwaveStatus {
    run_or_lost(
        || {
            // SAFETY: as the caller promises.
            let decoder = unsafe { live_mut(decoder) }?;
            let context_id = decoder.decoder.context_id();
            // SAFETY: as the caller promises.
            let textures = unsafe {
                [
                    plane_texture(y, Component::Y, "y", context_id)?,
                    plane_texture(cb, Component::Cb, "cb", context_id)?,
                    plane_texture(cr, Component::Cr, "cr", context_id)?,
                ]
            };
            distinct_metal(&textures)?;
            let targets = textures.map(|texture| &texture.view);
            decoder
                .decoder
                .decode_assembled_to_plane_textures(targets)?;
            Ok(())
        },
        // SAFETY: as the caller promises; the body's borrow has ended.
        || unsafe { lost(decoder) },
    )
}

/// An error if two of the plane textures `textures` wrap one Metal texture,
/// which `emberwave_texture_from_metal_plane` can make for different planes:
/// the decode would write all of them into it.
fn distinct_metal(textures: &[&EmberwaveTexture; 3]) -> Result<(), Failure> {
    let [y, cb, cr] = textures.map(|texture| metal_texture(&texture.texture));
    let same = |a: *mut c_void, b: *mut c_void| !a.is_null() && a == b;
    if same(y, cb) || same(y, cr) || same(cb, cr) {
        return Err(Failure::argument(
            "two of the planes are the same Metal texture",
        ));
    }
    Ok(())
}

/// `texture`, the argument `name`, unless it is `NULL`, on the device of
/// another context than `context_id`'s, or not for `plane`.
///
/// # Safety
///
/// `texture` is `NULL` or a live texture, which outlives the borrow.
unsafe fn plane_texture<'a>(
    texture: *const EmberwaveTexture,
    plane: Component,
    name: &str,
    context_id: u64,
) -> Result<&'a EmberwaveTexture, Failure> {
    // SAFETY: as the caller promises.
    let Some(texture) = (unsafe { texture.as_ref() }) else {
        return Err(Failure::argument(format!("{name} is NULL")));
    };
    // Not wgpu's device equality, which devices of two contexts can pass
    // (GpuContext::id).
    if texture.context_id != context_id {
        return Err(Failure::argument(format!(
            "{name} is on the GPU device of another context"
        )));
    }
    if texture.plane != Some(plane) {
        return Err(Failure::argument(format!(
            "{name} is not a texture for the {plane:?} plane"
        )));
    }
    Ok(texture)
}

/// The Metal texture of `texture`, an `id<MTLTexture>` in the format the
/// texture was made with, or `NULL` when its context does not run on Metal,
/// a Vulkan context on a Mac included, or `texture` is `NULL`.
///
/// The pointer is borrowed: `texture` keeps it, and destroying `texture`
/// releases it. Take it with `(__bridge id<MTLTexture>)` in Objective-C, or
/// `Unmanaged<MTLTexture>.fromOpaque(pointer).takeUnretainedValue()` in Swift,
/// never with a transfer of ownership, which would release emberwave's
/// reference. A command buffer that has encoded it keeps it until it
/// completes. Read it only once a decode into it has returned, and after
/// one that does not wait only in command buffers committed afterwards to
/// the context's queue (see `EmberwaveTexture`), and only on the
/// device of `emberwave_context_metal_device`.
///
/// # Safety
///
/// `texture` is `NULL` or a live texture.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_texture_metal(texture: *const EmberwaveTexture) -> *mut c_void {
    // SAFETY: as the caller promises.
    match unsafe { texture.as_ref() } {
        Some(texture) => metal_texture(&texture.texture),
        None => std::ptr::null_mut(),
    }
}

/// The Metal device of `context`, an `id<MTLDevice>`, or `NULL` when the
/// context does not run on Metal or `context` is `NULL`. The textures of its
/// decoders are on this device, so the caller's Metal work that reads them
/// belongs on it too: make the caller's queue and resources on it, or check
/// that the caller's device is this one.
///
/// The pointer is borrowed, as that of `emberwave_texture_metal`, and stays
/// valid while the context or any decoder, encoder or texture made on it is
/// alive.
///
/// # Safety
///
/// `context` is `NULL` or a live context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_context_metal_device(
    context: *const EmberwaveContext,
) -> *mut c_void {
    // SAFETY: as the caller promises.
    match unsafe { context.as_ref() } {
        Some(context) => metal_device(context.context.device()),
        None => std::ptr::null_mut(),
    }
}

/// The `id<MTLTexture>` behind `texture`, or `NULL` off Metal.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn metal_texture(texture: &wgpu::Texture) -> *mut c_void {
    // SAFETY: only the handle is taken out, for the caller's graphics API,
    // which reads the texture while no work of emberwave's writes it, as
    // emberwave_decoder_decode_texture asks.
    let hal = unsafe { texture.as_hal::<wgpu::hal::api::Metal>() };
    hal.map_or(std::ptr::null_mut(), |hal| {
        std::ptr::from_ref(hal.raw_handle()).cast_mut().cast()
    })
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn metal_texture(_texture: &wgpu::Texture) -> *mut c_void {
    std::ptr::null_mut()
}

/// The `id<MTLDevice>` behind `device`, or `NULL` off Metal.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn metal_device(device: &wgpu::Device) -> *mut c_void {
    // SAFETY: only the handle is taken out, for the caller's graphics API to
    // make its own objects on.
    let hal = unsafe { device.as_hal::<wgpu::hal::api::Metal>() };
    hal.map_or(std::ptr::null_mut(), |hal| {
        std::ptr::from_ref(&**hal.raw_device()).cast_mut().cast()
    })
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn metal_device(_device: &wgpu::Device) -> *mut c_void {
    std::ptr::null_mut()
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

    #[cfg(target_os = "macos")]
    #[test]
    fn a_plane_texture_the_app_may_write_takes_stores() {
        // tests/metal.rs sees the same planes either way: this pins that the
        // app's plane textures with MTLTextureUsageShaderWrite take the stores
        // where the decoder stores (decision D52), and the others the copies.
        use objc2::runtime::ProtocolObject;
        use objc2_metal::{MTLDevice, MTLPixelFormat, MTLTextureDescriptor, MTLTextureUsage};

        let mut context = std::ptr::null_mut();
        // SAFETY: NULL options and a pointer to store the context in.
        if unsafe { emberwave_context_create(std::ptr::null(), &mut context) }
            != EmberwaveStatus::Ok
        {
            return;
        }
        // SAFETY: live objects, and pointers to store results in.
        unsafe {
            let device = emberwave_context_metal_device(context);
            if !device.is_null() {
                let device: &ProtocolObject<dyn MTLDevice> = &*device.cast();
                let mut decoder = std::ptr::null_mut();
                assert_eq!(
                    emberwave_decoder_create(context, 64, 32, EMBERWAVE_CHROMA_420, &mut decoder),
                    EmberwaveStatus::Ok
                );
                let stores = (*decoder).decoder.stores_planes();
                for (usage, storage) in [
                    (MTLTextureUsage::ShaderRead, false),
                    (
                        MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
                        stores,
                    ),
                ] {
                    let descriptor =
                        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                            MTLPixelFormat::R8Unorm,
                            64,
                            32,
                            false,
                        );
                    descriptor.setUsage(usage);
                    let metal = device
                        .newTextureWithDescriptor(&descriptor)
                        .expect("a texture");
                    let mut texture = std::ptr::null_mut();
                    assert_eq!(
                        emberwave_texture_from_metal_plane(
                            decoder,
                            EMBERWAVE_PLANE_Y,
                            objc2::rc::Retained::as_ptr(&metal).cast_mut().cast(),
                            &mut texture
                        ),
                        EmberwaveStatus::Ok
                    );
                    let wrapped = (*texture)
                        .texture
                        .usage()
                        .contains(wgpu::TextureUsages::STORAGE_BINDING);
                    assert_eq!(wrapped, storage, "{usage:?}");
                    emberwave_texture_destroy(texture);
                }
                emberwave_decoder_destroy(decoder);
            }
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
