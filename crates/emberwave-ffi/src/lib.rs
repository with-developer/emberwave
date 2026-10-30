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
use std::sync::Arc;

use emberwave_core::bitstream::{
    ChromaSiting, ChromaSubsampling, ColorPrimaries, Component, Error as BitstreamError,
    FrameLayout, Header, Pwv1Header, SequenceHeader, TransferFunction, YcbcrRange, YcbcrTransform,
};
use emberwave_core::color::{ColorMatrix, ColorParams};
use emberwave_core::encode::{self, EncodeError, Signalling};
use emberwave_core::gpu::{
    Decoder, Encoder, GpuContext, GpuError, GpuOptions, QueueTurns, TexturePicture,
};

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

/// The luma plane, for `emberwave_texture_create_plane`, the functions that
/// wrap a plane of the caller's for decodes
/// (`emberwave_texture_from_metal_plane`,
/// `emberwave_texture_from_vulkan_plane`), and those that wrap one to encode
/// (`emberwave_texture_from_metal_source`,
/// `emberwave_texture_from_vulkan_source`).
pub const EMBERWAVE_PLANE_Y: u32 = 0;
/// The blue-difference chroma plane, for the same functions as
/// `EMBERWAVE_PLANE_Y`.
pub const EMBERWAVE_PLANE_CB: u32 = 1;
/// The red-difference chroma plane, for the same functions as
/// `EMBERWAVE_PLANE_Y`.
pub const EMBERWAVE_PLANE_CR: u32 = 2;
/// The two chroma planes interleaved in one texture of two unsigned
/// normalised 8-bit channels, Cb in the first and Cr in the second, as the
/// second plane of NV12 holds them: `MTLPixelFormatRG8Unorm` in Metal,
/// `VK_FORMAT_R8G8_UNORM` in Vulkan. Only for the textures encoders read,
/// `emberwave_texture_from_metal_source` and
/// `emberwave_texture_from_vulkan_source` (decision D62).
pub const EMBERWAVE_PLANE_CBCR: u32 = 3;

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
    /// Whether the context runs on the caller's own Vulkan device
    /// (`emberwave_context_create_vulkan`).
    vulkan: bool,
}

impl EmberwaveContext {
    /// A new context of `created`'s, on the caller's Metal queue with
    /// `shared_queue`, on the caller's Vulkan device with `vulkan`.
    fn boxed(created: GpuContext, shared_queue: bool, vulkan: bool) -> *mut Self {
        let info = created.adapter_info();
        let adapter = format!("{} ({:?})", info.name, info.backend).replace('\0', "");
        Box::into_raw(Box::new(Self {
            context: created,
            adapter: CString::new(adapter).unwrap_or_default(),
            shared_queue,
            vulkan,
        }))
    }
}

/// Lets go of `object`, made on a context of `turns`, under the caller's lock
/// of the queue if there is one: the last of wgpu's handles to a queue waits
/// for it to be idle (decision D61). A panic is caught.
fn let_go<T>(object: Box<T>, turns: Option<Arc<QueueTurns>>) {
    let _ = catch_unwind(AssertUnwindSafe(move || match turns {
        Some(turns) => turns.caller_locked(move || drop(object)),
        None => drop(object),
    }));
}

/// Decodes the frames of one stream: one picture size and chroma format.
pub struct EmberwaveDecoder {
    decoder: Decoder,
    /// Whether the decoder's context runs on the caller's own Metal queue,
    /// which the decoder keeps when the context is destroyed.
    shared_queue: bool,
    /// Whether the decoder's context runs on the caller's own Vulkan device.
    vulkan: bool,
    /// The turns at the queue of the decoder's context, if it has any.
    turns: Option<Arc<QueueTurns>>,
}

/// A texture on a decoder's GPU device that the decodes into textures fill
/// with the picture, `emberwave_decoder_decode_texture` and
/// `emberwave_decoder_submit_texture`, or with one of its planes,
/// `emberwave_decoder_decode_plane_textures` and
/// `emberwave_decoder_submit_plane_textures`, for the caller's graphics API
/// to read on the GPU. emberwave makes it (`emberwave_texture_create`,
/// `emberwave_texture_create_plane`), or wraps a Metal texture or a Vulkan
/// image of the caller's (`emberwave_texture_from_metal`,
/// `emberwave_texture_from_metal_plane`, `emberwave_texture_from_vulkan`,
/// `emberwave_texture_from_vulkan_plane`).
///
/// Or a texture of the caller's that holds a plane of a picture to encode,
/// for `emberwave_encoder_encode_textures` to read
/// (`emberwave_texture_from_metal_source`,
/// `emberwave_texture_from_vulkan_source`, decision D62). A texture of a
/// plane serves the decoders and encoders of its context whose pictures
/// have planes of its size, so planes a decode put into textures can be
/// encoded again.
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
    /// The context whose device it is on, which decodes into it and encodes
    /// from it must use.
    context_id: u64,
    /// What the texture holds. A plane of 4:4:4 has the size of the others,
    /// so only this tells them apart.
    holds: Holds,
    /// Whether Metal tracks the texture's hazards, which the decodes that do
    /// not wait need (decision D43). Those emberwave makes are tracked, and
    /// a caller's texture from a heap usually is not.
    tracked: bool,
    /// Whether the texture is on the caller's own Vulkan device
    /// (`emberwave_context_create_vulkan`), whose image the caller may use.
    /// Read only where there are such devices.
    #[cfg_attr(
        not(any(
            target_os = "linux",
            target_os = "android",
            all(windows, target_pointer_width = "64")
        )),
        allow(dead_code)
    )]
    vulkan: bool,
}

impl EmberwaveTexture {
    /// A new texture of `decoder`'s, for `plane` or for the RGB picture.
    fn boxed(
        decoder: &EmberwaveDecoder,
        texture: wgpu::Texture,
        plane: Option<Component>,
    ) -> *mut Self {
        let wrapping = Wrapping::decoder(decoder, plane);
        Box::into_raw(Box::new(Self::wrap(&wrapping, texture, true)))
    }

    /// `texture`, on the device of `wrapping` and for what it holds.
    fn wrap(wrapping: &Wrapping<'_>, texture: wgpu::Texture, tracked: bool) -> Self {
        let view = texture.create_view(&Default::default());
        Self {
            texture,
            view,
            context_id: wrapping.context_id,
            holds: wrapping.holds,
            tracked,
            vulkan: wrapping.vulkan,
        }
    }
}

/// What an `EmberwaveTexture` holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Holds {
    /// The RGB picture, which decodes into a texture put in.
    Picture,
    /// The Y, Cb or Cr plane.
    Plane(Component),
    /// Cb and Cr interleaved, which encoders read (decision D62).
    Chroma,
}

impl Holds {
    /// The texture's size in pictures of `layout`.
    fn size(self, layout: &FrameLayout) -> (u32, u32) {
        match self {
            Self::Picture => (layout.width(), layout.height()),
            Self::Plane(component) => layout.plane_size(component),
            Self::Chroma => layout.plane_size(Component::Cb),
        }
    }

    /// The texture, with its owner, for messages: "the plane's" or so.
    /// Read where Metal textures are checked.
    #[cfg_attr(not(any(target_os = "macos", target_os = "ios")), allow(dead_code))]
    fn whose(self) -> &'static str {
        match self {
            Self::Picture => "picture's",
            Self::Plane(_) => "plane's",
            Self::Chroma => "chroma planes'",
        }
    }
}

/// What a texture of the caller's is wrapped for: a decoder that fills it,
/// or an encoder that reads it.
struct Wrapping<'a> {
    device: &'a wgpu::Device,
    context_id: u64,
    /// Whether the context runs on the caller's own Vulkan device.
    vulkan: bool,
    /// The block layout of the pictures, whose sizes the texture has.
    layout: &'a FrameLayout,
    holds: Holds,
    /// Whether an encoder reads the texture, rather than a decoder fill it.
    read: bool,
    /// Whether the decoder stores planes into textures (decision D52), as
    /// it does on Metal only. Read where Metal textures are wrapped.
    #[cfg_attr(not(any(target_os = "macos", target_os = "ios")), allow(dead_code))]
    stores_planes: bool,
}

impl<'a> Wrapping<'a> {
    /// For the decodes of `decoder`, into the picture or `plane`.
    fn decoder(decoder: &'a EmberwaveDecoder, plane: Option<Component>) -> Self {
        Self {
            device: decoder.decoder.device(),
            context_id: decoder.decoder.context_id(),
            vulkan: decoder.vulkan,
            layout: decoder.decoder.layout(),
            holds: plane.map_or(Holds::Picture, Holds::Plane),
            read: false,
            stores_planes: decoder.decoder.stores_planes(),
        }
    }

    /// For the encodes of `encoder`, from what `holds` says.
    fn encoder(encoder: &'a EmberwaveEncoder, holds: Holds) -> Self {
        Self {
            device: encoder.encoder.device(),
            context_id: encoder.encoder.context_id(),
            vulkan: encoder.vulkan,
            layout: encoder.encoder.layout(),
            holds,
            read: true,
            stores_planes: false,
        }
    }

    /// The texture's size.
    fn size(&self) -> (u32, u32) {
        self.holds.size(self.layout)
    }
}

/// Encodes the pictures of one stream on the GPU: one picture size and
/// chroma format.
pub struct EmberwaveEncoder {
    encoder: Encoder,
    /// Whether the encoder's context runs on the caller's own Vulkan device.
    vulkan: bool,
    /// The turns at the queue of the encoder's context, if it has any.
    turns: Option<Arc<QueueTurns>>,
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
        unsafe { context.write(EmberwaveContext::boxed(created, false, false)) };
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
///   Apple M3 Max, 9 for a decode to a texture or to RGB, 6 to planes or to
///   plane textures and 18 for an encode, as wgpu makes one for each pass
///   and transition. The decoders and encoders of every context made on
///   the queue take turns at this, and hold none between calls. A full
///   queue makes a call wait until one of its command buffers has completed
///   and that buffer's completed handlers have returned; a queue without
///   room for 20 besides those the caller holds uncommitted can wait for
///   good. The default of 64 is plenty.
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
        unsafe { context.write(EmberwaveContext::boxed(created, true, false)) };
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
    // One at a time: from the making of its device until its first turn, a
    // context holds a command buffer of the queue outside any turn, and
    // several made at once beside a call that waits for room could each hold
    // one while waiting for that call's turn (decision D60).
    static MAKING: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _making = MAKING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

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
    let turns = queue_turns(queue_address, None, QueueTurns::new)?;
    Ok(GpuContext::from_device(adapter, device, queue).with_queue_turns(turns))
}

/// The turns of the contexts made on the queue at `address`, a Metal queue
/// or a `VkQueue`: the same for each, while a context, decoder or encoder
/// made on it lives, and made by `make` for the first. Those keep the
/// queue, so that no other queue takes its address meanwhile.
#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "linux",
    target_os = "android",
    all(windows, target_pointer_width = "64")
))]
fn queue_turns(
    address: usize,
    lock: LockKey,
    make: impl FnOnce() -> QueueTurns,
) -> Result<Arc<QueueTurns>, Failure> {
    use std::sync::{Mutex, PoisonError, Weak};
    static TURNS: Mutex<Vec<(usize, LockKey, Weak<QueueTurns>)>> = Mutex::new(Vec::new());
    let mut all = TURNS.lock().unwrap_or_else(PoisonError::into_inner);
    all.retain(|(_, _, turns)| turns.strong_count() > 0);
    if let Some((key, turns)) = all
        .iter()
        .filter(|&&(at, _, _)| at == address)
        .find_map(|(_, key, turns)| Some((*key, turns.upgrade()?)))
    {
        if key != lock {
            return Err(Failure::argument(
                "another context on this queue was made with other lock_queue, unlock_queue \
                 or user: every context on one queue takes the same",
            ));
        }
        return Ok(turns);
    }
    let turns = Arc::new(make());
    all.push((address, lock, Arc::downgrade(&turns)));
    Ok(turns)
}

/// The caller's lock of a queue, as the contexts on it were given it: the
/// addresses of `lock_queue`, `unlock_queue` and `user`, or none.
#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "linux",
    target_os = "android",
    all(windows, target_pointer_width = "64")
))]
type LockKey = Option<(usize, usize, usize)>;

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
unsafe fn metal_context(_device: *mut c_void, _queue: *mut c_void) -> Result<GpuContext, Failure> {
    Err(Failure(
        EmberwaveStatus::NoGpu,
        "Metal contexts are made on iPhones, iPads and Macs only".into(),
    ))
}

/// `vkGetInstanceProcAddr` of a Vulkan loader, as the C API takes it: cast
/// the loader's function to this type. Vulkan's calling convention is C's on
/// the 64-bit systems emberwave runs on.
pub type EmberwaveVkGetInstanceProcAddr = Option<
    unsafe extern "C" fn(
        instance: *mut c_void,
        name: *const c_char,
    ) -> Option<unsafe extern "C" fn()>,
>;

/// A Vulkan instance of the caller's (decision D61), made with the
/// extensions of `emberwave_vulkan_instance_extensions` among its own.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EmberwaveVulkanInstance {
    /// `vkGetInstanceProcAddr` of the loader the instance was made with.
    pub get_instance_proc_addr: EmberwaveVkGetInstanceProcAddr,
    /// The `VkInstance`.
    pub instance: *mut c_void,
    /// The `apiVersion` the instance was made with, such as
    /// `VK_API_VERSION_1_3`: at least the physical device's, up to 1.3,
    /// which wgpu uses.
    pub api_version: u32,
    /// The instance extensions it was made with: `extension_count` names, or
    /// `NULL` for none.
    pub extensions: *const *const c_char,
    /// How many names `extensions` holds.
    pub extension_count: u32,
}

/// A Vulkan device of the caller's, and the queue of it that emberwave is to
/// submit to (decision D61).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EmberwaveVulkanDevice {
    /// The instance the device was made with.
    pub instance: EmberwaveVulkanInstance,
    /// The `VkPhysicalDevice` the device was made on.
    pub physical_device: *mut c_void,
    /// The `VkDevice`, made with the extensions and features that
    /// `emberwave_vulkan_requirements_create` lists, and more if the caller
    /// needs them.
    pub device: *mut c_void,
    /// The device extensions the device was made with: `extension_count`
    /// names.
    pub extensions: *const *const c_char,
    /// How many names `extensions` holds.
    pub extension_count: u32,
    /// The family of the queue emberwave submits to, one with graphics and
    /// compute, which wgpu records work for.
    pub queue_family_index: u32,
    /// Which queue of that family, as for `vkGetDeviceQueue`.
    pub queue_index: u32,
    /// Takes the caller's lock of the queue, waiting for it to be free:
    /// Vulkan requires the uses of a queue to be externally synchronised.
    /// emberwave calls it with `user` around each submission, and while it
    /// lets go of the queue, never twice without `unlock_queue` between,
    /// from any thread. `NULL`, with `unlock_queue` `NULL` too, if nothing
    /// else uses the queue during emberwave's calls. Every context on one
    /// queue takes the same `lock_queue`, `unlock_queue` and `user`, which
    /// stay valid until everything made on those contexts is destroyed.
    pub lock_queue: Option<unsafe extern "C" fn(user: *mut c_void)>,
    /// Frees the caller's lock of the queue that `lock_queue` took.
    pub unlock_queue: Option<unsafe extern "C" fn(user: *mut c_void)>,
    /// Passed to `lock_queue` and `unlock_queue`.
    pub user: *mut c_void,
}

/// An image of the caller's Vulkan device, for
/// `emberwave_texture_from_vulkan` and `emberwave_texture_from_vulkan_plane`
/// (decision D61), and for `emberwave_texture_from_vulkan_source` (decision
/// D62).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EmberwaveVulkanImage {
    /// The `VkImage`.
    pub image: u64,
    /// Its `VkFormat`: `VK_FORMAT_B8G8R8A8_UNORM` or
    /// `VK_FORMAT_R8G8B8A8_UNORM` for the picture, `VK_FORMAT_R8_UNORM` for a
    /// plane, `VK_FORMAT_R8G8_UNORM` for Cb and Cr interleaved.
    pub format: u32,
    /// The `VkImageUsageFlags` it was made with.
    pub usage: u32,
    /// The `VkImageLayout` it is in now: `VK_IMAGE_LAYOUT_UNDEFINED`, whose
    /// contents a decode may discard, or
    /// `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`. For an image to encode,
    /// the layout it is in whenever an encode reads it, which is
    /// `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`.
    pub layout: u32,
    /// Called with `user` once emberwave and the GPU are done with the
    /// image, after the texture is destroyed and the work of every decode
    /// into it or encode from it has completed: from then on the caller may
    /// destroy the image and free its memory. From any thread, perhaps during another call of
    /// emberwave's on the context, or in the destroy of its last decoder,
    /// encoder or context. `NULL` if the caller knows that moment otherwise.
    pub release: Option<unsafe extern "C" fn(user: *mut c_void)>,
    /// Passed to `release`.
    pub user: *mut c_void,
}

/// Semaphores for a decode that does not wait, on a context of
/// `emberwave_context_create_vulkan` (decision D61): binary ones, or
/// timeline ones where `flags` says so.
///
/// A binary semaphore to wait for has a signal submitted already, or the
/// decode never runs, nor does any later work on the queue; one to signal
/// is waited for before it is signalled again. A wait that is never met
/// holds up every later submission to the queue, the caller's too, as the
/// decode's waits at every stage.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EmberwaveVulkanSync {
    /// A `VkSemaphore` the decode's GPU work waits for, such as one the
    /// caller signals once its reads of the textures are done, or 0 for
    /// none.
    pub wait_semaphore: u64,
    /// For a timeline `wait_semaphore`, the value to wait for.
    pub wait_value: u64,
    /// A `VkSemaphore` the decode signals once the textures hold the
    /// picture, for the caller's reads to wait for, or 0 for none.
    pub signal_semaphore: u64,
    /// For a timeline `signal_semaphore`, the value to signal it to.
    pub signal_value: u64,
    /// `EMBERWAVE_SYNC_WAIT_TIMELINE` and `EMBERWAVE_SYNC_SIGNAL_TIMELINE`
    /// combined with `|` for timeline semaphores, or 0 for binary ones.
    pub flags: u32,
}

/// `wait_semaphore` of an `EmberwaveVulkanSync` is a timeline semaphore, to
/// wait for at `wait_value`.
pub const EMBERWAVE_SYNC_WAIT_TIMELINE: u32 = 1 << 0;
/// `signal_semaphore` of an `EmberwaveVulkanSync` is a timeline semaphore,
/// to signal to `signal_value`.
pub const EMBERWAVE_SYNC_SIGNAL_TIMELINE: u32 = 1 << 1;

/// What emberwave needs a Vulkan device of the caller's to have enabled:
/// device extensions and features, from
/// `emberwave_vulkan_requirements_create`.
pub struct EmberwaveVulkanRequirements {
    /// The extensions' names, which live as long as the process.
    extensions: Vec<*const c_char>,
    /// A `VkPhysicalDeviceFeatures2`, whose chain `_held` keeps.
    features: *const c_void,
    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        all(windows, target_pointer_width = "64")
    ))]
    _held: vulkan::Held,
}

/// The instance extensions that wgpu uses, of those the loader of
/// `get_instance_proc_addr` offers: make the instance with them among its
/// own (decision D61). wgpu uses only those enabled, and without
/// `VK_KHR_get_physical_device_properties2` it learns less of a device and
/// uses fewer of its features. `VK_KHR_surface` goes with the
/// `VK_KHR_swapchain` that `emberwave_vulkan_requirements_create` lists on
/// devices that have it. With `names` `NULL`, stores their number in
/// `*count`; otherwise writes them to `names`, which has room for `*count`,
/// or gives `EMBERWAVE_STATUS_INVALID_ARGUMENT` with the number needed in
/// `*count` if that is too few. The names stay valid as long as the process.
/// Only where wgpu runs on Vulkan; elsewhere `EMBERWAVE_STATUS_NO_GPU`.
///
/// # Safety
///
/// `get_instance_proc_addr` is `NULL` or a loader's `vkGetInstanceProcAddr`,
/// `count` points to a `uint32_t`, and `names` is `NULL` or points to
/// `*count` pointers that may be written.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_vulkan_instance_extensions(
    get_instance_proc_addr: EmberwaveVkGetInstanceProcAddr,
    names: *mut *const c_char,
    count: *mut u32,
) -> EmberwaveStatus {
    run(|| {
        if count.is_null() {
            return Err(Failure::argument("count is NULL"));
        }
        let wanted = vulkan_instance_extensions(get_instance_proc_addr)?;
        // SAFETY: as the caller promises.
        let room = unsafe { count.read() };
        // SAFETY: as the caller promises.
        unsafe { count.write(wanted.len() as u32) };
        if names.is_null() {
            return Ok(());
        }
        if (room as usize) < wanted.len() {
            return Err(Failure::argument(format!(
                "names has room for {room} names, not the {} needed",
                wanted.len()
            )));
        }
        for (at, name) in wanted.iter().enumerate() {
            // SAFETY: room for `room` pointers, as the caller promises.
            unsafe { names.add(at).write(name.as_ptr()) };
        }
        Ok(())
    })
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    all(windows, target_pointer_width = "64")
))]
fn vulkan_instance_extensions(
    get_instance_proc_addr: EmberwaveVkGetInstanceProcAddr,
) -> Result<Vec<&'static std::ffi::CStr>, Failure> {
    vulkan::instance_extensions(get_instance_proc_addr)
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    all(windows, target_pointer_width = "64")
)))]
fn vulkan_instance_extensions(
    _get_instance_proc_addr: EmberwaveVkGetInstanceProcAddr,
) -> Result<Vec<&'static std::ffi::CStr>, Failure> {
    Err(vulkan_elsewhere())
}

/// Lists what a Vulkan device of the caller's must have enabled for
/// `emberwave_context_create_vulkan`, on the physical device
/// `physical_device` of `instance` (decision D61): the extensions, which
/// `emberwave_vulkan_requirements_extensions` gives, and the features, which
/// `emberwave_vulkan_requirements_features` gives. Make the device with
/// them, and with whatever else the caller needs. Only where wgpu runs on
/// Vulkan, on Linux, Android and Windows; elsewhere `EMBERWAVE_STATUS_NO_GPU`.
/// `EMBERWAVE_STATUS_NO_GPU` too for a physical device wgpu cannot use.
///
/// # Safety
///
/// `instance` is `NULL` or points to an `EmberwaveVulkanInstance` that
/// describes a live instance, and `physical_device` is `NULL` or a physical
/// device of it. `requirements` points to where the new requirements are
/// stored; it is set to `NULL` when the call fails.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_vulkan_requirements_create(
    instance: *const EmberwaveVulkanInstance,
    physical_device: *mut c_void,
    requirements: *mut *mut EmberwaveVulkanRequirements,
) -> EmberwaveStatus {
    run(|| {
        // SAFETY: as the caller promises.
        unsafe { output(requirements, "requirements", std::ptr::null_mut()) }?;
        // SAFETY: as the caller promises.
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return Err(Failure::argument("instance is NULL"));
        };
        // SAFETY: as the caller promises.
        let made = unsafe { vulkan_requirements(instance, physical_device) }?;
        // SAFETY: checked by the first output() above.
        unsafe { requirements.write(Box::into_raw(Box::new(made))) };
        Ok(())
    })
}

/// The names of the device extensions in `requirements`: `*count` of them,
/// in an array that lives as long as `requirements`, of names that live as
/// long as the process. `NULL`, and a count of 0, for `NULL`.
///
/// # Safety
///
/// `requirements` is `NULL` or live requirements, and `count` points to a
/// `uint32_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_vulkan_requirements_extensions(
    requirements: *const EmberwaveVulkanRequirements,
    count: *mut u32,
) -> *const *const c_char {
    // SAFETY: as the caller promises.
    let names = unsafe { requirements.as_ref() }.map_or(&[][..], |r| &r.extensions[..]);
    if !count.is_null() {
        // SAFETY: as the caller promises.
        unsafe { count.write(names.len() as u32) };
    }
    if names.is_empty() {
        std::ptr::null()
    } else {
        names.as_ptr()
    }
}

/// The features in `requirements`: a `VkPhysicalDeviceFeatures2` whose
/// `features`, and the structures of its `pNext` chain, have `VK_TRUE` for
/// each feature emberwave needs. They are what wgpu enables on a device of
/// its own on that physical device, and then counts on: some, such as
/// `robustBufferAccess`, for its own sake, others, such as the robustness of
/// images, for the bounds checks of the shaders it compiles. Set the same members to `VK_TRUE` in the
/// structures that make the device, or chain copies of the structures the
/// caller's chain lacks; where the caller's chain has
/// `VkPhysicalDeviceVulkan11Features`, `VkPhysicalDeviceVulkan12Features` or
/// `VkPhysicalDeviceVulkan13Features`, set the members there that hold the
/// same features instead, as Vulkan allows only one of the two. Valid while
/// `requirements` lives; `NULL` for `NULL`.
///
/// # Safety
///
/// `requirements` is `NULL` or live requirements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_vulkan_requirements_features(
    requirements: *const EmberwaveVulkanRequirements,
) -> *const c_void {
    // SAFETY: as the caller promises.
    unsafe { requirements.as_ref() }.map_or(std::ptr::null(), |r| r.features)
}

/// Destroys requirements. Destroying `NULL` does nothing.
///
/// # Safety
///
/// `requirements` is `NULL`, or requirements from
/// `emberwave_vulkan_requirements_create` that have not been destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_vulkan_requirements_destroy(
    requirements: *mut EmberwaveVulkanRequirements,
) {
    if !requirements.is_null() {
        // SAFETY: it came from Box::into_raw, as the caller promises.
        let boxed = unsafe { Box::from_raw(requirements) };
        let _ = catch_unwind(AssertUnwindSafe(move || drop(boxed)));
    }
}

/// Creates a context on the caller's own Vulkan device, for emberwave to
/// work on the caller's queue (decision D61). The device keeps belonging to
/// the caller, who destroys it after the context and everything made on it.
/// Only where wgpu runs on Vulkan, on Linux, Android and Windows; elsewhere
/// `EMBERWAVE_STATUS_NO_GPU`.
///
/// The device must have been made with the extensions and the features of
/// `emberwave_vulkan_requirements_create`: a lacking extension is
/// `EMBERWAVE_STATUS_INVALID_ARGUMENT`, and a lacking feature undefined
/// behaviour, as Vulkan cannot tell which were enabled. So is a queue family
/// without graphics and compute, a queue it does not have, and an instance
/// whose `apiVersion` is below the physical device's, up to 1.3.
///
/// The queue is shared:
/// - emberwave submits to it during its calls. Either nothing else uses the
///   queue during those, or the caller gives `lock_queue` and
///   `unlock_queue`, which emberwave holds around each submission, and
///   around each destroy of a context, decoder or encoder. The destroy that
///   lets go of the last hold on the queue waits for the queue to be idle
///   meanwhile, which holds up the caller's own submissions until it is. Do
///   not call emberwave while holding that lock; a context on a queue that
///   has contexts with other callbacks is
///   `EMBERWAVE_STATUS_INVALID_ARGUMENT`. A second queue of the family,
///   where the device has one, needs neither.
/// - The decoders and encoders of every context made on the queue take
///   turns at it, as on Metal (decision D60).
/// - The decodes that wait return once the GPU is done. The decodes that do
///   not wait, `emberwave_decoder_submit_texture` and
///   `emberwave_decoder_submit_plane_textures`, are for Metal contexts only.
///
/// # Safety
///
/// `device` is `NULL` or points to an `EmberwaveVulkanDevice` that describes
/// a live device made as above, with a queue of that family and index, and
/// callbacks that may be called from any thread. `context` points to where
/// the new context is stored; it is set to `NULL` when the call fails.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_context_create_vulkan(
    device: *const EmberwaveVulkanDevice,
    context: *mut *mut EmberwaveContext,
) -> EmberwaveStatus {
    run(|| {
        // SAFETY: as the caller promises.
        unsafe { output(context, "context", std::ptr::null_mut()) }?;
        // SAFETY: as the caller promises.
        let Some(device) = (unsafe { device.as_ref() }) else {
            return Err(Failure::argument("device is NULL"));
        };
        // SAFETY: as the caller promises.
        let created = unsafe { vulkan_context(device) }?;
        // SAFETY: checked by the first output() above.
        unsafe { context.write(EmberwaveContext::boxed(created, false, true)) };
        Ok(())
    })
}

/// The `VkImage` of `texture`, a texture on a context of
/// `emberwave_context_create_vulkan`, for the caller to read the picture or
/// the plane from on the GPU; 0 for `NULL` and for a texture on another
/// context (decision D61). The image belongs to the texture, which keeps it
/// until it is destroyed.
///
/// Between emberwave's calls the image is in the layout
/// `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`, from the first decode into it
/// on: a decode leaves it so, and the caller gives it back so before the next
/// call that uses it. It belongs to the context's queue family, and its
/// usage is `VK_IMAGE_USAGE_SAMPLED_BIT`, `VK_IMAGE_USAGE_TRANSFER_SRC_BIT`
/// and what the decodes need.
///
/// # Safety
///
/// `texture` is `NULL` or a live texture.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_texture_vulkan(texture: *const EmberwaveTexture) -> u64 {
    // SAFETY: as the caller promises.
    match unsafe { texture.as_ref() } {
        Some(texture) => catch_unwind(AssertUnwindSafe(|| vulkan_image(texture))).unwrap_or(0),
        None => 0,
    }
}

/// Wraps the caller's own Vulkan image for the decodes of `decoder` to put
/// the picture into, as into one of `emberwave_texture_create`'s, on a
/// context of `emberwave_context_create_vulkan` (decision D61); for a
/// decoder on another context, `EMBERWAVE_STATUS_INVALID_ARGUMENT`.
///
/// The image is on the context's device, 2D, of one level, layer and
/// sample, with optimal tiling and exclusive sharing on the context's queue
/// family, of the picture's size, and as `image` describes it. Its usage
/// has `VK_IMAGE_USAGE_SAMPLED_BIT`, as a decode leaves it in
/// `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`. A decode copies the picture
/// into it with `VK_IMAGE_USAGE_TRANSFER_DST_BIT` when its width is a
/// multiple of 64 pixels, and draws it with
/// `VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT` otherwise. An image that neither
/// can fill gives `EMBERWAVE_STATUS_INVALID_ARGUMENT`, and so does one that is
/// not as above where `image` says so; `texture` is then `NULL`.
///
/// emberwave never destroys the image, nor frees its memory: keep both
/// until `release` is called, or, without it, until the texture is
/// destroyed and the work of every decode into it has completed; a decode
/// that does not wait may still run after `emberwave_texture_destroy` has
/// returned. Between emberwave's calls the image is in
/// `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`, as for
/// `emberwave_texture_vulkan`.
///
/// # Safety
///
/// `decoder` is a live decoder that no other thread is using, `image` is
/// `NULL` or points to an `EmberwaveVulkanImage` that describes a live image
/// made as above, in the layout it gives. `texture` points to where the new
/// texture is stored.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_texture_from_vulkan(
    decoder: *const EmberwaveDecoder,
    image: *const EmberwaveVulkanImage,
    texture: *mut *mut EmberwaveTexture,
) -> EmberwaveStatus {
    // SAFETY: as the caller promises.
    unsafe { texture_from_vulkan(decoder, None, image, texture) }
}

/// Wraps the caller's own Vulkan image for the Y, Cb or Cr plane of the
/// pictures of `decoder`, as `emberwave_texture_create_plane` makes one:
/// `plane` is `EMBERWAVE_PLANE_Y`, `EMBERWAVE_PLANE_CB` or
/// `EMBERWAVE_PLANE_CR`. As for `emberwave_texture_from_vulkan`, but the
/// image is `VK_FORMAT_R8_UNORM` of the plane's size, and a decode copies
/// the plane into it, which needs `VK_IMAGE_USAGE_TRANSFER_DST_BIT`.
///
/// # Safety
///
/// As for `emberwave_texture_from_vulkan`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_texture_from_vulkan_plane(
    decoder: *const EmberwaveDecoder,
    plane: u32,
    image: *const EmberwaveVulkanImage,
    texture: *mut *mut EmberwaveTexture,
) -> EmberwaveStatus {
    // SAFETY: as the caller promises.
    unsafe { texture_from_vulkan(decoder, Some(plane), image, texture) }
}

/// Wraps the caller's own Vulkan image that holds a plane of a picture for
/// the encodes of `encoder` to read, `emberwave_encoder_encode_textures` and
/// `emberwave_encoder_encode_textures_vulkan`, on a context of
/// `emberwave_context_create_vulkan` (decision D62); for an encoder on
/// another context, `EMBERWAVE_STATUS_INVALID_ARGUMENT`. `plane` is
/// `EMBERWAVE_PLANE_Y`, `EMBERWAVE_PLANE_CB` or `EMBERWAVE_PLANE_CR` for an
/// image in `VK_FORMAT_R8_UNORM`, or `EMBERWAVE_PLANE_CBCR` for one in
/// `VK_FORMAT_R8G8_UNORM` with Cb and Cr interleaved.
///
/// The image is on the context's device, 2D, of one level, layer and
/// sample, with optimal tiling and exclusive sharing on the context's queue
/// family, of the plane's size as `emberwave_encoder_info` gives it (that of
/// Cb for `EMBERWAVE_PLANE_CBCR`), and as `image` describes it. Its usage
/// has `VK_IMAGE_USAGE_SAMPLED_BIT`, and its layout is
/// `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`, which it is in whenever an
/// encode reads it: from the first encode on, the caller's barriers start
/// from that layout and return the image to it before the next encode. An
/// image that is not as above where `image` says so gives
/// `EMBERWAVE_STATUS_INVALID_ARGUMENT`, and `texture` is then `NULL`.
///
/// emberwave never destroys the image, nor frees its memory: keep both
/// until `release` is called, or, without it, until the texture is
/// destroyed. Every encode waits for the GPU, so none still reads the image
/// then.
///
/// # Safety
///
/// `encoder` is a live encoder that no other thread is using, `image` is
/// `NULL` or points to an `EmberwaveVulkanImage` that describes a live image
/// made as above. `texture` points to where the new texture is stored.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_texture_from_vulkan_source(
    encoder: *const EmberwaveEncoder,
    plane: u32,
    image: *const EmberwaveVulkanImage,
    texture: *mut *mut EmberwaveTexture,
) -> EmberwaveStatus {
    run_or_lost(
        || {
            // SAFETY: as the caller promises.
            unsafe { output(texture, "texture", std::ptr::null_mut()) }?;
            let holds = source_holds(plane)?;
            // SAFETY: as the caller promises.
            let encoder =
                unsafe { encoder.as_ref() }.ok_or_else(|| Failure::argument("encoder is NULL"))?;
            // SAFETY: as the caller promises.
            let Some(image) = (unsafe { image.as_ref() }) else {
                return Err(Failure::argument("image is NULL"));
            };
            require_vulkan(encoder.vulkan, "encoder")?;
            // SAFETY: as the caller promises.
            let wrapped = unsafe { vulkan_import(&Wrapping::encoder(encoder, holds), image) }?;
            // SAFETY: checked by the first output() above.
            unsafe { texture.write(Box::into_raw(Box::new(wrapped))) };
            Ok(())
        },
        // SAFETY: as the caller promises.
        || unsafe { encoder_lost(encoder) },
    )
}

/// The work of `emberwave_texture_from_vulkan` and
/// `emberwave_texture_from_vulkan_plane`.
///
/// # Safety
///
/// As for `emberwave_texture_from_vulkan`.
unsafe fn texture_from_vulkan(
    decoder: *const EmberwaveDecoder,
    plane: Option<u32>,
    image: *const EmberwaveVulkanImage,
    texture: *mut *mut EmberwaveTexture,
) -> EmberwaveStatus {
    run_or_lost(
        || {
            // SAFETY: as the caller promises.
            unsafe { output(texture, "texture", std::ptr::null_mut()) }?;
            let plane = plane.map(plane_component).transpose()?;
            // SAFETY: as the caller promises.
            let decoder = unsafe { live(decoder) }?;
            // SAFETY: as the caller promises.
            let Some(image) = (unsafe { image.as_ref() }) else {
                return Err(Failure::argument("image is NULL"));
            };
            require_vulkan(decoder.vulkan, "decoder")?;
            // SAFETY: as the caller promises.
            let wrapped = unsafe { vulkan_import(&Wrapping::decoder(decoder, plane), image) }?;
            // SAFETY: checked by the first output() above.
            unsafe { texture.write(Box::into_raw(Box::new(wrapped))) };
            Ok(())
        },
        // SAFETY: as the caller promises.
        || unsafe { lost(decoder) },
    )
}

/// Decodes the frame received so far into `texture`, as
/// `emberwave_decoder_decode_texture` does, but returns once the work is
/// submitted to the queue, without waiting for it, on a context of
/// `emberwave_context_create_vulkan` (decision D61). Elsewhere
/// `EMBERWAVE_STATUS_INVALID_ARGUMENT`.
///
/// `sync`, if not `NULL`, has semaphores for the submission. The decode's
/// GPU work waits for `wait_semaphore`, at every stage, before it writes the
/// texture: the caller signals it once its own GPU work that reads the
/// texture is done, unless the queue orders that work before the decode
/// already. The decode signals `signal_semaphore` once the texture holds the
/// picture, in `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`: the caller's work
/// that reads the picture waits for it, unless that work goes to the same
/// queue after the call, with a barrier from all commands. Keep the
/// semaphores until the work that signals them has completed. When the call
/// fails, the semaphores may or may not have been waited for and signalled.
///
/// # Safety
///
/// As for `emberwave_decoder_decode_texture`, and `sync` is `NULL` or points
/// to an `EmberwaveVulkanSync` whose semaphores are of the context's device.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_decoder_submit_texture_vulkan(
    decoder: *mut EmberwaveDecoder,
    color: *const EmberwaveColor,
    texture: *const EmberwaveTexture,
    sync: *const EmberwaveVulkanSync,
) -> EmberwaveStatus {
    run_or_lost(
        || {
            // SAFETY: as the caller promises.
            let decoder = unsafe { live_mut(decoder) }?;
            require_vulkan(decoder.vulkan, "decoder")?;
            // SAFETY: as the caller promises.
            let texture = unsafe { picture_texture(texture, decoder.decoder.context_id()) }?;
            // SAFETY: as the caller promises.
            let params = unsafe { conversion(decoder, color) }?;
            // SAFETY: as the caller promises.
            let sync = unsafe { vulkan_sync(sync) }?;
            let mut semaphores = vulkan_semaphores(sync);
            decoder.decoder.submit_assembled_to_texture_with(
                &params,
                &texture.view,
                &mut semaphores,
            )?;
            Ok(())
        },
        // SAFETY: as the caller promises; the body's borrow has ended.
        || unsafe { lost(decoder) },
    )
}

/// Decodes the frame received so far into the plane textures `y`, `cb` and
/// `cr`, as `emberwave_decoder_decode_plane_textures` does, but returns once
/// the work is submitted, with the semaphores of `sync`, as
/// `emberwave_decoder_submit_texture_vulkan` does.
///
/// # Safety
///
/// As for `emberwave_decoder_decode_plane_textures`, and `sync` is `NULL` or
/// points to an `EmberwaveVulkanSync` whose semaphores are of the context's
/// device.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_decoder_submit_plane_textures_vulkan(
    decoder: *mut EmberwaveDecoder,
    y: *const EmberwaveTexture,
    cb: *const EmberwaveTexture,
    cr: *const EmberwaveTexture,
    sync: *const EmberwaveVulkanSync,
) -> EmberwaveStatus {
    run_or_lost(
        || {
            // SAFETY: as the caller promises.
            let decoder = unsafe { live_mut(decoder) }?;
            require_vulkan(decoder.vulkan, "decoder")?;
            let context_id = decoder.decoder.context_id();
            // SAFETY: as the caller promises.
            let textures = unsafe {
                [
                    plane_texture(y, Component::Y, "y", context_id)?,
                    plane_texture(cb, Component::Cb, "cb", context_id)?,
                    plane_texture(cr, Component::Cr, "cr", context_id)?,
                ]
            };
            distinct_images(&textures)?;
            // SAFETY: as the caller promises.
            let sync = unsafe { vulkan_sync(sync) }?;
            let mut semaphores = vulkan_semaphores(sync);
            let targets = textures.map(|texture| &texture.view);
            decoder
                .decoder
                .submit_assembled_to_plane_textures_with(targets, &mut semaphores)?;
            Ok(())
        },
        // SAFETY: as the caller promises; the body's borrow has ended.
        || unsafe { lost(decoder) },
    )
}

/// Encodes the picture whose planes are in the textures `y`, `cb` and `cr`,
/// as `emberwave_encoder_encode_textures` does, with the semaphores of
/// `sync` on its submission, on a context of
/// `emberwave_context_create_vulkan` (decision D62). Elsewhere
/// `EMBERWAVE_STATUS_INVALID_ARGUMENT`.
///
/// `sync`, if not `NULL`, has semaphores for the submission. The encode's
/// GPU work waits for `wait_semaphore`, at every stage, before it reads the
/// images: the caller signals it once its own GPU work that writes them is
/// done, such as the copy of a captured picture, unless the queue orders
/// that work before the encode already. The encode signals
/// `signal_semaphore` once the GPU has read the images and made the frame,
/// for the caller's work on other queues that writes them again. The call
/// waits for the GPU as the other encodes do, so it returns only once its
/// wait is met: by work submitted before the call, or from another thread.
/// Keep the semaphores until the work that signals them has completed. When
/// the call fails, the semaphores may or may not have been waited for and
/// signalled.
///
/// # Safety
///
/// As for `emberwave_encoder_encode_textures`, and `sync` is `NULL` or
/// points to an `EmberwaveVulkanSync` whose semaphores are of the context's
/// device.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_encoder_encode_textures_vulkan(
    encoder: *mut EmberwaveEncoder,
    y: *const EmberwaveTexture,
    cb: *const EmberwaveTexture,
    cr: *const EmberwaveTexture,
    sync: *const EmberwaveVulkanSync,
    frame: *mut u8,
    budget: usize,
    size: *mut usize,
) -> EmberwaveStatus {
    run_or_lost(
        || {
            // SAFETY: as the caller promises.
            unsafe { frame_arguments(frame, budget, size) }?;
            // SAFETY: as the caller promises.
            let encoder = unsafe { live_encoder(encoder) }?;
            require_vulkan(encoder.vulkan, "encoder")?;
            // SAFETY: as the caller promises.
            let picture = unsafe { source_picture(y, cb, cr, encoder.encoder.context_id()) }?;
            // SAFETY: as the caller promises.
            let sync = unsafe { vulkan_sync(sync) }?;
            let mut semaphores = vulkan_semaphores(sync);
            let encoded = encoder
                .encoder
                .encode_textures_with(picture, budget, &mut semaphores);
            // SAFETY: as the caller promises.
            unsafe { deliver(encoded, frame, size) }
        },
        // SAFETY: as the caller promises; the body's borrow has ended.
        || unsafe { encoder_lost(encoder) },
    )
}

/// The semaphores of `sync`, none for `NULL`, or an error for unknown flags.
///
/// # Safety
///
/// `sync` is `NULL` or points to an `EmberwaveVulkanSync`.
unsafe fn vulkan_sync(sync: *const EmberwaveVulkanSync) -> Result<EmberwaveVulkanSync, Failure> {
    // SAFETY: as the caller promises.
    let sync = unsafe { sync.as_ref() }.copied().unwrap_or_default();
    let known = EMBERWAVE_SYNC_WAIT_TIMELINE | EMBERWAVE_SYNC_SIGNAL_TIMELINE;
    if sync.flags & !known != 0 {
        return Err(Failure::argument(format!(
            "sync's flags {:#x} include unknown ones",
            sync.flags
        )));
    }
    Ok(sync)
}

/// An error unless the context of `owner`, a decoder or an encoder, runs on
/// the caller's own Vulkan device (`vulkan`).
fn require_vulkan(vulkan: bool, owner: &str) -> Result<(), Failure> {
    if vulkan {
        Ok(())
    } else {
        Err(Failure::argument(format!(
            "the {owner}'s context is not on the caller's Vulkan device \
             (emberwave_context_create_vulkan)"
        )))
    }
}

/// An error if two of the planes are the same Vulkan image.
fn distinct_images(textures: &[&EmberwaveTexture; 3]) -> Result<(), Failure> {
    let [y, cb, cr] = textures.map(vulkan_image);
    let same = |a: u64, b: u64| a != 0 && a == b;
    if same(y, cb) || same(y, cr) || same(cb, cr) {
        return Err(Failure::argument(
            "two of the planes are the same Vulkan image",
        ));
    }
    Ok(())
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    all(windows, target_pointer_width = "64")
))]
fn vulkan_image(texture: &EmberwaveTexture) -> u64 {
    vulkan::image(texture)
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    all(windows, target_pointer_width = "64")
)))]
fn vulkan_image(_texture: &EmberwaveTexture) -> u64 {
    0
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    all(windows, target_pointer_width = "64")
))]
unsafe fn vulkan_import(
    wrapping: &Wrapping<'_>,
    image: &EmberwaveVulkanImage,
) -> Result<EmberwaveTexture, Failure> {
    // SAFETY: as the caller promises.
    unsafe { vulkan::import(wrapping, image) }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    all(windows, target_pointer_width = "64")
)))]
unsafe fn vulkan_import(
    _wrapping: &Wrapping<'_>,
    _image: &EmberwaveVulkanImage,
) -> Result<EmberwaveTexture, Failure> {
    Err(vulkan_elsewhere())
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    all(windows, target_pointer_width = "64")
))]
fn vulkan_semaphores(sync: EmberwaveVulkanSync) -> vulkan::Semaphores {
    vulkan::Semaphores::new(sync)
}

/// Where there are no Vulkan contexts, nothing to add to a submission.
#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    all(windows, target_pointer_width = "64")
)))]
fn vulkan_semaphores(_sync: EmberwaveVulkanSync) -> impl emberwave_core::gpu::SubmissionHooks {
    struct Nothing;
    impl emberwave_core::gpu::SubmissionHooks for Nothing {
        fn before_submit(&mut self, _queue: &wgpu::Queue) {}
        fn after_submit(&mut self, _queue: &wgpu::Queue, _submitted: bool) {}
    }
    Nothing
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    all(windows, target_pointer_width = "64")
))]
unsafe fn vulkan_requirements(
    instance: &EmberwaveVulkanInstance,
    physical_device: *mut c_void,
) -> Result<EmberwaveVulkanRequirements, Failure> {
    // SAFETY: as the caller promises.
    unsafe { vulkan::requirements(instance, physical_device) }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    all(windows, target_pointer_width = "64")
)))]
unsafe fn vulkan_requirements(
    _instance: &EmberwaveVulkanInstance,
    _physical_device: *mut c_void,
) -> Result<EmberwaveVulkanRequirements, Failure> {
    Err(vulkan_elsewhere())
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    all(windows, target_pointer_width = "64")
))]
unsafe fn vulkan_context(device: &EmberwaveVulkanDevice) -> Result<GpuContext, Failure> {
    // SAFETY: as the caller promises.
    unsafe { vulkan::context(device) }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    all(windows, target_pointer_width = "64")
)))]
unsafe fn vulkan_context(_device: &EmberwaveVulkanDevice) -> Result<GpuContext, Failure> {
    Err(vulkan_elsewhere())
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    all(windows, target_pointer_width = "64")
)))]
fn vulkan_elsewhere() -> Failure {
    Failure(
        EmberwaveStatus::NoGpu,
        "Vulkan contexts are made on Linux, Android and 64-bit Windows only".into(),
    )
}

/// The Vulkan side of `emberwave_vulkan_requirements_create` and
/// `emberwave_context_create_vulkan`, through wgpu's Vulkan objects made
/// from the caller's (decision D61).
#[cfg(any(
    target_os = "linux",
    target_os = "android",
    all(windows, target_pointer_width = "64")
))]
mod vulkan {
    use std::ffi::{CStr, c_char, c_void};

    use ash::vk::{self, Handle};
    use emberwave_core::gpu::{
        CallerQueueLock, GpuContext, GpuOptions, PLANE_FORMAT, QueueTurns, SubmissionHooks,
    };
    use wgpu::hal::api::Vulkan;

    use super::{
        EmberwaveStatus, EmberwaveTexture, EmberwaveVkGetInstanceProcAddr, EmberwaveVulkanDevice,
        EmberwaveVulkanImage, EmberwaveVulkanInstance, EmberwaveVulkanRequirements,
        EmberwaveVulkanSync, Failure, Holds, Wrapping, queue_turns,
    };

    /// What keeps the chain of an `EmberwaveVulkanRequirements` alive.
    pub(super) type Held = (
        Box<wgpu::hal::vulkan::PhysicalDeviceFeatures>,
        Box<vk::PhysicalDeviceFeatures2<'static>>,
    );

    /// The `count` names at `names`, none for a count of 0.
    ///
    /// # Safety
    ///
    /// `names` is `NULL` or points to `count` pointers, each `NULL` or a C
    /// string that lives as long as the result.
    unsafe fn names<'a>(
        names: *const *const c_char,
        count: u32,
        what: &str,
    ) -> Result<Vec<&'a CStr>, Failure> {
        if count == 0 {
            return Ok(Vec::new());
        }
        if names.is_null() {
            return Err(Failure::argument(format!(
                "{what} is NULL, with a count of {count}"
            )));
        }
        // SAFETY: `count` pointers, as the caller promises.
        let pointers = unsafe { std::slice::from_raw_parts(names, count as usize) };
        pointers
            .iter()
            .map(|&name| {
                if name.is_null() {
                    Err(Failure::argument(format!("a name in {what} is NULL")))
                } else {
                    // SAFETY: a C string, as the caller promises.
                    Ok(unsafe { CStr::from_ptr(name) })
                }
            })
            .collect()
    }

    /// The instance extensions wgpu uses where the loader of
    /// `get_instance_proc_addr` offers them.
    pub(super) fn instance_extensions(
        get_instance_proc_addr: EmberwaveVkGetInstanceProcAddr,
    ) -> Result<Vec<&'static CStr>, Failure> {
        let Some(get_instance_proc_addr) = get_instance_proc_addr else {
            return Err(Failure::argument("get_instance_proc_addr is NULL"));
        };
        // SAFETY: as in `adapter`.
        let get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr =
            unsafe { std::mem::transmute(get_instance_proc_addr) };
        // SAFETY: a loader's vkGetInstanceProcAddr.
        let entry = unsafe {
            ash::Entry::from_static_fn(ash::StaticFn {
                get_instance_proc_addr,
            })
        };
        wgpu::hal::vulkan::Instance::desired_extensions(&entry, 0, wgpu::InstanceFlags::empty())
            .map_err(no_gpu)
    }

    fn no_gpu(error: impl std::fmt::Display) -> Failure {
        Failure(EmberwaveStatus::NoGpu, error.to_string())
    }

    /// wgpu's adapter on `physical_device` of the caller's `instance`.
    ///
    /// # Safety
    ///
    /// As for `emberwave_vulkan_requirements_create`.
    unsafe fn adapter(
        instance: &EmberwaveVulkanInstance,
        physical_device: *mut c_void,
    ) -> Result<wgpu::Adapter, Failure> {
        let Some(get_instance_proc_addr) = instance.get_instance_proc_addr else {
            return Err(Failure::argument("get_instance_proc_addr is NULL"));
        };
        if instance.instance.is_null() {
            return Err(Failure::argument("instance is NULL"));
        }
        if physical_device.is_null() {
            return Err(Failure::argument("physical_device is NULL"));
        }
        // SAFETY: the loader's vkGetInstanceProcAddr, as the caller
        // promises: that signature, with Vulkan's calling convention, which
        // is C's on the 64-bit systems emberwave runs on.
        let get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr =
            unsafe { std::mem::transmute(get_instance_proc_addr) };
        // SAFETY: a loader's vkGetInstanceProcAddr.
        let entry = unsafe {
            ash::Entry::from_static_fn(ash::StaticFn {
                get_instance_proc_addr,
            })
        };
        let raw_instance = vk::Instance::from_raw(instance.instance as usize as u64);
        // SAFETY: a live instance of that loader, as the caller promises.
        let ash_instance = unsafe { ash::Instance::load(entry.static_fn(), raw_instance) };
        let physical = vk::PhysicalDevice::from_raw(physical_device as usize as u64);
        // wgpu takes the version of the physical device, up to 1.3, as the
        // one it may use; the instance's limits what may be used.
        // SAFETY: a physical device of the instance, as the caller promises.
        let device_version =
            unsafe { ash_instance.get_physical_device_properties(physical) }.api_version;
        let needed = device_version.min(vk::API_VERSION_1_3);
        if instance.api_version < needed {
            return Err(Failure::argument(format!(
                "the instance's apiVersion is {}.{}, below the {}.{} of the physical device that \
                 wgpu uses: make the instance with at least that",
                vk::api_version_major(instance.api_version),
                vk::api_version_minor(instance.api_version),
                vk::api_version_major(needed),
                vk::api_version_minor(needed),
            )));
        }
        // SAFETY: as the caller promises.
        let enabled = unsafe {
            names(
                instance.extensions,
                instance.extension_count,
                "the instance's extensions",
            )
        }?;
        // The instance extensions wgpu uses, where the caller enabled them.
        let flags = wgpu::InstanceFlags::empty();
        let extensions: Vec<&'static CStr> =
            wgpu::hal::vulkan::Instance::desired_extensions(&entry, instance.api_version, flags)
                .map_err(no_gpu)?
                .into_iter()
                .filter(|name| enabled.contains(name))
                .collect();
        // SAFETY: the instance was made from `entry`, with `api_version`
        // and with `extensions` among its extensions, as the caller
        // promises. Given a drop callback, wgpu leaves it to the caller.
        let hal = unsafe {
            wgpu::hal::vulkan::Instance::from_raw(
                entry,
                ash_instance,
                instance.api_version,
                0,
                None,
                extensions,
                flags,
                Default::default(),
                false,
                Some(Box::new(|| {})),
            )
        }
        .map_err(no_gpu)?;
        let exposed = hal
            .expose_adapter(physical)
            .ok_or_else(|| no_gpu("wgpu cannot run on that physical device"))?;
        // SAFETY: an instance of wgpu's Vulkan.
        let wgpu_instance = unsafe { wgpu::Instance::from_hal::<Vulkan>(hal) };
        // SAFETY: an adapter of that instance.
        Ok(unsafe { wgpu_instance.create_adapter_from_hal(exposed) })
    }

    /// See `emberwave_vulkan_requirements_create`.
    ///
    /// # Safety
    ///
    /// As for `emberwave_vulkan_requirements_create`.
    pub(super) unsafe fn requirements(
        instance: &EmberwaveVulkanInstance,
        physical_device: *mut c_void,
    ) -> Result<EmberwaveVulkanRequirements, Failure> {
        // SAFETY: as the caller promises.
        let adapter = unsafe { adapter(instance, physical_device) }?;
        let descriptor = GpuContext::device_descriptor(&adapter, &GpuOptions::default());
        // SAFETY: only read.
        let hal =
            unsafe { adapter.as_hal::<Vulkan>() }.ok_or_else(|| no_gpu("not a Vulkan adapter"))?;
        let names = hal.required_device_extensions(descriptor.required_features);
        let mut features =
            Box::new(hal.physical_device_features(&names, descriptor.required_features));
        // wgpu chains the structures it needs; the boxes keep them where the
        // chain points.
        let chain = features
            .add_to_device_create(vk::DeviceCreateInfo::default())
            .p_next;
        let mut features2 =
            Box::new(vk::PhysicalDeviceFeatures2::default().features(features.get_core()));
        features2.p_next = chain.cast_mut();
        let pointer = std::ptr::from_ref::<vk::PhysicalDeviceFeatures2<'static>>(&features2).cast();
        Ok(EmberwaveVulkanRequirements {
            extensions: names.iter().map(|name| name.as_ptr()).collect(),
            features: pointer,
            _held: (features, features2),
        })
    }

    /// The caller's lock of a queue, through its callbacks.
    struct Callbacks {
        lock: unsafe extern "C" fn(user: *mut c_void),
        unlock: unsafe extern "C" fn(user: *mut c_void),
        user: *mut c_void,
    }

    // SAFETY: the caller promises that its callbacks may be called with
    // `user` from any thread.
    unsafe impl Send for Callbacks {}
    // SAFETY: as above.
    unsafe impl Sync for Callbacks {}

    impl CallerQueueLock for Callbacks {
        fn lock(&self) {
            // SAFETY: the caller's callback, as it promises.
            unsafe { (self.lock)(self.user) }
        }

        fn unlock(&self) {
            // SAFETY: as above.
            unsafe { (self.unlock)(self.user) }
        }
    }

    /// See `emberwave_context_create_vulkan`.
    ///
    /// # Safety
    ///
    /// As for `emberwave_context_create_vulkan`.
    pub(super) unsafe fn context(device: &EmberwaveVulkanDevice) -> Result<GpuContext, Failure> {
        if device.device.is_null() {
            return Err(Failure::argument("device is NULL"));
        }
        let lock = match (device.lock_queue, device.unlock_queue) {
            (Some(lock), Some(unlock)) => Some(Callbacks {
                lock,
                unlock,
                user: device.user,
            }),
            (None, None) => None,
            _ => {
                return Err(Failure::argument(
                    "lock_queue and unlock_queue are both given or both NULL",
                ));
            }
        };
        // SAFETY: as the caller promises.
        let enabled = unsafe {
            names(
                device.extensions,
                device.extension_count,
                "the device's extensions",
            )
        }?;
        // SAFETY: as the caller promises.
        let adapter = unsafe { adapter(&device.instance, device.physical_device) }?;
        let descriptor = GpuContext::device_descriptor(&adapter, &GpuOptions::default());
        let (open, turns) = {
            // SAFETY: only read.
            let hal = unsafe { adapter.as_hal::<Vulkan>() }
                .ok_or_else(|| no_gpu("not a Vulkan adapter"))?;
            let required = hal.required_device_extensions(descriptor.required_features);
            if let Some(missing) = required.iter().find(|name| !enabled.contains(name)) {
                return Err(Failure::argument(format!(
                    "the device lacks {}, which emberwave_vulkan_requirements_extensions lists",
                    missing.to_string_lossy()
                )));
            }
            let instance = hal.shared_instance().raw_instance();
            // wgpu records graphics, compute and copies for any queue: a
            // family without graphics would take stages it cannot run.
            // SAFETY: the adapter's physical device, of that instance.
            let families = unsafe {
                instance.get_physical_device_queue_family_properties(hal.raw_physical_device())
            };
            let family = families
                .get(device.queue_family_index as usize)
                .ok_or_else(|| Failure::argument("the device has no such queue family"))?;
            if !family
                .queue_flags
                .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            {
                return Err(Failure::argument(
                    "the queue family lacks graphics or compute, which wgpu records for",
                ));
            }
            if device.queue_index >= family.queue_count {
                return Err(Failure::argument(format!(
                    "the queue family has {} queues, not one of index {}",
                    family.queue_count, device.queue_index
                )));
            }
            // SAFETY: a live device of that instance, as the caller promises.
            let raw = unsafe {
                ash::Device::load(
                    instance.fp_v1_0(),
                    vk::Device::from_raw(device.device as usize as u64),
                )
            };
            // SAFETY: a queue the device was made with, as the caller
            // promises.
            let queue =
                unsafe { raw.get_device_queue(device.queue_family_index, device.queue_index) };
            // The queue's turns, before anything of wgpu's is made on it: a
            // context refused here makes nothing that would use the queue
            // without the lock as it goes.
            let key = lock
                .as_ref()
                .map(|lock| (lock.lock as usize, lock.unlock as usize, lock.user as usize));
            let turns = queue_turns(queue.as_raw() as usize, key, || match lock {
                Some(lock) => QueueTurns::with_caller_lock(Box::new(lock)),
                None => QueueTurns::new(),
            })?;
            // SAFETY: the device was made on the adapter's physical device,
            // with that queue, the required extensions among its own and
            // the features wgpu needs for these, as the caller promises.
            // Given a drop callback, wgpu leaves it to the caller.
            let open = unsafe {
                hal.device_from_raw(
                    raw,
                    Some(Box::new(|| {})),
                    &required,
                    descriptor.required_features,
                    &descriptor.required_limits,
                    &descriptor.memory_hints,
                    device.queue_family_index,
                    device.queue_index,
                )
            }
            .map_err(no_gpu)?;
            (open, turns)
        };
        // SAFETY: a device opened on this adapter with the descriptor's
        // features and limits.
        let (wgpu_device, wgpu_queue) =
            unsafe { adapter.create_device_from_hal(open, &descriptor) }.map_err(no_gpu)?;
        Ok(GpuContext::from_device(adapter, wgpu_device, wgpu_queue).with_queue_turns(turns))
    }

    /// The `VkImage` of `texture`, or 0 if it is not on the caller's own
    /// Vulkan device.
    pub(super) fn image(texture: &EmberwaveTexture) -> u64 {
        if !texture.vulkan {
            return 0;
        }
        // SAFETY: only read.
        unsafe { texture.texture.as_hal::<Vulkan>() }
            // SAFETY: the image of a live texture.
            .map_or(0, |hal| unsafe { hal.raw_handle() }.as_raw())
    }

    /// See `emberwave_texture_from_vulkan`.
    ///
    /// # Safety
    ///
    /// As for `emberwave_texture_from_vulkan`.
    pub(super) unsafe fn import(
        wrapping: &Wrapping<'_>,
        image: &EmberwaveVulkanImage,
    ) -> Result<EmberwaveTexture, Failure> {
        if image.image == 0 {
            return Err(Failure::argument("image is VK_NULL_HANDLE"));
        }
        let (width, height) = wrapping.size();
        let format = match (wrapping.holds, vk::Format::from_raw(image.format as i32)) {
            (Holds::Plane(_), vk::Format::R8_UNORM) => PLANE_FORMAT,
            (Holds::Chroma, vk::Format::R8G8_UNORM) => wgpu::TextureFormat::Rg8Unorm,
            (Holds::Picture, vk::Format::R8G8B8A8_UNORM) => wgpu::TextureFormat::Rgba8Unorm,
            (Holds::Picture, vk::Format::B8G8R8A8_UNORM) => wgpu::TextureFormat::Bgra8Unorm,
            (Holds::Plane(_), format) => {
                return Err(Failure::argument(format!(
                    "the image's format is {format:?}, not VK_FORMAT_R8_UNORM"
                )));
            }
            (Holds::Chroma, format) => {
                return Err(Failure::argument(format!(
                    "the image's format is {format:?}, not VK_FORMAT_R8G8_UNORM"
                )));
            }
            (Holds::Picture, format) => {
                return Err(Failure::argument(format!(
                    "the image's format is {format:?}, not VK_FORMAT_B8G8R8A8_UNORM or \
                     VK_FORMAT_R8G8B8A8_UNORM"
                )));
            }
        };
        let flags = vk::ImageUsageFlags::from_raw(image.usage);
        if !flags.contains(vk::ImageUsageFlags::SAMPLED) {
            return Err(Failure::argument(if wrapping.read {
                "the image lacks VK_IMAGE_USAGE_SAMPLED_BIT, which an encode reads it with"
            } else {
                "the image lacks VK_IMAGE_USAGE_SAMPLED_BIT: a decode leaves it in \
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL"
            }));
        }
        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING;
        for (flag, use_) in [
            (
                vk::ImageUsageFlags::TRANSFER_SRC,
                wgpu::TextureUsages::COPY_SRC,
            ),
            (
                vk::ImageUsageFlags::TRANSFER_DST,
                wgpu::TextureUsages::COPY_DST,
            ),
            (
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            ),
        ] {
            if flags.contains(flag) {
                usage |= use_;
            }
        }
        let copies = usage.contains(wgpu::TextureUsages::COPY_DST);
        let fillable = match wrapping.holds {
            Holds::Picture => {
                (copies && (width * 4).is_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT))
                    || usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
            }
            Holds::Plane(_) | Holds::Chroma => copies,
        };
        // An encode only samples the image (decision D62).
        if !wrapping.read && !fillable {
            return Err(Failure::argument(
                "neither a copy nor a draw can fill the image: a copy needs \
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT (and for the picture a width of a multiple \
                 of 64 pixels), a draw of the picture VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT",
            ));
        }
        let state = match vk::ImageLayout::from_raw(image.layout as i32) {
            // What an encode reads is in the image already: its layout is
            // the one every encode reads it in.
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => wgpu::TextureUses::RESOURCE,
            vk::ImageLayout::UNDEFINED if !wrapping.read => wgpu::TextureUses::UNINITIALIZED,
            other if wrapping.read => {
                return Err(Failure::argument(format!(
                    "the image's layout is {other:?}, not \
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, the layout an encode reads it in"
                )));
            }
            other => {
                return Err(Failure::argument(format!(
                    "the image's layout is {other:?}, not VK_IMAGE_LAYOUT_UNDEFINED or \
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL"
                )));
            }
        };
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let device = wrapping.device;
        let hal_texture = {
            // SAFETY: only read.
            let hal = unsafe { device.as_hal::<Vulkan>() }
                .ok_or_else(|| no_gpu("not a Vulkan device"))?;
            // What wgpu-core takes each usage of the texture's for.
            let mut uses = wgpu::TextureUses::RESOURCE;
            for (use_, internal) in [
                (wgpu::TextureUsages::COPY_SRC, wgpu::TextureUses::COPY_SRC),
                (wgpu::TextureUsages::COPY_DST, wgpu::TextureUses::COPY_DST),
                (
                    wgpu::TextureUsages::RENDER_ATTACHMENT,
                    wgpu::TextureUses::COLOR_TARGET,
                ),
            ] {
                if usage.contains(use_) {
                    uses |= internal;
                }
            }
            // wgpu calls a drop callback, rather than destroying the image,
            // once it and the GPU are done with it: the caller's `release`.
            let release = SendPointer(image.user);
            let release: wgpu::hal::DropCallback = match image.release {
                Some(callback) => Box::new(move || {
                    let release = release;
                    // SAFETY: the caller's callback with its `user`, which
                    // it promises may be called from any thread.
                    unsafe { callback(release.0) }
                }),
                None => Box::new(|| {}),
            };
            // SAFETY: an image of the device, made as described, as the
            // caller promises. Given a drop callback and external memory,
            // wgpu leaves both to the caller.
            unsafe {
                hal.texture_from_raw(
                    vk::Image::from_raw(image.image),
                    &wgpu::hal::TextureDescriptor {
                        label: Some("caller's image"),
                        size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: uses,
                        memory_flags: wgpu::hal::MemoryFlags::empty(),
                        view_formats: Vec::new(),
                    },
                    Some(release),
                    wgpu::hal::vulkan::TextureMemory::External,
                )
            }
        };
        // SAFETY: a texture of the device's, made as described, in the state
        // its layout gives, as the caller promises.
        let texture = unsafe {
            device.create_texture_from_hal::<Vulkan>(
                hal_texture,
                &wgpu::TextureDescriptor {
                    label: Some("caller's image"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                },
                state,
            )
        };
        Ok(EmberwaveTexture::wrap(wrapping, texture, true))
    }

    /// The caller's `user` of an `EmberwaveVulkanImage`, for its `release`
    /// on whichever thread wgpu calls it.
    struct SendPointer(*mut c_void);

    // SAFETY: the caller promises that `release` may be called with `user`
    // from any thread.
    unsafe impl Send for SendPointer {}
    // SAFETY: as above.
    unsafe impl Sync for SendPointer {}

    /// The semaphores of an `EmberwaveVulkanSync`, added to a decode's
    /// submission, and taken back if it was not made.
    pub(super) struct Semaphores {
        wait: Option<(vk::Semaphore, Option<u64>)>,
        signal: Option<(vk::Semaphore, Option<u64>)>,
    }

    impl Semaphores {
        pub(super) fn new(sync: EmberwaveVulkanSync) -> Self {
            let semaphore = |handle: u64, timeline: bool, value: u64| {
                (handle != 0).then(|| (vk::Semaphore::from_raw(handle), timeline.then_some(value)))
            };
            let timeline = |flag| sync.flags & flag != 0;
            Self {
                wait: semaphore(
                    sync.wait_semaphore,
                    timeline(super::EMBERWAVE_SYNC_WAIT_TIMELINE),
                    sync.wait_value,
                ),
                signal: semaphore(
                    sync.signal_semaphore,
                    timeline(super::EMBERWAVE_SYNC_SIGNAL_TIMELINE),
                    sync.signal_value,
                ),
            }
        }
    }

    impl SubmissionHooks for Semaphores {
        fn before_submit(&mut self, queue: &wgpu::Queue) {
            // SAFETY: only adds to the next submission, the decode's.
            if let Some(hal) = unsafe { queue.as_hal::<Vulkan>() } {
                if let Some((semaphore, value)) = self.wait {
                    hal.add_wait_semaphore(semaphore, value, vk::PipelineStageFlags::ALL_COMMANDS);
                }
                if let Some((semaphore, value)) = self.signal {
                    hal.add_signal_semaphore(semaphore, value);
                }
            }
        }

        fn after_submit(&mut self, queue: &wgpu::Queue, _submitted: bool) {
            // Whatever the call says: a submission that took them leaves
            // nothing to take back, and one that did not, as on a device lost
            // meanwhile, must not hand them to another.
            // SAFETY: only takes back what before_submit added.
            if let Some(hal) = unsafe { queue.as_hal::<Vulkan>() } {
                if let Some((semaphore, _)) = self.wait {
                    hal.remove_wait_semaphore(semaphore);
                }
                if let Some((semaphore, _)) = self.signal {
                    hal.remove_signal_semaphore(semaphore);
                }
            }
        }
    }
}

/// Destroys a context. Decoders and encoders created on it keep working, and
/// keep the GPU device until they are destroyed as well.
///
/// It may wait for the GPU, when it lets go of the last hold on the
/// context's queue. On a context of `emberwave_context_create_metal`, do not
/// call it inside a completion handler of a command buffer of that queue:
/// the wait never ends there.
///
/// # Safety
///
/// `context` is `NULL`, or a context from `emberwave_context_create`,
/// `emberwave_context_create_metal` or `emberwave_context_create_vulkan`
/// that has not been destroyed and that no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_context_destroy(context: *mut EmberwaveContext) {
    if !context.is_null() {
        // SAFETY: it came from Box::into_raw, as the caller promises.
        let boxed = unsafe { Box::from_raw(context) };
        let turns = boxed.context.queue_turns().cloned();
        let_go(boxed, turns);
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
            vulkan: context.vulkan,
            turns: context.context.queue_turns().cloned(),
        });
        // SAFETY: checked by the first output() above.
        unsafe { decoder.write(Box::into_raw(boxed)) };
        Ok(())
    })
}

/// Destroys a decoder and frees its GPU memory.
///
/// It may wait for the GPU, when it lets go of the last hold on the
/// context's queue. On a context of `emberwave_context_create_metal`, do not
/// call it inside a completion handler of a command buffer of that queue:
/// the wait never ends there.
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
        let turns = boxed.turns.clone();
        let_go(boxed, turns);
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
/// On a context of `emberwave_context_create_metal`, do not call it inside a
/// completion handler of a command buffer of the context's queue: the wait
/// never ends there.
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
/// On a context of `emberwave_context_create_metal`, do not call it inside a
/// completion handler of a command buffer of the context's queue: the wait
/// never ends there.
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
            let wrapped = unsafe { metal_import(&Wrapping::decoder(decoder, None), metal) }?;
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
            let wrapped =
                unsafe { metal_import(&Wrapping::decoder(decoder, Some(component)), metal) }?;
            // SAFETY: checked by the first output() above.
            unsafe { texture.write(Box::into_raw(Box::new(wrapped))) };
            Ok(())
        },
        // SAFETY: as the caller promises.
        || unsafe { lost(decoder) },
    )
}

/// Wraps the caller's own Metal texture, `metal`, an `id<MTLTexture>`, that
/// holds a plane of a picture for the encodes of `encoder` to read,
/// `emberwave_encoder_encode_textures` (decision D62): a texture of a
/// `CVPixelBuffer` from a `CVMetalTextureCache`, as a screen capture gives
/// them, or one the caller drew. `plane` is `EMBERWAVE_PLANE_Y`,
/// `EMBERWAVE_PLANE_CB` or `EMBERWAVE_PLANE_CR` for a texture in
/// `MTLPixelFormatR8Unorm`, or `EMBERWAVE_PLANE_CBCR` for one in
/// `MTLPixelFormatRG8Unorm` with Cb and Cr interleaved, as the second plane
/// of an NV12 pixel buffer holds them. Only on iPhones, iPads and Macs;
/// elsewhere `EMBERWAVE_STATUS_NO_GPU` for arguments that are not `NULL`.
///
/// The texture is on the device of the encoder's context
/// (`emberwave_context_metal_device`), 2D, of one level, layer and sample,
/// not a view of another texture nor memoryless, of the plane's size as
/// `emberwave_encoder_info` gives it (that of Cb for
/// `EMBERWAVE_PLANE_CBCR`), and has `MTLTextureUsageShaderRead`. A texture
/// that is not gives `EMBERWAVE_STATUS_INVALID_ARGUMENT`, and `texture` is
/// then `NULL`. On a context that does not run on Metal, a Vulkan context on
/// a Mac included, no texture is on the context's Metal device.
///
/// emberwave keeps a reference to `metal` until the texture is destroyed.
/// A texture may be wrapped for one encode and destroyed after it, as the
/// pixel buffers of a capture come and go, or kept for the encodes of many
/// pictures.
///
/// # Safety
///
/// `encoder` is a live encoder that no other thread is using, and `metal` is
/// `NULL` or an `id<MTLTexture>`. `texture` points to where the new texture
/// is stored.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_texture_from_metal_source(
    encoder: *const EmberwaveEncoder,
    plane: u32,
    metal: *mut c_void,
    texture: *mut *mut EmberwaveTexture,
) -> EmberwaveStatus {
    run_or_lost(
        || {
            // SAFETY: as the caller promises.
            unsafe { output(texture, "texture", std::ptr::null_mut()) }?;
            // SAFETY: as the caller promises.
            let encoder =
                unsafe { encoder.as_ref() }.ok_or_else(|| Failure::argument("encoder is NULL"))?;
            let holds = source_holds(plane)?;
            // SAFETY: as the caller promises.
            let wrapped = unsafe { metal_import(&Wrapping::encoder(encoder, holds), metal) }?;
            // SAFETY: checked by the first output() above.
            unsafe { texture.write(Box::into_raw(Box::new(wrapped))) };
            Ok(())
        },
        // SAFETY: as the caller promises.
        || unsafe { encoder_lost(encoder) },
    )
}

/// What a texture holds that an `EMBERWAVE_PLANE_*` value names for an
/// encoder, `EMBERWAVE_PLANE_CBCR` included.
fn source_holds(plane: u32) -> Result<Holds, Failure> {
    match plane {
        EMBERWAVE_PLANE_CBCR => Ok(Holds::Chroma),
        EMBERWAVE_PLANE_Y | EMBERWAVE_PLANE_CB | EMBERWAVE_PLANE_CR => {
            plane_component(plane).map(Holds::Plane)
        }
        other => Err(Failure::argument(format!(
            "plane is {other}, not EMBERWAVE_PLANE_Y, EMBERWAVE_PLANE_CB, EMBERWAVE_PLANE_CR \
             or EMBERWAVE_PLANE_CBCR"
        ))),
    }
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

/// The caller's `id<MTLTexture>` `metal` as a texture for `wrapping`, once
/// it has passed the checks of `emberwave_texture_from_metal` or, for an
/// encoder, of `emberwave_texture_from_metal_source`.
///
/// # Safety
///
/// `metal` is `NULL` or an `id<MTLTexture>`.
#[cfg(any(target_os = "macos", target_os = "ios"))]
unsafe fn metal_import(
    wrapping: &Wrapping<'_>,
    metal: *mut c_void,
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
    let device = wrapping.device;
    let owner = if wrapping.read { "encoder" } else { "decoder" };
    if !std::ptr::eq(
        Retained::as_ptr(&metal.device()).cast::<c_void>(),
        metal_device(device).cast_const(),
    ) {
        return Err(Failure::argument(format!(
            "the texture is not on the Metal device of the {owner}'s context"
        )));
    }

    let (width, height) = wrapping.size();
    let format = metal.pixelFormat();
    let format = match wrapping.holds {
        Holds::Plane(_) if format == MTLPixelFormat::R8Unorm => emberwave_core::gpu::PLANE_FORMAT,
        Holds::Chroma if format == MTLPixelFormat::RG8Unorm => wgpu::TextureFormat::Rg8Unorm,
        Holds::Picture if format == MTLPixelFormat::RGBA8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        Holds::Picture if format == MTLPixelFormat::BGRA8Unorm => wgpu::TextureFormat::Bgra8Unorm,
        Holds::Plane(_) => {
            return Err(Failure::argument(format!(
                "the texture's pixel format is {}, not MTLPixelFormatR8Unorm",
                format.0
            )));
        }
        Holds::Chroma => {
            return Err(Failure::argument(format!(
                "the texture's pixel format is {}, not MTLPixelFormatRG8Unorm",
                format.0
            )));
        }
        Holds::Picture => {
            return Err(Failure::argument(format!(
                "the texture's pixel format is {}, not MTLPixelFormatBGRA8Unorm or \
                 MTLPixelFormatRGBA8Unorm",
                format.0
            )));
        }
    };
    let whose = wrapping.holds.whose();
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
    if matches!(wrapping.holds, Holds::Plane(_))
        && wrapping.stores_planes
        && metal.usage().contains(MTLTextureUsage::ShaderWrite)
    {
        usage |= wgpu::TextureUsages::STORAGE_BINDING;
    }
    if wrapping.read {
        // An encode samples the texture (decision D62).
        if !usage.contains(wgpu::TextureUsages::TEXTURE_BINDING) {
            return Err(Failure::argument(
                "the texture lacks MTLTextureUsageShaderRead, which an encode reads it with",
            ));
        }
    } else {
        let fillable = match wrapping.holds {
            Holds::Picture => {
                (copies && (width * 4).is_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT))
                    || usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
            }
            Holds::Plane(_) | Holds::Chroma => copies,
        };
        if !fillable {
            return Err(Failure::argument(
                "neither a copy nor a draw can fill the texture: a copy needs a texture that \
                 is not framebuffer-only (and for the picture a width of a multiple of 64 \
                 pixels), a draw of the picture MTLTextureUsageRenderTarget",
            ));
        }
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
    // SAFETY: as above, and the usage is what the texture allows. Whatever
    // it holds, a decode writes all of it, and an encode reads what the
    // caller put in: wgpu takes a texture of the caller's as initialised.
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
    Ok(EmberwaveTexture::wrap(wrapping, texture, tracked))
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
unsafe fn metal_import(
    _wrapping: &Wrapping<'_>,
    _metal: *mut c_void,
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
/// `emberwave_texture_create_plane`, or one of the functions that wrap a
/// texture or an image of the caller's (`emberwave_texture_from_*`) that has
/// not been destroyed and that no other thread is using. A Metal texture of one emberwave made goes with
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
/// On a context of `emberwave_context_create_metal`, do not call it inside a
/// completion handler of a command buffer of the context's queue: the wait
/// never ends there.
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
    match texture.holds {
        Holds::Picture => Ok(texture),
        Holds::Plane(plane) => Err(Failure::argument(format!(
            "the texture is for the {plane:?} plane, which \
             emberwave_decoder_decode_plane_textures fills"
        ))),
        Holds::Chroma => Err(Failure::argument(
            "the texture is for Cb and Cr interleaved, which only encoders read",
        )),
    }
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
            "the decoder's context is not on the caller's Metal queue \
             (emberwave_context_create_metal), where the queue orders the \
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
/// On a context of `emberwave_context_create_metal`, do not call it inside a
/// completion handler of a command buffer of the context's queue: the wait
/// never ends there.
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
    if texture.holds != Holds::Plane(plane) {
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
        let boxed = Box::new(EmberwaveEncoder {
            encoder: created,
            vulkan: context.vulkan,
            turns: context.context.queue_turns().cloned(),
        });
        // SAFETY: checked by the first output() above.
        unsafe { encoder.write(Box::into_raw(boxed)) };
        Ok(())
    })
}

/// Destroys an encoder and frees its GPU memory.
///
/// It may wait for the GPU, when it lets go of the last hold on the
/// context's queue. On a context of `emberwave_context_create_metal`, do not
/// call it inside a completion handler of a command buffer of that queue:
/// the wait never ends there.
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
        let turns = boxed.turns.clone();
        let_go(boxed, turns);
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
/// On a context of `emberwave_context_create_metal`, do not call it inside a
/// completion handler of a command buffer of the context's queue: the wait
/// never ends there.
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
            unsafe { frame_arguments(frame, budget, size) }?;
            // SAFETY: as the caller promises.
            let encoder = unsafe { live_encoder(encoder) }?;
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
            let encoded = encoder.encoder.encode_strided(planes, strides, budget);
            // SAFETY: as the caller promises.
            unsafe { deliver(encoded, frame, size) }
        },
        // SAFETY: as the caller promises; the body's borrow has ended.
        || unsafe { encoder_lost(encoder) },
    )
}

/// Encodes the picture whose planes are in textures on the GPU, as
/// `emberwave_encoder_encode` encodes one in CPU memory and into the same
/// bytes (decision D62): `y` holds the Y plane, and `cb` and `cr` the Cb and
/// Cr planes, or `cb` both of them interleaved (`EMBERWAVE_PLANE_CBCR`) with
/// `cr` `NULL`. The textures are textures of planes on the encoder's context
/// with the planes' sizes: the caller's, from
/// `emberwave_texture_from_metal_source` or
/// `emberwave_texture_from_vulkan_source`, or textures that decodes put
/// planes into. Others give `EMBERWAVE_STATUS_INVALID_ARGUMENT` and encode
/// nothing, as does a budget too small, as for `emberwave_encoder_encode`.
///
/// Waits for the GPU: when it returns, the GPU has read the textures, and
/// the caller may write them again or hand them back to a capture's pool.
/// The GPU reads them after the caller's work on the context's queue that
/// Metal or Vulkan orders before the encode:
/// - On a context of `emberwave_context_create_metal`, the caller's command
///   buffers committed to the queue before the call, for textures whose
///   hazards Metal tracks.
/// - On a context of `emberwave_context_create_vulkan`, the caller's work
///   submitted to the queue before the call, with a barrier that makes its
///   writes visible to compute shaders and leaves the images in
///   `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`; semaphores order work on
///   other queues (`emberwave_encoder_encode_textures_vulkan`).
/// - Otherwise, on a context of `emberwave_context_create` or for a texture
///   whose hazards Metal does not track, only work that has completed: the
///   caller's GPU work that writes the textures completes before the call,
///   as it has for a capture's pixel buffer once it is delivered.
///
/// On a context of `emberwave_context_create_metal`, do not call it inside a
/// completion handler of a command buffer of the context's queue: the wait
/// never ends there.
///
/// # Safety
///
/// `encoder` is a live encoder that no other thread is using, `y` and `cb`
/// are live textures, and `cr` is `NULL` or a live texture. `frame` points
/// to `budget` bytes that may be written, and `size` to a `size_t`, which do
/// not overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn emberwave_encoder_encode_textures(
    encoder: *mut EmberwaveEncoder,
    y: *const EmberwaveTexture,
    cb: *const EmberwaveTexture,
    cr: *const EmberwaveTexture,
    frame: *mut u8,
    budget: usize,
    size: *mut usize,
) -> EmberwaveStatus {
    run_or_lost(
        || {
            // SAFETY: as the caller promises.
            unsafe { frame_arguments(frame, budget, size) }?;
            // SAFETY: as the caller promises.
            let encoder = unsafe { live_encoder(encoder) }?;
            // SAFETY: as the caller promises.
            let picture = unsafe { source_picture(y, cb, cr, encoder.encoder.context_id()) }?;
            let encoded = encoder.encoder.encode_textures(picture, budget);
            // SAFETY: as the caller promises.
            unsafe { deliver(encoded, frame, size) }
        },
        // SAFETY: as the caller promises; the body's borrow has ended.
        || unsafe { encoder_lost(encoder) },
    )
}

/// Checks the arguments where an encode puts the frame, and sets `*size`
/// to 0 for a failure.
///
/// # Safety
///
/// `size` is `NULL` or points to a `size_t`.
unsafe fn frame_arguments(frame: *mut u8, budget: usize, size: *mut usize) -> Result<(), Failure> {
    // SAFETY: as the caller promises.
    unsafe { output(size, "size", 0) }?;
    if frame.is_null() && budget > 0 {
        return Err(Failure::argument("frame is NULL"));
    }
    if budget > isize::MAX as usize {
        return Err(Failure::argument("budget is too large"));
    }
    Ok(())
}

/// Writes the frame of an encode to `frame` and its size to `*size`, or for
/// a budget too small the smallest frame's size.
///
/// # Safety
///
/// As `frame_arguments` checked: `frame` points to `budget` bytes, at least
/// the frame's, that may be written, and `size` to a `size_t`.
unsafe fn deliver(
    encoded: Result<Vec<u8>, GpuError>,
    frame: *mut u8,
    size: *mut usize,
) -> Result<(), Failure> {
    match encoded {
        Ok(bytes) => {
            // SAFETY: a frame is at most `budget` bytes, which the caller
            // promises may be written.
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
}

/// The picture of the textures `y`, `cb` and `cr` for an encoder on the
/// context `context_id`: three planes, or Y and Cb and Cr interleaved, with
/// `cr` `NULL`.
///
/// # Safety
///
/// `y`, `cb` and `cr` are `NULL` or live textures, which outlive the borrow.
unsafe fn source_picture<'a>(
    y: *const EmberwaveTexture,
    cb: *const EmberwaveTexture,
    cr: *const EmberwaveTexture,
    context_id: u64,
) -> Result<TexturePicture<'a>, Failure> {
    // SAFETY: as the caller promises.
    let luma = unsafe { plane_texture(y, Component::Y, "y", context_id) }?;
    // SAFETY: as the caller promises.
    let Some(chroma) = (unsafe { cb.as_ref() }) else {
        return Err(Failure::argument("cb is NULL"));
    };
    if chroma.holds == Holds::Chroma {
        // Not wgpu's device equality, which devices of two contexts can pass
        // (GpuContext::id).
        if chroma.context_id != context_id {
            return Err(Failure::argument(
                "cb is on the GPU device of another context",
            ));
        }
        if !cr.is_null() {
            return Err(Failure::argument(
                "cr is not NULL, though cb holds Cb and Cr interleaved",
            ));
        }
        return Ok(TexturePicture::LumaChroma {
            luma: &luma.view,
            chroma: &chroma.view,
        });
    }
    // SAFETY: as the caller promises.
    let (cb, cr) = unsafe {
        (
            plane_texture(cb, Component::Cb, "cb", context_id)?,
            plane_texture(cr, Component::Cr, "cr", context_id)?,
        )
    };
    Ok(TexturePicture::Planes([&luma.view, &cb.view, &cr.view]))
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
