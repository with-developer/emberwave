// Times HEVC through VideoToolbox, the device's own video encoder and decoder,
// on the pictures emberwave is timed on: the path of a Moonlight client today,
// beside emberwave's. Each frame is timed alone, as emberwave's are: decoding
// from the frame's bytes until VideoToolbox has handed over the picture,
// wrapped in Metal textures, as NV12 (Y, and Cb and Cr interleaved: the
// counterpart of emberwave's Y, Cb and Cr textures) or as BGRA (that of its
// RGB texture); encoding from the picture's planes in memory until the frame
// is out. Unlike emberwave's frames, HEVC's depend on those before them: the
// pictures in turn are encoded into one stream, a keyframe and then a frame
// for each one that is timed, and the stream is decoded in order. One session
// is open at a time, as the device's media engine serves them all.

import CoreMedia
import CoreVideo
import Foundation
import Metal
import VideoToolbox

/// A failed call of VideoToolbox or Core Media, with its status.
struct MediaFailure: Error, CustomStringConvertible {
    let description: String
    /// The device has no HEVC encoder or decoder for these pictures.
    var unavailable = false
}

/// Throws unless `status` is `noErr`.
func check(_ status: OSStatus, _ call: String) throws {
    if status != noErr {
        throw MediaFailure(description: "\(call): OSStatus \(status)")
    }
}

/// Makes and sets up a session with `setup`, whose failures, before anything
/// is timed, mean that the device has no such encoder or decoder: a simulator
/// or a virtual machine may have none, or a software one that refuses a
/// setting.
private func settingUp<Session>(_ setup: () throws -> Session) throws -> Session {
    do {
        return try setup()
    } catch var failure as MediaFailure {
        failure.unavailable = true
        throw failure
    }
}

/// How long a frame may take before VideoToolbox is deemed stuck, in seconds:
/// far longer than any does.
private let stuck = 10.0

/// The frames per second of the streams: their bit rates are per second.
let streamFrameRate = 60

/// A bit rate HEVC is timed at, and what it stands for.
struct HEVCRate {
    let bitsPerSecond: Int
    let meaning: String
    /// Everything is timed at this rate, or only decoding into NV12 textures,
    /// one frame after the other, and encoding.
    let full: Bool
}

/// The bit rates of `size`: Moonlight's default for it at 60 frames per
/// second, at which its clients stream unless told otherwise, and emberwave's
/// budget at 60 frames per second, the same bytes per frame.
func hevcRates(_ size: PictureSize) -> [HEVCRate] {
    // Moonlight's default: 10, 20 and 40 Mbit/s for 1080p, 1440p and 4K at
    // 30 frames per second, twice that at 60.
    let moonlight =
        size.height <= 1080 ? 20_000_000 : size.height <= 1440 ? 40_000_000 : 80_000_000
    return [
        HEVCRate(bitsPerSecond: moonlight, meaning: "Moonlight's default", full: true),
        HEVCRate(
            bitsPerSecond: size.budget * 8 * streamFrameRate, meaning: "emberwave's bytes",
            full: false),
    ]
}

/// What HEVC at one bit rate gave.
struct HEVCResult {
    /// The encoder VideoToolbox chose, and whether they are the hardware's,
    /// `nil` where the system does not say.
    let encoder: String
    let hardwareEncoder: Bool?
    let hardwareDecoder: Bool?
    /// The settings the encoder does not have, by their keys.
    let unsupported: [String]
    let keyframeBytes: Int
    /// The mean and the largest of the frames after the keyframe.
    let bytesPerFrame: Int
    let largestFrame: Int
    /// Of the luma of the last pictures decoded, against the pictures: a
    /// check that the stream decodes to them, not a measure of quality,
    /// which the scene's grain dominates.
    let psnr: Double
    let encode: Stats
    let decodeNV12: Stats
    /// Measured at a full rate only.
    let decodeBGRA: Stats?
    let decodeKeyframe: Stats?
    let pacedNV12: Stats?
    let pacedBGRA: Stats?
}

/// HEVC at one bit rate for one picture size: what it gave, or why it was not
/// timed.
struct HEVCOutcome {
    let size: PictureSize
    let rate: HEVCRate
    let result: Result<HEVCResult, Error>

    /// Whether the run failed, rather than found no HEVC on the device.
    var failed: Bool {
        if case .failure(let error) = result {
            return !((error as? MediaFailure)?.unavailable ?? false)
        }
        return false
    }

    /// Its lines in the report.
    var lines: [String] {
        let heading = "  HEVC at \(rate.bitsPerSecond / 1_000_000) Mbit/s, \(rate.meaning)"
        let result: HEVCResult
        switch self.result {
        case .failure(let error):
            return ["\(heading): \(failed ? "failed" : "not available"): \(error)"]
        case .success(let success):
            result = success
        }
        func kilobytes(_ bytes: Int) -> Int {
            Int((Double(bytes) / 1000).rounded())
        }
        func hardware(_ flag: Bool?) -> String {
            flag.map { $0 ? "hardware" : "software" } ?? "?"
        }
        var lines = [
            "\(heading) (\(kilobytes(result.bytesPerFrame)) kB a frame, at most "
                + "\(kilobytes(result.largestFrame)), keyframe \(kilobytes(result.keyframeBytes))):",
            "    \(result.encoder) (\(hardware(result.hardwareEncoder)))"
                + (result.unsupported.isEmpty
                    ? "" : ", without \(result.unsupported.joined(separator: ", "))")
                + "; decoder \(hardware(result.hardwareDecoder)); luma PSNR "
                + String(format: "%.1f dB", result.psnr),
            row("decode into NV12 textures", result.decodeNV12, indent: 4),
        ]
        if let bgra = result.decodeBGRA {
            lines.append(row("decode into a BGRA texture", bgra, indent: 4))
        }
        if let keyframe = result.decodeKeyframe {
            lines.append(row("decode a keyframe, NV12", keyframe, indent: 4))
        }
        lines.append(row("encode from memory", result.encode, indent: 4))
        if let nv12 = result.pacedNV12, let bgra = result.pacedBGRA {
            lines.append("    60 frames a second:")
            lines.append(row("decode into NV12 textures", nv12, indent: 6))
            lines.append(row("decode into a BGRA texture", bgra, indent: 6))
        }
        return lines
    }
}

/// Encodes `pictures`, of `size`, in turn into an HEVC stream at `rate`, and
/// decodes it, timing both. Textures are made on `device`, as a player that
/// draws with Metal makes them.
func runHEVC(
    _ size: PictureSize, _ rate: HEVCRate, _ pictures: [[[UInt8]]], _ plan: Plan,
    device: MTLDevice, progress: (String) -> Void
) -> HEVCOutcome {
    HEVCOutcome(
        size: size, rate: rate,
        result: Result {
            try timeHEVC(size, rate, pictures, plan, device: device, progress: progress)
        })
}

private func timeHEVC(
    _ size: PictureSize, _ rate: HEVCRate, _ pictures: [[[UInt8]]], _ plan: Plan,
    device: MTLDevice, progress: (String) -> Void
) throws -> HEVCResult {
    let name = "\(size.name): HEVC at \(rate.bitsPerSecond / 1_000_000) Mbit/s"
    progress("\(name), encoding")
    let encoder = try settingUp { try HEVCEncoder(size, rate, pictures) }
    // The warm-up's frames are not kept: the stream starts at a keyframe.
    let warm = now()
    var warmed = 0
    while warmed < plan.warmupFrames || now() - warm < plan.warmupMs {
        _ = try encoder.encode()
        warmed += 1
    }
    let first = encoder.picture
    var frames = [try encoder.encode(keyframe: true).bytes]
    var times: [Double] = []
    for _ in 0..<plan.frames {
        let frame = try encoder.encode()
        guard !frame.keyframe else {
            throw MediaFailure(description: "the encoder made a keyframe of its own")
        }
        frames.append(frame.bytes)
        times.append(frame.time)
    }
    // The loop also copies each frame out, which is not the encoder's time.
    let encode = Stats(times, total: times.reduce(0, +))
    let format = try encoder.parameterSets()
    encoder.invalidate()

    progress("\(name), decoding into NV12 textures")
    let nv12 = try settingUp {
        try HEVCDecoder(format, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, device)
    }
    defer { nv12.invalidate() }
    let decodeNV12 = try nv12.time(frames, plan)
    // The last picture of each, as decoded.
    var squared = 0.0
    for (offset, image) in nv12.last.enumerated() {
        let index = frames.count - 1 - offset
        squared += try lumaSquaredError(image, pictures[(first + index) % pictures.count][0], size)
    }
    let psnr = psnr(squaredError: squared, samples: Double(size.width * size.height * nv12.last.count))
    // Far below what any of the bit rates gives: the stream does not decode
    // to the pictures.
    guard psnr > 30 else {
        throw MediaFailure(description: String(format: "luma PSNR %.1f dB", psnr))
    }
    var decodeKeyframe: Stats?
    var pacedNV12: Stats?
    var decodeBGRA: Stats?
    var pacedBGRA: Stats?
    if rate.full {
        progress("\(name), decoding keyframes")
        // A keyframe depends on no other frame: the same one over again,
        // after a few untimed, as the decoder is warm.
        for _ in 0..<3 {
            _ = try nv12.decode(frames[0])
        }
        var keyframeTimes: [Double] = []
        for _ in 0..<max(plan.frames / 10, 6) {
            keyframeTimes.append(try nv12.decode(frames[0]).time)
        }
        decodeKeyframe = Stats(keyframeTimes, total: keyframeTimes.reduce(0, +))
        progress("\(name), decoding into NV12 textures, 60 frames a second")
        pacedNV12 = try nv12.time(frames, plan, paced: true)
        nv12.invalidate()

        progress("\(name), decoding into a BGRA texture")
        let bgra = try settingUp { try HEVCDecoder(format, kCVPixelFormatType_32BGRA, device) }
        defer { bgra.invalidate() }
        decodeBGRA = try bgra.time(frames, plan)
        progress("\(name), decoding into a BGRA texture, 60 frames a second")
        pacedBGRA = try bgra.time(frames, plan, paced: true)
    }

    let others = frames.dropFirst().map(\.count)
    return HEVCResult(
        encoder: encoder.name,
        hardwareEncoder: encoder.hardware,
        hardwareDecoder: nv12.hardware,
        unsupported: encoder.unsupported,
        keyframeBytes: frames[0].count,
        bytesPerFrame: others.reduce(0, +) / max(others.count, 1),
        largestFrame: others.max() ?? 0,
        psnr: psnr,
        encode: encode,
        decodeNV12: decodeNV12,
        decodeBGRA: decodeBGRA,
        decodeKeyframe: decodeKeyframe,
        pacedNV12: pacedNV12,
        pacedBGRA: pacedBGRA)
}

/// The value of a VideoToolbox session's property, `nil` if it has none.
private func property(_ session: VTSession, _ key: CFString) -> CFTypeRef? {
    var value: CFTypeRef?
    let status = withUnsafeMutablePointer(to: &value) {
        VTSessionCopyProperty(session, key: key, allocator: nil, valueOut: $0)
    }
    return status == noErr ? value : nil
}

/// An HEVC encoding session, and the pictures in NV12 in memory: the Y plane,
/// and Cb and Cr interleaved, rows packed.
private final class HEVCEncoder {
    let session: VTCompressionSession
    /// The encoder VideoToolbox chose, by its identifier, and whether it is the
    /// hardware's, `nil` where the system does not say.
    let name: String
    let hardware: Bool?
    /// The settings the encoder does not have, by their keys.
    let unsupported: [String]
    /// The picture the next frame is of.
    private(set) var picture = 0
    private let size: PictureSize
    private let pool: CVPixelBufferPool
    private let pictures: [(luma: [UInt8], chroma: [UInt8])]
    private var frameNumber: Int64 = 0
    /// The format of the last frame, with the stream's parameter sets.
    private var format: CMFormatDescription?

    init(_ size: PictureSize, _ rate: HEVCRate, _ pictures: [[[UInt8]]]) throws {
        self.size = size
        let attributes: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelBufferWidthKey: size.width,
            kCVPixelBufferHeightKey: size.height,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
        ]
        var session: VTCompressionSession?
        try check(
            VTCompressionSessionCreate(
                allocator: nil, width: Int32(size.width), height: Int32(size.height),
                codecType: kCMVideoCodecType_HEVC, encoderSpecification: nil,
                imageBufferAttributes: attributes as CFDictionary, compressedDataAllocator: nil,
                outputCallback: nil, refcon: nil, compressionSessionOut: &session),
            "VTCompressionSessionCreate")
        guard let session else {
            throw MediaFailure(description: "VTCompressionSessionCreate gave no session")
        }
        self.session = session
        // The encoder at its fastest, without frames held back for B-frames.
        // On an M3 Max, Apple's settings for low latency, which choose another
        // encoder, and real time at 60 frames per second took about twice as
        // long a frame: see BENCHMARKS.md.
        let properties: [(CFString, CFTypeRef)] = [
            (kVTCompressionPropertyKey_RealTime, kCFBooleanFalse),
            (kVTCompressionPropertyKey_MaximizePowerEfficiency, kCFBooleanFalse),
            (kVTCompressionPropertyKey_PrioritizeEncodingSpeedOverQuality, kCFBooleanTrue),
            (kVTCompressionPropertyKey_AllowFrameReordering, kCFBooleanFalse),
            (kVTCompressionPropertyKey_ProfileLevel, kVTProfileLevel_HEVC_Main_AutoLevel),
            (kVTCompressionPropertyKey_AverageBitRate, rate.bitsPerSecond as CFNumber),
            (kVTCompressionPropertyKey_ExpectedFrameRate, streamFrameRate as CFNumber),
            // Keyframes only where asked for, as in a game stream: far more
            // frames apart than are encoded.
            (kVTCompressionPropertyKey_MaxKeyFrameInterval, 100_000 as CFNumber),
            (kVTCompressionPropertyKey_ColorPrimaries, kCVImageBufferColorPrimaries_ITU_R_709_2),
            (
                kVTCompressionPropertyKey_TransferFunction,
                kCVImageBufferTransferFunction_ITU_R_709_2
            ),
            (kVTCompressionPropertyKey_YCbCrMatrix, kCVImageBufferYCbCrMatrix_ITU_R_709_2),
        ]
        var unsupported: [String] = []
        for (key, value) in properties {
            let status = VTSessionSetProperty(session, key: key, value: value)
            if status == kVTPropertyNotSupportedErr {
                unsupported.append(key as String)
            } else {
                try check(status, "\(key)")
            }
        }
        self.unsupported = unsupported
        try check(
            VTCompressionSessionPrepareToEncodeFrames(session),
            "VTCompressionSessionPrepareToEncodeFrames")
        name = property(session, kVTCompressionPropertyKey_EncoderID) as? String ?? "?"
        if #available(iOS 17.4, macOS 10.9, *) {
            hardware =
                property(session, kVTCompressionPropertyKey_UsingHardwareAcceleratedVideoEncoder)
                as? Bool ?? false
        } else {
            hardware = nil
        }
        guard let pool = VTCompressionSessionGetPixelBufferPool(session) else {
            throw MediaFailure(description: "the encoder has no pixel buffer pool")
        }
        self.pool = pool
        let (width, height) = (size.width / 2, size.height / 2)
        self.pictures = pictures.map { planes in
            var chroma = [UInt8](repeating: 0, count: 2 * width * height)
            for index in 0..<(width * height) {
                chroma[2 * index] = planes[1][index]
                chroma[2 * index + 1] = planes[2][index]
            }
            return (planes[0], chroma)
        }
    }

    deinit {
        invalidate()
    }

    private var invalidated = false

    /// Ends the session, which frees the encoder for others.
    func invalidate() {
        if !invalidated {
            VTCompressionSessionInvalidate(session)
            invalidated = true
        }
    }

    /// Encodes the next picture and waits for its frame, timed from taking the
    /// picture from memory into a pixel buffer of the session's until the
    /// frame is out: one frame at a time.
    func encode(keyframe: Bool = false) throws -> (bytes: [UInt8], keyframe: Bool, time: Double) {
        let done = DispatchSemaphore(value: 0)
        // Written by the output handler before it signals `done`.
        let output = Output()
        let options: CFDictionary? =
            keyframe ? [kVTEncodeFrameOptionKey_ForceKeyFrame: true] as CFDictionary : nil
        let start = now()
        var buffer: CVPixelBuffer?
        try check(
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer),
            "CVPixelBufferPoolCreatePixelBuffer")
        guard let buffer else {
            throw MediaFailure(description: "the pool gave no pixel buffer")
        }
        try copy(pictures[picture], to: buffer)
        try check(
            VTCompressionSessionEncodeFrame(
                session, imageBuffer: buffer,
                presentationTimeStamp: CMTime(
                    value: frameNumber, timescale: Int32(streamFrameRate)),
                duration: CMTime(value: 1, timescale: Int32(streamFrameRate)),
                frameProperties: options, infoFlagsOut: nil
            ) { status, flags, sample in
                output.end = now()
                output.status = status
                output.dropped = flags.contains(.frameDropped)
                output.sample = sample
                done.signal()
            }, "VTCompressionSessionEncodeFrame")
        guard done.wait(timeout: .now() + stuck) == .success else {
            invalidate()
            throw MediaFailure(description: "the encoder gave no frame in \(Int(stuck)) s")
        }
        try check(output.status, "the encoder's output")
        guard !output.dropped, let sample = output.sample,
            let data = CMSampleBufferGetDataBuffer(sample)
        else {
            throw MediaFailure(description: "the encoder dropped a frame")
        }
        format = CMSampleBufferGetFormatDescription(sample) ?? format
        var bytes = [UInt8](repeating: 0, count: CMBlockBufferGetDataLength(data))
        try bytes.withUnsafeMutableBytes {
            try check(
                CMBlockBufferCopyDataBytes(
                    data, atOffset: 0, dataLength: $0.count, destination: $0.baseAddress!),
                "CMBlockBufferCopyDataBytes")
        }
        picture = (picture + 1) % pictures.count
        frameNumber += 1
        return (bytes, isKeyframe(sample), output.end - start)
    }

    private final class Output {
        var end = 0.0
        var status: OSStatus = noErr
        var dropped = false
        var sample: CMSampleBuffer?
    }

    /// Copies `picture` into `buffer`, row by row: a pixel buffer's rows may
    /// be padded.
    private func copy(_ picture: (luma: [UInt8], chroma: [UInt8]), to buffer: CVPixelBuffer)
        throws
    {
        guard CVPixelBufferLockBaseAddress(buffer, []) == kCVReturnSuccess else {
            throw MediaFailure(description: "a pixel buffer could not be locked")
        }
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        for (plane, (bytes, rows)) in [
            (picture.luma, size.height), (picture.chroma, size.height / 2),
        ].enumerated() {
            guard let base = CVPixelBufferGetBaseAddressOfPlane(buffer, plane) else {
                throw MediaFailure(description: "a pixel buffer has no plane \(plane)")
            }
            let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, plane)
            // Both planes are as wide as the picture in bytes.
            bytes.withUnsafeBytes { source in
                for row in 0..<rows {
                    (base + row * stride).copyMemory(
                        from: source.baseAddress! + row * size.width, byteCount: size.width)
                }
            }
        }
    }

    /// The parameter sets of the stream, made into a format again as a client
    /// makes it from those it receives with a keyframe.
    func parameterSets() throws -> CMVideoFormatDescription {
        guard let format else {
            throw MediaFailure(description: "no frame encoded yet")
        }
        var count = 0
        var lengthSize: Int32 = 0
        try check(
            CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
                format, parameterSetIndex: 0, parameterSetPointerOut: nil,
                parameterSetSizeOut: nil, parameterSetCountOut: &count,
                nalUnitHeaderLengthOut: &lengthSize),
            "CMVideoFormatDescriptionGetHEVCParameterSetAtIndex")
        var sets: [[UInt8]] = []
        for index in 0..<count {
            var pointer: UnsafePointer<UInt8>?
            var length = 0
            try check(
                CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
                    format, parameterSetIndex: index, parameterSetPointerOut: &pointer,
                    parameterSetSizeOut: &length, parameterSetCountOut: nil,
                    nalUnitHeaderLengthOut: nil),
                "CMVideoFormatDescriptionGetHEVCParameterSetAtIndex")
            guard let pointer else {
                throw MediaFailure(description: "parameter set \(index) is missing")
            }
            sets.append(Array(UnsafeBufferPointer(start: pointer, count: length)))
        }
        return try withPointers(sets) { pointers in
            var description: CMFormatDescription?
            try check(
                CMVideoFormatDescriptionCreateFromHEVCParameterSets(
                    allocator: nil, parameterSetCount: sets.count, parameterSetPointers: pointers,
                    parameterSetSizes: sets.map(\.count), nalUnitHeaderLength: lengthSize,
                    extensions: nil, formatDescriptionOut: &description),
                "CMVideoFormatDescriptionCreateFromHEVCParameterSets")
            guard let description else {
                throw MediaFailure(description: "no format from the parameter sets")
            }
            return description
        }
    }
}

/// Calls `body` with pointers to the bytes of `arrays`, valid during the call.
private func withPointers<Result>(
    _ arrays: [[UInt8]], _ body: ([UnsafePointer<UInt8>]) throws -> Result
) rethrows -> Result {
    func nest(_ index: Int, _ pointers: [UnsafePointer<UInt8>]) throws -> Result {
        guard index < arrays.count else {
            return try body(pointers)
        }
        return try arrays[index].withUnsafeBufferPointer {
            try nest(index + 1, pointers + [$0.baseAddress!])
        }
    }
    return try nest(0, [])
}

/// Whether `sample` can be decoded without the frames before it.
private func isKeyframe(_ sample: CMSampleBuffer) -> Bool {
    guard
        let attachments = CMSampleBufferGetSampleAttachmentsArray(
            sample, createIfNecessary: false) as? [[CFString: Any]],
        let first = attachments.first
    else {
        return true
    }
    return !(first[kCMSampleAttachmentKey_NotSync] as? Bool ?? false)
}

/// An HEVC decoding session that gives its pictures in `pixelFormat`, in
/// IOSurfaces that it wraps in Metal textures, as a player that draws them
/// does: NV12 as an R8 texture of Y and an RG8 texture of Cb and Cr, BGRA as
/// one texture.
private final class HEVCDecoder {
    let session: VTDecompressionSession
    /// Whether the decoder is the hardware's, `nil` where the system does not
    /// say.
    let hardware: Bool?
    /// The pictures of the last frames timed, the last first, as many as there
    /// are pictures of the scene, kept with their textures as a player keeps
    /// what it shows.
    private(set) var last: [CVImageBuffer] = []
    private let format: CMVideoFormatDescription
    private let pixelFormat: OSType
    private let textureCache: CVMetalTextureCache
    private var shown: [CVMetalTexture] = []

    init(_ format: CMVideoFormatDescription, _ pixelFormat: OSType, _ device: MTLDevice) throws {
        self.format = format
        self.pixelFormat = pixelFormat
        var cache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(nil, nil, device, nil, &cache) == kCVReturnSuccess,
            let cache
        else {
            throw MediaFailure(description: "no Metal texture cache")
        }
        textureCache = cache
        let attributes: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: pixelFormat,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
            kCVPixelBufferMetalCompatibilityKey: true,
        ]
        var session: VTDecompressionSession?
        try check(
            VTDecompressionSessionCreate(
                allocator: nil, formatDescription: format, decoderSpecification: nil,
                imageBufferAttributes: attributes as CFDictionary, outputCallback: nil,
                decompressionSessionOut: &session),
            "VTDecompressionSessionCreate")
        guard let session else {
            throw MediaFailure(description: "VTDecompressionSessionCreate gave no session")
        }
        self.session = session
        if #available(iOS 17, macOS 10.9, *) {
            hardware =
                property(
                    session, kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder)
                as? Bool ?? false
        } else {
            hardware = nil
        }
    }

    deinit {
        invalidate()
    }

    private var invalidated = false

    /// Ends the session, which frees the decoder for others.
    func invalidate() {
        if !invalidated {
            VTDecompressionSessionInvalidate(session)
            invalidated = true
        }
    }

    /// Decodes `frame` from its bytes, as a client does with a frame it has
    /// received, asynchronously as Moonlight does, and waits for the picture:
    /// the time from handing over the bytes until the output handler has
    /// wrapped the picture in Metal textures.
    func decode(_ frame: [UInt8]) throws -> (picture: CVImageBuffer, time: Double) {
        let done = DispatchSemaphore(value: 0)
        // Written by the output handler before it signals `done`.
        let output = Output()
        let start = now()
        try frame.withUnsafeBytes { bytes in
            var block: CMBlockBuffer?
            // The frame's bytes, not copied.
            try check(
                CMBlockBufferCreateWithMemoryBlock(
                    allocator: nil,
                    memoryBlock: UnsafeMutableRawPointer(mutating: bytes.baseAddress!),
                    blockLength: bytes.count, blockAllocator: kCFAllocatorNull,
                    customBlockSource: nil, offsetToData: 0, dataLength: bytes.count, flags: 0,
                    blockBufferOut: &block),
                "CMBlockBufferCreateWithMemoryBlock")
            var sample: CMSampleBuffer?
            var length = bytes.count
            try check(
                CMSampleBufferCreateReady(
                    allocator: nil, dataBuffer: block, formatDescription: format,
                    sampleCount: 1, sampleTimingEntryCount: 0, sampleTimingArray: nil,
                    sampleSizeEntryCount: 1, sampleSizeArray: &length, sampleBufferOut: &sample),
                "CMSampleBufferCreateReady")
            guard let sample else {
                throw MediaFailure(description: "CMSampleBufferCreateReady gave no sample")
            }
            try check(
                VTDecompressionSessionDecodeFrame(
                    session, sampleBuffer: sample, flags: ._EnableAsynchronousDecompression,
                    infoFlagsOut: nil
                ) { status, flags, image, _, _ in
                    output.status = status
                    output.dropped = flags.contains(.frameDropped)
                    if status == noErr, let image {
                        output.picture = image
                        output.textures = self.wrap(image)
                    }
                    output.end = now()
                    done.signal()
                }, "VTDecompressionSessionDecodeFrame")
            // The bytes must outlive the decoding of the frame: a stuck
            // decoder is ended before they go.
            guard done.wait(timeout: .now() + stuck) == .success else {
                invalidate()
                throw MediaFailure(description: "the decoder gave no picture in \(Int(stuck)) s")
            }
        }
        try check(output.status, "the decoder's output")
        guard !output.dropped, let picture = output.picture, let textures = output.textures
        else {
            throw MediaFailure(description: "the decoder gave no picture")
        }
        shown = textures
        // Frees the textures of pictures no longer shown, as a player does.
        CVMetalTextureCacheFlush(textureCache, 0)
        return (picture, output.end - start)
    }

    private final class Output {
        var end = 0.0
        var status: OSStatus = noErr
        var dropped = false
        var picture: CVImageBuffer?
        var textures: [CVMetalTexture]?
    }

    /// Metal textures of `image`'s planes, without a copy; `nil` if Metal
    /// cannot read them.
    private func wrap(_ image: CVImageBuffer) -> [CVMetalTexture]? {
        let planes: [(MTLPixelFormat, Int)] =
            pixelFormat == kCVPixelFormatType_32BGRA
            ? [(.bgra8Unorm, 0)] : [(.r8Unorm, 0), (.rg8Unorm, 1)]
        var textures: [CVMetalTexture] = []
        for (format, plane) in planes {
            let width = CVPixelBufferIsPlanar(image)
                ? CVPixelBufferGetWidthOfPlane(image, plane) : CVPixelBufferGetWidth(image)
            let height = CVPixelBufferIsPlanar(image)
                ? CVPixelBufferGetHeightOfPlane(image, plane) : CVPixelBufferGetHeight(image)
            var texture: CVMetalTexture?
            guard
                CVMetalTextureCacheCreateTextureFromImage(
                    nil, textureCache, image, nil, format, width, height, plane, &texture)
                    == kCVReturnSuccess,
                let texture, CVMetalTextureGetTexture(texture) != nil
            else {
                return nil
            }
            textures.append(texture)
        }
        return textures
    }

    /// Times the decoding of `frames`, a keyframe and the frames after it: a
    /// warm-up over them in order, untimed, then the keyframe, untimed, which
    /// the decoder can take after any frame, and the others, each timed; one
    /// after the other, or with `paced` as a stream's frames arrive, the
    /// warm-up too.
    func time(_ frames: [[UInt8]], _ plan: Plan, paced: Bool = false) throws -> Stats {
        var clock = FrameClock()
        let warm = now()
        var index = 0
        while index < plan.warmupFrames || now() - warm < plan.warmupMs {
            if paced {
                clock.wait()
            }
            _ = try decode(frames[index % frames.count])
            index += 1
        }
        if paced {
            clock.wait()
        }
        _ = try decode(frames[0])
        var times: [Double] = []
        last = []
        for frame in frames.dropFirst() {
            if paced {
                clock.wait()
            }
            let (picture, time) = try decode(frame)
            times.append(time)
            last.insert(picture, at: 0)
            if last.count > 4 {
                last.removeLast()
            }
        }
        // The frames' own times: the loop also keeps the pictures, and paced
        // it mostly waits.
        return Stats(times, total: times.reduce(0, +))
    }
}

/// The sum of the squared differences of the luma of `image`, NV12, from
/// `luma`, the picture's Y plane.
private func lumaSquaredError(_ image: CVImageBuffer, _ luma: [UInt8], _ size: PictureSize)
    throws -> Double
{
    guard CVPixelBufferLockBaseAddress(image, .readOnly) == kCVReturnSuccess else {
        throw MediaFailure(description: "a decoded picture could not be locked")
    }
    defer { CVPixelBufferUnlockBaseAddress(image, .readOnly) }
    guard let plane = CVPixelBufferGetBaseAddressOfPlane(image, 0) else {
        throw MediaFailure(description: "a decoded picture has no luma plane")
    }
    let stride = CVPixelBufferGetBytesPerRowOfPlane(image, 0)
    let decoded = plane.assumingMemoryBound(to: UInt8.self)
    var sum = 0.0
    for row in 0..<size.height {
        var rowSum = 0
        for column in 0..<size.width {
            let difference = Int(decoded[row * stride + column]) - Int(luma[row * size.width + column])
            rowSum += difference * difference
        }
        sum += Double(rowSum)
    }
    return sum
}

/// The PSNR in dB of 8-bit samples whose squared differences sum to
/// `squaredError`.
func psnr(squaredError: Double, samples: Double) -> Double {
    squaredError == 0 ? .infinity : 10 * log10(255 * 255 * samples / squaredError)
}
