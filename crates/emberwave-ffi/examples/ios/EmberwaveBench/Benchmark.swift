// Times emberwave on this device's GPU through its C API, as an app uses it:
// encoding pictures from CPU memory, decoding frames into Metal textures, into
// Y, Cb and Cr textures, on the app's own queue, and to planes in CPU memory.
// Each frame is timed from handing over its bytes until the call returns with
// the GPU done, or, on the app's queue, where the call does not wait, until a
// command buffer of the app's that reads the picture has completed.

import Emberwave
import Foundation
import Metal

/// A failed call of the C API, with emberwave's description of why.
struct EmberwaveFailure: Error, CustomStringConvertible {
    let description: String
}

/// Throws unless `status` is `EMBERWAVE_STATUS_OK`.
func check(_ status: EmberwaveStatus, _ call: String) throws {
    if status != EMBERWAVE_STATUS_OK {
        throw EmberwaveFailure(description: "\(call): \(String(cString: emberwave_last_error()))")
    }
}

/// Milliseconds on the monotonic clock.
func now() -> Double {
    Double(DispatchTime.now().uptimeNanoseconds) / 1e6
}

/// Mean, median and 99th percentile (nearest rank) of per-frame times in
/// milliseconds, the mean from the time of the whole run.
struct Stats: CustomStringConvertible {
    let mean: Double
    let median: Double
    let p99: Double

    init(_ times: [Double], total: Double) {
        let sorted = times.sorted()
        func rank(_ percent: Double) -> Double {
            let index = Int((percent / 100 * Double(sorted.count)).rounded(.up))
            return sorted[min(max(index, 1), sorted.count) - 1]
        }
        mean = total / Double(times.count)
        median = rank(50)
        p99 = rank(99)
    }

    var description: String {
        String(format: "%.2f / %.2f / %.2f", mean, median, p99)
    }
}

/// A picture size and the bytes a frame of it gets: those of the native
/// benchmarks (BENCHMARKS.md) and of the browser's benchmark page.
struct PictureSize {
    let name: String
    let width: Int
    let height: Int
    let budget: Int
}

let pictureSizes = [
    PictureSize(name: "1080p", width: 1920, height: 1080, budget: 400_000),
    PictureSize(name: "1440p", width: 2560, height: 1440, budget: 700_000),
    PictureSize(name: "4K", width: 3840, height: 2160, budget: 1_200_000),
]

/// How long the benchmark measures.
struct Plan {
    /// Different pictures, encoded and decoded in turn.
    var pictures = 4
    /// Frames timed for each measurement.
    var frames = 300
    /// At least this many frames and milliseconds first, not timed: a GPU
    /// that has been idle runs the first moments of a load slower.
    var warmupFrames = 30
    var warmupMs = 1000.0
}

/// What one picture size gave.
struct SizeResult {
    let size: PictureSize
    let bytesPerFrame: Int
    let texture: Stats
    let planeTextures: Stats
    /// `nil` without a context on the app's queue.
    let appQueue: Stats?
    let planes: Stats
    let encode: Stats
    /// The last picture, decoded once more into a texture the app keeps.
    let picture: MTLTexture?
}

/// An emberwave context: the GPU, and its Metal device. Beside it, a context
/// on a command queue of the app's own on that device, as an app that draws
/// with Metal makes one (`emberwave_context_create_metal`).
final class Emberwave {
    let context: OpaquePointer
    let adapter: String
    let device: MTLDevice?
    /// The app's queue and the context on it, `nil` off Metal or if the
    /// context could not be made, which `appContextFailure` then says.
    let appQueue: MTLCommandQueue?
    let appContext: OpaquePointer?
    let appContextFailure: String?

    init() throws {
        var context: OpaquePointer?
        try check(emberwave_context_create(nil, &context), "emberwave_context_create")
        guard let context else {
            throw EmberwaveFailure(description: "emberwave_context_create gave no context")
        }
        self.context = context
        adapter = String(cString: emberwave_context_adapter(context))
        // Borrowed from the context, which this object keeps alive.
        let device = emberwave_context_metal_device(context).map {
            Unmanaged<MTLDevice>.fromOpaque($0).takeUnretainedValue()
        }
        self.device = device

        var appContext: OpaquePointer?
        var failure: String? = "no Metal device"
        let appQueue = device?.makeCommandQueue()
        if let device, let appQueue {
            // emberwave retains both for its context.
            do {
                try check(
                    emberwave_context_create_metal(
                        Unmanaged.passUnretained(device as AnyObject).toOpaque(),
                        Unmanaged.passUnretained(appQueue as AnyObject).toOpaque(), &appContext),
                    "emberwave_context_create_metal")
                failure = nil
            } catch {
                failure = "\(error)"
            }
        }
        self.appQueue = appQueue
        self.appContext = appContext
        appContextFailure = failure
    }

    deinit {
        emberwave_context_destroy(appContext)
        emberwave_context_destroy(context)
    }

    /// Times `step` with 0, 1, 2 and so on: first a warm-up, then
    /// `plan.frames` frames.
    private func time(_ plan: Plan, _ step: (Int) throws -> Void) throws -> Stats {
        let warm = now()
        var index = 0
        while index < plan.warmupFrames || now() - warm < plan.warmupMs {
            try step(index)
            index += 1
        }
        var times: [Double] = []
        let first = now()
        for offset in 0..<plan.frames {
            let start = now()
            try step(index + offset)
            times.append(now() - start)
        }
        return Stats(times, total: now() - first)
    }

    /// Makes the pictures of `size`, encodes them on the GPU within its
    /// budget, and times the encoding and the decoding of the frames, into
    /// textures and to planes in CPU memory.
    func run(_ size: PictureSize, _ plan: Plan, progress: (String) -> Void) throws -> SizeResult {
        progress("\(size.name): making \(plan.pictures) test pictures")
        let pictures = (0..<plan.pictures).map {
            scene(width: size.width, height: size.height, index: $0)
        }
        let (width, height) = (UInt32(size.width), UInt32(size.height))
        let chroma = UInt32(EMBERWAVE_CHROMA_420)

        var encoder: OpaquePointer?
        try check(
            emberwave_encoder_create(context, width, height, chroma, nil, &encoder),
            "emberwave_encoder_create")
        defer { emberwave_encoder_destroy(encoder) }
        var output = [UInt8](repeating: 0, count: size.budget)
        func encode(_ index: Int) throws -> [UInt8] {
            let picture = pictures[index % pictures.count]
            var written = 0
            try picture[0].withUnsafeBufferPointer { y in
                try picture[1].withUnsafeBufferPointer { cb in
                    try picture[2].withUnsafeBufferPointer { cr in
                        try output.withUnsafeMutableBufferPointer { frame in
                            try check(
                                emberwave_encoder_encode(
                                    encoder,
                                    EmberwaveSourcePlane(data: y.baseAddress, stride: size.width),
                                    EmberwaveSourcePlane(
                                        data: cb.baseAddress, stride: size.width / 2),
                                    EmberwaveSourcePlane(
                                        data: cr.baseAddress, stride: size.width / 2),
                                    frame.baseAddress, size.budget, &written),
                                "emberwave_encoder_encode")
                        }
                    }
                }
            }
            return Array(output[..<written])
        }
        progress("\(size.name): encoding")
        let frames = try (0..<plan.pictures).map(encode)
        let encodeTimes = try time(plan) { _ = try encode($0) }

        var decoder: OpaquePointer?
        try check(
            emberwave_decoder_create(context, width, height, chroma, &decoder),
            "emberwave_decoder_create")
        defer { emberwave_decoder_destroy(decoder) }
        // Gives frame `index` of the stream to the decoder, as a fresh start:
        // the frames repeat, so their numbers do not go on.
        func push(_ index: Int) throws {
            emberwave_decoder_clear(decoder)
            try frames[index % frames.count].withUnsafeBufferPointer {
                try check(
                    emberwave_decoder_push_packet(decoder, $0.baseAddress, $0.count),
                    "emberwave_decoder_push_packet")
            }
        }

        // Three textures in turn, as a player keeps them (see the C API's
        // README): nothing else reads them here, but that is the pattern.
        var textures: [OpaquePointer?] = []
        defer {
            for texture in textures {
                emberwave_texture_destroy(texture)
            }
        }
        for _ in 0..<3 {
            var texture: OpaquePointer?
            try check(
                emberwave_texture_create(decoder, UInt32(EMBERWAVE_TEXTURE_BGRA8), &texture),
                "emberwave_texture_create")
            textures.append(texture)
        }
        progress("\(size.name): decoding into textures")
        let textureTimes = try time(plan) { index in
            try push(index)
            try check(
                emberwave_decoder_decode_texture(decoder, nil, textures[index % textures.count]),
                "emberwave_decoder_decode_texture")
        }

        // Y, Cb and Cr, three sets in turn as well.
        var planeSets: [[OpaquePointer?]] = []
        defer {
            for texture in planeSets.joined() {
                emberwave_texture_destroy(texture)
            }
        }
        for _ in 0..<3 {
            var set: [OpaquePointer?] = []
            for plane in [EMBERWAVE_PLANE_Y, EMBERWAVE_PLANE_CB, EMBERWAVE_PLANE_CR] {
                var texture: OpaquePointer?
                try check(
                    emberwave_texture_create_plane(decoder, UInt32(plane), &texture),
                    "emberwave_texture_create_plane")
                set.append(texture)
            }
            planeSets.append(set)
        }
        progress("\(size.name): decoding into Y, Cb and Cr textures")
        let planeTextureTimes = try time(plan) { index in
            try push(index)
            let set = planeSets[index % planeSets.count]
            try check(
                emberwave_decoder_decode_plane_textures(decoder, set[0], set[1], set[2]),
                "emberwave_decoder_decode_plane_textures")
        }

        progress("\(size.name): decoding on the app's queue")
        let appQueueTimes = try timeOnAppQueue(size, plan, frames)

        let (cw, ch) = (size.width / 2, size.height / 2)
        // Three arrays, not one of three: Swift gives a closure exclusive
        // access to the whole of the array whose element it changes.
        var lumaPlane = [UInt8](repeating: 0, count: size.width * size.height)
        var cbPlane = [UInt8](repeating: 0, count: cw * ch)
        var crPlane = [UInt8](repeating: 0, count: cw * ch)
        progress("\(size.name): decoding to planes in CPU memory")
        let planeTimes = try time(plan) { index in
            try push(index)
            try lumaPlane.withUnsafeMutableBufferPointer { y in
                try cbPlane.withUnsafeMutableBufferPointer { cb in
                    try crPlane.withUnsafeMutableBufferPointer { cr in
                        try check(
                            emberwave_decoder_decode_yuv(
                                decoder,
                                EmberwavePlane(data: y.baseAddress, stride: size.width),
                                EmberwavePlane(data: cb.baseAddress, stride: cw),
                                EmberwavePlane(data: cr.baseAddress, stride: cw)),
                            "emberwave_decoder_decode_yuv")
                    }
                }
            }
        }

        // The last picture for the screen, in a texture of its own that the
        // app retains past emberwave's.
        var shown: OpaquePointer?
        try check(
            emberwave_texture_create(decoder, UInt32(EMBERWAVE_TEXTURE_BGRA8), &shown),
            "emberwave_texture_create")
        defer { emberwave_texture_destroy(shown) }
        try push(frames.count - 1)
        try check(
            emberwave_decoder_decode_texture(decoder, nil, shown),
            "emberwave_decoder_decode_texture")
        let picture = emberwave_texture_metal(shown).map {
            Unmanaged<MTLTexture>.fromOpaque($0).takeUnretainedValue()
        }

        return SizeResult(
            size: size,
            bytesPerFrame: frames.map(\.count).reduce(0, +) / frames.count,
            texture: textureTimes,
            planeTextures: planeTextureTimes,
            appQueue: appQueueTimes,
            planes: planeTimes,
            encode: encodeTimes,
            picture: picture)
    }

    /// Times `frames` decoded on the app's queue into one texture, each from
    /// the call until a command buffer of the app's that reads the picture, a
    /// copy of one pixel standing in for its draw, has completed. The call
    /// does not wait: the queue runs the app's command buffer after the
    /// decode. `nil` without a context on the app's queue.
    private func timeOnAppQueue(_ size: PictureSize, _ plan: Plan, _ frames: [[UInt8]]) throws
        -> Stats?
    {
        guard let appContext, let appQueue, let device else {
            return nil
        }
        var decoder: OpaquePointer?
        try check(
            emberwave_decoder_create(
                appContext, UInt32(size.width), UInt32(size.height),
                UInt32(EMBERWAVE_CHROMA_420), &decoder),
            "emberwave_decoder_create")
        defer { emberwave_decoder_destroy(decoder) }
        var texture: OpaquePointer?
        try check(
            emberwave_texture_create(decoder, UInt32(EMBERWAVE_TEXTURE_BGRA8), &texture),
            "emberwave_texture_create")
        defer { emberwave_texture_destroy(texture) }
        guard let raw = emberwave_texture_metal(texture),
            let pixel = device.makeBuffer(length: 4, options: .storageModeShared)
        else {
            throw EmberwaveFailure(description: "no Metal texture on the app's queue")
        }
        let metal = Unmanaged<MTLTexture>.fromOpaque(raw).takeUnretainedValue()
        return try time(plan) { index in
            emberwave_decoder_clear(decoder)
            try frames[index % frames.count].withUnsafeBufferPointer {
                try check(
                    emberwave_decoder_push_packet(decoder, $0.baseAddress, $0.count),
                    "emberwave_decoder_push_packet")
            }
            try check(
                emberwave_decoder_submit_texture(decoder, nil, texture),
                "emberwave_decoder_submit_texture")
            guard let commands = appQueue.makeCommandBuffer(),
                let blit = commands.makeBlitCommandEncoder()
            else {
                throw EmberwaveFailure(description: "no command buffer on the app's queue")
            }
            blit.copy(
                from: metal, sourceSlice: 0, sourceLevel: 0,
                sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                sourceSize: MTLSize(width: 1, height: 1, depth: 1), to: pixel,
                destinationOffset: 0, destinationBytesPerRow: 4, destinationBytesPerImage: 4)
            blit.endEncoding()
            commands.commit()
            commands.waitUntilCompleted()
        }
    }

    /// Checks the textures against the decodes to CPU memory, read back
    /// through Metal on queues of the app's own: the picture's texture, the
    /// Y, Cb and Cr textures, and a texture decoded into on the app's queue
    /// and read by the app's next command buffer there. A line each.
    func checkTextures() throws -> String {
        guard let device else {
            return "texture check: no Metal device (\(adapter))"
        }
        let size = PictureSize(name: "check", width: 640, height: 360, budget: 100_000)
        let frame = try checkFrame(size)
        let (width, height) = (UInt32(size.width), UInt32(size.height))
        let chroma = UInt32(EMBERWAVE_CHROMA_420)

        var decoder: OpaquePointer?
        try check(
            emberwave_decoder_create(context, width, height, chroma, &decoder),
            "emberwave_decoder_create")
        defer { emberwave_decoder_destroy(decoder) }
        func push() throws {
            emberwave_decoder_clear(decoder)
            try frame.withUnsafeBufferPointer {
                try check(
                    emberwave_decoder_push_packet(decoder, $0.baseAddress, $0.count),
                    "emberwave_decoder_push_packet")
            }
        }

        // The picture.
        var texture: OpaquePointer?
        try check(
            emberwave_texture_create(decoder, UInt32(EMBERWAVE_TEXTURE_RGBA8), &texture),
            "emberwave_texture_create")
        defer { emberwave_texture_destroy(texture) }
        try push()
        var rgb = [UInt8](repeating: 0, count: size.width * size.height * 3)
        try rgb.withUnsafeMutableBufferPointer {
            try check(
                emberwave_decoder_decode_rgb(
                    decoder, nil, EmberwavePlane(data: $0.baseAddress, stride: size.width * 3)),
                "emberwave_decoder_decode_rgb")
        }
        try push()
        try check(
            emberwave_decoder_decode_texture(decoder, nil, texture),
            "emberwave_decoder_decode_texture")
        let metal = try metalTexture(texture, on: device, "texture check")
        guard metal.pixelFormat == .rgba8Unorm, metal.width == size.width,
            metal.height == size.height
        else {
            throw EmberwaveFailure(
                description:
                    "texture check: \(metal.width)x\(metal.height), format \(metal.pixelFormat.rawValue)"
            )
        }
        let wrong = differences(read(metal, on: device), rgb, bgra: false)
        guard wrong == 0 else {
            throw EmberwaveFailure(
                description: "texture check: \(wrong) pixels differ from the RGB decode")
        }
        var lines = [
            "texture check: ok, \(size.width)x\(size.height) read through Metal equals the RGB decode"
        ]

        // The planes, against those decoded to CPU memory.
        let (cw, ch) = (size.width / 2, size.height / 2)
        var lumaPlane = [UInt8](repeating: 0, count: size.width * size.height)
        var cbPlane = [UInt8](repeating: 0, count: cw * ch)
        var crPlane = [UInt8](repeating: 0, count: cw * ch)
        try push()
        try lumaPlane.withUnsafeMutableBufferPointer { y in
            try cbPlane.withUnsafeMutableBufferPointer { cb in
                try crPlane.withUnsafeMutableBufferPointer { cr in
                    try check(
                        emberwave_decoder_decode_yuv(
                            decoder,
                            EmberwavePlane(data: y.baseAddress, stride: size.width),
                            EmberwavePlane(data: cb.baseAddress, stride: cw),
                            EmberwavePlane(data: cr.baseAddress, stride: cw)),
                        "emberwave_decoder_decode_yuv")
                }
            }
        }
        var planes: [OpaquePointer?] = []
        defer {
            for plane in planes {
                emberwave_texture_destroy(plane)
            }
        }
        for plane in [EMBERWAVE_PLANE_Y, EMBERWAVE_PLANE_CB, EMBERWAVE_PLANE_CR] {
            var texture: OpaquePointer?
            try check(
                emberwave_texture_create_plane(decoder, UInt32(plane), &texture),
                "emberwave_texture_create_plane")
            planes.append(texture)
        }
        try push()
        try check(
            emberwave_decoder_decode_plane_textures(decoder, planes[0], planes[1], planes[2]),
            "emberwave_decoder_decode_plane_textures")
        for (name, (plane, expected)) in zip(
            ["Y", "Cb", "Cr"], zip(planes, [lumaPlane, cbPlane, crPlane]))
        {
            let metal = try metalTexture(plane, on: device, "plane check")
            guard metal.pixelFormat == .r8Unorm, read(metal, on: device) == expected else {
                throw EmberwaveFailure(
                    description: "plane check: the \(name) texture differs from the decoded plane")
            }
        }
        lines.append("plane check: ok, Y, Cb and Cr read through Metal equal the planes decoded to memory")

        lines.append(try checkAppQueue(frame, size, rgb: rgb))
        return lines.joined(separator: "\n")
    }

    /// Decodes `frame` on the app's queue without waiting and reads the
    /// picture with the app's next command buffer on that queue: the queue
    /// alone orders the read after the decode.
    private func checkAppQueue(_ frame: [UInt8], _ size: PictureSize, rgb: [UInt8]) throws
        -> String
    {
        guard let appContext, let appQueue, let device else {
            return "queue check: no context on the app's queue (\(appContextFailure ?? "?"))"
        }
        var decoder: OpaquePointer?
        try check(
            emberwave_decoder_create(
                appContext, UInt32(size.width), UInt32(size.height),
                UInt32(EMBERWAVE_CHROMA_420), &decoder),
            "emberwave_decoder_create")
        defer { emberwave_decoder_destroy(decoder) }
        var texture: OpaquePointer?
        try check(
            emberwave_texture_create(decoder, UInt32(EMBERWAVE_TEXTURE_BGRA8), &texture),
            "emberwave_texture_create")
        defer { emberwave_texture_destroy(texture) }
        try frame.withUnsafeBufferPointer {
            try check(
                emberwave_decoder_push_packet(decoder, $0.baseAddress, $0.count),
                "emberwave_decoder_push_packet")
        }
        try check(
            emberwave_decoder_submit_texture(decoder, nil, texture),
            "emberwave_decoder_submit_texture")
        let metal = try metalTexture(texture, on: device, "queue check")
        // Metal orders the queue's work by the texture only while it tracks it.
        guard metal.hazardTrackingMode == .tracked else {
            throw EmberwaveFailure(description: "queue check: the texture is not tracked")
        }
        let wrong = differences(read(metal, on: device, queue: appQueue), rgb, bgra: true)
        guard wrong == 0 else {
            throw EmberwaveFailure(
                description: "queue check: \(wrong) pixels differ from the RGB decode")
        }
        return "queue check: ok, read by the app's next command buffer, the picture equals the RGB decode"
    }

    /// A frame of a 640x360 test picture, encoded on this device.
    private func checkFrame(_ size: PictureSize) throws -> [UInt8] {
        let picture = scene(width: size.width, height: size.height, index: 1)
        var encoder: OpaquePointer?
        try check(
            emberwave_encoder_create(
                context, UInt32(size.width), UInt32(size.height), UInt32(EMBERWAVE_CHROMA_420),
                nil, &encoder),
            "emberwave_encoder_create")
        defer { emberwave_encoder_destroy(encoder) }
        var frame = [UInt8](repeating: 0, count: size.budget)
        var written = 0
        try picture[0].withUnsafeBufferPointer { y in
            try picture[1].withUnsafeBufferPointer { cb in
                try picture[2].withUnsafeBufferPointer { cr in
                    try frame.withUnsafeMutableBufferPointer { out in
                        try check(
                            emberwave_encoder_encode(
                                encoder,
                                EmberwaveSourcePlane(data: y.baseAddress, stride: size.width),
                                EmberwaveSourcePlane(data: cb.baseAddress, stride: size.width / 2),
                                EmberwaveSourcePlane(data: cr.baseAddress, stride: size.width / 2),
                                out.baseAddress, size.budget, &written),
                            "emberwave_encoder_encode")
                    }
                }
            }
        }
        return Array(frame[..<written])
    }
}

/// The Metal texture of `texture`, which has to be on `device`.
func metalTexture(_ texture: OpaquePointer?, on device: MTLDevice, _ what: String) throws
    -> MTLTexture
{
    guard let raw = emberwave_texture_metal(texture) else {
        throw EmberwaveFailure(description: "\(what): no Metal texture on a Metal context")
    }
    let metal = Unmanaged<MTLTexture>.fromOpaque(raw).takeUnretainedValue()
    guard metal.device.registryID == device.registryID else {
        throw EmberwaveFailure(description: "\(what): the texture is on another device")
    }
    return metal
}

/// How many pixels of `texels`, four bytes each, differ from the RGB pixels
/// `rgb` with alpha 255, in RGBA order or with `bgra` in BGRA order.
func differences(_ texels: [UInt8], _ rgb: [UInt8], bgra: Bool) -> Int {
    var wrong = 0
    for index in 0..<(rgb.count / 3) {
        let (r, g, b) = (rgb[3 * index], rgb[3 * index + 1], rgb[3 * index + 2])
        let expected: [UInt8] = bgra ? [b, g, r, 255] : [r, g, b, 255]
        if texels.count < 4 * index + 4 || Array(texels[(4 * index)..<(4 * index + 4)]) != expected {
            wrong += 1
        }
    }
    return wrong
}

/// The pixels of `texture`, copied into memory the CPU can read by a blit on
/// `queue`, by default a command queue made here, as an app's own would be.
func read(_ texture: MTLTexture, on device: MTLDevice, queue: MTLCommandQueue? = nil) -> [UInt8] {
    let texel = texture.pixelFormat == .r8Unorm ? 1 : 4
    let row = texture.width * texel
    let bytes = row * texture.height
    guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared),
        let queue = queue ?? device.makeCommandQueue(),
        let commands = queue.makeCommandBuffer(),
        let blit = commands.makeBlitCommandEncoder()
    else {
        return []
    }
    blit.copy(
        from: texture, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
        sourceSize: MTLSize(width: texture.width, height: texture.height, depth: 1),
        to: buffer, destinationOffset: 0, destinationBytesPerRow: row,
        destinationBytesPerImage: bytes)
    blit.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    return [UInt8](UnsafeRawBufferPointer(start: buffer.contents(), count: bytes))
}

/// What the device is, for the report.
func deviceDescription() -> String {
    let environment = ProcessInfo.processInfo.environment
    if let simulated = environment["SIMULATOR_MODEL_IDENTIFIER"] {
        return "\(simulated) (simulator)"
    }
    var system = utsname()
    uname(&system)
    return withUnsafeBytes(of: &system.machine) { bytes in
        String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
    }
}

/// The report of a run, for the screen and to copy.
func makeReport(_ emberwave: Emberwave, _ plan: Plan, _ results: [SizeResult], check: String)
    -> String
{
    let formatter = ISO8601DateFormatter()
    let thermal: String
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: thermal = "nominal"
    case .fair: thermal = "fair"
    case .serious: thermal = "serious"
    case .critical: thermal = "critical"
    @unknown default: thermal = "unknown"
    }
    var lines = [
        "emberwave native benchmark, \(formatter.string(from: Date()))",
        "Device: \(deviceDescription()), \(ProcessInfo.processInfo.operatingSystemVersionString)",
        "GPU: \(emberwave.adapter); emberwave \(String(cString: emberwave_version()))",
        "\(plan.frames) frames after at least \(plan.warmupFrames) and \(Int(plan.warmupMs)) ms of warm-up, "
            + "\(plan.pictures) pictures in turn; thermal state at the end: \(thermal); "
            + "Low Power Mode: \(ProcessInfo.processInfo.isLowPowerModeEnabled ? "on" : "off")",
        "Per frame (ms): mean / median / 99th percentile, until the call returns with the GPU "
            + "done; on the app's queue, where the call does not wait, until the app's next "
            + "command buffer has read the picture",
    ]
    for result in results {
        // Rounded, as the browser's report rounds them.
        let kilobytes = Int((Double(result.bytesPerFrame) / 1000).rounded())
        lines.append("\(result.size.name) (\(kilobytes) kB a frame):")
        lines.append("  decode into a Metal texture     \(result.texture)")
        lines.append("  decode into Y, Cb, Cr textures  \(result.planeTextures)")
        if let appQueue = result.appQueue {
            lines.append("  decode on the app's queue       \(appQueue)")
        } else {
            lines.append(
                "  decode on the app's queue       not measured (\(emberwave.appContextFailure ?? "?"))")
        }
        lines.append("  decode to planes in memory      \(result.planes)")
        lines.append("  encode from memory              \(result.encode)")
    }
    lines.append(check)
    return lines.joined(separator: "\n")
}
