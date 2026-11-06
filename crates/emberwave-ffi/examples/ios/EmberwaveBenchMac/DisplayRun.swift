// The Mac's display-loop measurement (issue #81, `-display`): decodes at a
// stream's pace, 60 frames a second, as in the report's other paced rows, but
// with a display loop (DisplayLoop.swift) drawing the last picture over a
// screen at the screen's refresh, 60 and 120 Hz where the screen has both. Each
// way is also run without the loop, in the same run, so that the loop's cost is
// the difference and not a comparison with another day. Besides the time to
// decode, it reports the time from the frame's arrival until the display
// presented it.

import AppKit
import Emberwave
import Foundation
import IOKit.pwr_mgt
import Metal

extension Emberwave {
    /// Encodes `pictures`, of `size`, on the GPU within its budget: the frames
    /// of a stream.
    func encodeFrames(_ size: PictureSize, _ pictures: [[[UInt8]]]) throws -> [[UInt8]] {
        var encoder: OpaquePointer?
        try check(
            emberwave_encoder_create(
                context, UInt32(size.width), UInt32(size.height), UInt32(EMBERWAVE_CHROMA_420), nil,
                &encoder),
            "emberwave_encoder_create")
        defer { emberwave_encoder_destroy(encoder) }
        var output = [UInt8](repeating: 0, count: size.budget)
        return try pictures.map { picture in
            var written = 0
            try picture[0].withUnsafeBufferPointer { y in
                try picture[1].withUnsafeBufferPointer { cb in
                    try picture[2].withUnsafeBufferPointer { cr in
                        try output.withUnsafeMutableBufferPointer { frame in
                            try check(
                                emberwave_encoder_encode(
                                    encoder,
                                    EmberwaveSourcePlane(data: y.baseAddress, stride: size.width, size: y.count),
                                    EmberwaveSourcePlane(
                                        data: cb.baseAddress, stride: size.width / 2, size: cb.count),
                                    EmberwaveSourcePlane(
                                        data: cr.baseAddress, stride: size.width / 2, size: cr.count),
                                    frame.baseAddress, size.budget, &written),
                                "emberwave_encoder_encode")
                        }
                    }
                }
            }
            return Array(output[..<written])
        }
    }

    /// Decodes `frames` as a stream's arrive, into textures of `layout` kept
    /// three in turn, the warm-up too, handing each picture to `sink`. The
    /// time of a frame is from its arrival until the call returned with the
    /// GPU done, as `time(_:paced:_:)` has it.
    func pacedDecode(
        _ size: PictureSize, _ plan: Plan, _ frames: [[UInt8]], _ layout: PictureLayout,
        sink: PictureSink?
    ) throws -> Stats {
        guard let device else {
            throw EmberwaveFailure(description: "no Metal device")
        }
        var decoder: OpaquePointer?
        try check(
            emberwave_decoder_create(
                context, UInt32(size.width), UInt32(size.height), UInt32(EMBERWAVE_CHROMA_420),
                &decoder),
            "emberwave_decoder_create")
        defer { emberwave_decoder_destroy(decoder) }

        var owned: [OpaquePointer?] = []
        defer {
            for texture in owned {
                emberwave_texture_destroy(texture)
            }
        }
        var sets: [[OpaquePointer?]] = []
        var metal: [[MTLTexture]] = []
        for _ in 0..<3 {
            var set: [OpaquePointer?] = []
            let planes = layout == .rgb ? [nil] : [EMBERWAVE_PLANE_Y, EMBERWAVE_PLANE_CB, EMBERWAVE_PLANE_CR]
            for plane in planes {
                var texture: OpaquePointer?
                if let plane {
                    try check(
                        emberwave_texture_create_plane(decoder, UInt32(plane), &texture),
                        "emberwave_texture_create_plane")
                } else {
                    try check(
                        emberwave_texture_create(decoder, UInt32(EMBERWAVE_TEXTURE_BGRA8), &texture),
                        "emberwave_texture_create")
                }
                owned.append(texture)
                set.append(texture)
            }
            sets.append(set)
            metal.append(try set.map { try metalTexture($0, on: device, "display texture") })
        }

        func step(_ index: Int, timed: Bool) throws -> Double {
            let start = now()
            emberwave_decoder_clear(decoder)
            try frames[index % frames.count].withUnsafeBufferPointer {
                try check(
                    emberwave_decoder_push_packet(decoder, $0.baseAddress, $0.count),
                    "emberwave_decoder_push_packet")
            }
            let set = sets[index % sets.count]
            if layout == .rgb {
                try check(
                    emberwave_decoder_decode_texture(decoder, nil, set[0]),
                    "emberwave_decoder_decode_texture")
            } else {
                try check(
                    emberwave_decoder_decode_plane_textures(decoder, set[0], set[1], set[2]),
                    "emberwave_decoder_decode_plane_textures")
            }
            let end = now()
            sink?(start, layout, metal[index % metal.count], timed)
            return end - start
        }

        var clock = FrameClock(rate: displayStreamRate)
        let warm = now()
        var index = 0
        while index < plan.warmupFrames || now() - warm < plan.warmupMs {
            clock.wait()
            _ = try step(index, timed: false)
            index += 1
        }
        var times: [Double] = []
        for offset in 0..<plan.frames {
            clock.wait()
            times.append(try step(index + offset, timed: true))
        }
        return Stats(times, total: times.reduce(0, +))
    }
}

/// Frames a second in this run: a little under the stream's 60, so that the
/// frames drift across the refreshes (see `FrameClock.rate`), a full turn
/// about every two seconds.
private let displayStreamRate = 59.5

/// Frames in one beat between that stream and a 60 Hz loop: 2 s, 119 frames. A
/// run times whole beats, so that where the frames fall against the refreshes
/// averages out the same in every run.
let displayBeatFrames = 119

/// One way of decoding to measure with and without a display loop.
private struct Way {
    let name: String
    let run: (PictureSink?) throws -> Stats
}

/// Runs the whole measurement and returns its report. Call off the main
/// thread, which has to run the application's loop.
func runDisplayBenchmark(_ plan: Plan, progress: (String) -> Void) throws -> String {
    // The displays have to stay on, and be on: an unattended Mac sleeps them,
    // and then nothing reaches a screen.
    let activity = ProcessInfo.processInfo.beginActivity(
        options: [.idleDisplaySleepDisabled, .idleSystemSleepDisabled, .userInitiated],
        reason: "emberwave display-loop benchmark")
    defer { ProcessInfo.processInfo.endActivity(activity) }
    var wake: IOPMAssertionID = 0
    IOPMAssertionDeclareUserActivity(
        "emberwave display-loop benchmark" as CFString, kIOPMUserActiveLocal, &wake)
    defer { IOPMAssertionRelease(wake) }
    let emberwave = try Emberwave()
    guard let device = emberwave.device else {
        throw EmberwaveFailure(description: "no Metal device")
    }
    // A loop on each kind of screen: the screens' refresh rates, 60 Hz
    // and 120 Hz, which the Mac may have more than one of.
    let screens = DispatchQueue.main.sync { DisplayLoop.screens() }
    var loops: [(rate: Int, loop: DisplayLoop, how: String)] = []
    let isBuiltin = { (screen: NSScreen) -> Bool in
        let id = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID ?? 0
        return CGDisplayIsBuiltin(id) != 0
    }
    for rate in [60, 120] {
        let candidates = screens.filter { $0.maximumFramesPerSecond >= rate }
        // The slowest such screen, the built-in one of equals.
        guard
            let screen = candidates.min(by: {
                ($0.maximumFramesPerSecond, isBuiltin($0) ? 0 : 1)
                    < ($1.maximumFramesPerSecond, isBuiltin($1) ? 0 : 1)
            })
        else {
            continue
        }
        var loop = try DispatchQueue.main.sync { try DisplayLoop(device: device, screen: screen) }
        var how = "\(loop.screenName) at \(loop.maximumRate) Hz"
        // A screen that takes no window, off or on another input, is left for
        // every n-th refresh of a faster one.
        let takes = DispatchQueue.main.sync { () -> Bool in
            _ = loop.start(rate: rate)
            defer { _ = loop.stop() }
            return loop.isOnScreen
        }
        if !takes, let faster = screens.first(where: { $0.maximumFramesPerSecond > rate && isBuiltin($0) }) {
            loop = try DispatchQueue.main.sync { try DisplayLoop(device: device, screen: faster) }
            how = "every \(loop.maximumRate / rate)nd refresh of \(loop.screenName) at \(loop.maximumRate) Hz"
        }
        loops.append((rate, loop, how))
    }
    var lines = [
        "emberwave display-loop benchmark, \(ISO8601DateFormatter().string(from: Date()))",
        "Device: \(deviceDescription()), \(ProcessInfo.processInfo.operatingSystemVersionString)",
        "GPU: \(emberwave.adapter); emberwave \(String(cString: emberwave_version()))",
        "Screens: " + loops.map {
            "\($0.rate) Hz: \($0.how), drawn at "
                + "\(Int($0.loop.drawableSize.width))x\(Int($0.loop.drawableSize.height))"
        }.joined(separator: "; ")
            + "; Low Power Mode: \(ProcessInfo.processInfo.isLowPowerModeEnabled ? "on" : "off")",
        "\(plan.frames) frames after at least \(plan.warmupFrames) and \(Int(plan.warmupMs)) ms of warm-up, "
            + "\(plan.pictures) pictures in turn, arriving \(displayStreamRate) a second, a little under a stream's "
            + "\(streamFrameRate) so that they drift across the refreshes; HEVC at Moonlight's rate",
        "Per frame (ms): mean / median / 99th percentile. decode: from the frame's arrival until "
            + "its picture is in textures; to screen: from the frame's arrival until the display "
            + "presented it (presentedTime of its drawable, the first refresh that drew it). "
            + "not drawn: timed frames no refresh drew, because a newer one came first (the last "
            + "frame has a refresh to be drawn at); not presented: drawn, but the display never "
            + "presented the drawable. display link: the median time between its calls, and how "
            + "many came more than half an interval late, from the first timed frame on; the "
            + "calls' times carry the link thread's own jitter, so a few late ones say little. "
            + "The frames are whole beats of the stream's drift against a 60 Hz loop. Without a "
            + "loop, the window is there, black, with nothing drawn in it.",
    ]
    for size in pictureSizes {
        progress("\(size.name): making \(plan.pictures) test pictures")
        let pictures = makePictures(size, plan)
        let frames = try emberwave.encodeFrames(size, pictures)
        let rate = hevcRates(size)[0]
        progress("\(size.name): encoding HEVC at \(rate.bitsPerSecond / 1_000_000) Mbit/s")
        // A Mac without HEVC, as a virtual machine may be, is not a failure; anything
        // else is.
        let stream: HEVCStream?
        do {
            stream = try makeHEVCStream(size, rate, pictures, plan)
        } catch let failure as MediaFailure where failure.unavailable {
            stream = nil
        }
        if stream == nil {
            lines.append("\(size.name): HEVC not measured, VideoToolbox could not make the stream")
        }
        var ways = [
            Way(name: "emberwave into an RGB texture") {
                try emberwave.pacedDecode(size, plan, frames, .rgb, sink: $0)
            },
            Way(name: "emberwave into Y, Cb, Cr textures") {
                try emberwave.pacedDecode(size, plan, frames, .planes, sink: $0)
            },
        ]
        if let stream {
            ways.append(
                Way(name: "HEVC into NV12 textures") {
                    try hevcPaced(stream, plan, rate: displayStreamRate, device: device, sink: $0).stats
                })
        }
        lines.append("\(size.name), thermal state \(thermalState()):")
        for refresh in [nil] + loops.map({ Optional($0.rate) }) {
            lines.append(refresh.map { "  display loop at \($0) Hz:" } ?? "  no display loop:")
            // Without a loop the window is there, black, and nothing draws in it.
            let idle = refresh == nil ? loops.last?.loop : nil
            if let idle {
                DispatchQueue.main.sync { idle.showWindow() }
                progress("idle window: " + DispatchQueue.main.sync { idle.windowState })
            }
            defer {
                if let idle {
                    DispatchQueue.main.sync { idle.hideWindow() }
                }
            }
            for way in ways {
                progress("\(size.name): \(way.name), \(refresh.map { "display \($0) Hz" } ?? "no display")")
                guard let refresh, let loop = loops.first(where: { $0.rate == refresh })?.loop else {
                    lines.append(row(way.name, "decode \(try way.run(nil))", indent: 4))
                    continue
                }
                _ = DispatchQueue.main.sync { loop.start(rate: refresh) }
                guard DispatchQueue.main.sync(execute: { loop.isOnScreen }) else {
                    let state = DispatchQueue.main.sync { loop.windowState }
                    _ = DispatchQueue.main.sync { loop.stop() }
                    throw EmberwaveFailure(
                        description: "the display loop's window never reached the screen (is it asleep?): "
                            + state)
                }
                let decode: Stats
                do {
                    decode = try way.run(loop.offer)
                } catch {
                    _ = DispatchQueue.main.sync { loop.stop() }
                    throw error
                }
                let seen = DispatchQueue.main.sync { loop.stop() }
                var text = "decode \(decode)"
                if seen.toScreen.isEmpty {
                    text += "; nothing reached the screen"
                } else {
                    text += "; to screen \(Stats(seen.toScreen, total: seen.toScreen.reduce(0, +)))"
                }
                text += "; not drawn \(seen.skipped) of \(seen.offered), not presented \(seen.notPresented)"
                text += String(
                    format: "; display link every %.2f ms (%.0f Hz), %d of %d calls late",
                    seen.interval, 1000 / max(seen.interval, 0.001), seen.late, seen.ticks)
                text += "; no picture \(seen.noPicture), no drawable \(seen.noDrawable)"
                lines.append(row(way.name, text, indent: 4))
            }
        }
    }
    return lines.joined(separator: "\n")
}
