// A display loop for the Mac's measurements (issue #81): a borderless window
// over one screen that draws the last decoded picture, scaled to the screen,
// at the screen's refresh, through a CAMetalLayer and a CVDisplayLink, as a
// client's player does. (A CADisplayLink of an app that is not in front stays
// at 60 Hz on a 120 Hz screen; a CVDisplayLink follows the screen.) A paced decode runs beside it, and the loop notes when
// each frame first reaches the screen: the time from the frame's arrival until
// the display shows it.

import AppKit
import CoreVideo
import Foundation
import Metal
import QuartzCore

// Draws a picture over the whole screen: one triangle, and a fragment shader
// for each layout. The colours are BT.709, video range, as the pictures' are;
// whether they are exact does not matter to the timing.
private let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Vertex {
        float4 position [[position]];
        float2 uv;
    };

    vertex Vertex vertex_main(uint id [[vertex_id]]) {
        float2 corner = float2((id << 1) & 2, id & 2);
        Vertex out;
        out.position = float4(corner * 2.0 - 1.0, 0.0, 1.0);
        out.uv = float2(corner.x, 1.0 - corner.y);
        return out;
    }

    constexpr sampler linearSampler(filter::linear, address::clamp_to_edge);

    float3 bt709(float y, float cb, float cr) {
        float luma = (y * 255.0 - 16.0) / 219.0;
        float u = (cb * 255.0 - 128.0) / 224.0;
        float v = (cr * 255.0 - 128.0) / 224.0;
        return float3(luma + 1.5748 * v, luma - 0.1873 * u - 0.4681 * v, luma + 1.8556 * u);
    }

    fragment float4 rgb_main(Vertex in [[stage_in]], texture2d<float> color [[texture(0)]]) {
        return float4(color.sample(linearSampler, in.uv).rgb, 1.0);
    }

    fragment float4 planes_main(
        Vertex in [[stage_in]], texture2d<float> y [[texture(0)]],
        texture2d<float> cb [[texture(1)]], texture2d<float> cr [[texture(2)]]
    ) {
        return float4(bt709(y.sample(linearSampler, in.uv).r,
                            cb.sample(linearSampler, in.uv).r,
                            cr.sample(linearSampler, in.uv).r), 1.0);
    }

    fragment float4 nv12_main(
        Vertex in [[stage_in]], texture2d<float> y [[texture(0)]], texture2d<float> cbcr [[texture(1)]]
    ) {
        float2 chroma = cbcr.sample(linearSampler, in.uv).rg;
        return float4(bt709(y.sample(linearSampler, in.uv).r, chroma.x, chroma.y), 1.0);
    }
    """

/// What a display loop saw during one run.
struct DisplayOutcome {
    /// Milliseconds from a frame's arrival until the display presented it, for
    /// the frames that were drawn at all, those timed only.
    let toScreen: [Double]
    /// Frames offered by the decode, those timed only, and how many of them the
    /// loop never drew because a newer one had come before its next refresh.
    let offered: Int
    let skipped: Int
    /// Refreshes drawn; the time between the display link's calls, the median
    /// and the number of calls that came more than half an interval later,
    /// a sign of a loop that cannot keep up.
    let refreshes: Int
    let interval: Double
    let late: Int
    /// Calls of the display link, and how many of them found no picture yet or
    /// no drawable: what to look at when nothing was drawn.
    let ticks: Int
    let noPicture: Int
    let noDrawable: Int
}

final class DisplayLoop: NSObject {
    /// The screen the loop runs on, and its refresh rate in the mode it is in.
    let screenName: String
    let maximumRate: Int
    let screen: NSScreen

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let window: NSWindow
    private let view: NSView
    private let layer = CAMetalLayer()
    private var pipelines: [PictureLayout: MTLRenderPipelineState] = [:]
    private var link: CVDisplayLink?
    private let displayID: CGDirectDisplayID
    private var divider = 1
    private var counter = 0

    private struct Offered {
        var sequence: Int
        var arrived: Double
        var layout: PictureLayout
        var textures: [MTLTexture]
        var timed: Bool
    }

    // The decode thread offers pictures, the main thread draws them and
    // Metal's threads report when they were presented.
    private let lock = NSLock()
    private var latest: Offered?
    private var sequence = 0
    private var drawn = -1
    private var offeredTimed = 0
    private var skipped = 0
    private var refreshes = 0
    private var intervals: [Double] = []
    private var ticks = 0
    private var noPicture = 0
    private var noDrawable = 0
    private var toScreen: [Double] = []
    private var lastTick = 0.0

    /// The screens, the one with the highest refresh rate first.
    static func screens() -> [NSScreen] {
        NSScreen.screens.sorted { $0.maximumFramesPerSecond > $1.maximumFramesPerSecond }
    }

    /// Makes the window on `screen`, which is not shown until `start`. Call on
    /// the main thread.
    init(device: MTLDevice, screen: NSScreen) throws {
        self.screen = screen
        displayID = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID ?? 0
        guard let queue = device.makeCommandQueue() else {
            throw EmberwaveFailure(description: "no command queue")
        }
        self.device = device
        self.queue = queue
        screenName = screen.localizedName
        maximumRate = screen.maximumFramesPerSecond

        // In global coordinates: a rectangle given with a `screen` would be
        // relative to it, which puts a screen that is not the main one's
        // window off every screen.
        window = NSWindow(
            contentRect: screen.frame, styleMask: .borderless, backing: .buffered, defer: false)
        // Over the menu bar and the Dock, as a full-screen client is, and
        // out of the way of the mouse.
        window.level = .screenSaver
        window.isOpaque = true
        window.backgroundColor = .black
        window.ignoresMouseEvents = true
        window.hasShadow = false
        view = NSView(frame: NSRect(origin: .zero, size: screen.frame.size))
        layer.device = device
        layer.pixelFormat = .bgra8Unorm
        // Two drawables, as a client that minds latency has: the default three let a frame wait a refresh longer in the queue.
        layer.maximumDrawableCount = 2
        layer.framebufferOnly = true
        layer.frame = view.bounds
        layer.contentsScale = screen.backingScaleFactor
        layer.drawableSize = CGSize(
            width: screen.frame.width * screen.backingScaleFactor,
            height: screen.frame.height * screen.backingScaleFactor)
        view.wantsLayer = true
        view.layer = layer
        window.contentView = view
        super.init()

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: shaderSource, options: nil)
        } catch {
            throw EmberwaveFailure(description: "the display shaders: \(error)")
        }
        for (layout, fragment) in [
            (PictureLayout.rgb, "rgb_main"), (.planes, "planes_main"), (.nv12, "nv12_main"),
        ] {
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.vertexFunction = library.makeFunction(name: "vertex_main")
            descriptor.fragmentFunction = library.makeFunction(name: fragment)
            descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelines[layout] = try device.makeRenderPipelineState(descriptor: descriptor)
        }
    }

    /// Whether the window is on the screen now: a screen that is off, or showing
    /// another input, never takes it.
    var isOnScreen: Bool { window.occlusionState.contains(.visible) }

    /// The picture size the screen is drawn at, for the report.
    var drawableSize: CGSize { layer.drawableSize }

    /// Takes a picture from the decode, on any thread.
    func offer(
        _ arrived: Double, _ layout: PictureLayout, _ textures: [MTLTexture], _ timed: Bool
    ) {
        lock.lock()
        sequence += 1
        latest = Offered(
            sequence: sequence, arrived: arrived, layout: layout, textures: textures, timed: timed)
        if timed {
            offeredTimed += 1
        }
        lock.unlock()
    }

    /// Shows the window and draws at `rate` frames a second, or the screen's
    /// refresh if that is lower, until `stop`: on a screen of a higher refresh
    /// than `rate`, every second or third refresh and so on. Returns the rate
    /// drawn at. Call on the main thread.
    func start(rate: Int) -> Int {
        lock.lock()
        latest = nil
        sequence = 0
        drawn = -1
        offeredTimed = 0
        skipped = 0
        refreshes = 0
        intervals = []
        ticks = 0
        noPicture = 0
        noDrawable = 0
        toScreen = []
        lastTick = 0
        lock.unlock()
        divider = max(1, Int((Double(maximumRate) / Double(rate)).rounded()))
        counter = 0
        NSApp.activate()
        window.orderFrontRegardless()
        // The window server takes a moment to put it on the screen.
        let shown = Date(timeIntervalSinceNow: 2)
        while !window.occlusionState.contains(.visible), Date() < shown {
            RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.05))
        }
        var link: CVDisplayLink?
        CVDisplayLinkCreateWithCGDisplay(displayID, &link)
        if let link {
            // On Core Video's thread; the window itself stays the main thread's.
            CVDisplayLinkSetOutputHandler(link) { [unowned self] _, _, _, _, _ in
                self.counter += 1
                if self.counter % self.divider == 0 {
                    self.tick()
                }
                return kCVReturnSuccess
            }
            CVDisplayLinkStart(link)
        }
        self.link = link
        return maximumRate / divider
    }

    /// Stops drawing, hides the window and says what the loop saw, once the
    /// last pictures have been presented. Call on the main thread.
    func stop() -> DisplayOutcome {
        if let link {
            CVDisplayLinkStop(link)
        }
        link = nil
        // The last handlers run on Metal's threads a refresh or two later.
        RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.2))
        window.orderOut(nil)
        lock.lock()
        defer { lock.unlock() }
        latest = nil
        let median = intervals.sorted()[safe: intervals.count / 2] ?? 0
        let late = intervals.filter { $0 > median * 1.5 }.count
        return DisplayOutcome(
            toScreen: toScreen, offered: offeredTimed, skipped: skipped, refreshes: refreshes,
            interval: median, late: late, ticks: ticks, noPicture: noPicture,
            noDrawable: noDrawable)
    }

    private func tick() {
        let began = now()
        lock.lock()
        let frame = latest
        let previousTick = lastTick
        lastTick = began
        lock.unlock()
        lock.lock()
        ticks += 1
        if previousTick > 0 {
            intervals.append(began - previousTick)
        }
        lock.unlock()
        guard let frame else {
            lock.lock()
            noPicture += 1
            lock.unlock()
            return
        }
        guard let drawable = layer.nextDrawable(), let buffer = queue.makeCommandBuffer(),
            let pipeline = pipelines[frame.layout]
        else {
            lock.lock()
            noDrawable += 1
            lock.unlock()
            return
        }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = drawable.texture
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        guard let encoder = buffer.makeRenderCommandEncoder(descriptor: pass) else {
            return
        }
        encoder.setRenderPipelineState(pipeline)
        for (index, texture) in frame.textures.enumerated() {
            encoder.setFragmentTexture(texture, index: index)
        }
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()

        // A picture's first refresh is the one whose presentation is its way
        // to the screen.
        let fresh = frame.sequence != drawn
        lock.lock()
        refreshes += 1
        if fresh {
            if drawn >= 0, frame.timed {
                skipped += frame.sequence - drawn - 1
            }
            drawn = frame.sequence
        }
        lock.unlock()
        if fresh, frame.timed {
            let arrived = frame.arrived
            drawable.addPresentedHandler { [weak self] drawable in
                // 0 for a drawable that was never presented.
                guard drawable.presentedTime > 0, let self else {
                    return
                }
                let latency = drawable.presentedTime * 1000 - arrived
                self.lock.lock()
                self.toScreen.append(latency)
                self.lock.unlock()
            }
        }
        buffer.present(drawable)
        buffer.commit()
    }
}

extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
