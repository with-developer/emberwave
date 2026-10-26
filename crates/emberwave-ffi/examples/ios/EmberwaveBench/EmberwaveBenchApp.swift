// emberwave's benchmark for iPhones and iPads: it times decoding and encoding
// on the device's GPU through the C API of the XCFramework, beside HEVC through
// VideoToolbox, and shows the last decoded picture through the Metal texture
// the decoder drew it into.
//
// Started with the argument -autorun, as CI starts it in a simulator, it runs
// a short benchmark and the texture check without the screen, prints the
// report and exits: with status 0, or 1 if anything failed. With -run, it
// does the same with the whole benchmark, for a run started from a Mac with
// `xcrun devicectl device process launch --console`.

import SwiftUI
import UIKit

@main
struct EmberwaveBenchApp: App {
    init() {
        let arguments = ProcessInfo.processInfo.arguments
        if arguments.contains("-autorun") {
            DispatchQueue.global(qos: .userInitiated).async { autorun(.short) }
        } else if arguments.contains("-run") {
            // The screen stays on: a phone that locks halfway through spoils
            // the run. The application exists once the main queue runs.
            DispatchQueue.main.async { UIApplication.shared.isIdleTimerDisabled = true }
            DispatchQueue.global(qos: .userInitiated).async { autorun(Plan()) }
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

/// The benchmark without the screen, to `plan`.
private func autorun(_ plan: Plan) {
    do {
        let emberwave = try Emberwave()
        guard let device = emberwave.device else {
            throw EmberwaveFailure(description: "the context has no Metal device")
        }
        _ = try MetalView.Renderer.makePipeline(device: device)
        var results: [SizeResult] = []
        var hevc: [HEVCOutcome] = []
        for size in pictureSizes {
            let (result, outcomes) = try emberwave.runSize(size, plan) { print($0) }
            results.append(result)
            hevc.append(contentsOf: outcomes)
        }
        let check = try emberwave.checkTextures()
        print(makeReport(emberwave, plan, results, hevc: hevc, check: check))
        // A device without HEVC, as a simulator may be, is not a failure.
        if hevc.contains(where: \.failed) {
            print("failed: HEVC")
            fflush(stdout)
            exit(1)
        }
        fflush(stdout)
        exit(0)
    } catch {
        print("failed: \(error)")
        fflush(stdout)
        exit(1)
    }
}
