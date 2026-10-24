// emberwave's benchmark for iPhones and iPads: it times decoding and encoding
// on the device's GPU through the C API of the XCFramework, and shows the last
// decoded picture through the Metal texture the decoder drew it into.
//
// Started with the argument -autorun, as CI starts it in a simulator, it runs
// a short benchmark and the texture check without the screen, prints the
// report and exits: with status 0, or 1 if anything failed.

import SwiftUI

@main
struct EmberwaveBenchApp: App {
    init() {
        if ProcessInfo.processInfo.arguments.contains("-autorun") {
            DispatchQueue.global(qos: .userInitiated).async { autorun() }
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

/// The benchmark without the screen, shorter, for CI.
private func autorun() {
    do {
        let emberwave = try Emberwave()
        let plan = Plan(pictures: 2, frames: 60, warmupFrames: 10, warmupMs: 200)
        var results: [SizeResult] = []
        for size in pictureSizes {
            results.append(try emberwave.run(size, plan) { print($0) })
        }
        let check = try emberwave.checkTexture()
        print(makeReport(emberwave, plan, results, check: check))
        fflush(stdout)
        exit(0)
    } catch {
        print("failed: \(error)")
        fflush(stdout)
        exit(1)
    }
}
