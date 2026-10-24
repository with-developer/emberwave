// The screen: the last decoded picture, a button that runs the benchmark, its
// progress, and the report with a button that copies it.

import Metal
import SwiftUI
import UIKit

/// Runs the benchmark off the main thread and publishes what it does.
@MainActor
final class BenchModel: ObservableObject {
    @Published var status = "Tap Run. It takes about a minute; keep the app in front."
    @Published var report = ""
    @Published var running = false
    @Published var picture: MTLTexture?
    let emberwave: Emberwave?

    init() {
        do {
            emberwave = try Emberwave()
            status = "GPU: \(emberwave?.adapter ?? "none"). Tap Run; it takes about a minute."
        } catch {
            emberwave = nil
            status = "No GPU: \(error)"
        }
    }

    func run() {
        guard let emberwave, !running else {
            return
        }
        running = true
        report = ""
        // The screen stays on: a phone that locks halfway through spoils the run.
        UIApplication.shared.isIdleTimerDisabled = true
        Task.detached(priority: .userInitiated) {
            let plan = Plan()
            var results: [SizeResult] = []
            var outcome: String
            do {
                for size in pictureSizes {
                    let result = try emberwave.run(size, plan) { line in
                        Task { @MainActor in self.status = line }
                    }
                    results.append(result)
                    await MainActor.run { self.picture = result.picture }
                }
                let check = try emberwave.checkTexture()
                outcome = makeReport(emberwave, plan, results, check: check)
            } catch {
                outcome = makeReport(emberwave, plan, results, check: "failed: \(error)")
            }
            let finished = outcome
            await MainActor.run {
                self.report = finished
                self.status = "Done."
                self.running = false
                UIApplication.shared.isIdleTimerDisabled = false
            }
        }
    }
}

struct ContentView: View {
    @StateObject private var model = BenchModel()

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                if let device = model.emberwave?.device {
                    MetalView(device: device, picture: model.picture)
                        .aspectRatio(16 / 9, contentMode: .fit)
                        .background(Color.black)
                }
                HStack {
                    Button(model.running ? "Running…" : "Run") { model.run() }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.running || model.emberwave == nil)
                    if !model.report.isEmpty {
                        Button("Copy the results") { UIPasteboard.general.string = model.report }
                            .buttonStyle(.bordered)
                    }
                }
                Text(model.status)
                    .font(.callout)
                ScrollView {
                    Text(model.report)
                        .font(.system(.footnote, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding()
            .navigationTitle("emberwave")
        }
    }
}
