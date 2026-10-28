// The screen: the last decoded picture, a button that runs the benchmark, its
// progress, and the report with a button that copies it.

import Metal
import SwiftUI
import UIKit

/// Runs the benchmark off the main thread and publishes what it does.
@MainActor
final class BenchModel: ObservableObject {
    @Published var status = "Tap Run. It takes about five minutes; keep the app in front."
    @Published var report = ""
    @Published var running = false
    @Published var picture: MTLTexture?
    let emberwave: Emberwave?
    /// Started with -run, from a Mac: the run starts by itself once the app
    /// has stayed in front for `settle`; its progress, the app's moves to and
    /// from the front and the report go to the standard output as well, and
    /// the app then exits, with status 1 if anything failed or the run is not
    /// valid.
    let launchedToRun = ProcessInfo.processInfo.arguments.contains("-run")
    /// How long the app must have stayed in front before a run that -run asks
    /// for begins: an app a Mac has just launched may still leave the front
    /// for a moment, and a run during which it does is not valid.
    private static let settle = Duration.seconds(3)
    private let launched = ContinuousClock.now
    private var startedByLaunch = false
    private var pendingStart: Task<Void, Never>?
    /// What the run was doing when the app first left the front, if it did.
    private var interruptedDuring: String?
    private var observers: [any NSObjectProtocol] = []

    init() {
        do {
            emberwave = try Emberwave()
            status = "GPU: \(emberwave?.adapter ?? "none"). Tap Run; it takes about five minutes."
        } catch {
            emberwave = nil
            status = "No GPU: \(error)"
        }
        watchTheFront()
    }

    /// Starts the run that -run asks for if the app is in front already;
    /// otherwise `moved` does when it comes to the front.
    func runIfLaunchedToRun() {
        if UIApplication.shared.applicationState == .active {
            startOnceSettled()
        }
    }

    private func startOnceSettled() {
        guard launchedToRun, !startedByLaunch, pendingStart == nil else {
            return
        }
        // Without a GPU there is nothing to run: the Mac, which waits for
        // the app to end, learns why.
        guard emberwave != nil else {
            print("failed: \(status)")
            fflush(stdout)
            exit(1)
        }
        // The screen stays on while the run waits too: a phone that locks
        // then would put the run off until someone unlocks it.
        UIApplication.shared.isIdleTimerDisabled = true
        status = "Started from a Mac: the run begins once the app has stayed in front for \(Self.settle)."
        pendingStart = Task { [weak self] in
            try? await Task.sleep(for: Self.settle)
            guard let self, !Task.isCancelled else {
                return
            }
            self.pendingStart = nil
            self.startedByLaunch = true
            self.status = "Started from a Mac: running, keep the app in front."
            self.run()
        }
    }

    /// Follows the app to and from the front, as `moved` describes.
    private func watchTheFront() {
        let moves: [(Notification.Name, String)] = [
            (UIApplication.didBecomeActiveNotification, "came to the front"),
            (UIApplication.willResignActiveNotification, "is leaving the front"),
            (UIApplication.didEnterBackgroundNotification, "went to the background"),
            (UIApplication.willEnterForegroundNotification, "is coming back from the background"),
        ]
        for (name, move) in moves {
            observers.append(
                NotificationCenter.default.addObserver(forName: name, object: nil, queue: .main) {
                    [weak self] _ in
                    MainActor.assumeIsolated { self?.moved(name, move) }
                })
        }
    }

    /// Leaving the front marks a run as not valid and puts off a run that -run
    /// has not begun yet. For -run, each move is printed with the time since
    /// the launch: a Mac sees only the standard output.
    private func moved(_ name: Notification.Name, _ move: String) {
        if launchedToRun {
            print("\(stamp(since: launched))the app \(move); \(status)")
            fflush(stdout)
        }
        switch name {
        case UIApplication.didBecomeActiveNotification:
            startOnceSettled()
        case UIApplication.willResignActiveNotification:
            pendingStart?.cancel()
            pendingStart = nil
            if running, interruptedDuring == nil {
                interruptedDuring = status
            }
        default:
            break
        }
    }

    func run() {
        guard let emberwave, !running else {
            return
        }
        let printing = launchedToRun
        let launched = launched
        running = true
        report = ""
        // The screen stays on: a phone that locks halfway through spoils the run.
        UIApplication.shared.isIdleTimerDisabled = true
        // iOS runs no GPU work for an app in the background, and the calls do
        // not fail: those frames take no GPU time, and the one that waits for
        // the app to come back takes all of it. Anything else in front, even
        // briefly, also takes the GPU. As the browser's benchmark does with a
        // hidden page, such a run is marked (see `moved`).
        interruptedDuring = nil
        Task.detached(priority: .userInitiated) {
            let plan = Plan()
            var results: [SizeResult] = []
            var hevc: [HEVCOutcome] = []
            var outcome: String
            var failed = false
            do {
                for size in pictureSizes {
                    let (result, outcomes) = try emberwave.runSize(size, plan) { line in
                        if printing {
                            print(stamp(since: launched) + line)
                            fflush(stdout)
                        }
                        Task { @MainActor in self.status = line }
                    }
                    results.append(result)
                    hevc.append(contentsOf: outcomes)
                    await MainActor.run { self.picture = result.picture }
                }
                let check = try emberwave.checkTextures()
                outcome = makeReport(emberwave, plan, results, hevc: hevc, check: check)
                failed = hevc.contains(where: \.failed)
            } catch {
                outcome = makeReport(
                    emberwave, plan, results, hevc: hevc, check: "failed: \(error)")
                failed = true
            }
            let finished = outcome
            let anythingFailed = failed
            await MainActor.run {
                if let during = self.interruptedDuring {
                    let warning =
                        "The app left the front during the run, first at \"\(during)\", which stops "
                        + "or slows its GPU work: run it again with the app in front."
                    self.report = finished + "\nNot valid: " + warning
                    self.status = warning
                } else {
                    self.report = finished
                    self.status = "Done."
                }
                self.running = false
                UIApplication.shared.isIdleTimerDisabled = false
                if printing {
                    print(self.report)
                    fflush(stdout)
                    exit(anythingFailed || self.interruptedDuring != nil ? 1 : 0)
                }
            }
        }
    }
}

/// The time since `launched`, before a line printed for -run.
private func stamp(since launched: ContinuousClock.Instant) -> String {
    String(format: "%6.1f s  ", (ContinuousClock.now - launched) / .seconds(1))
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
            .onAppear { model.runIfLaunchedToRun() }
        }
    }
}
