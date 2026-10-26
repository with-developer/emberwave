// The benchmark app's measurements on a Mac, without the app: emberwave
// through its C API and HEVC through VideoToolbox, by the app's own code in
// ../EmberwaveBench, so that a Mac's numbers compare with a phone's. The
// report is the app's; the progress goes to the standard error.
// scripts/bench_app_on_mac.sh builds and runs it. With -autorun, it makes the
// short run of CI, and exits with status 1 if anything failed.

import Foundation

let plan = CommandLine.arguments.contains("-autorun") ? Plan.short : Plan()
do {
    let emberwave = try Emberwave()
    var results: [SizeResult] = []
    var hevc: [HEVCOutcome] = []
    for size in pictureSizes {
        let (result, outcomes) = try emberwave.runSize(size, plan) {
            FileHandle.standardError.write(Data(($0 + "\n").utf8))
        }
        results.append(result)
        hevc.append(contentsOf: outcomes)
    }
    let check = try emberwave.checkTextures()
    print(makeReport(emberwave, plan, results, hevc: hevc, check: check))
    // A Mac without HEVC, as a virtual machine may be, is not a failure.
    if hevc.contains(where: \.failed) {
        print("failed: HEVC")
        exit(1)
    }
} catch {
    print("failed: \(error)")
    exit(1)
}
