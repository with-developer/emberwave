#!/usr/bin/env bash
# Runs the measurements of the iPhone benchmark app
# (crates/emberwave-ffi/examples/ios) on this Mac, by the app's own code:
# emberwave through its C API, linked from the XCFramework's library for the
# Mac, and HEVC through VideoToolbox, with the pictures, the plan and the
# report of a phone's run (see BENCHMARKS.md). It builds the front-end,
# examples/ios/EmberwaveBenchMac, with swiftc into target/bench-app-mac.
#
# Usage: scripts/bench_app_on_mac.sh [-autorun | -display]
#
#   -autorun  the short run of CI: every size briefly, and the three texture
#             checks; fails unless the report is complete and they pass
#   -display  instead of the report above, decodes at 60 frames a second beside
#             a display loop (issue #81): a borderless window covers a screen
#             while it runs, a few minutes, and draws the last picture at 60
#             and at 120 Hz: on the slowest screen that has the rate, and on
#             the built-in screen when that one takes no window or has more.
#             It keeps the displays awake. Leave the Mac alone. -short is brief.
#
# Needs macOS with Xcode, and the XCFramework with its library for the Mac,
# which scripts/build_xcframework.sh makes. A full run takes a few minutes:
# leave the Mac alone meanwhile, as for scripts/bench.sh.
set -euo pipefail

cd "$(dirname "$0")/.."
SLICE=target/xcframework/Emberwave.xcframework/macos-arm64
APP=crates/emberwave-ffi/examples/ios
BUILD=target/bench-app-mac
if [ ! -f "$SLICE/libemberwave_ffi.a" ]; then
    echo "the XCFramework has no library for the Mac: run scripts/build_xcframework.sh first" >&2
    exit 1
fi

mkdir -p "$BUILD"
# The system libraries are words for the shell to split.
# shellcheck disable=SC2046
xcrun --sdk macosx swiftc -O -swift-version 5 -target arm64-apple-macos14.0 \
    -I "$SLICE/Headers/Emberwave" \
    "$APP/EmberwaveBench/Benchmark.swift" "$APP/EmberwaveBench/Scene.swift" \
    "$APP/EmberwaveBench/VideoToolbox.swift" "$APP/EmberwaveBenchMac/main.swift" \
    "$APP/EmberwaveBenchMac/DisplayLoop.swift" "$APP/EmberwaveBenchMac/DisplayRun.swift" \
    "$SLICE/libemberwave_ffi.a" $(cat target/xcframework/native-static-libs-macos-arm64.txt) \
    -o "$BUILD/emberwave-bench"

if [ "${1:-}" != -autorun ]; then
    exec "$BUILD/emberwave-bench" "$@"
fi
"$BUILD/emberwave-bench" -autorun | tee "$BUILD/autorun.log"
for check in texture plane queue; do
    grep -q "^$check check: ok" "$BUILD/autorun.log"
done
for size in 1080p 1440p 4K; do
    grep -q "^$size (" "$BUILD/autorun.log"
done
grep -q "^  HEVC at" "$BUILD/autorun.log"
echo "checked: the app's measurements ran on this Mac"
