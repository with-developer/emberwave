#!/usr/bin/env bash
# Builds the iPhone benchmark app (crates/emberwave-ffi/examples/ios) against
# the XCFramework that scripts/build_xcframework.sh made: for iPhones, which
# only checks that it links, and for the iOS Simulator, where it then runs
# with -autorun. That runs a short benchmark of every picture size, and a
# check that reads a texture of emberwave_decoder_decode_texture back through
# Metal on the app's own queue and finds the RGB picture. The check fails
# unless the app prints its report and "texture check: ok".
#
# Usage: scripts/check_ios_app.sh
#
# Needs macOS with Xcode and an iOS Simulator runtime, jq, and the
# XCFramework with both iOS libraries. Signs nothing.
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT=crates/emberwave-ffi/examples/ios/EmberwaveBench.xcodeproj
BUILD=target/ios-app
BUNDLE=io.github.with-developer.emberwave-bench
# shellcheck source=simulator.sh
source scripts/simulator.sh

for slice in ios-arm64 ios-arm64-simulator; do
    if [ ! -d "target/xcframework/Emberwave.xcframework/$slice" ]; then
        echo "the XCFramework has no $slice library: run scripts/build_xcframework.sh first" >&2
        exit 1
    fi
done

build() {
    xcodebuild -project "$PROJECT" -scheme EmberwaveBench -configuration Release \
        -destination "generic/platform=$1" -derivedDataPath "$BUILD" \
        CODE_SIGNING_ALLOWED=NO build -quiet
}
echo "== building for iPhones"
build iOS
echo "== building for the iOS Simulator"
build "iOS Simulator"

use_simulator
app="$BUILD/Build/Products/Release-iphonesimulator/EmberwaveBench.app"
xcrun simctl install "$device" "$app"
# --console-pty passes on what the app prints, until it exits.
xcrun simctl launch --console-pty --terminate-running-process "$device" "$BUNDLE" -autorun |
    tee "$BUILD/autorun.log"
# Not `! grep`: set -e does not stop at a negated command.
if grep -q "^failed:" "$BUILD/autorun.log"; then
    echo "the app failed" >&2
    exit 1
fi
grep -q "^texture check: ok" "$BUILD/autorun.log"
for size in 1080p 1440p 4K; do
    grep -q "^$size (" "$BUILD/autorun.log"
done
echo "checked: the app built for iPhones and ran in the simulator"
