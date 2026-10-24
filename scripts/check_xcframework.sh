#!/usr/bin/env bash
# Checks the XCFramework that scripts/build_xcframework.sh made, as apps use
# it: builds the C examples (crates/emberwave-ffi/examples/c) and the Swift
# example (examples/swift, with SwiftPM) against each of its libraries, linked
# statically with the system libraries rustc named. The programs for the Mac
# run here and those for the iOS Simulator in a simulator: the decoding
# examples decode every test vector, which emberwave-cli compares with
# upstream's FP32 decode at the 80 dB of decision D9, and the encoding example
# encodes two sources and fails below 40 dB, as in scripts/check_c_example.sh.
# The programs for devices are only linked, as no device is at hand.
#
# Usage: scripts/check_xcframework.sh CLI
#
#   CLI  emberwave-cli built for this Mac, relative to the repository root
#        or absolute, for example target/debug/emberwave-cli
#
# Needs macOS with Xcode. For the iOS Simulator library it needs an iOS
# Simulator runtime (Xcode > Settings > Components): it uses a booted iPhone
# simulator, or boots one of the newest runtime and shuts it down at the end.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
CLI=$1
case $CLI in
    /*) ;;
    *) CLI=$ROOT/$CLI ;;
esac
XCFRAMEWORK=$ROOT/target/xcframework/Emberwave.xcframework
BUILD=$ROOT/target/xcframework-check
rm -rf target/xcframework-check

# shellcheck source=simulator.sh
source scripts/simulator.sh

# The SDK and the target triple of a slice, at the oldest system its library
# is for (see build_xcframework.sh).
slice_sdk() {
    case $1 in
        ios-arm64) echo "iphoneos arm64-apple-ios12.0" ;;
        ios-arm64-simulator) echo "iphonesimulator arm64-apple-ios14.0-simulator" ;;
        macos-arm64) echo "macosx arm64-apple-macos11.0" ;;
        *)
            echo "unknown slice $1" >&2
            return 1
            ;;
    esac
}

# Builds the C and Swift examples for SLICE into $BUILD/SLICE.
build() {
    local slice=$1 spec sdk triple example bin
    spec=$(slice_sdk "$slice")
    read -r sdk triple <<< "$spec"
    echo "== building for $slice"
    mkdir -p "$BUILD/$slice"
    for example in decode encode; do
        # The system libraries are words for the shell to split.
        # shellcheck disable=SC2046
        xcrun --sdk "$sdk" clang -target "$triple" -std=c99 -Wall -Wextra -Werror -pedantic \
            -I "$XCFRAMEWORK/$slice/Headers/Emberwave" "crates/emberwave-ffi/examples/c/$example.c" \
            "$XCFRAMEWORK/$slice/libemberwave_ffi.a" \
            $(cat "target/xcframework/native-static-libs-$slice.txt") \
            -o "$BUILD/$slice/emberwave_$example"
    done
    swift build -c release --package-path crates/emberwave-ffi/examples/swift \
        --scratch-path "$BUILD/swift-$slice" --triple "$triple" \
        --sdk "$(xcrun --sdk "$sdk" --show-sdk-path)"
    bin=$(swift build -c release --package-path crates/emberwave-ffi/examples/swift \
        --scratch-path "$BUILD/swift-$slice" --triple "$triple" \
        --sdk "$(xcrun --sdk "$sdk" --show-sdk-path)" --show-bin-path)
    cp "$bin/emberwave-decode" "$BUILD/$slice/emberwave_decode_swift"
}

# Runs the examples in DIRECTORY with RUN, a command that runs a program and
# its arguments: on_mac or on_simulator.
exercise() {
    local run=$1 dir=$2 vector name program
    for vector in "$ROOT"/testdata/*.pwv1; do
        name=$(basename "$vector" .pwv1)
        for program in emberwave_decode emberwave_decode_swift; do
            echo "== $name, $program"
            "$run" "$dir/$program" "$vector" "$dir/$name.y4m"
            "$CLI" compare "$dir/$name.y4m" "$ROOT/testdata/$name.ref-fp32.y4m" --min-psnr 80 |
                tail -n 1
        done
    done
    for name in scene_200x120_420 scene_131x67_444; do
        echo "== $name, encoded"
        "$run" "$dir/emberwave_encode" "$ROOT/testdata/encoder/$name.source.y4m" 20000 40 |
            tail -n 1
    done
}

on_mac() {
    "$@"
}

# A simulator runs programs of the Mac's file system, with their paths.
on_simulator() {
    xcrun simctl spawn "$device" "$@"
}

slices=()
for slice in ios-arm64 ios-arm64-simulator macos-arm64; do
    if [ -d "$XCFRAMEWORK/$slice" ]; then
        slices+=("$slice")
    fi
done
if [ ${#slices[@]} -eq 0 ]; then
    echo "no XCFramework at $XCFRAMEWORK: run scripts/build_xcframework.sh first" >&2
    exit 1
fi
for slice in "${slices[@]}"; do
    build "$slice"
done

for slice in "${slices[@]}"; do
    case $slice in
        macos-arm64)
            echo "== on this Mac"
            exercise on_mac "$BUILD/$slice"
            ;;
        ios-arm64-simulator)
            use_simulator
            # A failure in the simulator must fail the check: the example
            # without arguments exits with 2.
            if on_simulator "$BUILD/$slice/emberwave_decode" 2> /dev/null; then
                echo "simctl spawn does not pass on the exit status of the program" >&2
                exit 1
            fi
            exercise on_simulator "$BUILD/$slice"
            ;;
    esac
done
echo "checked: ${slices[*]}"
