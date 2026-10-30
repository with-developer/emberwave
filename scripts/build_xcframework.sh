#!/usr/bin/env bash
# Builds the C library (emberwave-ffi) as static libraries for iPhones and
# iPads, for the iOS Simulator and for Macs, all with Apple silicon, and
# bundles them with the header into target/xcframework/Emberwave.xcframework,
# which Xcode and SwiftPM link like any other XCFramework.
#
# The header sits in Headers/Emberwave with a module map: Swift imports it as
# `import Emberwave`, C includes <Emberwave/emberwave.h>. A module map in a
# directory of its own does not clash with those of other XCFrameworks, as
# Xcode copies the headers of all of them into one directory.
#
# Next to the XCFramework, native-static-libs-SLICE.txt names the system
# libraries and frameworks that a program linking the library of SLICE needs
# as well, as rustc reports them, and Emberwave.xcframework.zip holds the
# XCFramework at the top of a zip archive, the shape a SwiftPM binary target
# downloads: releases publish it as it is.
#
# Usage: scripts/build_xcframework.sh [SLICE...]
#
#   SLICE  ios-arm64 (devices), ios-arm64-simulator or macos-arm64; all three
#          by default
#
# Needs macOS with Xcode, and Rust's standard library for each slice's
# target, which rustup installs with
#     rustup target add aarch64-apple-ios aarch64-apple-ios-sim
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=target/xcframework

# The Rust target of a slice, and the oldest system its library is for: iOS
# 12, the oldest that Xcode 26 builds for; iOS 14, the first Simulator on
# Apple silicon; macOS 11, the first on Apple silicon. wgpu checks at run time
# for what newer systems add.
slice_target() {
    case $1 in
        ios-arm64) echo "aarch64-apple-ios IPHONEOS_DEPLOYMENT_TARGET=12.0" ;;
        ios-arm64-simulator) echo "aarch64-apple-ios-sim IPHONEOS_DEPLOYMENT_TARGET=14.0" ;;
        macos-arm64) echo "aarch64-apple-darwin MACOSX_DEPLOYMENT_TARGET=11.0" ;;
        *)
            echo "unknown slice $1: ios-arm64, ios-arm64-simulator or macos-arm64" >&2
            return 1
            ;;
    esac
}

slices=("$@")
if [ ${#slices[@]} -eq 0 ]; then
    slices=(ios-arm64 ios-arm64-simulator macos-arm64)
fi
sysroot=$(rustc --print sysroot)
for slice in "${slices[@]}"; do
    # An assignment, so that an unknown slice ends the script.
    spec=$(slice_target "$slice")
    read -r target _ <<< "$spec"
    if [ ! -d "$sysroot/lib/rustlib/$target" ]; then
        echo "Rust's standard library for $target is missing: rustup target add $target" >&2
        exit 1
    fi
done

rm -rf target/xcframework
mkdir -p "$OUT/headers/Emberwave"
cp crates/emberwave-ffi/include/emberwave.h "$OUT/headers/Emberwave/"
cat > "$OUT/headers/Emberwave/module.modulemap" <<'EOF'
module Emberwave {
    header "emberwave.h"
    export *
}
EOF

libraries=()
for slice in "${slices[@]}"; do
    read -r target deployment <<< "$(slice_target "$slice")"
    echo "== $slice ($target, $deployment)"
    # Without --color never the line ends in a colour escape code. A build
    # that is up to date prints the line again.
    env "$deployment" cargo rustc --release --locked --color never -p emberwave-ffi --lib \
        --target "$target" --crate-type staticlib -- --print native-static-libs 2>&1 |
        tee "$OUT/$slice.log"
    sed -n 's/.*native-static-libs: //p' "$OUT/$slice.log" | tail -n 1 > "$OUT/native-static-libs-$slice.txt"
    test -s "$OUT/native-static-libs-$slice.txt"
    libraries+=(-library "target/$target/release/libemberwave_ffi.a" -headers "$OUT/headers")
done

xcodebuild -create-xcframework "${libraries[@]}" -output "$OUT/Emberwave.xcframework"
# ditto zips a bundle as Apple's tools do, here without the extended
# attributes that macOS adds to built files, which would go in as ._ files.
(cd "$OUT" && ditto -c -k --keepParent --norsrc --noextattr --noacl \
    Emberwave.xcframework Emberwave.xcframework.zip)
for slice in "${slices[@]}"; do
    echo "$slice needs: $(cat "$OUT/native-static-libs-$slice.txt")"
done
