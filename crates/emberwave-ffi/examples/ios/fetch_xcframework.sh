#!/usr/bin/env bash
# Puts the XCFramework of an emberwave release where the benchmark app's Xcode
# project looks for it, target/xcframework/Emberwave.xcframework, after
# checking it against the release's SHA256SUMS. For a Mac without Rust's
# standard library for iOS, which scripts/build_xcframework.sh needs.
#
# Usage: crates/emberwave-ffi/examples/ios/fetch_xcframework.sh [TAG]
#
#   TAG  a release, such as v0.9.0; by default that of the version in
#        Cargo.toml
set -euo pipefail

cd "$(dirname "$0")/../../../.."
tag=${1:-v$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n 1)}
name="emberwave-ffi-$tag-xcframework"
url="https://github.com/with-developer/emberwave/releases/download/$tag"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
curl -fsSL -o "$work/$name.zip" "$url/$name.zip"
curl -fsSL -o "$work/SHA256SUMS" "$url/SHA256SUMS"
(cd "$work" && grep " $name.zip\$" SHA256SUMS | shasum -a 256 -c -)
unzip -q "$work/$name.zip" -d "$work"
# The app is written for the C API of this checkout, which a release made
# before a change to the API does not have.
if ! cmp -s "$work/$name/Emberwave.xcframework/ios-arm64/Headers/Emberwave/emberwave.h" \
    crates/emberwave-ffi/include/emberwave.h; then
    echo "warning: the C API of $tag differs from that of this checkout, so the app may not build" \
        "against it: check out $tag, or build the XCFramework with scripts/build_xcframework.sh" >&2
fi

mkdir -p target/xcframework
rm -rf target/xcframework/Emberwave.xcframework
cp -R "$work/$name/Emberwave.xcframework" target/xcframework/
echo "target/xcframework/Emberwave.xcframework is that of $tag"
