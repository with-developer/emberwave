#!/usr/bin/env bash
# Builds the C examples of the C API (crates/emberwave-ffi/examples/c) with
# CMake against the shared library, decodes every test vector with the
# decoding example and compares the result with upstream's FP32 decode, at the
# 80 dB of decision D9, and encodes the sources of two encoder vectors with the
# encoding example, which decodes its packets again and fails below 40 dB.
#
# Usage: scripts/check_c_example.sh [CARGO_OUT_DIR]
#
#   CARGO_OUT_DIR  where cargo put emberwave-ffi and emberwave-cli, relative
#                  to the repository root or absolute (default target/debug);
#                  build both first, for example with
#                  cargo build -p emberwave-ffi -p emberwave-cli
#
# Needs CMake and a C compiler. The example uses whatever GPU the machine
# has, or a software driver.
set -euo pipefail

# Paths stay relative to the root: under Git Bash on Windows, absolute paths
# look like /d/a/..., which native programs such as CMake do not understand.
cd "$(dirname "$0")/.."
CARGO_OUT=${1:-target/debug}
BUILD=target/c-example

cmake -S crates/emberwave-ffi/examples/c -B "$BUILD" \
    -DEMBERWAVE_LIB_DIR="$CARGO_OUT" -DCMAKE_BUILD_TYPE=Debug
cmake --build "$BUILD" --config Debug

# Single-configuration generators put the programs in the build directory,
# Visual Studio's in a directory per configuration.
program() {
    local candidate
    for candidate in "$BUILD/$1" "$BUILD/Debug/$1.exe"; do
        if [ -x "$candidate" ]; then
            echo "$candidate"
            return
        fi
    done
    echo "$1 was not built in $BUILD" >&2
    exit 1
}
example=$(program emberwave_decode)
encoder=$(program emberwave_encode)
cli=$CARGO_OUT/emberwave-cli
if [ -x "$cli.exe" ]; then
    cli=$cli.exe
fi

for vector in testdata/*.pwv1; do
    name=$(basename "$vector" .pwv1)
    echo "== $name"
    "$example" "$vector" "$BUILD/$name.y4m"
    "$cli" compare "$BUILD/$name.y4m" "testdata/$name.ref-fp32.y4m" --min-psnr 80 | tail -n 1
done

for source in scene_200x120_420 scene_131x67_444; do
    echo "== $source, encoded"
    "$encoder" "testdata/encoder/$source.source.y4m" 20000 40 | tail -n 1
done
