#!/usr/bin/env bash
# Builds the pinned upstream PyroWave Metal library and the refgen tool.
#
# Meant to be sourced by the scripts that need them. It sets ROOT, CACHE and
# REFGEN and leaves the upstream checkout in testdata/_cache/ (git-ignored).
# See testdata/README.md for the pinned revision and what was verified.

UPSTREAM_URL=https://github.com/Themaister/pyrowave.git
UPSTREAM_REV=79a3232a2199281f33f7a2e2659fc4fac62b022a

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CACHE=$ROOT/testdata/_cache
REFGEN=$CACHE/refgen

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    echo "this needs Apple Silicon macOS (upstream Metal port)" >&2
    exit 1
fi

mkdir -p "$CACHE"

if [ ! -d "$CACHE/pyrowave/.git" ]; then
    git clone --quiet "$UPSTREAM_URL" "$CACHE/pyrowave"
fi
git -C "$CACHE/pyrowave" checkout --quiet "$UPSTREAM_REV"

BUILD=$CACHE/build/pyrowave-metal
cmake -S "$CACHE/pyrowave/metal" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$BUILD" -j > /dev/null

xcrun clang++ -std=c++17 -O2 -fobjc-arc \
    -I"$CACHE/pyrowave/metal" \
    "$ROOT/scripts/refgen/refgen.mm" \
    -L"$BUILD" -lpyrowave-metal -Wl,-rpath,"$BUILD" \
    -framework Metal -framework Foundation \
    -o "$REFGEN"
