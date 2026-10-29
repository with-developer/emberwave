#!/usr/bin/env bash
# Builds the pinned upstream PyroWave Metal library and the refgen tool.
#
# Meant to be sourced by the scripts that need them. It sets ROOT, CACHE and
# REFGEN and leaves the upstream checkout in testdata/_cache/ (git-ignored).
# See testdata/README.md for the pinned revision and what was verified.

UPSTREAM_URL=https://github.com/Themaister/pyrowave.git
UPSTREAM_REV=7fe0e2667cd3076b8b2bf04bcb80bc8d691b800b

# The repository root, from this file's location. BASH_SOURCE is empty when
# another shell sources this file; zsh puts the file in $0 instead. A wrong
# root would clone and build outside the repository, so check it first.
REFGEN_DIR=$(dirname "${BASH_SOURCE[0]:-$0}")
ROOT=$(cd "$REFGEN_DIR/../.." && pwd)
if [ ! -f "$ROOT/scripts/refgen/refgen.mm" ]; then
    echo "cannot find the repository from $REFGEN_DIR; run the scripts with bash" >&2
    return 1 2>/dev/null || exit 1
fi
CACHE=$ROOT/testdata/_cache
REFGEN=$CACHE/refgen

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    echo "this needs Apple Silicon macOS (upstream Metal port)" >&2
    return 1 2>/dev/null || exit 1
fi

mkdir -p "$CACHE"

if [ ! -d "$CACHE/pyrowave/.git" ]; then
    git clone --quiet "$UPSTREAM_URL" "$CACHE/pyrowave"
fi
# A checkout cloned before the pin moved lacks the pinned commit.
if ! git -C "$CACHE/pyrowave" cat-file -e "$UPSTREAM_REV^{commit}" 2>/dev/null; then
    git -C "$CACHE/pyrowave" fetch --quiet "$UPSTREAM_URL"
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
