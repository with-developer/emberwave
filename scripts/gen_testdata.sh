#!/usr/bin/env bash
# Regenerates the small committed test vectors and their reference decodes.
#
# Vectors are encoded by the upstream PyroWave encoder and the reference
# pictures are decoded by the upstream PyroWave decoder, both from the pinned
# revision below (the Metal port, so this needs Apple Silicon macOS).
# See testdata/README.md.
#
# Every vector gets two reference decodes of the same bitstream:
#   NAME.ref.y4m        upstream's default precision (FP16 wavelet storage)
#   NAME.ref-fp32.y4m   upstream's FP32 precision (PYROWAVE_PRECISION=2)
#
# Usage: scripts/gen_testdata.sh [output-dir]      (default: testdata/)
#
# The encoder does not produce byte-for-byte reproducible bitstreams (padding
# bits in the last sign byte of a block are leftover GPU memory), so
# regenerating changes the committed files. Only do it when a vector is added
# or its parameters change; tests run against the committed bytes.
set -euo pipefail

UPSTREAM_URL=https://github.com/Themaister/pyrowave.git
UPSTREAM_REV=79a3232a2199281f33f7a2e2659fc4fac62b022a

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CACHE=$ROOT/testdata/_cache
OUT=${1:-$ROOT/testdata}

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    echo "gen_testdata.sh needs Apple Silicon macOS (upstream Metal port)" >&2
    exit 1
fi

mkdir -p "$CACHE" "$OUT"

if [ ! -d "$CACHE/pyrowave/.git" ]; then
    git clone --quiet "$UPSTREAM_URL" "$CACHE/pyrowave"
fi
git -C "$CACHE/pyrowave" checkout --quiet "$UPSTREAM_REV"

BUILD=$CACHE/build/pyrowave-metal
cmake -S "$CACHE/pyrowave/metal" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$BUILD" -j > /dev/null

REFGEN=$CACHE/refgen
xcrun clang++ -std=c++17 -O2 -fobjc-arc \
    -I"$CACHE/pyrowave/metal" \
    "$ROOT/scripts/refgen/refgen.mm" \
    -L"$BUILD" -lpyrowave-metal -Wl,-rpath,"$BUILD" \
    -framework Metal -framework Foundation \
    -o "$REFGEN"

# gen NAME encode-options...: encode a vector, then decode it with the reference.
gen() {
    local name=$1
    shift
    echo "== $name"
    "$REFGEN" encode --out "$OUT/$name.pwv1" "$@"
    "$REFGEN" decode "$OUT/$name.pwv1" "$OUT/$name.ref.y4m"
    PYROWAVE_PRECISION=2 "$REFGEN" decode "$OUT/$name.pwv1" "$OUT/$name.ref-fp32.y4m"
}

# Sizes are chosen so the aligned size differs from the picture size, which
# exercises cropping and partial 8x8 / 32x32 blocks in the small sub-bands.
gen solid_160x96_420 --pattern solid --width 160 --height 96 --chroma 420 --frames 3 --bytes-per-frame 40000
gen gradient_200x120_420 --pattern gradient --width 200 --height 120 --chroma 420 --frames 3 --bytes-per-frame 60000
gen noise_192x136_420 --pattern noise --width 192 --height 136 --chroma 420 --frames 3 --bytes-per-frame 48000
# Tight budgets force coarser quantisation (non-trivial quant codes and plane counts).
gen noise_192x136_420_lowrate --pattern noise --width 192 --height 136 --chroma 420 --frames 3 --bytes-per-frame 12000
# 4:4:4 with odd dimensions.
gen noise_131x67_444 --pattern noise --width 131 --height 67 --chroma 444 --frames 2 --bytes-per-frame 20000
