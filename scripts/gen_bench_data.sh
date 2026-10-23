#!/usr/bin/env bash
# Generates large vectors for benchmarks and for checking big pictures, with
# the upstream PyroWave encoder (the Metal port, so this needs Apple Silicon
# macOS). The files are not committed: they go to testdata/generated/, which is
# git-ignored, and can be regenerated at any time.
#
# Usage: scripts/gen_bench_data.sh [--frames N] [--sizes "1080p 1440p 4k"] [--noise]
#
#   --frames N   frames per vector (default 60)
#   --sizes      which resolutions to generate (default: all three)
#   --noise      also generate a 4K stream of white noise, the worst case for
#                the amount of data per frame
#
# The pictures are the synthetic "scene" of scripts/refgen: colour bars, a
# gradient, a moving disc and a fine grid. The byte budget per frame is roughly
# what a 400 to 800 Mbit/s stream at 60 fps would carry.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=refgen/build.sh
source "$ROOT/scripts/refgen/build.sh"

FRAMES=60
SIZES="1080p 1440p 4k"
NOISE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --frames) FRAMES=$2; shift 2 ;;
        --sizes) SIZES=$2; shift 2 ;;
        --noise) NOISE=1; shift ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

OUT=$ROOT/testdata/generated
mkdir -p "$OUT"

# gen NAME PATTERN WIDTH HEIGHT BYTES_PER_FRAME
gen() {
    echo "== $1 ($3x$4, $FRAMES frames)"
    "$REFGEN" encode --out "$OUT/$1.pwv1" --pattern "$2" --width "$3" --height "$4" \
        --chroma 420 --frames "$FRAMES" --bytes-per-frame "$5"
}

for size in $SIZES; do
    case "$size" in
        1080p) gen scene_1080p_420 scene 1920 1080 400000 ;;
        1440p) gen scene_1440p_420 scene 2560 1440 700000 ;;
        4k) gen scene_4k_420 scene 3840 2160 1200000 ;;
        *) echo "unknown size $size (use 1080p, 1440p or 4k)" >&2; exit 2 ;;
    esac
done

if [ "$NOISE" = 1 ]; then
    gen noise_4k_420 noise 3840 2160 3000000
fi

ls -l "$OUT"/*.pwv1
