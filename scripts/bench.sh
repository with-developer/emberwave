#!/usr/bin/env bash
# Measures decode times on this machine and rewrites the generated tables of
# BENCHMARKS.md and README.md.
#
# Usage: scripts/bench.sh [--frames N] [--no-upstream]
#
#   --frames N     frames measured per number (default 600: ten passes over
#                  each 60-frame stream, so that the 99th percentile is not
#                  simply the slowest frame)
#   --no-upstream  do not time upstream's Metal decoder for comparison
#
# The streams come from scripts/gen_bench_data.sh, which is run first if they
# are missing (that, and timing upstream, need Apple Silicon macOS). Raw
# results stay in target/bench/ as JSON. Needs python3 for the report.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
FRAMES=600
UPSTREAM=1
while [ $# -gt 0 ]; do
    case "$1" in
        --frames) FRAMES=$2; shift 2 ;;
        --no-upstream) UPSTREAM=0; shift ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

STREAMS="scene_1080p_420 scene_1440p_420 scene_4k_420 noise_4k_420"
GENERATED=$ROOT/testdata/generated
OUT=$ROOT/target/bench
mkdir -p "$OUT"

for stream in $STREAMS; do
    if [ ! -f "$GENERATED/$stream.pwv1" ]; then
        "$ROOT/scripts/gen_bench_data.sh" --noise
        break
    fi
done

cargo build --release -p emberwave-cli --manifest-path "$ROOT/Cargo.toml"
CLI=${CARGO_TARGET_DIR:-$ROOT/target}/release/emberwave-cli
if [ "$UPSTREAM" = 1 ]; then
    # shellcheck source=refgen/build.sh
    source "$ROOT/scripts/refgen/build.sh"
fi

python3 "$ROOT/scripts/bench_report.py" environment "$OUT/environment.json"
for stream in $STREAMS; do
    input=$GENERATED/$stream.pwv1
    echo "== $stream"
    "$CLI" bench "$input" --frames "$FRAMES" --output yuv --json > "$OUT/$stream.yuv.json"
    "$CLI" bench "$input" --frames "$FRAMES" --output rgb --json > "$OUT/$stream.rgb.json"
    "$CLI" bench "$input" --frames "$FRAMES" --output rgb --readback --json > "$OUT/$stream.rgb-readback.json"
    if [ "$UPSTREAM" = 1 ]; then
        "$REFGEN" bench "$input" --frames "$FRAMES" --json > "$OUT/$stream.upstream.json"
        PYROWAVE_PRECISION=2 "$REFGEN" bench "$input" --frames "$FRAMES" --json \
            > "$OUT/$stream.upstream-fp32.json"
    fi
done

python3 "$ROOT/scripts/bench_report.py" write "$OUT" "$ROOT/BENCHMARKS.md" "$ROOT/README.md"
echo "updated BENCHMARKS.md and README.md from $OUT"
