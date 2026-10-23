#!/usr/bin/env bash
# Measures decode and encode times on this machine and rewrites the generated
# tables of BENCHMARKS.md and README.md.
#
# Usage: scripts/bench.sh [--frames N] [--no-upstream]
#
#   --frames N     frames measured per number (default 600: ten passes over
#                  each 60-frame stream, so that the 99th percentile is not
#                  simply the slowest frame); the CPU encoder, which takes a
#                  quarter of a second for a 4K frame, gets a tenth of it
#   --no-upstream  do not time upstream's Metal decoder and encoder for
#                  comparison
#
# The streams and pictures come from scripts/gen_bench_data.sh, which is run
# first if they are missing (that, and timing upstream, need Apple Silicon
# macOS). Raw results stay in target/bench/ as JSON. Needs python3 for the
# report.
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
# The pictures the encoder benchmarks encode, and the bytes per frame they
# get: those of the decoder's streams of the same size.
ENCODE="scene_1080p_420:400000 scene_1440p_420:700000 scene_4k_420:1200000"
for entry in $ENCODE; do
    if [ ! -f "$GENERATED/${entry%%:*}.y4m" ]; then
        "$ROOT/scripts/gen_bench_data.sh" --sources
        break
    fi
done

cargo build --release -p emberwave-cli --manifest-path "$ROOT/Cargo.toml"
CLI=${CARGO_TARGET_DIR:-$ROOT/target}/release/emberwave-cli
if [ "$UPSTREAM" = 1 ]; then
    # shellcheck source=refgen/build.sh
    source "$ROOT/scripts/refgen/build.sh"
fi

# measure FILE COMMAND...: runs the command and keeps its JSON in FILE, which
# only appears when the command succeeded. The report reads only results
# newer than this run's environment.json, so none of an earlier run's slips
# in next to this run's.
measure() {
    local file=$1
    shift
    "$@" > "$file.partial"
    mv "$file.partial" "$file"
}

python3 "$ROOT/scripts/bench_report.py" environment "$OUT/environment.json"
for stream in $STREAMS; do
    input=$GENERATED/$stream.pwv1
    echo "== $stream"
    measure "$OUT/$stream.yuv.json" "$CLI" bench "$input" --frames "$FRAMES" --output yuv --json
    measure "$OUT/$stream.rgb.json" "$CLI" bench "$input" --frames "$FRAMES" --output rgb --json
    measure "$OUT/$stream.rgb-readback.json" \
        "$CLI" bench "$input" --frames "$FRAMES" --output rgb --readback --json
    if [ "$UPSTREAM" = 1 ]; then
        measure "$OUT/$stream.upstream.json" "$REFGEN" bench "$input" --frames "$FRAMES" --json
        PYROWAVE_PRECISION=2 measure "$OUT/$stream.upstream-fp32.json" \
            "$REFGEN" bench "$input" --frames "$FRAMES" --json
    fi
done

# The CPU encoder takes a quarter of a second for a 4K frame: a tenth of the
# frames, one at least.
CPU_FRAMES=$(( FRAMES >= 10 ? FRAMES / 10 : 1 ))
for entry in $ENCODE; do
    source=${entry%%:*}
    budget=${entry##*:}
    input=$GENERATED/$source.y4m
    echo "== $source, encoded"
    measure "$OUT/$source.encode-gpu.json" \
        "$CLI" bench-encode "$input" --bytes-per-frame "$budget" --gpu --frames "$FRAMES" --json
    measure "$OUT/$source.encode-cpu.json" \
        "$CLI" bench-encode "$input" --bytes-per-frame "$budget" --frames "$CPU_FRAMES" --json
    if [ "$UPSTREAM" = 1 ]; then
        measure "$OUT/$source.encode-upstream.json" \
            "$REFGEN" bench-encode "$input" --bytes-per-frame "$budget" --frames "$FRAMES" --json
        PYROWAVE_PRECISION=2 measure "$OUT/$source.encode-upstream-fp32.json" \
            "$REFGEN" bench-encode "$input" --bytes-per-frame "$budget" --frames "$FRAMES" --json
    fi
done

python3 "$ROOT/scripts/bench_report.py" write "$OUT" "$ROOT/BENCHMARKS.md" "$ROOT/README.md"
echo "updated BENCHMARKS.md and README.md from $OUT"
