#!/usr/bin/env bash
# Decodes 1000 4K frames and watches the memory of the process: half of the
# leak check of milestone M5. The other half, which counts the GPU resources,
# is the ignored test a_thousand_4k_frames_leave_no_trace in
# crates/emberwave-core/tests/gpu.rs.
#
# Usage: scripts/soak.sh [frames]      (default 1000)
#
# Runs `emberwave-cli bench --output rgb --readback` on the 4K stream of
# scripts/gen_bench_data.sh (generated first if missing), samples the
# resident memory twice a second, and fails if any later sample is more than
# LIMIT_MIB above the first sample after the first second. The peak counts,
# not the last sample, which may be taken while the process shuts down.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
FRAMES=${1:-1000}
LIMIT_MIB=16
STREAM=$ROOT/testdata/generated/scene_4k_420.pwv1
[ -f "$STREAM" ] || "$ROOT/scripts/gen_bench_data.sh" --sizes 4k

cargo build --release -p emberwave-cli --manifest-path "$ROOT/Cargo.toml"
"${CARGO_TARGET_DIR:-$ROOT/target}/release/emberwave-cli" bench "$STREAM" --frames "$FRAMES" --output rgb \
    --readback --warmup 10 --warmup-seconds 0 > /dev/null &
pid=$!

# Resident set size in KiB, twice a second, while the decoder runs.
samples=()
while kill -0 "$pid" 2>/dev/null; do
    rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    if [ -n "$rss" ]; then
        samples+=("$rss")
    fi
    sleep 0.5
done
wait "$pid"

count=${#samples[@]}
if [ "$count" -lt 4 ]; then
    echo "only $count samples: the run was too short to judge" >&2
    exit 1
fi
# The first second is start-up: buffers are created then.
first=${samples[2]}
last=${samples[$((count - 1))]}
peak=$(printf '%s\n' "${samples[@]:2}" | sort -n | tail -1)
echo "resident memory over $FRAMES frames ($count samples): $((first / 1024)) MiB after" \
    "the first second, peak $((peak / 1024)) MiB, $((last / 1024)) MiB at the end"
growth_kib=$((peak - first))
if [ "$growth_kib" -gt $((LIMIT_MIB * 1024)) ]; then
    echo "it grew by $((growth_kib / 1024)) MiB, more than $LIMIT_MIB MiB" >&2
    exit 1
fi
echo "no growth beyond $LIMIT_MIB MiB"
