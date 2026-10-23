#!/usr/bin/env bash
# Times the decoder on Mesa's lavapipe, a software Vulkan driver, in a Linux
# container: reference numbers for machines without a GPU, and for CI. The
# results go to target/bench/ next to those of scripts/bench.sh, which puts
# them into BENCHMARKS.md (or run `scripts/bench_report.py write` yourself).
#
# Usage: scripts/bench_lavapipe.sh [--frames N]     (default 10: lavapipe is slow)
#
# Needs Docker (any engine that runs Linux containers) and the streams of
# scripts/gen_bench_data.sh. The first run builds emberwave-cli inside the
# container, which takes a few minutes; named volumes keep the build for the
# next run.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
FRAMES=10
while [ $# -gt 0 ]; do
    case "$1" in
        --frames) FRAMES=$2; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

# Debian 13 for a recent Mesa, and the Rust release this was last run with.
IMAGE=rust:1.94-trixie
OUT=$ROOT/target/bench
mkdir -p "$OUT"

docker run --rm \
    -v "$ROOT:/src:ro" \
    -v "$OUT:/out" \
    -v emberwave-lavapipe-registry:/usr/local/cargo/registry \
    -v emberwave-lavapipe-target:/target \
    -e FRAMES="$FRAMES" \
    "$IMAGE" bash -euo pipefail -c '
        apt-get update -qq
        apt-get install -y -qq --no-install-recommends mesa-vulkan-drivers libvulkan1 > /dev/null
        CARGO_TARGET_DIR=/target cargo build --release --locked -p emberwave-cli \
            --manifest-path /src/Cargo.toml
        cli=/target/release/emberwave-cli
        for stream in scene_1080p_420 scene_1440p_420 scene_4k_420; do
            echo "== $stream"
            "$cli" bench "/src/testdata/generated/$stream.pwv1" --backend vulkan --software \
                --output rgb --frames "$FRAMES" --warmup 2 --warmup-seconds 0 --json \
                > "/out/$stream.lavapipe.json"
        done
        mesa=$(dpkg-query -W -f "\${Version}" mesa-vulkan-drivers)
        printf "{\"date\":\"%s\",\"os\":\"Debian %s\",\"arch\":\"%s\",\"cpus\":%s,\"mesa\":\"%s\"}\n" \
            "$(date +%F)" "$(cat /etc/debian_version)" "$(uname -m)" "$(nproc)" "$mesa" \
            > /out/lavapipe-environment.json
    '
echo "lavapipe results are in $OUT"
