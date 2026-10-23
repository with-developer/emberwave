#!/usr/bin/env bash
# Decodes a 4K stream with emberwave-cli and compares it with the original
# decoder: the acceptance check of the project (CLAUDE.md, section 9).
#
# It encodes a short 4K stream with the upstream encoder, decodes it with the
# upstream decoder at its default and its FP32 precision, decodes it with
# emberwave-cli to PNG and to Y4M, and requires at least 60 dB PSNR against the
# default-precision decode and 80 dB against the FP32 one (decision D9).
#
# Usage: scripts/verify_4k.sh [frames]      (default 2; needs Apple Silicon macOS)
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=refgen/build.sh
source "$ROOT/scripts/refgen/build.sh"

FRAMES=${1:-2}
WORK=$ROOT/testdata/generated/verify_4k
rm -rf "$WORK"
mkdir -p "$WORK"

"$REFGEN" encode --out "$WORK/scene_4k.pwv1" --pattern scene --width 3840 --height 2160 \
    --chroma 420 --frames "$FRAMES" --bytes-per-frame 1200000
"$REFGEN" decode "$WORK/scene_4k.pwv1" "$WORK/reference.y4m"
PYROWAVE_PRECISION=2 "$REFGEN" decode "$WORK/scene_4k.pwv1" "$WORK/reference-fp32.y4m"

cargo build --release -p emberwave-cli --manifest-path "$ROOT/Cargo.toml"
CLI=${CARGO_TARGET_DIR:-$ROOT/target}/release/emberwave-cli

"$CLI" decode "$WORK/scene_4k.pwv1" --out-dir "$WORK/out" --format png
"$CLI" decode "$WORK/scene_4k.pwv1" --out-dir "$WORK/out" --format y4m

echo "== against the default-precision decode (needs 60 dB)"
"$CLI" compare "$WORK/out/scene_4k.y4m" "$WORK/reference.y4m" --min-psnr 60
echo "== against the FP32 decode (needs 80 dB)"
"$CLI" compare "$WORK/out/scene_4k.y4m" "$WORK/reference-fp32.y4m" --min-psnr 80

ls -l "$WORK"/out/*.png
echo "4K verification passed"
