#!/usr/bin/env bash
# Makes the test vectors of emberwave's encoder in testdata/encoder/, from the
# list in testdata/encoder/vectors.txt. For each:
#
#   NAME.source.y4m    the pictures, synthesised by refgen (scripts/refgen)
#   NAME.pwv1          emberwave's CPU encoder's stream of them
#   NAME.ref-fp32.y4m  upstream's decoder's FP32 decode of that stream
#
# tests/encoder.rs encodes the sources again and expects the same bytes, and
# compares emberwave's decode of the streams with upstream's. Decoder
# conformance stays with the upstream-encoded vectors in testdata/ (decision
# D28). Needs Apple Silicon macOS for upstream's Metal decoder.
#
# Usage: scripts/gen_encoder_vectors.sh
set -euo pipefail

# shellcheck source=refgen/build.sh
source "$(dirname "$0")/refgen/build.sh"

cargo build --release --locked -p emberwave-cli --manifest-path "$ROOT/Cargo.toml"
CLI=${CARGO_TARGET_DIR:-$ROOT/target}/release/emberwave-cli
OUT=$ROOT/testdata/encoder
SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT

grep -v '^#' "$OUT/vectors.txt" | while read -r name pattern width height chroma frames budget; do
    [ -n "$name" ] || continue
    echo "== $name"
    # refgen encodes with upstream too; only the pictures are kept, so it gets
    # a budget that any picture fits.
    "$REFGEN" encode --out "$SCRATCH/$name.upstream.pwv1" --pattern "$pattern" \
        --width "$width" --height "$height" --chroma "$chroma" --frames "$frames" \
        --bytes-per-frame 4000000 --source "$OUT/$name.source.y4m"
    "$CLI" encode "$OUT/$name.source.y4m" --out "$OUT/$name.pwv1" --bytes-per-frame "$budget"
    PYROWAVE_PRECISION=2 "$REFGEN" decode "$OUT/$name.pwv1" "$OUT/$name.ref-fp32.y4m"
done
