#!/usr/bin/env bash
# Builds the browser player into web/: the wasm module (web/pkg/) and the
# sample streams (web/samples/). Serve web/ with any static file server to
# try it, for example: python3 -m http.server -d web 8000
#
# Usage: scripts/build_web.sh
#
# Needs the wasm32-unknown-unknown target (rustup target add
# wasm32-unknown-unknown) and wasm-bindgen-cli at the version of the
# wasm-bindgen crate in Cargo.lock, which the two must agree on.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
# Where cargo puts what it builds, however that is configured: an
# environment variable or a cargo configuration file.
TARGET_DIR=$(cargo metadata --format-version 1 --no-deps --manifest-path "$ROOT/Cargo.toml" |
    sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')
test -n "$TARGET_DIR"

WANT=$(awk '/^name = "wasm-bindgen"$/ { getline; gsub(/"/, "", $3); print $3 }' "$ROOT/Cargo.lock")
HAVE=$(wasm-bindgen --version 2>/dev/null | awk '{ print $2 }' || true)
if [ "$HAVE" != "$WANT" ]; then
    echo "needs wasm-bindgen-cli $WANT (found ${HAVE:-none}):" \
        "cargo install wasm-bindgen-cli --version $WANT" >&2
    exit 1
fi

cargo build --release --locked --target wasm32-unknown-unknown -p emberwave-web \
    --manifest-path "$ROOT/Cargo.toml"
wasm-bindgen --target web --no-typescript --out-dir "$ROOT/web/pkg" \
    "$TARGET_DIR/wasm32-unknown-unknown/release/emberwave_web.wasm"

mkdir -p "$ROOT/web/samples"
cp "$ROOT"/testdata/*.pwv1 "$ROOT/web/samples/"
echo "built web/"
