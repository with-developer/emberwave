#!/usr/bin/env bash
# Builds the browser player's module as an npm package into
# target/npm/emberwave-web/, for web pages that take it as a dependency: the
# module and its JavaScript glue, as scripts/build_web.sh builds them for the
# pages, with TypeScript declarations this time, a package.json at the crate's
# version, the package's README (crates/emberwave-web/npm/README.md), LICENSE
# and THIRD_PARTY_NOTICES.md. scripts/check_npm.sh checks what it built.
#
# The package is ready to publish, and not published: publishing it, and the
# name it takes on npm, are the maintainer's (decision D76).
#
# Usage: scripts/build_npm.sh
#
# Needs what scripts/build_web.sh needs (the wasm32-unknown-unknown target and
# wasm-bindgen-cli at the version of the wasm-bindgen crate in Cargo.lock),
# and Node.
set -euo pipefail

# The name proposed for npm (D76), which the maintainer confirms before the
# first publish. The package's README names it too, which is checked below.
NAME=emberwave-web

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/target/npm/emberwave-web
README=$ROOT/crates/emberwave-web/npm/README.md

# Where cargo puts what it builds, however that is configured, and the
# crate's version, which is the workspace's.
TARGET_DIR=
VERSION=
{ IFS= read -r TARGET_DIR; IFS= read -r VERSION; } < <(
    cargo metadata --format-version 1 --no-deps --manifest-path "$ROOT/Cargo.toml" |
        node -e 'const metadata = JSON.parse(require("fs").readFileSync(0, "utf8"));
            console.log(metadata.target_directory);
            console.log(metadata.packages.find((p) => p.name === "emberwave-web").version);'
) || true
if [ -z "$TARGET_DIR" ] || [ -z "$VERSION" ]; then
    echo "found no target directory or version in cargo's metadata (needs cargo and node)" >&2
    exit 1
fi

WANT=$(awk '/^name = "wasm-bindgen"$/ { getline; gsub(/"/, "", $3); print $3 }' "$ROOT/Cargo.lock")
HAVE=$(wasm-bindgen --version 2>/dev/null | awk '{ print $2 }' || true)
if [ "$HAVE" != "$WANT" ]; then
    echo "needs wasm-bindgen-cli $WANT (found ${HAVE:-none}):" \
        "cargo install wasm-bindgen-cli --version $WANT" >&2
    exit 1
fi

# The examples of the README import the package by its name.
if ! grep -q "^npm install $NAME\$" "$README" || ! grep -q "from \"$NAME\";" "$README"; then
    echo "$README does not name the package $NAME in its examples" >&2
    exit 1
fi

# The wasm keeps the paths of the source files its panic messages name,
# which on a personal machine are under the user's home directory. The
# package is built with them remapped, so that it says nothing about the
# machine that built it; the check after wasm-bindgen fails if one is left.
# scripts/build_web.sh remaps them alike, so that the pages' wasm is this
# one and cargo builds it once. Last match wins: the home directory, then
# cargo's sources, then this repository. Separated by 0x1f, so that a path
# may hold spaces; this replaces RUSTFLAGS for this build.
CARGO_SOURCES=${CARGO_HOME:-$HOME/.cargo}
SEP=$'\x1f'
export CARGO_ENCODED_RUSTFLAGS="--remap-path-prefix=$HOME=/home${SEP}--remap-path-prefix=$CARGO_SOURCES=/cargo${SEP}--remap-path-prefix=$ROOT=/emberwave"
cargo build --release --locked --target wasm32-unknown-unknown -p emberwave-web \
    --manifest-path "$ROOT/Cargo.toml"
unset CARGO_ENCODED_RUSTFLAGS

rm -rf "$OUT"
# The pages' target, an ES module whose default export fetches the wasm from
# beside it or takes it from the caller, so that a page uses the package as
# web/main.js uses web/pkg. The declarations of the wasm's own exports,
# emberwave_web_bg.wasm.d.ts, are for importing the wasm directly, which
# this target does not do: they do not ship.
wasm-bindgen --target web --out-dir "$OUT" \
    "$TARGET_DIR/wasm32-unknown-unknown/release/emberwave_web.wasm"
rm "$OUT/emberwave_web_bg.wasm.d.ts"
for path in "$HOME/" "$ROOT/"; do
    if LC_ALL=C grep -q -a -F -- "$path" "$OUT/emberwave_web_bg.wasm"; then
        echo "the wasm names $path, which the package must not ship" >&2
        rm -rf "$OUT"
        exit 1
    fi
done

cp "$README" "$OUT/README.md"
cp "$ROOT/LICENSE" "$ROOT/THIRD_PARTY_NOTICES.md" "$OUT/"

# "files" is what ships, beside package.json; check_npm.sh checks the archive
# against the same list. The glue has no effect on import beyond defining
# its exports, hence "sideEffects": false.
cat > "$OUT/package.json" <<EOF
{
  "name": "$NAME",
  "version": "$VERSION",
  "description": "Decode PyroWave video in the browser with WebGPU: the WebAssembly module of emberwave's player",
  "license": "MIT",
  "repository": {
    "type": "git",
    "url": "git+https://github.com/with-developer/emberwave.git",
    "directory": "crates/emberwave-web"
  },
  "homepage": "https://github.com/with-developer/emberwave#readme",
  "bugs": "https://github.com/with-developer/emberwave/issues",
  "keywords": [
    "pyrowave",
    "video",
    "codec",
    "decoder",
    "wavelet",
    "webgpu",
    "wasm",
    "webassembly",
    "game-streaming"
  ],
  "type": "module",
  "main": "./emberwave_web.js",
  "module": "./emberwave_web.js",
  "types": "./emberwave_web.d.ts",
  "exports": {
    ".": {
      "types": "./emberwave_web.d.ts",
      "default": "./emberwave_web.js"
    },
    "./emberwave_web_bg.wasm": "./emberwave_web_bg.wasm",
    "./package.json": "./package.json"
  },
  "files": [
    "emberwave_web.js",
    "emberwave_web.d.ts",
    "emberwave_web_bg.wasm",
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md"
  ],
  "sideEffects": false
}
EOF
echo "built $OUT: $NAME $VERSION"
