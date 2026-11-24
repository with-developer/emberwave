#!/usr/bin/env bash
# Checks the npm package that scripts/build_npm.sh built into
# target/npm/emberwave-web/ (decision D76): lists what `npm publish` would
# upload, packs it, installs the archive into a new project as a page's
# dependency, from the file alone and without the registry, and runs
# scripts/check_web.mjs on the package installed there, which imports it by
# its name and checks its files. CI runs it after the build.
#
# Usage: scripts/check_npm.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PACKAGE=$ROOT/target/npm/emberwave-web
if [ ! -f "$PACKAGE/package.json" ]; then
    echo "no package in $PACKAGE: scripts/build_npm.sh builds it" >&2
    exit 1
fi
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# What would ship, and the sizes.
(cd "$PACKAGE" && npm pack --dry-run)

ARCHIVE=$(cd "$PACKAGE" && npm pack --silent --pack-destination "$WORK")
mkdir "$WORK/project"
echo '{ "private": true }' > "$WORK/project/package.json"
(cd "$WORK/project" &&
    npm install --offline --ignore-scripts --no-audit --no-fund "$WORK/$ARCHIVE")
node "$ROOT/scripts/check_web.mjs" "$WORK/project"
