// Instantiates the module of the browser pages, which scripts/build_web.sh
// built into web/pkg/, outside a browser: every import of the glue
// wasm-bindgen generated must resolve, and the classes the pages use must be
// there and callable. CI runs it after the build, so that a broken module
// shows before it is published.
//
//     node scripts/check_web.mjs

import { readFileSync } from "node:fs";

const pkg = new URL("../web/pkg/", import.meta.url);
// The glue is an ES module in a .js file, which Node takes for CommonJS
// without a package.json saying otherwise: imported from its text instead.
const glue = readFileSync(new URL("emberwave_web.js", pkg));
const module = await import(`data:text/javascript;base64,${glue.toString("base64")}`);
module.initSync({ module: readFileSync(new URL("emberwave_web_bg.wasm", pkg)) });

const { Demo, Player, Bench } = module;
for (const [name, value] of [
  ["Player.create", Player?.create],
  ["Player.max_file_bytes", Player?.max_file_bytes],
  ["Bench.create", Bench?.create],
  ["Demo", Demo],
  ["Demo.frames", Demo?.frames],
]) {
  if (typeof value !== "function") {
    throw new Error(`${name} is missing from the module`);
  }
}
const most = Player.max_file_bytes();
if (!(most > 0)) {
  throw new Error(`Player.max_file_bytes() gave ${most}`);
}
console.log(`the browser module instantiates; the player loads files of up to ${most / 2 ** 20} MiB`);

// The demo's pictures are drawn and encoded on the CPU, which needs no GPU:
// two of them make a PWV1 file of two frames.
const demo = new Demo();
demo.encode_next();
const encoded = demo.encode_next();
const file = demo.take_file();
demo.free();
const magic = new TextDecoder().decode(file.subarray(0, 4));
const sizes = [];
for (let at = 24; at + 4 <= file.length;) {
  const size = new DataView(file.buffer, file.byteOffset + at, 4).getUint32(0, true);
  if (size === 0 || at + 4 + size > file.length) {
    throw new Error(`the demo's frame ${sizes.length + 1} has ${size} bytes, past the file's end`);
  }
  sizes.push(size);
  at += 4 + size;
}
if (encoded !== 2 || magic !== "PWV1" || sizes.length !== 2 || !(Demo.frames() > 2)) {
  throw new Error(`the demo made ${encoded} pictures, a file starting "${magic}" with ${sizes.length} frames`);
}
console.log(`the demo encodes: ${sizes.join(" and ")} bytes for its first two of ${Demo.frames()} pictures`);
