// Instantiates the module of the browser pages, which scripts/build_web.sh
// built into web/pkg/, outside a browser: every import of the glue
// wasm-bindgen generated must resolve, and the classes the pages use must be
// there and callable. CI runs it after the build, so that a broken module
// shows before it is published.
//
//     node scripts/check_web.mjs
//
// Given a scratch project's directory, it checks the npm package installed
// there instead, the project's one dependency (decision D76): its files are
// those that ship, every file its package.json points to is one of them, its
// declarations declare the module's exports, and the module, imported by the
// package's name as a page's code would, does what the pages' does.
// scripts/check_npm.sh makes such a project; the check writes one file into
// it, the module that imports the package.
//
//     node scripts/check_web.mjs PROJECT_DIR

import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";

// What the package ships, as its package.json's "files" says, with
// package.json itself.
const PACKAGE_FILES = [
  "LICENSE",
  "README.md",
  "THIRD_PARTY_NOTICES.md",
  "emberwave_web.d.ts",
  "emberwave_web.js",
  "emberwave_web_bg.wasm",
  "package.json",
];

// The module of the pages, from web/pkg/.
async function pagesModule() {
  const pkg = new URL("../web/pkg/", import.meta.url);
  // The glue is an ES module in a .js file, which Node takes for CommonJS
  // without a package.json saying otherwise: imported from its text instead.
  const glue = readFileSync(new URL("emberwave_web.js", pkg));
  const module = await import(`data:text/javascript;base64,${glue.toString("base64")}`);
  module.initSync({ module: readFileSync(new URL("emberwave_web_bg.wasm", pkg)) });
  return module;
}

// The module of the npm package installed in `project`.
async function packageModule(project) {
  const manifest = JSON.parse(readFileSync(join(project, "package.json"), "utf8"));
  const names = Object.keys(manifest.dependencies ?? {});
  if (names.length !== 1) {
    throw new Error(`${project} has ${names.length} dependencies, not the package alone`);
  }
  const [name] = names;
  // Node resolves these through the package's "exports".
  const require = createRequire(join(project, "package.json"));
  const wasm = require.resolve(`${name}/emberwave_web_bg.wasm`);
  const own = require.resolve(`${name}/package.json`);
  const root = dirname(own);
  const files = readdirSync(root).sort();
  if (files.join() !== PACKAGE_FILES.join()) {
    throw new Error(`the package holds ${files.join(", ")}, not ${PACKAGE_FILES.join(", ")}`);
  }
  const shipped = JSON.parse(readFileSync(own, "utf8"));
  // Checked here, as Node since 20.19 and 22.7 loads the glue as an ES module
  // without it too, by its syntax; bundlers and older Node do not.
  if (shipped.type !== "module") {
    throw new Error(`the package's "type" is ${shipped.type}, not "module"`);
  }
  // Every path that package.json gives, "exports" included, under any
  // condition, is one of the files.
  const leaves = (value) =>
    typeof value === "string" ? [value] : Object.values(value ?? {}).flatMap(leaves);
  for (const path of [shipped.main, shipped.module, shipped.types, ...leaves(shipped.exports)]) {
    if (typeof path !== "string" || !existsSync(join(root, path))) {
      throw new Error(`the package's package.json points to ${path}, which it does not hold`);
    }
  }
  // TypeScript takes the first condition that applies, so "types" comes
  // first, and it is the declarations that are checked below.
  const main = shipped.exports?.["."] ?? {};
  if (Object.keys(main)[0] !== "types" || main.types !== shipped.types) {
    throw new Error(`the package's exports["."] does not start with "types": ${JSON.stringify(shipped.types)}`);
  }
  const text = readFileSync(join(root, shipped.types), "utf8");
  for (const declaration of [
    "export class Player ",
    "export class Demo ",
    "export class Bench ",
    "export function packet_ends(",
    "export function initSync(",
    "export default function ",
  ]) {
    if (!text.includes(declaration)) {
      throw new Error(`${shipped.types} lacks "${declaration.trim()}"`);
    }
  }
  // Imported by its name, from a module in the project, so that Node's ES
  // module resolution finds it as it would a page's import.
  const importer = join(project, "check-import.mjs");
  writeFileSync(importer, `export * from ${JSON.stringify(name)};\n` +
    `export { default } from ${JSON.stringify(name)};\n`);
  const module = await import(pathToFileURL(importer).href);
  module.initSync({ module: readFileSync(wasm) });
  console.log(`the npm package ${name} ${shipped.version} installs with its ${files.length} files and declarations`);
  return module;
}

const project = process.argv[2];
const module = project ? await packageModule(project) : await pagesModule();

const { Demo, Player, Bench, packet_ends } = module;
for (const [name, value] of [
  ["init", module.default],
  ["initSync", module.initSync],
  ["Player.create", Player?.create],
  ["Player.max_file_bytes", Player?.max_file_bytes],
  ["Player.prototype.set_partial_share", Player?.prototype.set_partial_share],
  ["Player.prototype.set_concealment", Player?.prototype.set_concealment],
  ["Player.prototype.received_blocks", Player?.prototype.received_blocks],
  ["Player.prototype.total_blocks", Player?.prototype.total_blocks],
  ["Bench.create", Bench?.create],
  ["Demo", Demo],
  ["Demo.frames", Demo?.frames],
  ["Demo.frame_budget", Demo?.frame_budget],
  ["packet_ends", packet_ends],
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
if (!sizes.every((size) => size <= Demo.frame_budget())) {
  throw new Error(`the demo's frames take ${sizes.join(" and ")} bytes, over ${Demo.frame_budget()}`);
}
console.log(`the demo encodes: ${sizes.join(" and ")} bytes for its first two of ${Demo.frames()} pictures`);

// The packets a frame splits into, for playing it as a network delivers it:
// they end in turn at the frame's end. A block larger than the size asked
// for goes alone, so a packet can be longer; most are not.
const first = file.subarray(28, 28 + sizes[0]);
const ends = packet_ends(first, 1200);
let start = 0;
let over = 0;
for (const end of ends) {
  if (!(end > start)) {
    throw new Error(`the packets end at ${[...ends].join(", ")}, not in turn`);
  }
  over += end - start > 1200 ? 1 : 0;
  start = end;
}
if (start !== first.length || ends.length < 2 || over * 2 > ends.length) {
  throw new Error(`a frame of ${first.length} bytes splits into ${ends.length} packets ending at ` +
    `${start}, ${over} of them over 1200 bytes`);
}
console.log(`a frame of ${first.length} bytes splits into ${ends.length} packets, ${over} of them over 1200 bytes`);
