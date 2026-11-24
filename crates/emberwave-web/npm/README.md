<!-- The README of the npm package: scripts/build_npm.sh copies it into the package (decision D76 in ARCHITECTURE.md). -->

# emberwave-web

Decode [PyroWave](https://github.com/Themaister/pyrowave) video in a web page, on the GPU through WebGPU. This is the WebAssembly module of [emberwave](https://github.com/with-developer/emberwave)'s browser player: it plays PWV1 files (`.pwv1`), or a stream packet by packet as a client receives it, and draws the pictures on a canvas. [Try the player](https://with-developer.github.io/emberwave/).

It needs a browser with WebGPU: `Player.create` fails without it. The pictures are decoded on the GPU and drawn there, straight onto the canvas.

## Install

```sh
npm install emberwave-web
```

## Play a file

```js
import init, { Player } from "emberwave-web";

// Fetches emberwave_web_bg.wasm from beside the module. Where a bundler moves
// the module, pass the file's URL: init({ module_or_path: url }); the package
// exports the file as "emberwave-web/emberwave_web_bg.wasm".
await init();

const canvas = document.querySelector("canvas");
const player = await Player.create(canvas);
const bytes = new Uint8Array(await (await fetch("stream.pwv1")).arrayBuffer());
console.log(player.load(bytes)); // the size, the frames and the frame rate
canvas.width = player.width();
canvas.height = player.height();

// The file's frames in turn, at the stream's frame rate, looping.
const interval = 1000 / player.frame_rate();
let due = 0;
requestAnimationFrame(function frame(now) {
  if (now >= due) {
    due += interval;
    if (due < now) {
      due = now + interval; // after a pause, start again rather than catch up
    }
    player.draw_next();
  }
  requestAnimationFrame(frame);
});
```

## Play a stream as packets arrive

```js
player.open_stream(1920, 1080, false, 60); // 4:2:0; true for 4:4:4

// For each packet of the frame being assembled, in any order. The transport
// numbers its frames: it ignores packets of frames older than this one.
player.push_packet(packet);

// Before the first packet of a newer frame: draw this one, whole, or in part
// if enough of it arrived, then let go of its packets.
if (player.is_ready(true)) {
  player.draw_assembled();
}
player.drop_packets();
```

`packet_ends(frame, maxBytes)` says where a host would split an encoded frame into packets. `set_partial_share` and `set_concealment` choose how a frame that lost packets is drawn, and `received_blocks` and `total_blocks` count what arrived of it.

## Good to know

- Methods throw an `Error` with a message for a file, a packet or a call they cannot take.
- A player holds GPU memory: call `player.free()` when done with it. Loading another file or opening another stream lets go of the last one's.
- If `player.is_lost()`, the GPU device is gone: make a new player.
- `Demo` draws and encodes a demo stream on the CPU, and `Bench` times the decoding on the browser's GPU, as the player's pages do.
- The TypeScript declarations, `emberwave_web.d.ts`, document every method.
- The version is emberwave's. Before 1.0.0 the API may change between minor versions; [the changelog](https://github.com/with-developer/emberwave/blob/main/CHANGELOG.md) says what changed.

## License

MIT, in `LICENSE`. `THIRD_PARTY_NOTICES.md` holds PyroWave's license.
