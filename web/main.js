// The page of the emberwave player: loads a .pwv1 file and plays it with the
// wasm module that scripts/build_web.sh builds into pkg/.
import init, { Player } from "./pkg/emberwave_web.js";

// The streams in testdata/, copied to samples/ by scripts/build_web.sh.
const SAMPLES = [
  "gradient_200x120_420",
  "noise_192x136_420",
  "noise_192x136_420_lowrate",
  "noise_131x67_444",
  "solid_160x96_420",
];

const canvas = document.getElementById("screen");
const info = document.getElementById("info");
const stats = document.getElementById("stats");

let player;
// Set once the module failed for good: a panic in it leaves its objects in
// use, so that every later call fails, until the page is loaded again.
let broken = false;
// While a player is made anew for one whose GPU was lost.
let renewing = null;
// Bumped for every file, so that the loop of the previous one stops.
let generation = 0;
// Bumped for every file asked for, so that only the last one asked for plays
// when reading an earlier one takes longer.
let request = 0;

function fail(message) {
  generation += 1;
  info.textContent = message;
  info.className = "error";
  stats.textContent = "";
}

// Reports what went wrong with a file. A panic in the module, which
// WebAssembly reports as a RuntimeError, stops the player for good; a lost
// GPU only until the next file, which gets a new player.
function failWith(name, error) {
  if (error instanceof WebAssembly.RuntimeError ||
      String(error).includes("recursive use of an object")) {
    broken = true;
    fail(`${name}: the player stopped working (${error}). Reload the page to start over.`);
    for (const control of document.querySelectorAll("input, button")) {
      control.disabled = true;
    }
    return;
  }
  if (player?.is_lost()) {
    console.error(error);
    fail(`${name}: the GPU device was lost. The next file or sample starts again on a new one.`);
    return;
  }
  fail(`${name}: ${error}`);
}

// Makes a new player if the GPU of the one there is was lost, or if making
// one failed before.
async function renewIfLost() {
  if (renewing) {
    await renewing;
  }
  if (player && !player.is_lost()) {
    return;
  }
  // The loop of the old file draws with the old player: it stops first.
  generation += 1;
  renewing = (async () => {
    player?.free();
    player = null;
    player = await Player.create(canvas);
  })();
  try {
    await renewing;
  } finally {
    renewing = null;
  }
}

async function play(bytes, name) {
  if (broken) {
    return;
  }
  // Checked before the bytes are copied into the module's memory.
  const most = Player.max_file_bytes();
  if (bytes.byteLength > most) {
    fail(`${name}: the file has ${Math.floor(bytes.byteLength / 2 ** 20)} MiB, ` +
      `and the player loads at most ${most / 2 ** 20} MiB`);
    return;
  }
  let description;
  try {
    await renewIfLost();
    if (broken) {
      return;
    }
    description = player.load(new Uint8Array(bytes));
  } catch (error) {
    failWith(name, error);
    return;
  }
  // The canvas has the size of the picture, so that a pixel is a fragment.
  canvas.width = player.width();
  canvas.height = player.height();
  info.className = "";
  info.textContent = `${name}: ${description}. GPU: ${player.adapter()}.`;
  stats.textContent = "";

  const mine = ++generation;
  const interval = 1000 / player.frame_rate();
  // When the next frame is due. Advancing it by the interval, not to the
  // time of drawing, keeps a rate that does not divide the display's.
  let due = 0;
  let frames = 0;
  let cpu = 0;
  let since = performance.now();

  function frame(now) {
    if (mine !== generation) {
      return;
    }
    // A millisecond of slack, as animation frames do not come exactly on time.
    if (now >= due - 1) {
      due += interval;
      // After a pause, such as a hidden tab, start over rather than catch up.
      if (due < now) {
        due = now + interval;
      }
      try {
        cpu += player.draw_next();
      } catch (error) {
        failWith(name, error);
        return;
      }
      frames += 1;
      if (now - since >= 1000) {
        stats.textContent =
          `${(frames * 1000 / (now - since)).toFixed(1)} frames per second · ` +
          `CPU ${(cpu / frames).toFixed(2)} ms · ` +
          `submitted to done ${player.take_gpu_ms().toFixed(2)} ms per frame`;
        frames = 0;
        cpu = 0;
        since = now;
      }
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// Reads a file the user chose and plays it, unless another was chosen since.
async function playFile(file) {
  if (broken) {
    return;
  }
  const mine = ++request;
  let bytes;
  try {
    bytes = await file.arrayBuffer();
  } catch (error) {
    if (mine === request) {
      fail(`${file.name}: ${error}`);
    }
    return;
  }
  if (mine === request) {
    await play(bytes, file.name);
  }
}

async function playSample(name) {
  if (broken) {
    return;
  }
  const mine = ++request;
  let bytes;
  try {
    const response = await fetch(`samples/${name}.pwv1`);
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    bytes = await response.arrayBuffer();
  } catch (error) {
    if (mine === request) {
      fail(`${name}: ${error}`);
    }
    return;
  }
  if (mine === request) {
    await play(bytes, name);
  }
}

async function start() {
  // A file dropped anywhere is the player's; without this, a drop outside the
  // drop zone, or before the player is ready, would open the file instead.
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());

  if (!navigator.gpu) {
    fail("This browser has no WebGPU. A recent Chrome or Edge has it.");
    return;
  }
  try {
    await init();
    player = await Player.create(canvas);
  } catch (error) {
    fail(`WebGPU did not start: ${error}`);
    return;
  }
  info.textContent = `GPU: ${player.adapter()}. Open a file or play a sample.`;

  const input = document.getElementById("file");
  input.addEventListener("change", () => {
    const file = input.files[0];
    // Cleared, so that choosing the same file again plays it again.
    input.value = "";
    if (file) {
      playFile(file);
    }
  });

  const drop = document.getElementById("drop");
  drop.addEventListener("dragover", () => drop.classList.add("over"));
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  window.addEventListener("drop", (event) => {
    drop.classList.remove("over");
    const file = event.dataTransfer.files[0];
    if (file) {
      playFile(file);
    }
  });

  const samples = document.getElementById("samples");
  for (const name of SAMPLES) {
    const button = document.createElement("button");
    button.textContent = name;
    button.addEventListener("click", () => playSample(name));
    samples.append(button);
  }

  // ?play=NAME starts a sample straight away, for links to one.
  const requested = new URLSearchParams(location.search).get("play");
  if (SAMPLES.includes(requested)) {
    playSample(requested);
  }
}

start();
