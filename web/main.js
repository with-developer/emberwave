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

function play(bytes, name) {
  let description;
  try {
    description = player.load(new Uint8Array(bytes));
  } catch (error) {
    fail(`${name}: ${error}`);
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
        fail(`${name}: ${error}`);
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
    play(bytes, file.name);
  }
}

async function playSample(name) {
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
    play(bytes, name);
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
