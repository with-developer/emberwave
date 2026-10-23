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

function fail(message) {
  info.textContent = message;
  info.className = "error";
}

let player;
// Bumped for every file, so that the loop of the previous one stops.
let generation = 0;

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
  let last = -Infinity;
  let frames = 0;
  let cpu = 0;
  let gpu = 0;
  let since = performance.now();

  function frame(now) {
    if (mine !== generation) {
      return;
    }
    if (now - last >= interval - 1) {
      last = now;
      try {
        cpu += player.draw_next();
        gpu += player.last_gpu_ms();
      } catch (error) {
        fail(`${name}: ${error}`);
        return;
      }
      frames += 1;
      if (now - since >= 1000) {
        stats.textContent =
          `${(frames * 1000 / (now - since)).toFixed(1)} frames per second · ` +
          `CPU ${(cpu / frames).toFixed(2)} ms · ` +
          `submitted to done ${(gpu / frames).toFixed(2)} ms per frame`;
        frames = 0;
        cpu = 0;
        gpu = 0;
        since = now;
      }
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

async function start() {
  if (!navigator.gpu) {
    fail("This browser has no WebGPU. A recent Chrome or Edge has it.");
    return;
  }
  await init();
  try {
    player = await Player.create(canvas);
  } catch (error) {
    fail(`WebGPU did not start: ${error}`);
    return;
  }
  info.textContent = `GPU: ${player.adapter()}. Open a file or play a sample.`;

  document.getElementById("file").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (file) {
      play(await file.arrayBuffer(), file.name);
    }
  });

  const drop = document.getElementById("drop");
  drop.addEventListener("dragover", (event) => {
    event.preventDefault();
    drop.classList.add("over");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", async (event) => {
    event.preventDefault();
    drop.classList.remove("over");
    const file = event.dataTransfer.files[0];
    if (file) {
      play(await file.arrayBuffer(), file.name);
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

async function playSample(name) {
  const response = await fetch(`samples/${name}.pwv1`);
  if (!response.ok) {
    fail(`${name}: ${response.status} ${response.statusText}`);
    return;
  }
  play(await response.arrayBuffer(), name);
}

start();
