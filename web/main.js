// The page of the emberwave player: loads a .pwv1 file and plays it with the
// wasm module that scripts/build_web.sh builds into pkg/.
import init, { Demo, Player, packet_ends } from "./pkg/emberwave_web.js";

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
const network = document.getElementById("network");
const asPackets = document.getElementById("packets");
const packetBytes = document.getElementById("packet-bytes");
const loss = document.getElementById("loss");

// What playing as packets has met since it was last shown.
const tally = { frames: 0, whole: 0, partial: 0, skipped: 0, packets: 0, lost: 0, malformed: 0 };

function resetTally() {
  Object.keys(tally).forEach((key) => { tally[key] = 0; });
}

// Sends `frame` to the player as a network would deliver it: cut at its
// blocks into packets of up to the chosen size, shuffled, and with the
// chosen share of them lost. Then draws what arrived, if enough did, and
// returns the CPU's milliseconds for the draw, or null for a frame skipped.
function playAsPackets(frame) {
  const ends = packet_ends(frame, Number(packetBytes.value));
  const packets = [];
  let start = 0;
  for (const end of ends) {
    packets.push(frame.subarray(start, end));
    start = end;
  }
  for (let i = packets.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [packets[i], packets[j]] = [packets[j], packets[i]];
  }
  const share = Number(loss.value);
  for (const packet of packets) {
    tally.packets += 1;
    if (Math.random() < share) {
      tally.lost += 1;
      continue;
    }
    try {
      player.push_packet(packet);
    } catch {
      tally.malformed += 1;
    }
  }
  tally.frames += 1;
  if (!player.is_ready(true)) {
    // Too little arrived: the frame is skipped, and the last one stays.
    tally.skipped += 1;
    player.drop_packets();
    return null;
  }
  if (player.is_ready(false)) {
    tally.whole += 1;
  } else {
    tally.partial += 1;
  }
  const cpu = player.draw_assembled();
  player.drop_packets();
  return cpu;
}

function showTally() {
  if (tally.frames === 0) {
    network.textContent = "";
    return;
  }
  network.textContent =
    `As packets: ${(tally.packets / tally.frames).toFixed(1)} a frame, ` +
    `${(100 * tally.lost / tally.packets).toFixed(1)} % lost · frames drawn whole ${tally.whole}, ` +
    `in part ${tally.partial}, skipped ${tally.skipped}` +
    (tally.malformed ? ` · malformed packets ${tally.malformed}` : "");
  resetTally();
}

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
  network.textContent = "";
  resetTally();
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
  // The next of the file's frames to play as packets, which the player hands
  // out, so that the page keeps no copy of the file.
  const count = player.frames();
  let next = 0;
  network.textContent = "";
  resetTally();
  // When the next frame is due. Advancing it by the interval, not to the
  // time of drawing, keeps a rate that does not divide the display's.
  let due = 0;
  let drawn = 0;
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
        // A frame skipped as packets draws nothing, and is not counted.
        let spent = null;
        if (asPackets.checked && count > 0) {
          spent = playAsPackets(player.frame(next));
          next = (next + 1) % count;
        } else {
          spent = player.draw_next();
        }
        if (spent !== null) {
          cpu += spent;
          drawn += 1;
        }
      } catch (error) {
        failWith(name, error);
        return;
      }
      if (now - since >= 1000) {
        stats.textContent =
          `${(drawn * 1000 / (now - since)).toFixed(1)} frames drawn per second · ` +
          `CPU ${(cpu / Math.max(drawn, 1)).toFixed(2)} ms · ` +
          `submitted to done ${player.take_gpu_ms().toFixed(2)} ms per frame`;
        showTally();
        drawn = 0;
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

// Draws and encodes the demo stream in this page, a picture at a time so that
// the progress shows and a file chosen meanwhile takes over, then plays it.
async function playDemo() {
  if (broken) {
    return;
  }
  const mine = ++request;
  let demo;
  let bytes;
  try {
    demo = new Demo();
    const total = Demo.frames();
    for (let done = 0; done < total;) {
      done = demo.encode_next();
      info.className = "";
      info.textContent = `Drawing a demo stream and encoding it in this page: ${done} of ${total} pictures.`;
      await new Promise((resolve) => setTimeout(resolve, 0));
      // Meanwhile another stream may have been asked for, or the module may
      // have stopped working, which the page says already.
      if (broken || mine !== request) {
        return;
      }
    }
    bytes = demo.take_file();
  } catch (error) {
    if (mine === request) {
      failWith("demo", error);
    }
    return;
  } finally {
    // A module that panicked cannot free what it was using.
    if (!broken) {
      demo?.free();
    }
  }
  if (mine === request) {
    await play(bytes, "Demo, encoded in this page");
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
  info.textContent = `GPU: ${player.adapter()}. Open a file, or play the demo or a sample.`;

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

  document.getElementById("demo").addEventListener("click", () => playDemo());

  const samples = document.getElementById("samples");
  for (const name of SAMPLES) {
    const button = document.createElement("button");
    button.textContent = name;
    button.addEventListener("click", () => playSample(name));
    samples.append(button);
  }

  // ?packets plays as packets (?packets=0 does not), and ?loss=0.05 loses
  // that share of them, from 0 to 1, for links to the network's view.
  const params = new URLSearchParams(location.search);
  if (params.has("packets") && !["0", "false"].includes(params.get("packets"))) {
    asPackets.checked = true;
  }
  const lossAsked = Number(params.get("loss"));
  if (params.has("loss") && Number.isFinite(lossAsked)) {
    const share = Math.min(1, Math.max(0, lossAsked));
    let option = [...loss.options].find((candidate) => Number(candidate.value) === share);
    if (!option) {
      option = new Option(`${+(share * 100).toFixed(2)} %`, String(share));
      loss.add(option);
    }
    loss.value = option.value;
  }

  // ?play=NAME starts a sample straight away, for links to one; otherwise
  // the demo plays, so that the page shows a stream without a file.
  const requested = params.get("play");
  if (SAMPLES.includes(requested)) {
    playSample(requested);
  } else {
    playDemo();
  }
}

start();
