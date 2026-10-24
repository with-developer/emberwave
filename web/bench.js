// The benchmark page: decodes test streams frame after frame with the wasm
// module (scripts/build_web.sh builds it into pkg/) and reports the times, in
// the way `emberwave-cli bench --output rgb` reports them natively.
import init, { Bench } from "./pkg/emberwave_web.js";

// The pictures of the native benchmarks (BENCHMARKS.md), at their budgets.
const SIZES = [
  { name: "1080p", width: 1920, height: 1080, budget: 400000 },
  { name: "1440p", width: 2560, height: 1440, budget: 700000 },
  { name: "4K", width: 3840, height: 2160, budget: 1200000 },
];
// Different pictures, decoded in turn.
const PICTURES = 4;
const FRAMES = 300;
// At least this many frames and this much time first, uncounted: a GPU
// runs slower for the first moments of a load.
const WARMUP_FRAMES = 30;
const WARMUP_MS = 1000;

const status = document.getElementById("status");
const results = document.getElementById("results");
const report = document.getElementById("report");
const copy = document.getElementById("copy");
const start = document.getElementById("start");

function say(text, error = false) {
  status.textContent = text;
  status.className = error ? "error" : "";
}

// Lets the browser show what changed before a long synchronous call.
function paint() {
  return new Promise((resolve) => setTimeout(resolve, 50));
}

// The smallest step of performance.now(), which browsers coarsen.
function timerStep() {
  let smallest = Infinity;
  for (let i = 0; i < 200; i++) {
    const a = performance.now();
    let b = performance.now();
    while (b === a) {
      b = performance.now();
    }
    smallest = Math.min(smallest, b - a);
  }
  return smallest;
}

// Nearest-rank percentile of sorted values.
function percentile(sorted, p) {
  const rank = Math.ceil((p / 100) * sorted.length);
  return sorted[Math.min(sorted.length, Math.max(1, rank)) - 1];
}

async function decodeAndWait(bench) {
  const cpu = bench.decode_next();
  await bench.work_done();
  return cpu;
}

async function measure(bench) {
  const warm = performance.now();
  for (let n = 0; n < WARMUP_FRAMES || performance.now() - warm < WARMUP_MS; n++) {
    await decodeAndWait(bench);
  }
  const each = [];
  let cpu = 0;
  const first = performance.now();
  for (let i = 0; i < FRAMES; i++) {
    const before = performance.now();
    cpu += await decodeAndWait(bench);
    each.push(performance.now() - before);
  }
  // The mean from the whole run, which a coarse timer cannot spoil.
  const mean = (performance.now() - first) / FRAMES;
  each.sort((a, b) => a - b);
  return {
    mean,
    median: percentile(each, 50),
    p99: percentile(each, 99),
    cpu: cpu / FRAMES,
  };
}

function row(cells) {
  const tr = document.createElement("tr");
  for (const cell of cells) {
    const td = document.createElement("td");
    td.textContent = cell;
    tr.append(td);
  }
  return tr;
}

async function run() {
  start.disabled = true;
  // A phone that locks its screen halfway through stops the run.
  let wakeLock = null;
  try {
    wakeLock = await navigator.wakeLock?.request("screen");
  } catch {
    // Without one, the run is still worth having.
  }
  results.hidden = true;
  report.hidden = true;
  copy.hidden = true;
  const body = results.querySelector("tbody");
  body.replaceChildren();

  if (!navigator.gpu) {
    say("This browser has no WebGPU.", true);
    start.disabled = false;
    return;
  }
  let bench;
  try {
    await init();
    bench = await Bench.create();
  } catch (error) {
    say(`WebGPU did not start: ${error}`, true);
    start.disabled = false;
    return;
  }
  const step = timerStep();
  const lines = [
    `emberwave benchmark, ${new Date().toISOString().slice(0, 16).replace("T", " ")} UTC`,
    `Browser: ${navigator.userAgent}`,
    `GPU: ${bench.adapter()}`,
    `Timer step: ${step.toFixed(3)} ms; ${FRAMES} frames after at least ${WARMUP_FRAMES} and ${WARMUP_MS} ms of warm-up, ${PICTURES} pictures in turn`,
    "Decode to RGB in GPU memory, per frame (ms): mean, median, 99th percentile, CPU",
  ];
  results.hidden = false;
  for (const size of SIZES) {
    try {
      say(`${size.name}: making ${PICTURES} test pictures with the encoder (this takes a moment)…`);
      await paint();
      const bytes = bench.prepare(size.width, size.height, PICTURES, size.budget);
      say(`${size.name}: decoding…`);
      await paint();
      const time = await measure(bench);
      const cells = [
        `${size.name} (${Math.round(bytes / PICTURES / 1000)} kB a frame)`,
        time.mean.toFixed(2),
        time.median.toFixed(2),
        time.p99.toFixed(2),
        time.cpu.toFixed(2),
        String(FRAMES),
      ];
      body.append(row(cells));
      lines.push(`${cells[0]}: ${cells.slice(1, 5).join(" / ")}`);
    } catch (error) {
      say(`${size.name}: ${error}`, true);
      lines.push(`${size.name}: failed: ${error}`);
      break;
    }
  }
  if (!status.classList.contains("error")) {
    say("Done.");
  }
  report.textContent = lines.join("\n");
  report.hidden = false;
  copy.hidden = false;
  start.disabled = false;
  wakeLock?.release().catch(() => {});
}

start.addEventListener("click", run);
// ?run starts straight away, for links and for testing.
if (new URLSearchParams(location.search).has("run")) {
  run();
}
copy.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(report.textContent);
    copy.textContent = "Copied";
  } catch {
    copy.textContent = "Select the text above and copy it";
  }
});
