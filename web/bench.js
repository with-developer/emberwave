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

// The smallest step of performance.now(), which browsers coarsen: from up to
// 200 steps, for about 50 ms, as each waits a step on a blocked page, and a
// million readings at most for a step that does not come.
function timerStep() {
  let smallest = Infinity;
  const until = performance.now() + 50;
  for (let i = 0; i < 200 && performance.now() < until; i++) {
    const a = performance.now();
    let b = a;
    for (let spins = 0; b === a && spins < 1e6; spins++) {
      b = performance.now();
    }
    if (b !== a) {
      smallest = Math.min(smallest, b - a);
    }
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
  // A frame that raised a GPU error was not decoded. The browser reports
  // errors as events of their own, which can come after the work is done:
  // they get a moment, so that the last frames' count for this size.
  await bench.work_done();
  await paint();
  bench.check();
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
  results.hidden = true;
  report.hidden = true;
  copy.hidden = true;
  copy.textContent = "Copy the results";
  const body = results.querySelector("tbody");
  body.replaceChildren();
  // A page that goes to the background, or a phone that locks its screen,
  // slows the frames down: such a run is marked.
  let hidden = document.hidden;
  const watch = () => {
    hidden ||= document.hidden;
  };
  document.addEventListener("visibilitychange", watch);
  let wakeLock = null;
  let bench = null;
  try {
    try {
      wakeLock = await navigator.wakeLock?.request("screen");
    } catch {
      // Without one, the run is still worth having.
    }
    if (!navigator.gpu) {
      say("This browser has no WebGPU.", true);
      return;
    }
    try {
      await init();
      bench = await Bench.create();
    } catch (error) {
      say(`WebGPU did not start: ${error}`, true);
      return;
    }
    await measureAll(bench, body, () => hidden);
  } finally {
    document.removeEventListener("visibilitychange", watch);
    wakeLock?.release().catch(() => {});
    start.disabled = false;
    try {
      bench?.free();
    } catch {
      // Its device goes with the page then.
    }
  }
}

// Measures every size and shows the results. `wasHidden` says whether the
// page has been hidden since the run began.
async function measureAll(bench, body, wasHidden) {
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
  if (wasHidden()) {
    const warning = "The page was hidden during the run, which slows the frames down: " +
      "run it again with the page in front.";
    lines.push(`Not valid: ${warning}`);
    if (!status.classList.contains("error")) {
      say(warning, true);
    }
  } else if (!status.classList.contains("error")) {
    say("Done.");
  }
  report.textContent = lines.join("\n");
  report.hidden = false;
  copy.hidden = false;
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
