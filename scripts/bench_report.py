#!/usr/bin/env python3
"""Benchmark report helper for scripts/bench.sh and scripts/bench_lavapipe.sh.

    bench_report.py environment OUT.json
        Records the machine the benchmark runs on.
    bench_report.py write RESULTS_DIR BENCHMARKS.md README.md README.ko.md
        Rewrites the generated blocks of the three files, the tables of
        BENCHMARKS.md and the summary of each README in its language, from the
        JSON results that `emberwave-cli bench --json`, `emberwave-cli
        bench-encode --json`, `refgen bench --json` and `refgen bench-encode
        --json` wrote.

Only the Python standard library is used.
"""

import datetime
import json
import pathlib
import platform
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The streams of scripts/gen_bench_data.sh, in the order of the tables.
STREAMS = [
    ("scene_1080p_420", "1080p scene"),
    ("scene_1440p_420", "1440p scene"),
    ("scene_4k_420", "4K scene"),
    ("noise_4k_420", "4K noise"),
]

# Frames per stream, as gen_bench_data.sh writes them.
STREAM_FRAMES = 60

# The pictures of the encoder benchmarks, gen_bench_data.sh --sources.
ENCODE_SOURCES = [
    ("scene_1080p_420", "1080p scene"),
    ("scene_1440p_420", "1440p scene"),
    ("scene_4k_420", "4K scene"),
]


def run(*command):
    """The output of a command, or None if it cannot run."""
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def wgpu_version():
    lock = (ROOT / "Cargo.lock").read_text()
    match = re.search(r'name = "wgpu"\nversion = "([^"]+)"', lock)
    return match.group(1) if match else "unknown"


def environment():
    env = {
        "date": datetime.date.today().isoformat(),
        "commit": run("git", "-C", str(ROOT), "rev-parse", "--short", "HEAD") or "unknown",
        "local_changes": bool(
            run("git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no")
        ),
        "rustc": run("rustc", "--version") or "unknown",
        "wgpu": wgpu_version(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores": None,
        "memory_gib": None,
        "gpu_cores": None,
        "power": None,
    }
    if platform.system() == "Darwin":
        env["os"] = "macOS {} ({})".format(
            run("sw_vers", "-productVersion"), run("sw_vers", "-buildVersion")
        )
        env["cpu"] = run("sysctl", "-n", "machdep.cpu.brand_string") or env["cpu"]
        cores = run("sysctl", "-n", "hw.ncpu")
        env["cpu_cores"] = int(cores) if cores else None
        memory = run("sysctl", "-n", "hw.memsize")
        env["memory_gib"] = round(int(memory) / 2**30) if memory else None
        displays = run("system_profiler", "SPDisplaysDataType", "-json")
        if displays:
            for gpu in json.loads(displays).get("SPDisplaysDataType", []):
                if "sppci_cores" in gpu:
                    env["gpu_cores"] = int(gpu["sppci_cores"])
        power = run("pmset", "-g", "batt")
        if power:
            match = re.search(r"'(.+?)'", power.splitlines()[0])
            env["power"] = match.group(1) if match else None
    return env


# Results older than this run's environment.json are an earlier run's: when
# bench.sh times less this time, they must not reach the report. Set by
# main(); lavapipe's results, which have an environment of their own, are
# read whatever their age.
RUN_START = None


def load(directory, name):
    path = directory / f"{name}.json"
    if not path.exists():
        return None
    if (
        RUN_START is not None
        and "lavapipe" not in name
        and name != "environment"
        and path.stat().st_mtime < RUN_START
    ):
        return None
    return json.loads(path.read_text())


def value(result, series, key="mean"):
    if not result:
        return None
    numbers = result["series"].get(series)
    return numbers[key] if numbers else None


def ms(number):
    return "–" if number is None else f"{number:.2f}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def machine(env, backend):
    computer = env["cpu"]
    details = []
    if env["cpu_cores"]:
        details.append(f"{env['cpu_cores']} CPU cores")
    if env["gpu_cores"]:
        details.append(f"{env['gpu_cores']} GPU cores")
    if env["memory_gib"]:
        details.append(f"{env['memory_gib']} GiB")
    if details:
        computer += ", " + ", ".join(details)
    os_line = env["os"] + (f", on {env['power']}" if env["power"] else "")
    return table(
        ["", ""],
        [
            ["Computer", computer],
            ["OS", os_line],
            ["GPU API", f"{backend} through wgpu {env['wgpu']}"],
            ["Compiler", env["rustc"]],
        ],
    )


def benchmarks_block(results, env):
    first = next(
        (r for r in (load(results, f"{s}.rgb") for s, _ in STREAMS) if r), None
    )
    if first is None:
        raise SystemExit(f"no results in {results}")
    backend = first["backend"]
    frames = first["frames"]
    changes = " plus local changes" if env["local_changes"] else ""

    parts = [
        f"Measured on {env['date']} at commit `{env['commit']}`{changes} with `scripts/bench.sh`.",
        machine(env, backend),
        f"Every number is over {frames} frames ({frames // STREAM_FRAMES} passes over the "
        f"{STREAM_FRAMES}-frame stream) after a warm-up of at least 10 frames and one second. "
        "Milliseconds per frame.",
        "### Decode to RGB in GPU memory\n\n"
        "What a player waits for before it can draw the frame. `total` is CPU preparation "
        "plus GPU wall-clock time; the pass times come from GPU timestamps.",
    ]
    rows = []
    for stem, label in STREAMS:
        r = load(results, f"{stem}.rgb")
        if r:
            rows.append(
                [label]
                + [ms(value(r, "total", k)) for k in ("mean", "p50", "p99")]
                + [
                    ms(value(r, s))
                    for s in (
                        "cpu prepare",
                        "gpu wall",
                        "gpu (timestamps)",
                        "dequant",
                        "idwt",
                        "yuv2rgb",
                    )
                ]
            )
    parts.append(
        table(
            [
                "Stream",
                "total mean",
                "p50",
                "p99",
                "CPU prepare",
                "GPU wall",
                "GPU passes",
                "dequant",
                "idwt",
                "yuv2rgb",
            ],
            rows,
        )
    )

    parts.append(
        "### Decode to YUV planes in GPU memory, next to upstream\n\n"
        "Upstream's own decoder (its Metal port, timed the same way by `refgen bench`) writes "
        "the same 8-bit planes. By default it keeps the wavelet pyramid in FP16; with "
        "`PYROWAVE_PRECISION=2` it uses FP32, as emberwave does. Its GPU time is the span of "
        "its command buffer."
    )
    rows = []
    for stem, label in STREAMS:
        ours = load(results, f"{stem}.yuv")
        if not ours:
            continue
        row = [label, ms(value(ours, "total")), ms(value(ours, "total", "p99")), ms(value(ours, "gpu (timestamps)"))]
        for name in ("upstream", "upstream-fp32"):
            theirs = load(results, f"{stem}.{name}")
            row += [
                ms(value(theirs, "total")),
                ms(value(theirs, "total", "p99")),
                ms(value(theirs, "gpu (command buffer)")),
            ]
        rows.append(row)
    parts.append(
        table(
            [
                "Stream",
                "emberwave mean",
                "p99",
                "GPU",
                "upstream FP16 mean",
                "p99",
                "GPU",
                "upstream FP32 mean",
                "p99",
                "GPU",
            ],
            rows,
        )
    )

    parts.append(
        "### Decode to RGB and copy it to the CPU\n\n"
        "What `emberwave-cli decode` does before it writes a PNG, without the PNG."
    )
    rows = []
    for stem, label in STREAMS:
        r = load(results, f"{stem}.rgb-readback")
        if r:
            rows.append([label] + [ms(value(r, "total", k)) for k in ("mean", "p50", "p99")])
    parts.append(table(["Stream", "mean", "p50", "p99"], rows))

    lavapipe_env = load(results, "lavapipe-environment")
    rows = []
    for stem, label in STREAMS:
        r = load(results, f"{stem}.lavapipe")
        if r:
            rows.append([label] + [ms(value(r, "total", k)) for k in ("mean", "p50", "p99")] + [str(r["frames"])])
    if rows and lavapipe_env:
        parts.append(
            "### Software rendering: lavapipe\n\n"
            "Decode to RGB in GPU memory on Mesa's lavapipe, a software Vulkan driver, measured "
            f"on {lavapipe_env['date']} by `scripts/bench_lavapipe.sh` in a "
            f"{lavapipe_env['os']} container ({lavapipe_env['arch']}, {lavapipe_env['cpus']} CPUs, "
            f"Mesa {lavapipe_env['mesa']}) on the computer above. Reference only: "
            "a software driver is for machines without a GPU, and for CI."
        )
        parts.append(table(["Stream", "mean", "p50", "p99", "frames"], rows))

    parts += encode_parts(results)
    return "\n\n".join(parts) + "\n"


def encode_parts(results):
    """The tables of the encoder benchmarks, if there are results."""
    rows = []
    for stem, label in ENCODE_SOURCES:
        gpu = load(results, f"{stem}.encode-gpu")
        if not gpu:
            continue
        cpu = load(results, f"{stem}.encode-cpu")
        row = [label, f"{gpu['bytes_per_frame'] // 1000} kB"]
        row += [ms(value(gpu, "total")), ms(value(gpu, "total", "p99"))]
        for name in ("encode-upstream", "encode-upstream-fp32"):
            theirs = load(results, f"{stem}.{name}")
            row += [ms(value(theirs, "total")), ms(value(theirs, "total", "p99"))]
        row.append(ms(value(cpu, "total")))
        rows.append(row)
    if not rows:
        return []
    parts = [
        "### Encode, next to upstream\n\n"
        "From the picture's planes in CPU memory to the frame in CPU memory, the upload and the "
        "read-back included, which is what `Encoder::encode` does and what upstream's encoder "
        "does with `pyrowave_encoder_encode_cpu` and `pyrowave_encoder_packetize`, timed by "
        "`refgen bench-encode`. The pictures are 20 frames of the scene, read into memory first "
        "and taken in turn. Upstream's encoder keeps its wavelet pyramid in FP16 by default and "
        "in FP32 with `PYROWAVE_PRECISION=2`, as emberwave's does. The CPU encoder is timed over "
        "a tenth of the frames.",
        table(
            [
                "Picture",
                "budget",
                "emberwave GPU mean",
                "p99",
                "upstream FP16 mean",
                "p99",
                "upstream FP32 mean",
                "p99",
                "emberwave CPU mean",
            ],
            rows,
        ),
        "Where the GPU encoder's time goes, the pass times from GPU timestamps:",
    ]
    rows = []
    for stem, label in ENCODE_SOURCES:
        gpu = load(results, f"{stem}.encode-gpu")
        if gpu:
            rows.append(
                [label]
                + [
                    ms(value(gpu, s))
                    for s in (
                        "cpu prepare",
                        "gpu wall",
                        "gpu (timestamps)",
                        "transform",
                        "quantisation",
                        "rate control",
                        "packing",
                    )
                ]
            )
    parts.append(
        table(
            [
                "Picture",
                "CPU prepare",
                "GPU wall",
                "GPU passes",
                "transform",
                "quantisation",
                "rate control",
                "packing",
            ],
            rows,
        )
    )
    rows = []
    for stem, label in ENCODE_SOURCES:
        planes = load(results, f"{stem}.encode-gpu")
        textures = load(results, f"{stem}.encode-gpu-textures")
        if planes and textures:
            rows.append(
                [label]
                + [
                    ms(value(result, *key))
                    for result in (planes, textures)
                    for key in (("total",), ("total", "p99"), ("cpu prepare",), ("transform",))
                ]
            )
    if rows:
        parts.append(
            "From pictures on the GPU already, an R8 texture per plane as a screen capture "
            "would hand them over (`Encoder::encode_textures`, `bench-encode --textures`), "
            "against the same pictures in CPU memory. `gather.wgsl` packs the textures into "
            "the planes that the upload fills otherwise, within the transform's time, and the "
            "CPU only prepares the GPU work. A capture that would have to be read back to CPU "
            "memory first, which is not measured here, saves more:"
        )
        parts.append(
            table(
                [
                    "Picture",
                    "from CPU memory mean",
                    "p99",
                    "CPU prepare",
                    "transform",
                    "from textures mean",
                    "p99",
                    "CPU prepare",
                    "transform",
                ],
                rows,
            )
        )
    return parts


def readme_results(results):
    """What the READMEs' summaries are made of: the decode to RGB at 1080p,
    1440p and 4K as table rows, its backend, and the 4K frame's planes and
    encode next to upstream's at FP32, each None if it was not measured."""
    rows = []
    for stem, label in STREAMS[:3]:
        r = load(results, f"{stem}.rgb")
        if r:
            rows.append([label.split()[0], ms(value(r, "total")), ms(value(r, "total", "p99"))])
    return {
        "rows": rows,
        "backend": load(results, f"{STREAMS[0][0]}.rgb")["backend"],
        "planes": load(results, "scene_4k_420.yuv"),
        "planes_upstream": load(results, "scene_4k_420.upstream-fp32"),
        "encode": load(results, "scene_4k_420.encode-gpu"),
        "encode_upstream": load(results, "scene_4k_420.encode-upstream-fp32"),
    }


def readme_block(results, env):
    r = readme_results(results)
    text = (
        f"Decode time per frame on an {env['cpu']} ({r['backend']}), RGB output kept in GPU "
        f"memory, measured on {env['date']}, in milliseconds:\n\n"
        + table(["Stream", "mean", "p99"], r["rows"])
        + "\n"
    )
    if r["planes"] and r["planes_upstream"]:
        text += (
            f"\nUpstream's own Metal decoder is faster on the same machine: "
            f"{ms(value(r['planes_upstream'], 'total'))} ms for a 4K frame at FP32 against "
            f"emberwave's {ms(value(r['planes'], 'total'))} ms (planes only). "
            "[`BENCHMARKS.md`](BENCHMARKS.md) has the method and all the numbers.\n"
        )
    if r["encode"]:
        text += (
            f"\nEncoding a 4K frame within 1.2 MB takes {ms(value(r['encode'], 'total'))} ms on "
            "the GPU, from the planes in CPU memory to the frame in CPU memory"
            + (
                f", and {ms(value(r['encode_upstream'], 'total'))} ms with upstream's encoder at FP32."
                if r["encode_upstream"]
                else "."
            )
            + "\n"
        )
    return text


def readme_block_ko(results, env):
    """The summary of readme_block in Korean, for README.ko.md."""
    r = readme_results(results)
    text = (
        f"{env['cpu']}({r['backend']})에서 측정한 프레임당 디코딩 시간입니다. 단위는 밀리초이고, "
        f"RGB 출력은 GPU 메모리에 둔 채로 쟀습니다({env['date']} 측정).\n\n"
        + table(["스트림", "평균", "p99"], r["rows"])
        + "\n"
    )
    if r["planes"] and r["planes_upstream"]:
        text += (
            "\n같은 기기에서 업스트림의 Metal 디코더는 이보다 빠릅니다. 4K 프레임 하나를 평면까지 "
            f"디코딩하는 데 업스트림은 FP32로 {ms(value(r['planes_upstream'], 'total'))} ms, "
            f"emberwave는 {ms(value(r['planes'], 'total'))} ms가 걸립니다. "
            "측정 방법과 전체 결과는 [`BENCHMARKS.md`](BENCHMARKS.md)에 있습니다.\n"
        )
    if r["encode"]:
        text += (
            f"\n4K 프레임 하나를 1.2 MB 이내로 인코딩하는 데는 GPU에서 {ms(value(r['encode'], 'total'))} ms가 "
            "걸립니다. CPU 메모리의 평면을 받아 CPU 메모리에 프레임을 내놓기까지의 시간"
            + (
                f"이며, 업스트림 인코더는 FP32로 {ms(value(r['encode_upstream'], 'total'))} ms가 "
                "걸립니다."
                if r["encode_upstream"]
                else "입니다."
            )
            + "\n"
        )
    return text


def replace_block(path, tag, body):
    text = path.read_text()
    pattern = re.compile(rf"(<!-- {tag}:begin[^>]*-->\n).*?(<!-- {tag}:end -->)", re.S)
    if not pattern.search(text):
        raise SystemExit(f"{path} has no {tag} markers")
    path.write_text(pattern.sub(lambda m: m.group(1) + body + m.group(2), text))


def main(argv):
    if len(argv) == 3 and argv[1] == "environment":
        pathlib.Path(argv[2]).write_text(json.dumps(environment(), indent=2) + "\n")
    elif len(argv) == 6 and argv[1] == "write":
        global RUN_START
        results = pathlib.Path(argv[2])
        env = load(results, "environment")
        if env is None:
            raise SystemExit(f"{results}/environment.json is missing; run scripts/bench.sh")
        RUN_START = (results / "environment.json").stat().st_mtime
        replace_block(pathlib.Path(argv[3]), "bench", benchmarks_block(results, env))
        replace_block(pathlib.Path(argv[4]), "bench-summary", readme_block(results, env))
        replace_block(pathlib.Path(argv[5]), "bench-summary", readme_block_ko(results, env))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
