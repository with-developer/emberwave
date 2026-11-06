#!/usr/bin/env python3
"""A performance model for emberwave: estimates, never measurements.

    time = fixed + per_pixel * pixels

The two terms are fitted per way (decode into planes, decode into RGB, encode)
on the Apple M3 Max, from numbers BENCHMARKS.md already holds. The iPhone 17's
numbers, measured the same way, then check the model: its per-pixel cost should
sit near the M3 Max's times the ratio of their memory bandwidths, and its fixed
cost near the M3 Max's. Estimates for other chips scale the per-pixel term by
bandwidth and keep the fixed term.

What this cannot give: decoding at a stream's pace (the GPU's power states, see
issue #81), thermal throttling, or anything that is not memory-bound. Every
figure it prints is a model output; report it as "estimated", with this
script's name and the date of the measurements below.

Usage:
    scripts/perf_model.py              # fit, check, estimates, as text
    scripts/perf_model.py --markdown   # the estimated table for BENCHMARKS.md
    scripts/perf_model.py --write BENCHMARKS.md   # put that table between the file's perf-model markers
Standard library only.
"""

import argparse
import sys

# Frame sizes of the benchmark streams (BENCHMARKS.md, "What is measured").
SIZES = {"1080p": 1920 * 1080, "1440p": 2560 * 1440, "4K": 3840 * 2160}
# Bytes per frame of those streams: the per-pixel term also carries the data.
BUDGET_KB = {"1080p": 400, "1440p": 700, "4K": 1200}

# Mean milliseconds per frame, frame after frame, through the C API in the
# benchmark app (BENCHMARKS.md, "Against HEVC on the same device"), XCFramework
# of release 0.11.0, measured 2026-10-26 and 2026-10-27. Each value is the
# lowest and the highest of the runs; the model takes their middle. The M3 Max's
# 1440p decode into RGB leaves out the run with a few slow frames (0.85 ms):
# the section's own text gives 0.69 to 0.70 ms for the others.
MEASURED = {
    "Apple M3 Max": {
        "decode to planes": {"1080p": (0.42, 0.43), "1440p": (0.56, 0.56), "4K": (1.00, 1.01)},
        "decode to RGB": {"1080p": (0.53, 0.56), "1440p": (0.69, 0.70), "4K": (1.29, 1.30)},
        "encode": {"1080p": (0.81, 0.82), "1440p": (1.11, 1.12), "4K": (1.90, 1.92)},
    },
    "iPhone 17": {
        "decode to planes": {"1080p": (1.27, 1.27), "1440p": (2.10, 2.11), "4K": (4.51, 4.54)},
        "decode to RGB": {"1080p": (1.61, 1.62), "1440p": (2.71, 2.72), "4K": (5.83, 5.84)},
        "encode": {"1080p": (2.24, 2.24), "1440p": (3.82, 3.84), "4K": (7.88, 7.90)},
    },
}
MEASURED_ON = "2026-10-26 and 2026-10-27, release 0.11.0"

# Memory bandwidth in GB/s, the figure of Apple's own announcement for each chip
# (the A19's 68 GB/s is the figure the project has used since 2026-11-04; Apple
# does not publish it). Chips with several GPU configurations are listed by the
# one whose bandwidth differs.
BANDWIDTH = [
    ("iPhone 17 (A19)", 68.0, "check"),
    ("Apple M1", 68.25, ""),
    ("Apple M2", 100.0, ""),
    ("Apple M3", 100.0, ""),
    ("Apple M4", 120.0, ""),
    ("Apple M4 Pro", 273.0, ""),
    ("Apple M3 Max (40-core GPU)", 400.0, "fit"),
    ("Apple M4 Max (32-core GPU)", 410.0, ""),
    ("Apple M4 Max (40-core GPU)", 546.0, ""),
]
REFERENCE = "Apple M3 Max"
REFERENCE_BANDWIDTH = 400.0
CHECK = "iPhone 17"
CHECK_BANDWIDTH = 68.0


def middle(pair):
    return (pair[0] + pair[1]) / 2


def fit(points):
    """Least squares for t = a + b * megapixels; returns (a, b, worst error in %)."""
    xs = [SIZES[k] / 1e6 for k in points]
    ys = [middle(points[k]) for k in points]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    worst = max(abs(a + b * x - y) / y * 100 for x, y in zip(xs, ys))
    return a, b, worst


def predict(a, b, bandwidth_ratio, megapixels):
    """Fixed term kept, per-pixel term scaled by how much slower the memory is."""
    return a + b * bandwidth_ratio * megapixels


def text_report():
    fits = {dev: {way: fit(pts) for way, pts in ways.items()} for dev, ways in MEASURED.items()}
    ratio = REFERENCE_BANDWIDTH / CHECK_BANDWIDTH
    print(f"Measured {MEASURED_ON}; sizes 1080p/1440p/4K = "
          + "/".join(f"{SIZES[k] / 1e6:.2f}" for k in SIZES) + " Mpixel")
    print("\nFit, t = fixed + per_pixel x megapixels (ms, ms per Mpixel):")
    print(f"{'device':14} {'way':18} {'fixed':>6} {'per Mpx':>8} {'worst error':>12}")
    for dev, ways in fits.items():
        for way, (a, b, worst) in ways.items():
            print(f"{dev:14} {way:18} {a:6.2f} {b:8.3f} {worst:11.1f}%")
    print(f"\nCheck on the {CHECK} (bandwidth ratio {REFERENCE_BANDWIDTH:.0f}/{CHECK_BANDWIDTH:.0f} = {ratio:.2f}):")
    print(f"{'way':18} {'per-pixel ratio':>16} {'fixed iPhone/Mac':>17} "
          f"{'1080p err':>10} {'1440p err':>10} {'4K err':>8}")
    for way in MEASURED[CHECK]:
        a_m, b_m, _ = fits[REFERENCE][way]
        a_p, b_p, _ = fits[CHECK][way]
        errs = []
        for size in SIZES:
            est = predict(a_m, b_m, ratio, SIZES[size] / 1e6)
            got = middle(MEASURED[CHECK][way][size])
            errs.append((est - got) / got * 100)
        print(f"{way:18} {b_p / b_m:16.2f} {a_p / a_m:17.2f} "
              + " ".join(f"{e:+9.1f}%" for e in errs))
    print("\n(error = model minus measured, on the iPhone's own measurements)")
    print("\nEstimates, ms per frame, frame after frame (1080p / 4K):")
    ways = list(MEASURED[REFERENCE])
    print(f"{'chip':28} {'GB/s':>6} " + " ".join(f"{w:>18}" for w in ways))
    for chip, bw, note in BANDWIDTH:
        row = []
        for way in ways:
            a, b, _ = fits[REFERENCE][way]
            cells = [predict(a, b, REFERENCE_BANDWIDTH / bw, SIZES[size] / 1e6) for size in ("1080p", "4K")]
            row.append(f"{cells[0]:7.2f} / {cells[1]:6.2f}")
        print(f"{chip:28} {bw:6.1f} " + " ".join(f"{c:>18}" for c in row)
              + (f"  ({note})" if note else ""))


def markdown_table(out_lines=None):
    fits = {way: fit(pts) for way, pts in MEASURED[REFERENCE].items()}
    ways = [("decode to planes", "Decode to planes"), ("decode to RGB", "Decode to RGB"), ("encode", "Encode")]
    out = []
    out.append("| Chip | Memory GB/s | " + " | ".join(
        f"{label}, {size}" for _, label in ways for size in ("1080p", "4K")) + " |")
    out.append("|---|---|" + "---|" * (len(ways) * 2))
    for chip, bw, note in BANDWIDTH:
        cells = []
        for way, _ in ways:
            a, b, _w = fits[way]
            for size in ("1080p", "4K"):
                cells.append(f"{predict(a, b, REFERENCE_BANDWIDTH / bw, SIZES[size] / 1e6):.2f}")
        tag = {"fit": " (fitted on)", "check": " (the check)"}.get(note, "")
        out.append(f"| {chip}{tag} | {bw:g} | " + " | ".join(cells) + " |")
    return "\n".join(out)


BEGIN = "<!-- perf-model:begin"
END = "<!-- perf-model:end -->"


def write_into(path):
    """Replace what stands between the markers; the begin marker's line is kept."""
    text = open(path, encoding="utf-8").read()
    head, _, rest = text.partition(BEGIN)
    marker_rest, found, tail = rest.partition(END)
    if not found:
        sys.exit(f"{path}: no perf-model markers")
    begin_line = BEGIN + marker_rest.split("\n", 1)[0] + "\n"
    body = f"Estimated by `scripts/perf_model.py`, not measured. Milliseconds per frame, frame after frame.\n\n{markdown_table()}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + begin_line + body + END + tail)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--markdown", action="store_true", help="print the estimated table for BENCHMARKS.md")
    parser.add_argument("--write", metavar="FILE", help="put the table between FILE's perf-model markers")
    args = parser.parse_args()
    if args.write:
        write_into(args.write)
    elif args.markdown:
        print(markdown_table())
    else:
        text_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
