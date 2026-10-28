#!/usr/bin/env python3
"""Draws the README's banner and charts as SVG into docs/images/.

    readme_figures.py

The charts take their numbers from BENCHMARKS.md, copied into DECODE and
ENCODE below with the section each comes from: when a run there changes
them, change them here and run the script again. Each chart is drawn twice,
for GitHub's light and dark themes, and the README picks one with
<picture>. The SVG files use the viewer's system font.

Only the Python standard library is used.
"""

import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"

# Decoding a frame to the RGB picture in GPU memory, frames one after the
# other, mean ms: (row, API, 1080p, 4K), a pair being the lowest and highest
# mean over several runs.
DECODE = [
    # BENCHMARKS.md, "Decode to RGB in GPU memory", the run of 2026-10-26.
    ("Apple M3 Max", "Metal, native", 0.53, 1.22),
    # BENCHMARKS.md, "In the browser, phones included", Chrome 155, four runs.
    ("Apple M3 Max", "Chrome, WebGPU", (0.70, 1.00), (1.69, 1.88)),
    # BENCHMARKS.md, "On the iPhone, natively", into a Metal texture, two runs.
    ("iPhone 17", "Metal, native app", (1.62, 1.63), (6.21, 6.29)),
    # BENCHMARKS.md, "In the browser, phones included", Safari 27.0.1.
    ("iPhone 17", "Safari, WebGPU", 2.36, 8.05),
]
DECODE_NOTES = [
    "Apple M3 Max: macOS 27.0.1, emberwave-cli bench on 2026-10-26, Chrome 155 headless on 2026-10-24.",
    "iPhone 17: iOS 27.0.1, the benchmark app on 2026-10-26, Safari on 2026-10-24. Shaded: the spread between runs.",
    "When frames arrive 60 a second, the GPU idles between them and a decode takes up to 5 times as long.",
]

# Encoding a 4K frame from the picture in CPU memory to the frame's bytes, on
# the Apple M3 Max, frames one after the other, mean ms: (label, ms, ours).
ENCODE = [
    # BENCHMARKS.md, "Encode, next to upstream", the run of 2026-10-26.
    ("emberwave, on the GPU", 1.90, True),
    ("Upstream PyroWave, FP16 (its default)", 3.17, False),
    ("Upstream PyroWave, FP32", 3.33, False),
    # BENCHMARKS.md, "Against HEVC on the same device", the M3 Max, three runs.
    ("Apple's HEVC encoder, at 80 Mbit/s", (12.76, 13.00), False),
]
ENCODE_NOTES = [
    "PyroWave at 1.2 MB a frame (BENCHMARKS.md, 2026-10-26); HEVC through VideoToolbox at its fastest",
    "settings, Moonlight's default rate for 4K, three runs. A PC's NVENC is not measured here.",
]

# The validated two-series palette (categorical slots 1 and 2) and the inks,
# for GitHub's light (#ffffff) and dark (#0d1117) pages.
THEMES = {
    "light": {
        "series": ["#2a78d6", "#eb6834"],
        "neutral": "#898781",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#6f6d67",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "card": "#ffffff",
        "border": "#d1d9e0",
    },
    "dark": {
        "series": ["#3987e5", "#d95926"],
        "neutral": "#898781",
        "primary": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#9d9b94",
        "grid": "#2c2c2a",
        "axis": "#4a4a46",
        "card": "#0d1117",
        "border": "#30363d",
    },
}


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def span(value):
    """The lowest and highest of a value, which is one number or a pair."""
    return value if isinstance(value, tuple) else (value, value)


def label(value):
    low, high = span(value)
    return f"{low:.2f} ms" if low == high else f"{low:.2f}–{high:.2f} ms"


def bar(x0, x1, y, h, fill, rounded=True, opacity=None):
    """A horizontal bar from the baseline x0, its end at x1 rounded by 4 px."""
    extra = f' fill-opacity="{opacity}"' if opacity is not None else ""
    r = min(4.0, h / 2, max(x1 - x0, 0) / 2) if rounded else 0
    if r <= 0:
        return f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(x1 - x0, 0):.1f}" height="{h}" fill="{fill}"{extra}/>'
    return (
        f'<path d="M{x0:.1f},{y:.1f}H{x1 - r:.1f}A{r},{r} 0 0 1 {x1:.1f},{y + r:.1f}'
        f'V{y + h - r:.1f}A{r},{r} 0 0 1 {x1 - r:.1f},{y + h:.1f}H{x0:.1f}Z" fill="{fill}"{extra}/>'
    )


def ranged_bar(x0, scale, value, y, h, fill):
    """A bar to the lowest value, and a lighter one on to the highest."""
    low, high = span(value)
    parts = []
    if high > low:
        parts.append(bar(x0, x0 + high * scale, y, h, fill, opacity=0.35))
        parts.append(bar(x0, x0 + low * scale, y, h, fill, rounded=False))
    else:
        parts.append(bar(x0, x0 + low * scale, y, h, fill))
    return "".join(parts)


def text(x, y, content, size, fill, weight=400, anchor="start", extra=""):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}"{extra}>{esc(content)}</text>'
    )


def svg(width, height, title, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}" '
        f'font-family="{FONT}">\n<title>{esc(title)}</title>\n{body}\n</svg>\n'
    )


def card(width, height, theme):
    """The chart's own background, GitHub's page colour in its theme, so that
    it stays legible where a viewer gets the other theme's file."""
    t = THEMES[theme]
    return (f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="12" '
            f'fill="{t["card"]}" stroke="{t["border"]}"/>')


def axis(left, right, top, bottom, top_value, step, theme):
    """Vertical hairlines at every step, labelled under the plot, and the baseline."""
    t = THEMES[theme]
    scale = (right - left) / top_value
    out = []
    value = 0.0
    while value <= top_value + 1e-9:
        x = left + value * scale
        if value > 0:
            out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="{t["grid"]}" stroke-width="1"/>')
        unit = " ms" if value + step > top_value + 1e-9 else ""
        out.append(text(x, bottom + 18, f"{value:g}{unit}", 12, t["muted"], anchor="middle"))
        value += step
    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{t["axis"]}" stroke-width="1"/>')
    return out, scale


def decode_chart(theme):
    t = THEMES[theme]
    width, left, right = 960, 236, 836
    top = 118
    bar_h, gap, row_gap = 18, 2, 22
    band = 2 * bar_h + gap
    bottom = top + len(DECODE) * (band + row_gap) - row_gap + 12
    height = bottom + 36 + 18 * len(DECODE_NOTES)
    out = [
        card(width, height, theme),
        text(24, 38, "Decode time per frame, lower is better", 20, t["primary"], 600),
        text(24, 62, "One Rust + WGSL decoder: a frame's bytes to the RGB picture in GPU memory, frames one after the other.", 14, t["secondary"]),
    ]
    for i, name in enumerate(["1080p", "4K"]):
        x = 24 + i * 92
        out.append(f'<rect x="{x}" y="80" width="12" height="12" rx="3" fill="{t["series"][i]}"/>')
        out.append(text(x + 18, 91, name, 13, t["secondary"]))
    grid, scale = axis(left, right, top - 8, bottom, 9.0, 2.0, theme)
    out.extend(grid)
    for row, (device, api, *values) in enumerate(DECODE):
        y = top + row * (band + row_gap)
        out.append(text(24, y + band / 2 - 3, device, 14, t["primary"], 600))
        out.append(text(24, y + band / 2 + 15, api, 13, t["secondary"]))
        for i, value in enumerate(values):
            by = y + i * (bar_h + gap)
            out.append(ranged_bar(left, scale, value, by, bar_h, t["series"][i]))
            out.append(text(left + span(value)[1] * scale + 6, by + 13, label(value), 12, t["secondary"],
                            extra=' font-variant-numeric="tabular-nums"'))
    for i, note in enumerate(DECODE_NOTES):
        out.append(text(24, bottom + 44 + 18 * i, note, 12, t["muted"]))
    return svg(width, height, "Decode time per frame on an Apple M3 Max and an iPhone 17, natively and in the browser", "\n".join(out))


def encode_chart(theme):
    t = THEMES[theme]
    width, left, right = 960, 300, 836
    top = 86
    bar_h, row_gap = 18, 18
    bottom = top + len(ENCODE) * (bar_h + row_gap) - row_gap + 12
    height = bottom + 36 + 18 * len(ENCODE_NOTES)
    out = [
        card(width, height, theme),
        text(24, 38, "Encoding a 4K frame on an Apple M3 Max, lower is better", 20, t["primary"], 600),
        text(24, 62, "From the picture in memory to the frame's bytes, frames one after the other.", 14, t["secondary"]),
    ]
    grid, scale = axis(left, right, top - 8, bottom, 14.0, 2.0, theme)
    out.extend(grid)
    for row, (name, value, ours) in enumerate(ENCODE):
        y = top + row * (bar_h + row_gap)
        out.append(text(24, y + 13, name, 14, t["primary"], 600 if ours else 400))
        fill = t["series"][1] if ours else t["neutral"]
        out.append(ranged_bar(left, scale, value, y, bar_h, fill))
        out.append(text(left + span(value)[1] * scale + 6, y + 13, label(value), 12, t["secondary"],
                        extra=' font-variant-numeric="tabular-nums"'))
    for i, note in enumerate(ENCODE_NOTES):
        out.append(text(24, bottom + 44 + 18 * i, note, 12, t["muted"]))
    return svg(width, height, "Encoding a 4K frame on an Apple M3 Max: emberwave, upstream PyroWave and Apple's HEVC encoder", "\n".join(out))


# A 4K frame's way through the encoder and the decoder on the Apple M3 Max,
# ms: BENCHMARKS.md, the pass times of the GPU encoder and of the decode to
# RGB in GPU memory, the run of 2026-10-26. (name, detail, ms, on the GPU)
PIPELINE = [
    ("Encode", [
        ("Picture", "Y, Cb, Cr planes", None, None),
        ("Wavelet transform", "CDF 9/7, 5 levels", 0.40, True),
        ("Quantisation", "8x8 blocks", 0.59, True),
        ("Rate control", "fits the budget", 0.13, True),
        ("Packing", "bit-planes", 0.18, True),
        ("Frame", "at most 1.2 MB", None, None),
    ]),
    ("Decode", [
        ("Packets", "as they arrive", None, None),
        ("Assembly", "on the CPU", 0.10, False),
        ("Dequantisation", "bit-planes to floats", 0.19, True),
        ("Inverse transform", "5 levels", 0.40, True),
        ("YUV to RGB", "the stream's colours", 0.24, True),
        ("Picture", "in GPU memory", None, None),
    ]),
]


def pipeline(theme):
    t = THEMES[theme]
    width, left = 960, 24
    box_w, box_h, arrow = 138, 64, 16
    rows_top = [96, 232]
    height = 386
    out = [
        card(width, height, theme),
        text(24, 38, "A 4K frame through emberwave", 20, t["primary"], 600),
        text(24, 62, "Every frame is coded on its own: no frame refers to another. Times on an Apple M3 Max.", 14, t["secondary"]),
        "<defs>"
        f'<marker id="head-{theme}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto">'
        f'<path d="M0,0L8,4L0,8Z" fill="{t["axis"]}"/></marker></defs>',
    ]
    for (title, stages), top in zip(PIPELINE, rows_top):
        colour = t["series"][1] if title == "Encode" else t["series"][0]
        # The decoder's title goes under its row, out of the network's way.
        label_y = top - 12 if title == "Encode" else top + box_h + 22
        out.append(text(left, label_y, title + (" on the host's GPU" if title == "Encode" else " on the client's GPU"), 13, t["secondary"], 600))
        for i, (name, detail, ms, gpu) in enumerate(stages):
            x = left + i * (box_w + arrow + 2)
            if ms is None:
                out.append(f'<rect x="{x}" y="{top}" width="{box_w}" height="{box_h}" rx="10" fill="none" '
                           f'stroke="{t["axis"]}" stroke-dasharray="4 3"/>')
            else:
                fill = colour if gpu else t["neutral"]
                out.append(f'<rect x="{x}" y="{top}" width="{box_w}" height="{box_h}" rx="10" fill="{fill}" '
                           f'fill-opacity="0.12" stroke="{fill}" stroke-width="1.5"/>')
            out.append(text(x + box_w / 2, top + 24, name, 13, t["primary"], 600, anchor="middle"))
            out.append(text(x + box_w / 2, top + 42, detail, 12, t["secondary"], anchor="middle"))
            if ms is not None:
                out.append(text(x + box_w / 2, top + 58, f"{ms:.2f} ms", 12, t["secondary"], anchor="middle",
                                extra=' font-variant-numeric="tabular-nums"'))
            if i + 1 < len(stages):
                ax = x + box_w + 2
                out.append(f'<line x1="{ax}" y1="{top + box_h / 2}" x2="{ax + arrow - 2}" y2="{top + box_h / 2}" '
                           f'stroke="{t["axis"]}" stroke-width="1.5" marker-end="url(#head-{theme})"/>')
    # The frame goes over the network, which is the application's.
    x_frame = left + 5 * (box_w + arrow + 2) + box_w / 2
    x_packets = left + box_w / 2
    y0, y1 = rows_top[0] + box_h, rows_top[1]
    mid = (y0 + y1) / 2 + 6
    out.append(f'<path d="M{x_frame},{y0 + 2}V{mid}H{x_packets}V{y1 - 4}" fill="none" stroke="{t["axis"]}" '
               f'stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#head-{theme})"/>')
    out.append(text((x_frame + x_packets) / 2, mid - 8, "packets over the network: the application's own transport", 12, t["muted"], anchor="middle"))
    out.append(text(24, height - 32, "Stage times from GPU timestamps. The whole frame takes 1.90 ms to encode, memory to memory, and 1.22 ms to decode:", 12, t["muted"]))
    out.append(text(24, height - 14, "the rest is the CPU's part, copies and waiting (BENCHMARKS.md, 2026-10-26).", 12, t["muted"]))
    return svg(width, height, "A 4K frame through emberwave's encoder and decoder, with the time of each stage on an Apple M3 Max", "\n".join(out))


def wavelet(cx, cy, half_width, amplitude, cycles, points=240):
    """A Morlet wavelet across 2 * half_width, as a path."""
    out = []
    for i in range(points + 1):
        u = -1 + 2 * i / points
        y = math.cos(math.pi * cycles * u) * math.exp(-((u / 0.42) ** 2))
        out.append(f"{'M' if i == 0 else 'L'}{cx + u * half_width:.1f},{cy - y * amplitude:.1f}")
    return "".join(out)


def banner():
    width, height = 1280, 340
    cx, cy = 250, 176
    out = [
        "<defs>",
        '<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#17100d"/><stop offset="0.55" stop-color="#1d1311"/>'
        '<stop offset="1" stop-color="#100c12"/></linearGradient>',
        '<radialGradient id="glow" cx="0.5" cy="0.5" r="0.5">'
        '<stop offset="0" stop-color="#ff6a1a" stop-opacity="0.42"/>'
        '<stop offset="1" stop-color="#ff6a1a" stop-opacity="0"/></radialGradient>',
        '<linearGradient id="fire" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" stop-color="#ffd166"/><stop offset="0.5" stop-color="#ff7a1a"/>'
        '<stop offset="1" stop-color="#e8302a"/></linearGradient>',
        '<filter id="blur" x="-20%" y="-50%" width="140%" height="200%"><feGaussianBlur stdDeviation="9"/></filter>',
        '<filter id="soft" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="1.6"/></filter>',
        "</defs>",
        f'<rect width="{width}" height="{height}" rx="24" fill="url(#bg)"/>',
        f'<ellipse cx="{cx}" cy="{cy}" rx="250" ry="150" fill="url(#glow)"/>',
    ]
    # The wavelet, its glow, and two fainter echoes: one wave at three scales.
    main = wavelet(cx, cy, 190, 92, 4.2)
    out.append(f'<path d="{main}" fill="none" stroke="url(#fire)" stroke-width="16" stroke-linecap="round" opacity="0.55" filter="url(#blur)"/>')
    out.append(f'<path d="{wavelet(cx, cy, 190, 50, 4.2)}" fill="none" stroke="url(#fire)" stroke-width="3" stroke-linecap="round" opacity="0.35"/>')
    out.append(f'<path d="{wavelet(cx, cy, 190, 24, 4.2)}" fill="none" stroke="url(#fire)" stroke-width="2" stroke-linecap="round" opacity="0.22"/>')
    out.append(f'<path d="{main}" fill="none" stroke="url(#fire)" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>')
    # Embers rising from the crests, placed by a fixed sequence so that the
    # file does not change from run to run.
    seed = 7
    for _ in range(16):
        seed = (seed * 1103515245 + 12345) % 2**31
        u = (seed % 1000) / 1000 * 2 - 1
        seed = (seed * 1103515245 + 12345) % 2**31
        rise = 20 + (seed % 1000) / 1000 * 95
        seed = (seed * 1103515245 + 12345) % 2**31
        r = 1.4 + (seed % 1000) / 1000 * 2.6
        crest = math.exp(-((u / 0.42) ** 2)) * 92
        x, y = cx + u * 190, cy - crest - rise
        colour = "#ffc46b" if r < 2.6 else "#ff8a3d"
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{colour}" opacity="{0.35 + r / 8:.2f}" filter="url(#soft)"/>')
    x = 520
    out.append(text(x, 158, "emberwave", 96, "#fff5ec", 700, extra=' letter-spacing="-2"'))
    out.append(text(x + 4, 208, "PyroWave's wavelet video codec, on every GPU", 30, "#f0d9c8"))
    chips = ["Metal", "Vulkan", "DX12", "WebGPU", "iPhone", "C API"]
    cx_chip = x + 4
    for chip in chips:
        w = 26 + 10.6 * len(chip)
        out.append(
            f'<rect x="{cx_chip:.1f}" y="236" width="{w:.1f}" height="36" rx="18" fill="#ffffff" fill-opacity="0.06" '
            f'stroke="#ffb27a" stroke-opacity="0.45"/>'
        )
        out.append(text(cx_chip + w / 2, 260, chip, 18, "#fbe6d6", 500, anchor="middle"))
        cx_chip += w + 12
    return svg(width, height, "emberwave: PyroWave's wavelet video codec, on every GPU", "\n".join(out))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"banner.svg": banner()}
    for theme in THEMES:
        files[f"decode-{theme}.svg"] = decode_chart(theme)
        files[f"encode-{theme}.svg"] = encode_chart(theme)
        files[f"pipeline-{theme}.svg"] = pipeline(theme)
    for name, content in files.items():
        (OUT / name).write_text(content)
        print(f"wrote {(OUT / name).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
