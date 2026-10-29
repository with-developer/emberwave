#!/usr/bin/env python3
"""Draws the README's picture of what a frame budget buys, and prints the
table of BENCHMARKS.md's "Picture quality".

    quality_figure.py SOURCE.y4m OUT.png BYTES... [--figure BYTES,...]
        [--region X,Y] [--title TEXT] [--cli PATH]

SOURCE.y4m holds one 8-bit 4:2:0 picture. emberwave-cli encodes it on the
GPU within each budget of BYTES and decodes it again, in a temporary
directory. The table has a row per budget: the frame's size, its bit rate at
60 frames a second and the PSNR of each plane over the whole picture. The
picture shows the source in full, then a region of it enlarged twice, from
the source and from the budgets of --figure (all of them by default), each
with its bit rate and luma PSNR. The region is the one that loses the most
at the smallest budget, on a grid of its own size, unless --region gives its
top left corner.

The README's picture is a frame of a game, which the repository does not
hold, so this script is run by hand and not in CI. Unlike readme_figures.py
it needs numpy and Pillow.
"""

import argparse
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parent.parent

FRAME_RATE = 60
# The region, in the source's pixels, and how much it is enlarged.
CROP = (320, 180)
ZOOM = 2
COLUMNS = 3
GAP = 12
RADIUS = 14
# The region's outline in the full picture: the banner's orange, on a dark
# edge that keeps it visible on light parts of the picture.
BOX = (255, 122, 26)
# PWV1: a 24-byte file header, then each frame after its size, a
# little-endian u32 (crates/emberwave-core/src/bitstream/pwv1.rs).
PWV1_HEADER_BYTES = 24
FONTS = [
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def read_y4m(path):
    """The first picture of an 8-bit 4:2:0 Y4M file: its Y, Cb and Cr planes."""
    with open(path, "rb") as f:
        header = f.readline().split()
        if not header or header[0] != b"YUV4MPEG2":
            sys.exit(f"{path}: not a Y4M file")
        params = {p[:1]: p[1:] for p in header[1:]}
        width, height = int(params[b"W"]), int(params[b"H"])
        if not params.get(b"C", b"420").startswith(b"420"):
            sys.exit(f"{path}: not 8-bit 4:2:0")
        if not f.readline().startswith(b"FRAME"):
            sys.exit(f"{path}: no picture")
        cw, ch = (width + 1) // 2, (height + 1) // 2
        data = f.read(width * height + 2 * cw * ch)
    if len(data) < width * height + 2 * cw * ch:
        sys.exit(f"{path}: the picture is cut short")
    y = np.frombuffer(data, np.uint8, width * height).reshape(height, width)
    cb = np.frombuffer(data, np.uint8, cw * ch, width * height).reshape(ch, cw)
    cr = np.frombuffer(data, np.uint8, cw * ch, width * height + cw * ch).reshape(ch, cw)
    return y, cb, cr


def psnr(source, decoded):
    error = np.mean((source.astype(np.float64) - decoded) ** 2)
    return float("inf") if error == 0 else 10 * np.log10(255**2 / error)


def to_rgb(planes):
    """The picture in 8-bit RGB: BT.709 in video range, as the streams
    signal, the chroma upsampled bilinearly."""
    y, cb, cr = planes
    height, width = y.shape

    def chroma(plane):
        full = Image.fromarray(plane.astype(np.float32)).resize((width, height), Image.BILINEAR)
        return (np.asarray(full) - 128) / 224

    luma = (y.astype(np.float32) - 16) / 219
    blue, red = chroma(cb), chroma(cr)
    rgb = np.stack([luma + 1.5748 * red, luma - 0.1873 * blue - 0.4681 * red, luma + 1.8556 * blue], -1)
    return Image.fromarray(np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8))


def round_trip(cli, source, budget, scratch):
    """Encodes the source within `budget` bytes and decodes it again: the
    frame's size and the decoded planes."""
    stream = scratch / f"{budget}.pwv1"
    for command in (
        [cli, "encode", source, "--out", stream, "--bytes-per-frame", str(budget), "--frames", "1", "--gpu"],
        [cli, "decode", stream, "--out-dir", scratch, "--format", "y4m"],
    ):
        done = subprocess.run([str(part) for part in command], capture_output=True, text=True)
        if done.returncode != 0:
            sys.exit(f"{' '.join(map(str, command))} failed:\n{done.stdout}{done.stderr}")
    data = stream.read_bytes()
    size = int.from_bytes(data[PWV1_HEADER_BYTES : PWV1_HEADER_BYTES + 4], "little")
    return size, read_y4m(scratch / f"{budget}.y4m")


def worst_region(source, decoded):
    """The top left corner of the region, on a grid of its size, whose luma
    loses the most."""
    cw, ch = CROP
    error = (source.astype(np.float64) - decoded) ** 2
    rows, cols = error.shape[0] // ch, error.shape[1] // cw
    tiles = error[: rows * ch, : cols * cw].reshape(rows, ch, cols, cw).mean(axis=(1, 3))
    row, col = np.unravel_index(np.argmax(tiles), tiles.shape)
    return int(col) * cw, int(row) * ch


def font(size):
    for path in FONTS:
        try:
            face = ImageFont.truetype(path, size)
        except OSError:
            continue
        try:
            face.set_variation_by_name("Semibold")
        except (OSError, ValueError):
            pass
        return face
    return ImageFont.load_default(size)


def label(panel, text):
    """The panel, its corners rounded, with `text` on a dark plate in its
    top left corner."""
    face = font(30)
    ascent, descent = face.getmetrics()
    plate = Image.new("RGBA", panel.size)
    draw = ImageDraw.Draw(plate)
    left, top, pad = 14, 14, 12
    width = draw.textlength(text, font=face)
    draw.rounded_rectangle(
        [left, top, left + width + 2 * pad, top + ascent + descent + 2 * pad - 4], radius=10, fill=(12, 12, 12, 176)
    )
    draw.text((left + pad, top + pad - 2), text, font=face, fill=(255, 255, 255, 255), anchor="la")
    panel = Image.alpha_composite(panel.convert("RGBA"), plate)
    corners = Image.new("L", panel.size, 0)
    ImageDraw.Draw(corners).rounded_rectangle([0, 0, panel.width - 1, panel.height - 1], radius=RADIUS, fill=255)
    panel.putalpha(corners)
    return panel


def figure(source, rows, region, title):
    """The full picture with the region outlined, then the region from the
    source and from each row's decoded picture."""
    height, width = source[0].shape
    cw, ch = CROP
    pw, ph = cw * ZOOM, ch * ZOOM
    x0, y0 = region
    original = to_rgb(source)

    full = original.resize((pw, ph), Image.LANCZOS)
    draw = ImageDraw.Draw(full)
    sx, sy = pw / width, ph / height
    box = [x0 * sx, y0 * sy, (x0 + cw) * sx, (y0 + ch) * sy]
    draw.rectangle([box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2], outline=(0, 0, 0), width=7)
    draw.rectangle(box, outline=BOX, width=3)

    def enlarged(picture):
        return picture.crop((x0, y0, x0 + cw, y0 + ch)).resize((pw, ph), Image.NEAREST)

    panels = [label(full, title), label(enlarged(original), "Source")]
    for size, planes, y_psnr in rows:
        rate = size * 8 * FRAME_RATE / 1e6
        panels.append(label(enlarged(to_rgb(planes)), f"{rate:.0f} Mbit/s  ·  {y_psnr:.1f} dB"))

    lines = (len(panels) + COLUMNS - 1) // COLUMNS
    sheet = Image.new("RGBA", (COLUMNS * pw + (COLUMNS - 1) * GAP, lines * ph + (lines - 1) * GAP))
    for i, panel in enumerate(panels):
        sheet.alpha_composite(panel, ((i % COLUMNS) * (pw + GAP), (i // COLUMNS) * (ph + GAP)))
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("budgets", type=int, nargs="+", metavar="BYTES")
    parser.add_argument("--figure", help="the budgets in the picture, comma-separated")
    parser.add_argument("--region", help="the region's top left corner, X,Y")
    parser.add_argument("--title", default="Source", help="the full picture's label")
    parser.add_argument("--cli", type=pathlib.Path, default=ROOT / "target" / "release" / "emberwave-cli")
    args = parser.parse_args()

    source = read_y4m(args.source)
    height, width = source[0].shape
    shown = [int(b) for b in args.figure.split(",")] if args.figure else args.budgets
    if not set(shown) <= set(args.budgets):
        sys.exit("--figure names a budget that is not given")

    results = {}
    with tempfile.TemporaryDirectory() as scratch:
        for budget in args.budgets:
            results[budget] = round_trip(args.cli.resolve(), args.source.resolve(), budget, pathlib.Path(scratch))

    print(f"{width}x{height}, {FRAME_RATE} frames a second\n")
    print("| Budget | Frame | at 60 frames a second | bits per pixel | Y | Cb | Cr |")
    print("|---|---|---|---|---|---|---|")
    for budget, (size, planes) in results.items():
        db = [psnr(s, d) for s, d in zip(source, planes)]
        print(
            f"| {budget:,} | {size:,} | {size * 8 * FRAME_RATE / 1e6:.0f} Mbit/s | {size * 8 / (width * height):.2f}"
            f" | {db[0]:.2f} | {db[1]:.2f} | {db[2]:.2f} |"
        )

    smallest = results[min(shown)][1][0]
    region = tuple(int(v) for v in args.region.split(",")) if args.region else worst_region(source[0], smallest)
    rows = [(results[b][0], results[b][1], psnr(source[0], results[b][1][0])) for b in shown]
    figure(source, rows, region, args.title).save(args.out, optimize=True)
    print(f"\nregion {region[0]},{region[1]}, {CROP[0]}x{CROP[1]}; wrote {args.out}")


if __name__ == "__main__":
    main()
