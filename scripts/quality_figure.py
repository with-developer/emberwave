#!/usr/bin/env python3
"""Draws the README's picture of what a frame budget buys, and prints the
table of BENCHMARKS.md's "Picture quality".

    quality_figure.py SOURCE.y4m OUT.png BYTES... [--figure BYTES,...]
        [--region X,Y] [--title TEXT] [--credit TEXT] [--gpu] [--cli PATH]

SOURCE.y4m holds an 8-bit 4:2:0 picture of at least 320x180 with an even
width and height; only its first picture is used. emberwave-cli encodes it
within each budget of BYTES, on the CPU, which makes the same bytes on every
machine, or with --gpu on the GPU, and decodes it again, in a temporary
directory. Build it first with `cargo build --release -p emberwave-cli`, or
name another with --cli.

The table has a row per budget: the frame's size, its bit rate at 60 frames
a second and the PSNR of each plane over the whole picture. The picture
shows the source in full, then a region of it enlarged twice, from the
source and from the budgets of --figure (all of them by default), each with
its bit rate and luma PSNR. The region is the one that loses the most at the
smallest budget shown, on a grid of its own size, unless --region gives its
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
# Behind a full picture that is not 16:9.
MARGIN = (24, 24, 24)
# The 8-bit 4:2:0 colour spaces of Y4M, as emberwave-cli reads them
# (crates/emberwave-cli/src/y4m.rs); a header without one means 420jpeg.
CHROMA_420 = {b"420", b"420jpeg", b"420mpeg2", b"420paldv"}
# PWV1: a 24-byte file header, then each frame after its size, a
# little-endian u32 (crates/emberwave-core/src/bitstream/pwv1.rs).
PWV1_HEADER_BYTES = 24
FONTS = [
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def read_y4m(path):
    """The first picture of an 8-bit 4:2:0 Y4M file, its Y, Cb and Cr
    planes, and whether it is in full range rather than video range."""
    with open(path, "rb") as f:
        header = f.readline().split()
        if not header or header[0] != b"YUV4MPEG2":
            sys.exit(f"{path}: not a Y4M file")
        params = {p[:1]: p[1:] for p in header[1:] if not p.startswith(b"X")}
        width, height = int(params[b"W"]), int(params[b"H"])
        if params.get(b"C", b"420jpeg") not in CHROMA_420:
            sys.exit(f"{path}: not 8-bit 4:2:0")
        full_range = b"XCOLORRANGE=FULL" in header
        if not f.readline().startswith(b"FRAME"):
            sys.exit(f"{path}: no picture")
        cw, ch = (width + 1) // 2, (height + 1) // 2
        data = f.read(width * height + 2 * cw * ch)
    if len(data) < width * height + 2 * cw * ch:
        sys.exit(f"{path}: the picture is cut short")
    y = np.frombuffer(data, np.uint8, width * height).reshape(height, width)
    cb = np.frombuffer(data, np.uint8, cw * ch, width * height).reshape(ch, cw)
    cr = np.frombuffer(data, np.uint8, cw * ch, width * height + cw * ch).reshape(ch, cw)
    return (y, cb, cr), full_range


def psnr(source, decoded):
    error = np.mean((source.astype(np.float64) - decoded) ** 2)
    return float("inf") if error == 0 else 10 * np.log10(255**2 / error)


def to_rgb(planes, full_range):
    """The picture in 8-bit RGB: BT.709, in video range or full range as the
    source is, the chroma upsampled bilinearly."""
    y, cb, cr = planes
    height, width = y.shape
    if full_range:
        luma_scale, luma_offset, chroma_scale = 255, 0, 255
    else:
        luma_scale, luma_offset, chroma_scale = 219, 16, 224

    def chroma(plane):
        full = Image.fromarray(plane.astype(np.float32)).resize((width, height), Image.BILINEAR)
        return (np.asarray(full) - 128) / chroma_scale

    luma = (y.astype(np.float32) - luma_offset) / luma_scale
    blue, red = chroma(cb), chroma(cr)
    rgb = np.stack([luma + 1.5748 * red, luma - 0.1873 * blue - 0.4681 * red, luma + 1.8556 * blue], -1)
    return Image.fromarray(np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8))


def round_trip(cli, source, budget, gpu, scratch):
    """Encodes the source within `budget` bytes and decodes it again: the
    frame's size and the decoded planes."""
    stream = scratch / f"{budget}.pwv1"
    encode = [cli, "encode", source, "--out", stream, "--bytes-per-frame", str(budget), "--frames", "1"]
    for command in (
        encode + (["--gpu"] if gpu else []),
        [cli, "decode", stream, "--out-dir", scratch, "--format", "y4m"],
    ):
        try:
            done = subprocess.run([str(part) for part in command], capture_output=True, text=True)
        except OSError as error:
            sys.exit(f"cannot run {cli}: {error}; build it with `cargo build --release -p emberwave-cli`")
        if done.returncode != 0:
            sys.exit(f"{' '.join(map(str, command))} failed:\n{done.stdout}{done.stderr}")
    data = stream.read_bytes()
    size = int.from_bytes(data[PWV1_HEADER_BYTES : PWV1_HEADER_BYTES + 4], "little")
    return size, read_y4m(scratch / f"{budget}.y4m")[0]


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


def plate(draw, text, size, corner, panel_size):
    """`text` on a dark plate, in the panel's top left corner, or its bottom
    right one for `corner` "br"."""
    face = font(size)
    ascent, descent = face.getmetrics()
    margin, pad = 14, round(size * 0.4)
    width = draw.textlength(text, font=face) + 2 * pad
    height = ascent + descent + 2 * pad - 4
    left, top = margin, margin
    if corner == "br":
        left, top = panel_size[0] - margin - width, panel_size[1] - margin - height
    draw.rounded_rectangle([left, top, left + width, top + height], radius=10, fill=(12, 12, 12, 176))
    draw.text((left + pad, top + pad - 2), text, font=face, fill=(255, 255, 255, 255), anchor="la")


def label(panel, text, credit=None):
    """The panel, its corners rounded, with `text` in its top left corner
    and `credit`, if any, smaller, in its bottom right one."""
    overlay = Image.new("RGBA", panel.size)
    draw = ImageDraw.Draw(overlay)
    plate(draw, text, 30, "tl", panel.size)
    if credit:
        plate(draw, credit, 22, "br", panel.size)
    panel = Image.alpha_composite(panel.convert("RGBA"), overlay)
    corners = Image.new("L", panel.size, 0)
    ImageDraw.Draw(corners).rounded_rectangle([0, 0, panel.width - 1, panel.height - 1], radius=RADIUS, fill=255)
    panel.putalpha(corners)
    return panel


def figure(source, full_range, rows, region, title, credit):
    """The full picture with the region outlined, then the region from the
    source and from each row's decoded picture."""
    height, width = source[0].shape
    cw, ch = CROP
    pw, ph = cw * ZOOM, ch * ZOOM
    x0, y0 = region
    original = to_rgb(source, full_range)

    # The full picture, fitted into a panel of the region's shape.
    scale = min(pw / width, ph / height)
    tw, th = round(width * scale), round(height * scale)
    ox, oy = (pw - tw) // 2, (ph - th) // 2
    full = Image.new("RGB", (pw, ph), MARGIN)
    full.paste(original.resize((tw, th), Image.LANCZOS), (ox, oy))
    draw = ImageDraw.Draw(full)
    box = [ox + x0 * scale, oy + y0 * scale, ox + (x0 + cw) * scale, oy + (y0 + ch) * scale]
    draw.rectangle([box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2], outline=(0, 0, 0), width=7)
    draw.rectangle(box, outline=BOX, width=3)

    def enlarged(picture):
        return picture.crop((x0, y0, x0 + cw, y0 + ch)).resize((pw, ph), Image.NEAREST)

    panels = [label(full, title, credit), label(enlarged(original), "Source")]
    for size, planes, y_psnr in rows:
        rate = size * 8 * FRAME_RATE / 1e6
        panels.append(label(enlarged(to_rgb(planes, full_range)), f"{rate:.0f} Mbit/s  ·  {y_psnr:.1f} dB"))

    lines = (len(panels) + COLUMNS - 1) // COLUMNS
    sheet = Image.new("RGBA", (COLUMNS * pw + (COLUMNS - 1) * GAP, lines * ph + (lines - 1) * GAP))
    for i, panel in enumerate(panels):
        sheet.alpha_composite(panel, ((i % COLUMNS) * (pw + GAP), (i // COLUMNS) * (ph + GAP)))
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=pathlib.Path, help="an 8-bit 4:2:0 Y4M file; its first picture is used")
    parser.add_argument("out", type=pathlib.Path, help="the PNG file to write")
    parser.add_argument("budgets", type=int, nargs="+", metavar="BYTES", help="the frame budgets, in bytes")
    parser.add_argument("--figure", help="the budgets in the picture, comma-separated (default: all)")
    parser.add_argument("--region", help="the enlarged region's top left corner, X,Y (default: the worst)")
    parser.add_argument("--title", default="Source", help="the full picture's label")
    parser.add_argument("--credit", help="a smaller line in the full picture's bottom right corner")
    parser.add_argument("--gpu", action="store_true", help="encode on the GPU instead of the CPU")
    parser.add_argument(
        "--cli",
        type=pathlib.Path,
        default=ROOT / "target" / "release" / "emberwave-cli",
        help="the emberwave-cli to run (default: target/release/emberwave-cli)",
    )
    args = parser.parse_args()

    source, full_range = read_y4m(args.source)
    height, width = source[0].shape
    if width < CROP[0] or height < CROP[1] or width % 2 or height % 2:
        sys.exit(f"{width}x{height}: the picture must be at least {CROP[0]}x{CROP[1]}, with an even width and height")
    shown = [int(b) for b in args.figure.split(",")] if args.figure else args.budgets
    if not set(shown) <= set(args.budgets):
        sys.exit("--figure names a budget that is not given")
    if args.region:
        region = tuple(int(v) for v in args.region.split(","))
        if not (0 <= region[0] <= width - CROP[0] and 0 <= region[1] <= height - CROP[1]):
            sys.exit(f"--region {args.region} does not fit a {CROP[0]}x{CROP[1]} region in {width}x{height}")

    results = {}
    with tempfile.TemporaryDirectory() as scratch:
        for budget in args.budgets:
            results[budget] = round_trip(
                args.cli.resolve(), args.source.resolve(), budget, args.gpu, pathlib.Path(scratch)
            )

    encoder = "GPU" if args.gpu else "CPU"
    print(f"{width}x{height}, the {encoder} encoder, {FRAME_RATE} frames a second\n")
    print("| Budget | Frame | at 60 frames a second | bits per pixel | Y | Cb | Cr |")
    print("|---|---|---|---|---|---|---|")
    for budget, (size, planes) in results.items():
        db = [psnr(s, d) for s, d in zip(source, planes)]
        print(
            f"| {budget:,} | {size:,} | {size * 8 * FRAME_RATE / 1e6:.0f} Mbit/s | {size * 8 / (width * height):.2f}"
            f" | {db[0]:.2f} | {db[1]:.2f} | {db[2]:.2f} |"
        )

    if not args.region:
        region = worst_region(source[0], results[min(shown)][1][0])
    rows = [(results[b][0], results[b][1], psnr(source[0], results[b][1][0])) for b in shown]
    figure(source, full_range, rows, region, args.title, args.credit).save(args.out, optimize=True)
    print(f"\nregion {region[0]},{region[1]}, {CROP[0]}x{CROP[1]}; wrote {args.out}")


if __name__ == "__main__":
    main()
