"""Summarize inversion runs: a Markdown table and one labeled contact sheet per input image.

    python experiments/summarize.py results/cpu
"""

import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from llm_img_reverse.invert import parse_size, psnr
from llm_img_reverse.vision import load_image, save_image

STREAM_ORDER = ["main", "ds8", "ds16", "ds24", "all"]


def main(root):
    runs = {}
    for path in sorted(root.glob("*/metrics.json")):
        metrics = json.loads(path.read_text())
        image, streams = Path(metrics["args"]["input"]).stem, metrics["args"]["streams"]
        runs.setdefault(image, {})[streams] = (path.parent, metrics)

    lines = [
        "| image | matched streams | PSNR (dB) | gray PSNR (dB) | rel. error main | ds8 | ds16 | ds24 | minutes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for image, by_streams in runs.items():
        any_metrics = next(iter(by_streams.values()))[1]
        original = load_image(any_metrics["args"]["input"], *parse_size(any_metrics["args"]["size"]))
        gray = psnr(torch.full_like(original, 0.5), original)
        tiles = [("original", original)]
        for streams in sorted(by_streams, key=lambda s: STREAM_ORDER.index(s) if s in STREAM_ORDER else 99):
            run_dir, m = by_streams[streams]
            errors = " | ".join(f"{m['tokens'][s]['relative_error']:.3f}" for s in STREAM_ORDER[:4])
            lines.append(
                f"| {image} | {streams} | {m['psnr']:.1f} | {gray:.1f} | {errors} | {m['seconds'] / 60:.1f} |"
            )
            tiles.append((streams, load_image(run_dir / "reconstruction.png", *original.shape[1:])))
        save_image(torch.cat([t for _, t in tiles], dim=2), root / f"{image}.png")
        label(root / f"{image}.png", [name for name, _ in tiles], original.shape[2])
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def label(path, names, tile_width):
    """Add a caption strip with each tile's name above the sheet."""
    sheet = Image.open(path)
    out = Image.new("RGB", (sheet.width, sheet.height + 20), "white")
    out.paste(sheet, (0, 20))
    draw = ImageDraw.Draw(out)
    for i, name in enumerate(names):
        draw.text((i * tile_width + 6, 4), name, fill="black")
    out.save(path)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
