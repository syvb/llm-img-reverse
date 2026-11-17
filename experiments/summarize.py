"""Summarize inversion runs: a Markdown table, and per input image a labeled contact sheet and,
when every run of that image saved frames (--save-frames), a video of the optimization.

    python experiments/summarize.py results/cpu
"""

import json
import sys
from pathlib import Path

import torch

from llm_img_reverse.invert import parse_size, psnr
from llm_img_reverse.vision import load_image
from llm_img_reverse.video import run_name, sheet, write_video

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
        ordered = sorted(by_streams, key=lambda s: STREAM_ORDER.index(s) if s in STREAM_ORDER else 99)
        for streams in ordered:
            m = by_streams[streams][1]
            errors = " | ".join(f"{m['tokens'][s]['relative_error']:.3f}" for s in STREAM_ORDER[:4])
            lines.append(
                f"| {image} | {streams} | {m['psnr']:.1f} | {gray:.1f} | {errors} | {m['seconds'] / 60:.1f} |"
            )
        names = ["original", *ordered]
        tiles = [original] + [load_image(by_streams[s][0] / "reconstruction.png", *original.shape[1:]) for s in ordered]
        sheet(tiles, names).save(root / f"{image}.png")
        frame_files = [by_streams[s][0] / "frames.pt" for s in ordered]
        if all(f.exists() for f in frame_files):
            recorded = [torch.load(f) for f in frame_files]
            write_video(
                root / f"{image}.mp4", original, ["original", *map(run_name, ordered)],
                [(r["frames"], r["losses"].tolist()) for r in recorded],
            )
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
