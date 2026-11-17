"""Summarize inversion runs: a Markdown table, and per input image a labeled contact sheet and,
when every run of that image saved frames (--save-frames), a video of the optimization.

    python experiments/summarize.py results/cpu
"""

import json
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from llm_img_reverse.invert import parse_size, psnr
from llm_img_reverse.vision import load_image

STREAM_ORDER = ["main", "ds8", "ds16", "ds24", "all"]
FONT = ImageFont.load_default(size=13)
STRIP = 20  # caption strip height in pixels
FPS, HOLD_SECONDS = 30, 2


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
        run_dirs = [by_streams[s][0] for s in ordered]
        if all((d / "frames.pt").exists() for d in run_dirs):
            write_video(root / f"{image}.mp4", original, names, run_dirs)
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def sheet(tiles, names, captions=None):
    """[3, H, W] images in [0, 1] side by side, each name above its tile and each caption below it."""
    height, width = tiles[0].shape[1:]
    out = Image.new("RGB", (width * len(tiles), height + STRIP * (2 if captions else 1)), "white")
    draw = ImageDraw.Draw(out)
    for i, tile in enumerate(tiles):
        pixels = (tile.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
        out.paste(Image.fromarray(pixels), (i * width, STRIP))
        draw.text((i * width + 6, 3), names[i], fill="black", font=FONT)
        if captions:
            draw.text((i * width + 6, STRIP + height + 3), captions[i], fill="black", font=FONT)
    return out


def write_video(path, original, names, run_dirs):
    """One video frame per vision-tower evaluation; each tile shows its run's lowest-loss image so far,
    which skips L-BFGS line-search trials that were rejected. The last frame is held."""
    recorded = [torch.load(d / "frames.pt") for d in run_dirs]
    best_index = [running_best(r["losses"].tolist()) for r in recorded]
    count = max(len(b) for b in best_index)
    width, height = original.shape[2] * len(names), original.shape[1] + 2 * STRIP
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
         str(path)],
        stdin=subprocess.PIPE,
    )
    for n in range(count + FPS * HOLD_SECONDS):
        tiles, captions = [original], [f"evaluation {min(n + 1, count)} / {count}"]
        for r, indices in zip(recorded, best_index):
            i = indices[min(n, len(indices) - 1)]
            tile = r["frames"][i].float() / 255
            tiles.append(tile)
            captions.append(f"{psnr(tile, original):.1f} dB, loss {r['losses'][i]:.3f}")
        ffmpeg.stdin.write(sheet(tiles, names, captions).tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait():
        raise RuntimeError(f"ffmpeg failed writing {path}")


def running_best(losses):
    """For each evaluation n, the index of the lowest loss among evaluations 0..n."""
    best, indices = 0, []
    for i, loss in enumerate(losses):
        best = i if loss < losses[best] else best
        indices.append(best)
    return indices


if __name__ == "__main__":
    main(Path(sys.argv[1]))
