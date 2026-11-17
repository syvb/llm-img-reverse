"""Contact sheets and videos of inversion runs."""

import subprocess

import torch
from PIL import Image, ImageDraw, ImageFont

from .invert import psnr

FONT = ImageFont.load_default(size=13)
STRIP = 20  # caption strip height in pixels
FPS, HOLD_SECONDS = 30, 2


def sheet(tiles, names, captions=None):
    """[3, H, W] images in [0, 1] side by side, each name above its tile and each caption below it."""
    height, width = tiles[0].shape[1:]
    out = Image.new("RGB", (width * len(tiles), height + STRIP * (2 if captions else 1)), "white")
    draw = ImageDraw.Draw(out)
    for i, tile in enumerate(tiles):
        pixels = (tile.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        out.paste(Image.fromarray(pixels), (i * width, STRIP))
        draw.text((i * width + 6, 3), names[i], fill="black", font=FONT)
        if captions:
            draw.text((i * width + 6, STRIP + height + 3), captions[i], fill="black", font=FONT)
    return out


def write_video(path, original, names, runs):
    """Write an MP4 of runs optimizing toward `original`, one video frame per evaluation.

    `runs` holds (frames, losses) per run: the [E, 3, H, W] uint8 images and the losses that
    --save-frames records. Each tile shows its run's lowest-loss image so far, which skips L-BFGS
    line-search trials that were rejected. `names` labels the original and then each run. The last
    frame is held for HOLD_SECONDS.
    """
    best_index = [running_best(list(losses)) for _, losses in runs]
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
        for (frames, losses), indices in zip(runs, best_index):
            i = indices[min(n, len(indices) - 1)]
            tile = frames[i].float() / 255
            tiles.append(tile)
            captions.append(f"{psnr(tile, original):.1f} dB, loss {float(losses[i]):.3f}")
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
