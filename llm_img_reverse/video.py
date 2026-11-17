"""Contact sheets and videos of inversion runs."""

import subprocess

import torch
from PIL import Image, ImageDraw, ImageFont

from .invert import psnr

FONT = ImageFont.load_default(size=13)
STRIP = 20  # caption strip height in pixels
FPS, HOLD_SECONDS = 30, 2


def run_name(streams):
    """A run's title in videos, from the streams it matched: deepstack8 for ds8, and so on."""
    return streams.replace("ds", "deepstack")


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


def write_video(path, original, names, runs, evaluations=None):
    """Write an MP4 of runs optimizing toward `original`, one video frame per evaluation.

    `runs` holds (frames, losses) per run: the [E, 3, H, W] uint8 images and the losses that
    --save-frames records. Each tile shows its run's lowest-loss image so far, which skips L-BFGS
    line-search trials that were rejected. `names` labels the original and then each run; when the
    target tokens came from no image, `original` is None and only the runs are shown, without PSNR.
    `evaluations` numbers the frames when they are a sample of a longer run (by default frame n is
    evaluation n + 1). The last frame is held for HOLD_SECONDS.
    """
    best_index = [running_best(list(losses)) for _, losses in runs]
    count = max(len(b) for b in best_index)
    labels = evaluations or range(1, count + 1)
    height, width = runs[0][0].shape[2:]
    width, height = width * len(names), height + 2 * STRIP
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
         str(path)],
        stdin=subprocess.PIPE,
    )

    def progress(last):
        return f"evaluation {labels[last]} / {labels[-1]}"

    for n in range(count + FPS * HOLD_SECONDS):
        tiles, captions = ([original], [progress(min(n, count - 1))]) if original is not None else ([], [])
        for (frames, losses), indices in zip(runs, best_index):
            last = min(n, len(indices) - 1)  # a run that ended early holds its last frame
            tile = frames[indices[last]].float() / 255
            tiles.append(tile)
            score = f"{psnr(tile, original):.1f} dB" if original is not None else progress(last)
            captions.append(f"{score}, loss {float(losses[indices[last]]):.3f}")
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
