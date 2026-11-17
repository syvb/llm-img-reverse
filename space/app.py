"""Hugging Face Space: reconstruct an image from Qwen3-VL's soft tokens, live."""

import base64
import io
import os
import queue
import tempfile
import threading
import time
from pathlib import Path

import gradio as gr
import spaces
import torch
from PIL import Image

from llm_img_reverse.invert import invert, parse_streams, psnr, token_metrics
from llm_img_reverse.video import write_video
from llm_img_reverse.vision import encode, load_image, load_vision_tower, stream_names

# On ZeroGPU the model is placed on cuda at import time; a real GPU is attached inside @spaces.GPU.
DEVICE = "cuda" if os.getenv("SPACES_ZERO_GPU") or torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")
MODEL = load_vision_tower(device=DEVICE)
# Per-dimension mean and standard deviation of each stream's tokens over COCO photos, from
# experiments/token_stats.py.
STATS = torch.load(Path(__file__).with_name("token_stats.pt"))

IMAGE, RANDOM = "Encode an image", "Random: the mean token plus noise"
STREAMS = {
    "ds8: DeepStack after vision block 8": "ds8",
    "ds16: DeepStack after vision block 16": "ds16",
    "ds24: DeepStack after vision block 24": "ds24",
    "main: final soft tokens (the LLM's input)": "main",
    "all four": "all",
}
PREVIEW_SECONDS = 0.25
VIDEO_FRAMES = 600  # longer runs are sampled down to this many video frames


def gpu_seconds(size, streams, steps, original=None, targets=None):
    """GPU time to reserve for one run: 15 s to start, plus 1.5 times what an evaluation took on the
    Space, about 30 ms up to block 8 and 50 ms (256 px) or 75 ms (384 px) through the whole tower."""
    per_eval = 0.03 if streams == "ds8" else 0.05 if size == 256 else 0.075
    return int(15 + 1.5 * steps * per_eval)


@spaces.GPU(duration=gpu_seconds)
def optimize(size, streams, steps, original=None, targets=None):
    """Yield (uint8 frame, loss) after each evaluation, then a dict of token metrics. The target tokens
    are those of the image `original`, or `targets` when there is no image."""
    with torch.no_grad():
        if original is not None:
            targets = encode(MODEL, original.to(DEVICE))
        targets = {name: tokens.to(DEVICE) for name, tokens in targets.items()}
    weights = parse_streams(streams, stream_names(MODEL))
    updates = queue.Queue()

    def record(image, loss):
        updates.put(((image.clamp(0, 1) * 255).round().to(torch.uint8).cpu(), loss))

    def work():
        try:
            image, _ = invert(MODEL, targets, (3, size, size), weights, steps, on_eval=record)
            updates.put(token_metrics(MODEL, image, targets))
        except Exception as error:  # surfaced in the generator below
            updates.put(error)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = updates.get()
        if isinstance(item, Exception):
            raise item
        yield item
        if isinstance(item, dict):
            return


def as_html(frame):
    """The image inline as a data URL. A gr.Image output makes the browser download each update as a
    file, and at GPU speed every download was replaced before it finished, so the preview froze."""
    buffer = io.BytesIO()
    Image.fromarray(frame.permute(1, 2, 0).numpy()).save(buffer, format="WEBP", quality=90)
    data = base64.b64encode(buffer.getvalue()).decode()
    return f'<img src="data:image/webp;base64,{data}" style="display: block; width: 100%; max-width: 384px; margin: auto">'


def random_targets(streams, size, noise, seed):
    """Target tokens on the matched streams for a size x size image: the mean token of COCO photos
    plus Gaussian noise, scaled in each dimension by `noise` times the spread of real tokens."""
    generator = torch.Generator().manual_seed(seed)
    count = (size // 32) ** 2
    return {
        name: STATS["mean"][name] + noise * STATS["std"][name] * torch.randn(count, len(STATS["mean"][name]), generator=generator)
        for name in parse_streams(streams, stream_names(MODEL))
    }


def reconstruct(target, path, noise, seed, streams_label, steps, size):
    size, steps, streams = int(size), int(steps), STREAMS[streams_label]
    original = targets = None
    if target == IMAGE:
        if path is None:
            raise gr.Error("Upload an image or pick an example first.")
        original = load_image(path, size, size)
    else:
        targets = random_targets(streams, size, noise, int(seed))
    # (best image so far, its loss, evaluation) at every stride-th evaluation, for the video
    stride, samples, count = -(-steps // VIDEO_FRAMES), [], 0
    start = time.time()
    last_preview = start - PREVIEW_SECONDS  # show the first frame right away
    for item in optimize(size, streams, steps, original=original, targets=targets):
        if isinstance(item, dict):
            metrics = item
            break
        count += 1
        if count == 1 or item[1] < best_loss:
            best, best_loss = item
        if (count - 1) % stride == 0:
            samples.append((best, best_loss, count))
        if time.time() - last_preview >= PREVIEW_SECONDS:
            last_preview = time.time()
            yield as_html(best), f"Evaluation {count} of {steps}, loss {best_loss:.3f}", None
    seconds = time.time() - start
    if samples[-1][2] != count:
        samples.append((best, best_loss, count))
    frames, losses, evaluations = zip(*samples)

    if original is not None:
        gray = psnr(torch.full_like(original, 0.5), original)
        score = f"**PSNR {psnr(best.float() / 255, original):.1f} dB** (a flat gray image scores {gray:.1f})."
    else:
        score = f"**Random target tokens**: the mean token plus noise of {noise:g} × the spread of real tokens, seed {int(seed)}."
    rows = "\n".join(
        f"| {name}{' (matched)' if streams in (name, 'all') else ''} | {m['cosine']:.3f} | {m['relative_error']:.3f} |"
        for name, m in metrics.items()
    )
    stopped = f" (L-BFGS found no lower loss and stopped before {steps})" if count < steps else ""
    summary = (
        f"{score} Loss {losses[0]:.3f} at the gray start, {best_loss:.3f} at best; "
        f"{count} evaluations in {seconds:.0f} s{stopped}.\n\n"
        f"| stream | token cosine | relative error |\n|---|---|---|\n{rows}"
    )
    video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    names = ["original", streams] if original is not None else [f"{streams}, random tokens"]
    write_video(video, original, names, [(torch.stack(frames), list(losses))], list(evaluations))
    yield as_html(best), summary, video


with gr.Blocks(title="Qwen3-VL soft-token inversion") as demo:
    gr.Markdown(
        "# Running Qwen3-VL's vision stack backwards\n"
        "Encode an image with the vision tower of "
        "[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), then start from a gray "
        "image and optimize it with L-BFGS until its soft tokens match. Watch what the tokens keep. "
        "DeepStack layer 8 usually gives a photo back; the final tokens alone usually stay gray. "
        "Or start from random tokens: the average token of 200 COCO photos plus Gaussian noise, scaled in "
        "each of the 4096 dimensions by how much real tokens vary there. "
        "Code and findings: [syvb/llm-img-reverse](https://github.com/syvb/llm-img-reverse)."
    )
    with gr.Row():
        with gr.Column():
            target_in = gr.Radio([IMAGE, RANDOM], value=IMAGE, label="Target tokens")
            image_in = gr.Image(type="filepath", label="Image (center-cropped to a square)")
            with gr.Row(visible=False) as random_in:
                noise_in = gr.Slider(0, 2, value=1, step=0.05, label="Noise, in standard deviations of real tokens")
                seed_in = gr.Number(0, precision=0, label="Seed")
            streams_in = gr.Radio(list(STREAMS), value=next(iter(STREAMS)), label="Soft tokens to match")
            steps_in = gr.Slider(
                50, 5000, value=300, step=50, label="L-BFGS evaluations",
                info="30 to 75 ms each on the GPU; long runs can exceed a free account's ZeroGPU limits.",
            )
            size_in = gr.Radio([256, 384], value=256, label="Size in pixels (64 or 144 tokens)")
            run = gr.Button("Reconstruct", variant="primary")
        with gr.Column():
            image_out = gr.HTML(label="Reconstruction (best so far)", show_label=True, container=True, min_height=300)
            status = gr.Markdown()
            video_out = gr.Video(label="Optimization", show_label=False)  # the label would cover the tile titles
    gr.Examples([f"examples/{name}.png" for name in ["chelsea", "coffee", "astronaut", "testcard"]], inputs=image_in)
    target_in.change(
        lambda target: (gr.update(visible=target == IMAGE), gr.update(visible=target == RANDOM)),
        target_in, [image_in, random_in],
    )
    run.click(
        reconstruct, [target_in, image_in, noise_in, seed_in, streams_in, steps_in, size_in], [image_out, status, video_out]
    )

if __name__ == "__main__":
    demo.launch()
