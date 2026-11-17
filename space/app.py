"""Hugging Face Space: reconstruct an image from Qwen3-VL's soft tokens, live."""

import base64
import io
import os
import queue
import tempfile
import threading
import time

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

STREAMS = {
    "ds8: DeepStack after vision block 8": "ds8",
    "ds16: DeepStack after vision block 16": "ds16",
    "ds24: DeepStack after vision block 24": "ds24",
    "main: final soft tokens (the LLM's input)": "main",
    "all four": "all",
}
PREVIEW_SECONDS = 0.25


def gpu_seconds(original, streams, steps):
    """Generous GPU time for one run: per-evaluation cost scales with the number of pixels."""
    return int(20 + steps * 0.15 * original.shape[1] * original.shape[2] / 256**2)


@spaces.GPU(duration=gpu_seconds)
def optimize(original, streams, steps):
    """Yield (uint8 frame, loss) after each evaluation, then a dict of token metrics."""
    original = original.to(DEVICE)
    with torch.no_grad():
        targets = encode(MODEL, original)
    weights = parse_streams(streams, stream_names(MODEL))
    updates = queue.Queue()

    def record(image, loss):
        updates.put(((image.clamp(0, 1) * 255).round().to(torch.uint8).cpu(), loss))

    def work():
        try:
            image, _ = invert(MODEL, targets, tuple(original.shape), weights, steps, on_eval=record)
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


def reconstruct(path, streams_label, steps, size):
    if path is None:
        raise gr.Error("Upload an image or pick an example first.")
    original = load_image(path, int(size), int(size))
    streams, steps = STREAMS[streams_label], int(steps)
    frames, losses, best = [], [], 0
    start = time.time()
    last_preview = start - PREVIEW_SECONDS  # show the first frame right away
    for item in optimize(original, streams, steps):
        if isinstance(item, dict):
            metrics = item
            break
        frames.append(item[0])
        losses.append(item[1])
        best = len(losses) - 1 if losses[-1] < losses[best] else best
        if time.time() - last_preview >= PREVIEW_SECONDS:
            last_preview = time.time()
            yield as_html(frames[best]), f"Evaluation {len(frames)} of {steps}, loss {losses[best]:.3f}", None
    seconds = time.time() - start

    image = frames[best].float() / 255
    gray = psnr(torch.full_like(original, 0.5), original)
    rows = "\n".join(
        f"| {name}{' (matched)' if streams in (name, 'all') else ''} | {m['cosine']:.3f} | {m['relative_error']:.3f} |"
        for name, m in metrics.items()
    )
    summary = (
        f"**PSNR {psnr(image, original):.1f} dB** (a flat gray image scores {gray:.1f}). "
        f"{len(frames)} evaluations in {seconds:.0f} s.\n\n"
        f"| stream | token cosine | relative error |\n|---|---|---|\n{rows}"
    )
    video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    write_video(video, original, ["original", streams], [(torch.stack(frames), losses)])
    yield as_html(frames[best]), summary, video


with gr.Blocks(title="Qwen3-VL soft-token inversion") as demo:
    gr.Markdown(
        "# Running Qwen3-VL's vision stack backwards\n"
        "Encode an image with the vision tower of "
        "[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), then start from a gray "
        "image and optimize it with L-BFGS until its soft tokens match. Watch what the tokens keep. "
        "DeepStack layer 8 usually gives a photo back; the final tokens alone usually stay gray. "
        "Code and findings: [syvb/llm-img-reverse](https://github.com/syvb/llm-img-reverse)."
    )
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="filepath", label="Image (center-cropped to a square)")
            streams_in = gr.Radio(list(STREAMS), value=next(iter(STREAMS)), label="Soft tokens to match")
            steps_in = gr.Slider(50, 500, value=300, step=25, label="L-BFGS evaluations")
            size_in = gr.Radio([256, 384], value=256, label="Size in pixels (64 or 144 tokens)")
            run = gr.Button("Reconstruct", variant="primary")
        with gr.Column():
            image_out = gr.HTML(label="Reconstruction (best so far)", show_label=True, container=True, min_height=300)
            status = gr.Markdown()
            video_out = gr.Video(label="Optimization")
    gr.Examples([f"examples/{name}.png" for name in ["chelsea", "coffee", "astronaut", "testcard"]], inputs=image_in)
    run.click(reconstruct, [image_in, streams_in, steps_in, size_in], [image_out, status, video_out])

if __name__ == "__main__":
    demo.launch()
