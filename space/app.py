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

from llm_img_reverse.invert import DEFAULT_LR, invert, parse_streams, psnr, token_metrics
from llm_img_reverse.video import run_name, write_video
from llm_img_reverse.vision import encode, load_image, load_vision_tower, stream_names

# On ZeroGPU the model is placed on cuda at import time; a real GPU is attached inside @spaces.GPU.
DEVICE = "cuda" if os.getenv("SPACES_ZERO_GPU") or torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")
MODEL = load_vision_tower(device=DEVICE)
# Statistics of each stream's tokens over COCO photos, from experiments/token_stats.py: per-dimension
# mean and std, and the top principal components with their variances and neighbor correlations.
STATS = torch.load(Path(__file__).with_name("token_stats.pt"))

IMAGE = "Encode an image"
RANDOM = "Random, independent: the mean token plus noise"
CORRELATED = "Random, correlated like real tokens (PCA)"
# Checkbox label -> the streams that one run matches.
RUNS = {
    "deepstack8: after vision block 8": "ds8",
    "deepstack16: after vision block 16": "ds16",
    "deepstack24: after vision block 24": "ds24",
    "main: final soft tokens (the LLM's input)": "main",
    "all four, matched together": "all",
}
# Radio label -> the optimizer's name in invert.
OPTIMIZERS = {"L-BFGS": "lbfgs", "Adam": "adam", "Muon": "muon"}
PREVIEW_SECONDS = 0.25
# A compact summary table, so that its columns fit beside the inputs without breaking words.
CSS = "#summary table { font-size: 13px } #summary th, #summary td { padding: 4px 6px; white-space: nowrap } #summary { overflow-x: auto }"
VIDEO_FRAMES = 600  # longer runs are sampled down to this many video frames


def gpu_seconds(size, runs, steps, optimizer, lr, original=None, targets=None):
    """GPU time to reserve: 15 s to start, plus 1.5 times what an evaluation took on the Space, about
    12 ms (256 px) or 22 ms (384 px) up to block 8 and 30 or 60 ms through the whole tower, for every run
    and with any optimizer."""
    per_eval = sum((0.012 if size == 256 else 0.022) if streams == "ds8" else 0.03 if size == 256 else 0.06 for streams in runs)
    return int(15 + 1.5 * steps * per_eval)


@spaces.GPU(duration=gpu_seconds)
def optimize(size, runs, steps, optimizer, lr, original=None, targets=None):
    """Run one inversion with `optimizer` and `lr` (None for its default) per entry of `runs`, the streams it matches such as "ds8" or "all", one after
    another. Yield (run, uint8 frame, loss) after each evaluation and (run, dict of token metrics) when a
    run ends. The target tokens are those of the image `original`, or `targets` when there is no image."""
    with torch.no_grad():
        if original is not None:
            targets = encode(MODEL, original.to(DEVICE))
        targets = {name: tokens.to(DEVICE) for name, tokens in targets.items()}
    updates = queue.Queue()

    def work():
        try:
            for run, streams in enumerate(runs):
                def record(image, loss):
                    updates.put((run, (image.clamp(0, 1) * 255).round().to(torch.uint8).cpu(), loss))

                weights = parse_streams(streams, stream_names(MODEL))
                image, _ = invert(MODEL, targets, (3, size, size), weights, steps, optimizer=optimizer, lr=lr, on_eval=record)
                updates.put((run, token_metrics(MODEL, image, targets)))
            updates.put(None)
        except Exception as error:  # surfaced in the generator below
            updates.put(error)

    threading.Thread(target=work, daemon=True).start()
    while (item := updates.get()) is not None:
        if isinstance(item, Exception):
            raise item
        yield item


def as_html(tiles):
    """(label, uint8 image or None) tiles side by side, the images inline as data URLs. A gr.Image
    output makes the browser download each update as a file, and at GPU speed every download was
    replaced before it finished, so the preview froze."""
    cells = []
    for label, frame in tiles:
        image = '<div style="aspect-ratio: 1; background: #8882"></div>'  # a run that has not started
        if frame is not None:
            buffer = io.BytesIO()
            Image.fromarray(frame.permute(1, 2, 0).numpy()).save(buffer, format="WEBP", quality=90)
            data = base64.b64encode(buffer.getvalue()).decode()
            image = f'<img src="data:image/webp;base64,{data}" style="display: block; width: 100%">'
        cells.append(f'<div style="flex: 1; max-width: 384px; text-align: center; font-size: 13px">{label}{image}</div>')
    return f'<div style="display: flex; gap: 8px; justify-content: center">{"".join(cells)}</div>'


def random_targets(size, noise, seed):
    """Target tokens on every stream for a size x size image: the mean token of COCO photos plus
    Gaussian noise, scaled in each dimension by `noise` times the spread of real tokens."""
    generator = torch.Generator().manual_seed(seed)
    count = (size // 32) ** 2
    return {
        name: STATS["mean"][name] + noise * STATS["std"][name] * torch.randn(count, len(STATS["mean"][name]), generator=generator)
        for name in stream_names(MODEL)
    }


def correlated_targets(size, noise, seed):
    """Target tokens on every stream for a size x size image, from a Gaussian fitted to real tokens:
    the mean token plus the top principal components, each at `noise` times its standard deviation.
    Each component's coefficients form a random field over the token grid whose correlation between
    neighbors matches COCO photos and falls off as correlation ** distance."""
    generator = torch.Generator().manual_seed(seed)
    side = size // 32
    cells = torch.cartesian_prod(torch.arange(side), torch.arange(side)).double()  # row-major, like tokens
    distance = torch.cdist(cells, cells)
    targets = {}
    for name in stream_names(MODEL):
        correlation = STATS["neighbor_correlation"][name].double().clamp(0, 0.99)
        field = torch.linalg.cholesky(correlation[:, None, None] ** distance)  # [components, tokens, tokens]
        white = torch.randn(len(correlation), side * side, 1, dtype=torch.float64, generator=generator)
        coefficients = (field @ white)[..., 0].T.float() * STATS["variance"][name].sqrt()  # [tokens, components]
        targets[name] = STATS["mean"][name] + noise * coefficients @ STATS["components"][name].float()
    return targets


def reconstruct(target, path, noise, seed, run_labels, optimizer_label, lr, steps, size):
    runs = [streams for label, streams in RUNS.items() if label in run_labels]
    if not runs:
        raise gr.Error("Check at least one set of soft tokens to match.")
    optimizer = OPTIMIZERS[optimizer_label]
    if optimizer == "lbfgs":
        lr, method = None, "L-BFGS"
    elif not lr or lr <= 0:
        raise gr.Error("The learning rate must be above 0.")
    else:
        method = f"{optimizer_label}, learning rate {lr:g} decaying along a cosine"
    size, steps = int(size), int(steps)
    original = targets = None
    if target == IMAGE:
        if path is None:
            raise gr.Error("Upload an image or pick an example first.")
        original = load_image(path, size, size)
    else:
        targets = (random_targets if target == RANDOM else correlated_targets)(size, noise, int(seed))
    names = [run_name(streams) for streams in runs]
    first = [("original", (original * 255).round().to(torch.uint8))] if original is not None else []
    # Per run: the best image so far, its loss, the evaluations so far, the token metrics at the end,
    # and (best image so far, its loss, evaluation) at every stride-th evaluation, for the video.
    best, best_loss, count, metrics = [None] * len(runs), [None] * len(runs), [0] * len(runs), [None] * len(runs)
    stride, samples = -(-steps // VIDEO_FRAMES), [[] for _ in runs]
    start = time.time()
    last_preview = start - PREVIEW_SECONDS  # show the first frame right away
    for run, *item in optimize(size, runs, steps, optimizer, lr, original=original, targets=targets):
        if len(item) == 1:  # the run ended
            metrics[run] = item[0]
            if samples[run][-1][2] != count[run]:
                samples[run].append((best[run], best_loss[run], count[run]))
            continue
        frame, loss = item
        count[run] += 1
        if count[run] == 1 or loss < best_loss[run]:
            best[run], best_loss[run] = frame, loss
        if (count[run] - 1) % stride == 0:
            samples[run].append((best[run], best_loss[run], count[run]))
        if time.time() - last_preview >= PREVIEW_SECONDS:
            last_preview = time.time()
            status = f"Run {run + 1} of {len(runs)}, {names[run]}: evaluation {count[run]} of {steps}, loss {best_loss[run]:.3f}"
            yield as_html(first + list(zip(names, best))), status, None
    seconds = time.time() - start

    if original is not None:
        header = f"PSNR against the original; a flat gray image scores {psnr(torch.full_like(original, 0.5), original):.1f} dB."
    elif target == RANDOM:
        header = f"**Random target tokens**: the mean token plus noise of {noise:g} × the spread of real tokens, seed {int(seed)}."
    else:
        header = (
            f"**Correlated random target tokens**: the mean token plus the top {len(STATS['variance']['main'])} principal "
            f"components at {noise:g} × their spread, correlated between neighboring tokens like COCO photos, seed {int(seed)}."
        )
    stopped = [f"{names[run]} after {count[run]}" for run in range(len(runs)) if count[run] < steps]
    columns = ["run", *(["PSNR"] if original is not None else []), "loss", *map(run_name, stream_names(MODEL))]
    rows = []
    for run, streams in enumerate(runs):
        matched = parse_streams(streams, stream_names(MODEL))
        # Random targets of streams a run did not match are unrelated noise, so their errors mean nothing.
        errors = [
            f"**{m['relative_error']:.3f}**" if name in matched else f"{m['relative_error']:.3f}" if original is not None else "–"
            for name, m in metrics[run].items()
        ]
        score = [f"{psnr(best[run].float() / 255, original):.1f} dB"] if original is not None else []
        rows.append([names[run], *score, f"{samples[run][0][1]:.3f} → {best_loss[run]:.3f}", *errors])
    summary = (
        f"{header} {sum(count)} evaluations in {seconds:.0f} s with {method}"
        + (f"; L-BFGS found no lower loss and stopped {', '.join(stopped)}" if stopped else "")
        + ". Loss goes from the gray start to the best image; the last columns are the relative error of each "
        "stream's tokens, matched ones in bold.\n\n"
        + "\n".join("| " + " | ".join(row) + " |" for row in [columns, ["---"] * len(columns), *rows])
    )
    video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    tags = [tag for tag in ({RANDOM: "random", CORRELATED: "correlated"}.get(target), optimizer != "lbfgs" and optimizer_label) if tag]
    suffix = f" ({', '.join(tags)})" if tags else ""
    write_video(
        video, original, [label for label, _ in first] + [name + suffix for name in names],
        [(torch.stack([s[0] for s in run_samples]), [s[1] for s in run_samples]) for run_samples in samples],
        [s[2] for s in max(samples, key=len)],
    )
    yield as_html(first + list(zip(names, best))), summary, video


with gr.Blocks(title="Qwen3-VL soft-token inversion") as demo:
    gr.Markdown(
        "# Running Qwen3-VL's vision stack backwards\n"
        "Encode an image with the vision tower of "
        "[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), then start from a gray "
        "image and optimize it with L-BFGS, Adam or Muon until its soft tokens match. Watch what the tokens keep. "
        "DeepStack layer 8 usually gives a photo back. The final tokens alone stay gray with L-BFGS; Muon gets "
        "the cat back from them after about 2000 evaluations. "
        "Check several sets of tokens to compare them side by side. "
        "Or start from random tokens, built from the tokens of 200 COCO photos: either their average token "
        "plus Gaussian noise, scaled in each of the 4096 dimensions by how much real tokens vary there, or a "
        "sample along their top principal components, correlated between neighboring tokens like a photo's. "
        "Code and findings: [syvb/llm-img-reverse](https://github.com/syvb/llm-img-reverse)."
    )
    with gr.Row():
        with gr.Column():
            target_in = gr.Radio([IMAGE, RANDOM, CORRELATED], value=IMAGE, label="Target tokens")
            image_in = gr.Image(type="filepath", label="Image (center-cropped to a square)")
            with gr.Row(visible=False) as random_in:
                noise_in = gr.Slider(0, 2, value=1, step=0.05, label="Noise, in standard deviations of real tokens")
                seed_in = gr.Number(0, precision=0, label="Seed")
            runs_in = gr.CheckboxGroup(
                list(RUNS), value=[next(iter(RUNS))], label="Soft tokens to match",
                info="Each checked option is its own run, one after another, and its own tile in the video.",
            )
            with gr.Row():
                optimizer_in = gr.Radio(
                    list(OPTIMIZERS), value="L-BFGS", label="Optimizer", scale=3,
                    info="L-BFGS stops early once it finds no lower loss. Adam and Muon take every step, with the "
                    "learning rate decaying along a cosine; Muon orthogonalizes each step of the image's spectrum.",
                )
                lr_in = gr.Number(DEFAULT_LR["adam"], minimum=0, step=0.0001, label="Learning rate", visible=False, scale=1)
            steps_in = gr.Slider(
                50, 5000, value=300, step=50, label="Evaluations per run",
                info="12 to 60 ms each on the GPU; long runs can exceed a free account's ZeroGPU limits.",
            )
            size_in = gr.Radio([256, 384], value=256, label="Size in pixels (64 or 144 tokens)")
            run = gr.Button("Reconstruct", variant="primary")
        with gr.Column():
            image_out = gr.HTML(label="Reconstructions (best so far)", show_label=True, container=True, min_height=150)
            status = gr.Markdown(elem_id="summary")
            video_out = gr.Video(label="Optimization", show_label=False)  # the label would cover the tile titles
    gr.Examples([f"examples/{name}.png" for name in ["chelsea", "coffee", "astronaut", "testcard"]], inputs=image_in)
    target_in.change(
        lambda target: (gr.update(visible=target == IMAGE), gr.update(visible=target != IMAGE)),
        target_in, [image_in, random_in],
    )
    optimizer_in.change(
        lambda label: gr.update(visible=label != "L-BFGS", value=DEFAULT_LR[OPTIMIZERS[label]]), optimizer_in, lr_in
    )
    run.click(
        reconstruct, [target_in, image_in, noise_in, seed_in, runs_in, optimizer_in, lr_in, steps_in, size_in],
        [image_out, status, video_out],
    )

if __name__ == "__main__":
    demo.launch(css=CSS)
