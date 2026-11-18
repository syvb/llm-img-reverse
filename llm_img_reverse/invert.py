"""Reconstruct an image from Qwen3-VL soft tokens by optimizing an image until its tokens match.

By default L-BFGS optimizes a 1/f-scaled Fourier spectrum in decorrelated color space; --param pixel
and --optimizer adam are the naive baselines, which mostly recover edges and noise.

    python -m llm_img_reverse.invert examples/testcard.png --streams main --out results/testcard_main
    python -m llm_img_reverse.invert tokens.pt --streams ds8 --out results/edited

An image input is encoded to get the target tokens (and --save-tokens writes them out, so they can be
edited and inverted later); a .pt input holds tokens saved that way.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.optim.lbfgs import _strong_wolfe

from .vision import encode, load_image, load_vision_tower, save_image, stream_names


def relative_error(actual, target):
    return (actual - target).square().sum() / target.square().sum()


def total_variation(image):
    return (image[:, 1:] - image[:, :-1]).square().mean() + (image[:, :, 1:] - image[:, :, :-1]).square().mean()


# Square root of the RGB color covariance of natural images (ImageNet), from Lucid's feature
# visualization code; columns scaled so the largest has norm 1.
COLOR_SQRT = torch.tensor([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]])
COLOR_SQRT = COLOR_SQRT / COLOR_SQRT.norm(dim=0).max()


class FourierImage(torch.nn.Module):
    """Image parameterized by its spectrum, scaled by 1/frequency, in decorrelated color space.

    Gradient steps then change low frequencies and natural color directions about as easily as
    edges, which plain pixels do not (see Olah et al., "Feature Visualization", Distill 2017).
    """

    def __init__(self, shape, generator):
        super().__init__()
        c, h, w = shape
        freqs = (torch.fft.fftfreq(h)[:, None] ** 2 + torch.fft.rfftfreq(w)[None] ** 2).sqrt()
        self.register_buffer("scale", 1 / freqs.clamp(min=1 / max(h, w)))
        self.register_buffer("color", COLOR_SQRT.clone())
        self.spectrum = torch.nn.Parameter(0.01 * torch.randn(c, h, w // 2 + 1, 2, generator=generator))
        self.size = (h, w)

    def forward(self):
        spectrum = torch.view_as_complex(self.spectrum) * self.scale
        image = torch.fft.irfft2(spectrum, s=self.size, norm="ortho")
        return torch.sigmoid(torch.einsum("ij,jhw->ihw", self.color, image))


class PixelImage(torch.nn.Module):
    """Plain pixels (through a sigmoid, so any parameter value is a valid image)."""

    def __init__(self, shape, generator):
        super().__init__()
        self.logits = torch.nn.Parameter(0.04 * torch.randn(shape, generator=generator))

    def forward(self):
        return torch.sigmoid(self.logits)


PARAMETERIZATIONS = {"fourier": FourierImage, "pixel": PixelImage}


def cuda_graph(fn, warmup=2):
    """Capture `fn`, which takes no arguments, in a CUDA graph and return a function that replays it.

    A replay launches all of fn's kernels at once, without Python or per-kernel launch overhead, which
    dominates at these small sizes. It returns the same output tensors every time, overwritten by the
    next replay, and computes from whatever the tensors fn read hold then, such as updated parameters.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):  # outside the capture, select kernels and allocate workspaces
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    # Only this thread is checked for calls that break capture, so concurrent runs (ZeroGPU runs
    # handlers in parallel) can keep working while one captures.
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        outputs = fn()

    def replay():
        graph.replay()
        return outputs

    return replay


@torch.no_grad()
def lbfgs(closure, param, max_eval, lr=1.0, history_size=50):
    """Minimize closure() over the tensor `param` with L-BFGS in at most `max_eval` evaluations.

    The steps of torch.optim.LBFGS with line_search_fn="strong_wolfe" and zero tolerances. Its two-loop
    recursion makes the CPU wait for the GPU twice per stored pair, about 4 ms a step on the Space's
    GPU; here the history is a fixed buffer whose unused rows are zero, which change nothing, so the
    recursion has no syncs and on CUDA runs as one graph. `closure` returns the loss and leaves the
    gradient in param.grad.
    """
    size = param.numel()
    steps, changes = param.new_zeros(history_size, size), param.new_zeros(history_size, size)  # oldest first
    rho, gamma, grad = param.new_zeros(history_size), param.new_ones(()), param.new_zeros(size)

    def two_loop():  # the direction -H grad, with H the inverse Hessian estimate from the history
        q, alpha = grad.neg(), [None] * history_size
        for i in reversed(range(history_size)):
            alpha[i] = steps[i].dot(q) * rho[i]
            q.addcmul_(changes[i], alpha[i], value=-1)
        r = q * gamma
        for i in range(history_size):
            r.addcmul_(steps[i], alpha[i] - changes[i].dot(r) * rho[i])
        return r

    if param.is_cuda:
        two_loop = cuda_graph(two_loop)

    def evaluate(x, t, d):  # loss and gradient at x + t d, for the line search
        param.copy_(x + t * d.view_as(param))
        with torch.enable_grad():
            loss = float(closure())
        return loss, param.grad.flatten().clone()

    with torch.enable_grad():
        loss = float(closure())
    flat_grad, evals, first = param.grad.flatten().clone(), 1, True
    if flat_grad.abs().max() == 0:
        return
    while True:
        if first:
            d = flat_grad.neg()
            t = min(1.0, 1.0 / flat_grad.abs().sum()) * lr
        else:
            change, step = flat_grad - prev_grad, d * t
            curvature = change.dot(step)
            if curvature > 1e-10:
                for buffer, new in [(steps, step), (changes, change), (rho, 1 / curvature)]:
                    buffer[:-1] = buffer[1:].clone()
                    buffer[-1] = new
                gamma.copy_(curvature / change.dot(change))
            grad.copy_(flat_grad)
            d, t = two_loop(), lr
        first, prev_grad = False, flat_grad
        slope = flat_grad.dot(d)
        if slope > 0:
            return
        x = param.clone()
        loss, flat_grad, t, line_evals = _strong_wolfe(evaluate, x, t, d, loss, flat_grad, slope, max_ls=max_eval - evals)
        param.copy_(x + t * d.view_as(param))
        evals += line_evals
        if evals >= max_eval or flat_grad.abs().max() == 0 or (d * t).abs().max() == 0:
            return


def invert(
    model, targets, shape, weights, steps=300, optimizer="lbfgs", lr=None, tv=0.0, param="fourier", seed=0,
    log_every=25, log=lambda line: print(line, flush=True), on_eval=None,
):
    """Find a [3, H, W] image in [0, 1] whose soft tokens match `targets` on the weighted streams.

    `steps` counts evaluations of the vision tower (forward and backward), for either optimizer.
    `on_eval(image, loss)` is called after each one; with L-BFGS this includes line-search trials. On
    CUDA the image tensor is overwritten by the next evaluation, so copy what you keep.
    """
    # Initialized on the CPU generator so a seed gives the same start on every device.
    device = next(iter(targets.values())).device
    image_fn = PARAMETERIZATIONS[param](shape, torch.Generator().manual_seed(seed)).to(device)
    history = []

    def evaluate():
        """The image, loss and per-stream errors at the current parameters, and the gradient in .grad."""
        image_fn.zero_grad()
        image = image_fn()
        tokens = encode(model, image, streams=list(weights))
        errors = {name: relative_error(tokens[name], targets[name]) for name in weights}
        loss = sum(w * errors[name] for name, w in weights.items()) + tv * total_variation(image)
        loss.backward()
        return image.detach(), loss.detach(), {name: error.detach() for name, error in errors.items()}

    if device.type == "cuda":
        with torch.enable_grad():
            evaluate = cuda_graph(evaluate)

    def closure():
        image, loss, errors = evaluate()
        evaluation = len(history) + 1
        history.append({"eval": evaluation, "loss": loss.item(), **{k: v.item() for k, v in errors.items()}})
        if evaluation % log_every == 0:
            log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in history[-1].items()))
        if on_eval:
            on_eval(image, loss.item())
        return loss

    if optimizer == "lbfgs":
        (param,) = image_fn.parameters()
        lbfgs(closure, param, steps, lr=lr or 1)
    else:
        opt = torch.optim.Adam(image_fn.parameters(), lr=lr or 0.05)
        schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
        for _ in range(steps):
            opt.step(closure)
            schedule.step()
    with torch.no_grad():
        return image_fn(), history[log_every - 1 :: log_every]


@torch.no_grad()
def token_metrics(model, image, targets):
    """Per stream: relative squared error and mean per-token cosine similarity to the targets."""
    tokens = encode(model, image)
    return {
        name: {
            "relative_error": relative_error(tokens[name], targets[name]).item(),
            "cosine": torch.cosine_similarity(tokens[name], targets[name], dim=-1).mean().item(),
        }
        for name in targets
    }


def psnr(a, b):
    return (10 * torch.log10(1 / (a - b).square().mean())).item()


def parse_streams(text, names):
    """'main' or 'main,ds8=0.5' or 'all' -> {stream: weight}."""
    if text == "all":
        return dict.fromkeys(names, 1.0)
    weights = {}
    for part in text.split(","):
        name, _, weight = part.partition("=")
        if name not in names:
            raise SystemExit(f"unknown stream {name!r}; choose from {', '.join(names)} or all")
        weights[name] = float(weight or 1)
    return weights


def parse_size(text):
    height, _, width = text.partition("x")
    return int(height), int(width or height)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="image file, or .pt tokens saved with --save-tokens")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--size", default="256", help="HxW (or one number for square), multiples of 32")
    parser.add_argument("--streams", default="main", help="streams to match: main, ds8, ds16, ds24, comma list with =weight, or all")
    parser.add_argument("--steps", type=int, default=300, help="forward/backward evaluations of the vision tower")
    parser.add_argument("--optimizer", choices=["lbfgs", "adam"], default="lbfgs")
    parser.add_argument("--lr", type=float, help="learning rate (default 1 for lbfgs, 0.05 for adam)")
    parser.add_argument("--tv", type=float, default=0.0, help="total variation weight")
    parser.add_argument("--param", choices=list(PARAMETERIZATIONS), default="fourier", help="how the image is optimized")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-tokens", type=Path, help="write the target tokens of an image input here")
    parser.add_argument("--threads", type=int, help="torch CPU threads")
    parser.add_argument("--save-frames", action="store_true", help="save every evaluated image to frames.pt")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    model = load_vision_tower()
    original = None
    if args.input.suffix == ".pt":
        saved = torch.load(args.input)
        targets, shape = saved["tokens"], tuple(saved["shape"])
    else:
        original = load_image(args.input, *parse_size(args.size))
        shape = tuple(original.shape)
        with torch.no_grad():
            targets = encode(model, original)
        if args.save_tokens:
            torch.save({"tokens": targets, "shape": shape}, args.save_tokens)
    weights = parse_streams(args.streams, stream_names(model))

    args.out.mkdir(parents=True, exist_ok=True)
    frames, losses = [], []

    def record(image, loss):
        frames.append((image.clamp(0, 1) * 255).round().to(torch.uint8))
        losses.append(loss)

    start = time.time()
    image, history = invert(
        model, targets, shape, weights, args.steps, args.optimizer, args.lr, args.tv, args.param, args.seed,
        on_eval=record if args.save_frames else None,
    )
    if args.save_frames:
        torch.save({"frames": torch.stack(frames), "losses": torch.tensor(losses)}, args.out / "frames.pt")
    metrics = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "seconds": round(time.time() - start, 1),
        "tokens": token_metrics(model, image, targets),
        "history": history,
    }
    save_image(image, args.out / "reconstruction.png")
    if original is not None:
        metrics["psnr"] = psnr(image, original)
        save_image(torch.cat([original, image], dim=2), args.out / "comparison.png")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("seconds", "psnr", "tokens") if k in metrics}, indent=2))


if __name__ == "__main__":
    main()
