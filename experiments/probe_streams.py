"""Forward-only probes of what each stream responds to, written to results/probes.md.

    python experiments/probe_streams.py examples/testcard.png examples/chelsea.png

- Top-8 token energy: the share of a stream's squared norm in its 8 largest of 64 tokens.
- Line probe: relative error along the straight line from a gray image (alpha 0) to the original
  (alpha 1). A steady fall means a descent path from gray to the original exists.
- Perturbations: relative error after adding 2% Gaussian pixel noise, shifting 4 px, converting to
  grayscale, or lowering brightness by 0.2.
"""

import sys
from pathlib import Path

import torch

from llm_img_reverse.invert import relative_error
from llm_img_reverse.vision import encode, load_image, load_vision_tower, stream_names

ALPHAS = [0.1, 0.3, 0.5, 0.9]
PERTURBATIONS = {
    "noise 2%": lambda x: (x + 0.02 * torch.randn_like(x)).clamp(0, 1),
    "shift 4px": lambda x: torch.roll(x, 4, dims=2),
    "grayscale": lambda x: x.mean(0, keepdim=True).expand_as(x),
    "brightness −0.2": lambda x: (x - 0.2).clamp(0, 1),
}


@torch.no_grad()
def main(paths):
    torch.manual_seed(0)
    model = load_vision_tower()
    lines = [
        "| image | stream | top-8 token energy | "
        + " | ".join(f"α={a}" for a in ALPHAS) + " | " + " | ".join(PERTURBATIONS) + " |",
        "|---|---|---|" + "---|" * (len(ALPHAS) + len(PERTURBATIONS)),
    ]
    for path in paths:
        original = load_image(path, 256, 256)
        gray = torch.full_like(original, 0.5)
        targets = encode(model, original)
        variants = [encode(model, gray + a * (original - gray)) for a in ALPHAS]
        variants += [encode(model, perturb(original)) for perturb in PERTURBATIONS.values()]
        for name in stream_names(model):
            energy = targets[name].square().sum(-1)
            top8 = (energy.topk(8).values.sum() / energy.sum()).item()
            errors = " | ".join(f"{relative_error(v[name], targets[name]).item():.3f}" for v in variants)
            lines.append(f"| {Path(path).stem} | {name} | {top8:.0%} | {errors} |")
    out = Path("results/probes.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
