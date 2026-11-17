"""Statistics of each stream's soft tokens over COCO photos, which the Space uses to make random target
tokens. Writes space/token_stats.pt (needs pyarrow).

    python experiments/token_stats.py --images 200 --components 64

Each photo is encoded at 256x256 and 384x384, the Space's two sizes, and the statistics pool every
token of every photo at both sizes:

- mean and std: per dimension.
- components and variance: the top principal components of the tokens and their variances.
- neighbor_correlation: for each component, the correlation between the coefficients of tokens that
  are horizontal or vertical neighbors in the grid.
"""

import argparse
import io
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from llm_img_reverse.vision import encode, load_image, load_vision_tower

DATASET, SHARD = "detection-datasets/coco", "data/val-00000-of-00002-c4f2e391ee4aba11.parquet"
OUT = Path(__file__).resolve().parent.parent / "space" / "token_stats.pt"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=int, default=200)
    parser.add_argument("--components", type=int, default=64)
    args = parser.parse_args()

    model = load_vision_tower()
    table = pq.read_table(hf_hub_download(DATASET, SHARD, repo_type="dataset"), columns=["image"])
    # Running sums of tokens, of their outer products, and of outer products of neighboring tokens.
    total, products, pair_products = {}, {}, {}
    count = pairs = 0
    for i, row in enumerate(table.column("image").to_pylist()[: args.images]):
        for size in (256, 384):
            tokens = encode(model, load_image(io.BytesIO(row["bytes"]), size, size))
            side = size // 32
            for name, t in tokens.items():
                grid = t.reshape(side, side, -1)
                left, right = grid[:, :-1].flatten(0, 1), grid[:, 1:].flatten(0, 1)
                top, bottom = grid[:-1].flatten(0, 1), grid[1:].flatten(0, 1)
                total[name] = total.get(name, 0) + t.double().sum(0)
                products[name] = products.get(name, 0) + (t.T @ t).double()
                pair_products[name] = pair_products.get(name, 0) + (left.T @ right + top.T @ bottom).double()
            count += side * side
            pairs += 2 * side * (side - 1)
        print(f"{i + 1}/{args.images}", flush=True)

    stats = {key: {} for key in ("mean", "std", "components", "variance", "neighbor_correlation")}
    for name in total:
        mean = total[name] / count
        covariance = products[name] / count - mean.outer(mean)
        neighbors = pair_products[name] / pairs - mean.outer(mean)
        variance, vectors = torch.linalg.eigh(covariance)  # ascending
        variance, vectors = variance.flip(0)[: args.components], vectors.flip(1)[:, : args.components]
        # Covariance of neighbors' coefficients along each component, over the component's variance.
        correlation = ((vectors.T @ neighbors) * vectors.T).sum(1) / variance
        stats["mean"][name] = mean.float()
        stats["std"][name] = covariance.diagonal().sqrt().float()
        stats["components"][name] = vectors.T.half()
        stats["variance"][name] = variance.float()
        stats["neighbor_correlation"][name] = correlation.float()
        explained = variance.cumsum(0) / covariance.trace()
        print(
            f"{name}: |mean| {mean.norm():.1f}, sqrt(sum var) {covariance.trace().sqrt():.1f}, "
            f"explained by top 16/64/{args.components}: {explained[15]:.2f}/{explained[63]:.2f}/{explained[-1]:.2f}, "
            f"neighbor correlation of components 1/8/32/{args.components}: "
            + "/".join(f"{correlation[k]:.2f}" for k in (0, 7, 31, args.components - 1))
            + f", of all variance: {neighbors.trace() / covariance.trace():.2f}"
        )
    torch.save({**stats, "images": args.images}, OUT)


if __name__ == "__main__":
    main()
