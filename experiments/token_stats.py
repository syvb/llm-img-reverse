"""Per-dimension mean and standard deviation of each stream's soft tokens over COCO photos, which the
Space uses to make random target tokens. Writes space/token_stats.pt (needs pyarrow).

    python experiments/token_stats.py --images 200

Each photo is encoded at 256x256 and 384x384, the Space's two sizes, and the statistics pool every
token of every photo at both sizes.
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
    args = parser.parse_args()

    model = load_vision_tower()
    table = pq.read_table(hf_hub_download(DATASET, SHARD, repo_type="dataset"), columns=["image"])
    total, squares, count = {}, {}, 0
    for i, row in enumerate(table.column("image").to_pylist()[: args.images]):
        for size in (256, 384):
            tokens = encode(model, load_image(io.BytesIO(row["bytes"]), size, size))
            for name, t in tokens.items():
                total[name] = total.get(name, 0) + t.double().sum(0)
                squares[name] = squares.get(name, 0) + t.double().square().sum(0)
            count += len(t)
        print(f"{i + 1}/{args.images}", flush=True)
    mean = {name: total[name] / count for name in total}
    std = {name: (squares[name] / count - mean[name].square()).sqrt() for name in total}
    torch.save(
        {"mean": {k: v.float() for k, v in mean.items()}, "std": {k: v.float() for k, v in std.items()}, "images": args.images},
        OUT,
    )
    for name in mean:
        print(f"{name}: |mean| {mean[name].norm():.1f}, sqrt(sum var) {std[name].norm():.1f}")


if __name__ == "__main__":
    main()
