# llm-img-reverse

Running Qwen3-VL-8B's vision stack backwards: given the soft tokens the vision tower produces for an
image, optimize a new image until its soft tokens match them, and look at what comes back.

## What gets inverted

For each 32×32 block of pixels, the vision tower of [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
produces four 4096-d vectors, which this repo calls streams:

| stream | where it comes from | where the LLM uses it |
|---|---|---|
| `main` | final merger after all 27 vision blocks | the input embeddings at the image positions |
| `ds8`, `ds16`, `ds24` | DeepStack mergers after blocks 8, 16 and 24 | added to the hidden state after LLM layers 0, 1 and 2 |

`llm_img_reverse/vision.py` loads only the vision tower (0.58B parameters, from shard 4 of the
checkpoint) in fp32. It reproduces the image processor's patch layout as a differentiable function,
which the tests check against `AutoImageProcessor`, and computes any subset of the streams, stopping
the tower at the last block those streams need.

`llm_img_reverse/invert.py` minimizes the relative squared error between the image's tokens and the
targets, summed over the chosen streams with weights. By default it runs L-BFGS over a 1/f-scaled
Fourier spectrum in decorrelated color space (the parameterization from Olah et al., *Feature
Visualization*, 2017). With plain pixels, or with Adam, the result is mostly edges on top of noise.

## Usage

```sh
uv venv && uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -e '.[dev]'
pytest tests

# Invert the test card from its main tokens only (downloads about 2.7 GB of weights on first use).
python -m llm_img_reverse.invert examples/testcard.png --streams main --out results/testcard_main

# Weighted mix of streams, non-square size, and save the target tokens for later.
python -m llm_img_reverse.invert examples/chelsea.png --size 256x320 --streams ds8,main=0.5 \
    --save-tokens chelsea.pt --out results/chelsea

# Invert tokens from a file, such as ones you edited.
python -m llm_img_reverse.invert chelsea.pt --streams all --out results/chelsea_edited
```

Each run writes `reconstruction.png`, `comparison.png` (original next to the reconstruction, for
image inputs) and `metrics.json` (PSNR, and the relative error and mean cosine similarity of every
stream, including those that were not optimized). `--param pixel`, `--optimizer adam`, `--tv` and
`--lr` switch to the baselines and regularizers.

`experiments/cpu_sweep.sh` reruns the CPU tests below, and `experiments/summarize.py` builds the
table and contact sheets from any directory of runs.

## CPU notes

On 4 cores, one forward and backward pass at 256×256 takes about 1.5 s through the whole tower and
about 0.55 s up to block 8, so 300 steps take 3 to 8 minutes. Each run peaks at about 3 GB of RAM;
four runs at once ran out of memory on a 15 GB machine.
