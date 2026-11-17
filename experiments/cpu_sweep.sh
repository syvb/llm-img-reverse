#!/bin/bash
# CPU sweep at 256x256 (64 soft tokens per stream), 300 L-BFGS evaluations per run.
# Which streams carry what: the test card against each stream alone and all four together;
# the photos against the main stream alone and all four. Then summarize into results/cpu.
set -euo pipefail
cd "$(dirname "$0")/.."
out=results/cpu
run() {  # image streams
    local name=$(basename "$1" .png)_$2
    [ -f "$out/$name/metrics.json" ] && return
    echo "== $name"
    .venv/bin/python -m llm_img_reverse.invert "$1" --size 256 --streams "$2" --steps 300 --out "$out/$name" > "$out/$name.log" 2>&1
}
mkdir -p "$out"
for streams in main ds8 ds16 ds24 all; do run examples/testcard.png $streams; done
for image in astronaut chelsea coffee; do
    for streams in main all; do run examples/$image.png $streams; done
done
.venv/bin/python experiments/summarize.py "$out"
