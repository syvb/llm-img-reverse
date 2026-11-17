---
title: Qwen3-VL Soft-Token Inversion
emoji: 🔁
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.17.3
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Reconstruct images from Qwen3-VL vision soft tokens
---

Encodes an image with the vision tower of Qwen3-VL-8B-Instruct, then optimizes a new image from gray
until its soft tokens match: the final tokens the LLM reads, one of the three DeepStack streams, or
all four. Code, method and CPU results: https://github.com/syvb/llm-img-reverse

This folder is deployed with `python space/deploy.py <owner>/<space>`, which uploads it together
with the `llm_img_reverse` package and the example images.
