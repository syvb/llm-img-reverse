"""Qwen3-VL vision tower: weight loading, differentiable preprocessing, and encoding."""

import json

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
PREFIX = "model.visual."
# Qwen3-VL normalizes pixels with mean 0.5 and std 0.5 (preprocessor_config.json).
MEAN, STD = 0.5, 0.5


def load_vision_tower(model_id=MODEL_ID, dtype=torch.float32, device="cpu"):
    """Load only the vision tower (about 0.6B parameters) from the model's safetensors shards."""
    config = AutoConfig.from_pretrained(model_id).vision_config
    config._attn_implementation = "sdpa"
    with open(hf_hub_download(model_id, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    state = {}
    for shard in sorted({v for k, v in weight_map.items() if k.startswith(PREFIX)}):
        with safe_open(hf_hub_download(model_id, shard), "pt") as f:
            for key in f.keys():
                if key.startswith(PREFIX):
                    state[key[len(PREFIX) :]] = f.get_tensor(key).to(dtype)
    model = Qwen3VLVisionModel(config).to(dtype)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def stream_names(model):
    return ["main"] + [f"ds{i}" for i in model.config.deepstack_visual_indexes]


def to_pixel_values(image, config):
    """Turn a [3, H, W] image in [0, 1] into the (pixel_values, grid_thw) the processor produces.

    Same layout as Qwen2VLImageProcessorFast with do_resize=False, but differentiable.
    H and W must be multiples of patch_size * spatial_merge_size (32).
    """
    p, t, m = config.patch_size, config.temporal_patch_size, config.spatial_merge_size
    c, h, w = image.shape
    gh, gw = h // p, w // p
    if gh * p != h or gw * p != w or gh % m or gw % m:
        raise ValueError(f"image size {h}x{w} must be a multiple of {p * m}")
    x = ((image - MEAN) / STD).unsqueeze(0).expand(t, c, h, w)  # a still image fills both frames
    x = x.reshape(t, c, gh // m, m, p, gw // m, m, p)
    x = x.permute(2, 5, 3, 6, 1, 0, 4, 7)  # merge blocks, then 2x2 patches, then C, T, P, P
    return x.reshape(gh * gw, c * t * p * p), torch.tensor([[1, gh, gw]], device=image.device)


def encode(model, image, streams=None):
    """Soft tokens for an image: {"main": [N, 4096], "ds8": ..., "ds16": ..., "ds24": ...}.

    Tokens are ordered row-major over the (H/32) x (W/32) grid of merged patches. Pass `streams`
    to compute only those; the tower then stops after the last block they need. This follows
    Qwen3VLVisionModel.forward for a single image.
    """
    names = stream_names(model)
    streams = names if streams is None else streams
    ds_layers = {f"ds{i}": i for i in model.config.deepstack_visual_indexes}
    last_block = max(len(model.blocks) - 1 if s == "main" else ds_layers[s] for s in streams)

    pixel_values, grid_thw = to_pixel_values(image, model.config)
    hidden = model.patch_embed(pixel_values) + model.fast_pos_embed_interpolate(grid_thw)
    rotary = model.rot_pos_emb(grid_thw).reshape(hidden.shape[0], -1)
    emb = torch.cat((rotary, rotary), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    cu_seqlens = torch.tensor([0, hidden.shape[0]], dtype=torch.int32, device=hidden.device)

    out = {}
    for layer, block in enumerate(model.blocks[: last_block + 1]):
        hidden = block(hidden, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
        name = f"ds{layer}"
        if name in streams:
            out[name] = model.deepstack_merger_list[model.config.deepstack_visual_indexes.index(layer)](hidden)
    if "main" in streams:
        out["main"] = model.merger(hidden)
    return {name: out[name] for name in names if name in out}


def load_image(path, height, width):
    """Load an image, scale it to cover height x width, and center-crop. Returns [3, H, W] in [0, 1]."""
    img = Image.open(path).convert("RGB")
    scale = max(height / img.height, width / img.width)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BICUBIC)
    left, top = (img.width - width) // 2, (img.height - height) // 2
    img = img.crop((left, top, left + width, top + height))
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255


def save_image(image, path):
    array = (image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)
