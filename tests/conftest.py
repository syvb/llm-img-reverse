import pytest
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionConfig, Qwen3VLVisionModel


@pytest.fixture
def tiny_tower():
    """A small randomly initialized vision tower with the real layout (2x2 merge, 16px patches)."""
    config = Qwen3VLVisionConfig(
        depth=5, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=96,
        num_position_embeddings=64, deepstack_visual_indexes=[1, 3],
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return Qwen3VLVisionModel(config).eval().requires_grad_(False)
