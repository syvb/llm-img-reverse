import pytest
import torch

from llm_img_reverse.invert import invert, relative_error
from llm_img_reverse.vision import encode


@pytest.mark.parametrize("optimizer,param", [("lbfgs", "fourier"), ("lbfgs", "pixel"), ("adam", "fourier"), ("adam", "pixel")])
def test_invert_moves_tokens_toward_targets(tiny_tower, optimizer, param):
    torch.manual_seed(1)
    original = torch.rand(3, 64, 64)
    with torch.no_grad():
        targets = encode(tiny_tower, original)
        start = relative_error(encode(tiny_tower, torch.full_like(original, 0.5))["ds1"], targets["ds1"]).item()

    image, _ = invert(tiny_tower, targets, original.shape, {"ds1": 1.0}, steps=40, optimizer=optimizer, param=param, log=lambda _: None)

    assert image.shape == original.shape and 0 <= image.min() and image.max() <= 1
    with torch.no_grad():
        final = relative_error(encode(tiny_tower, image)["ds1"], targets["ds1"]).item()
    assert final < 0.5 * start
