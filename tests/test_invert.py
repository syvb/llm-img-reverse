import pytest
import torch

from llm_img_reverse.invert import invert, lbfgs, relative_error
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


def test_lbfgs_takes_the_steps_of_torch_lbfgs():
    # A small history, so that old pairs are dropped within the run.
    generator = torch.Generator().manual_seed(0)
    a, b = torch.randn(40, 30, generator=generator, dtype=torch.float64), torch.randn(40, generator=generator, dtype=torch.float64)
    runs = []
    for mine in (False, True):
        x, losses = torch.zeros(30, dtype=torch.float64, requires_grad=True), []

        def closure():
            x.grad = None
            loss = (a @ x - b).square().sum() + 0.1 * x.pow(4).sum() + torch.sin(3 * x).sum()
            loss.backward()
            losses.append(loss.item())
            return loss.detach()

        if mine:
            lbfgs(closure, x, 60, history_size=3)
        else:
            torch.optim.LBFGS(
                [x], max_iter=60, max_eval=60, history_size=3, line_search_fn="strong_wolfe", tolerance_grad=0, tolerance_change=0,
            ).step(closure)
        runs.append(losses)
    assert len(runs[1]) == len(runs[0]) == 60
    torch.testing.assert_close(runs[1], runs[0], rtol=1e-10, atol=0)
