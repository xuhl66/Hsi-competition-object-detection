from __future__ import annotations

import torch

from hod26.model import HyperDetModel, update_ema


def test_model_inference_and_loss_backward() -> None:
    model = HyperDetModel(pretrained_rgb=None)
    model.eval()
    inputs = torch.randn(1, 16, 64, 128)
    with torch.inference_mode():
        predictions = model(inputs)
    decoded = predictions[0] if isinstance(predictions, tuple) else predictions
    assert decoded.shape == (1, 22, 680)

    model.train()
    batch = {
        "img": inputs,
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
        "batch_idx": torch.tensor([0.0]),
    }
    loss, items = model(batch)
    assert torch.isfinite(loss)
    assert items.shape == (3,)
    loss.backward()
    assert model.ssrm.group_logits.grad is not None
    assert torch.isfinite(model.ssrm.group_logits.grad).all()


def test_ema_uses_early_update_ramp() -> None:
    source = torch.nn.Linear(2, 1)
    ema = torch.nn.Linear(2, 1)
    with torch.no_grad():
        source.weight.fill_(2.0)
        source.bias.fill_(2.0)
        ema.weight.zero_()
        ema.bias.zero_()
    effective = update_ema(ema, source, decay=0.9999, updates=1, tau=2000)
    assert effective < 0.001
    torch.testing.assert_close(
        ema.weight, torch.full_like(ema.weight, 2.0), atol=0.002, rtol=0
    )
