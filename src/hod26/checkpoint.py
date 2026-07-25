from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Sequence

import torch

from .model import HyperDetModel, build_model


def load_checkpoint_model(
    config: dict[str, Any],
    classes: Sequence[str],
    checkpoint_path: str | Path,
    device: torch.device,
    use_ema: bool = True,
) -> tuple[HyperDetModel, dict[str, Any]]:
    """Recreate the detector and load an exact full-model training checkpoint."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise ValueError(f"Not a HOD26 training checkpoint: {checkpoint_path}")

    # A complete checkpoint already contains every tensor. Avoid downloading and
    # transiently loading the declared COCO initializer during evaluation.
    model_config = copy.deepcopy(config)
    model_config["model"]["pretrained_rgb"] = None
    model = build_model(model_config, classes)
    state_key = "ema" if use_ema and checkpoint.get("ema") is not None else "model"
    incompatible = model.load_state_dict(checkpoint[state_key], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.to(device).eval()
    return model, checkpoint

