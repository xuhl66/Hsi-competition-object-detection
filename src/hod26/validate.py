from __future__ import annotations

import argparse
import json

import torch

from .checkpoint import load_checkpoint_model
from .config import load_config
from .evaluation import evaluate_model
from .runtime import build_train_loaders, load_data_assets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a HOD26 checkpoint")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--raw-weights",
        action="store_true",
        help="Evaluate the instantaneous weights instead of EMA",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    assets = load_data_assets(config)
    _, validation_loader, _ = build_train_loaders(config, assets)
    if not len(validation_loader.dataset):
        raise ValueError("The configured fold has no validation images")
    model, checkpoint = load_checkpoint_model(
        config,
        assets["classes"],
        args.checkpoint,
        device,
        use_ema=not args.raw_weights,
    )
    metrics = evaluate_model(
        model,
        validation_loader,
        assets["classes"],
        device,
        confidence=float(config["inference"]["confidence"]),
        nms_iou=float(config["inference"]["nms_iou"]),
        max_detections=int(config["inference"]["max_detections"]),
        multi_label=bool(config["inference"].get("multi_label", True)),
        amp=bool(config["train"]["amp"]),
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "weights": (
            "ema"
            if not args.raw_weights and checkpoint.get("ema") is not None
            else "model"
        ),
        **metrics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
