from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_checkpoint_model
from .config import load_config, resolve_path
from .evaluation import postprocess_predictions
from .runtime import build_inference_loader, load_data_assets
from .submission import validate_submission, write_submission


@torch.inference_mode()
def infer(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    device_name: str | None = None,
    raw_weights: bool = False,
) -> tuple[Path, dict[str, Any]]:
    device = torch.device(
        device_name or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    assets = load_data_assets(config)
    samples = assets["test_manifest"]["samples"]
    loader = build_inference_loader(config, assets, samples)
    model, checkpoint = load_checkpoint_model(
        config,
        assets["classes"],
        checkpoint_path,
        device,
        use_ema=not raw_weights,
    )

    rows: list[dict[str, Any]] = []
    amp = bool(config["train"]["amp"]) and device.type == "cuda"
    for batch in loader:
        images = batch["img"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            output = model(images)
        detections = postprocess_predictions(
            output,
            batch["metadata"],
            num_classes=len(assets["classes"]),
            confidence=float(config["inference"]["confidence"]),
            nms_iou=float(config["inference"]["nms_iou"]),
            max_detections=int(config["inference"]["max_detections"]),
            multi_label=bool(config["inference"].get("multi_label", True)),
        )
        for image_id, image_detections in zip(
            batch["image_id"], detections, strict=True
        ):
            for x1, y1, x2, y2, confidence, class_id in image_detections.cpu().tolist():
                rows.append(
                    {
                        "image_id": image_id,
                        "class_id": int(class_id),
                        "confidence": confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )

    rows.sort(key=lambda row: (int(row["image_id"]), -float(row["confidence"])))
    destination = resolve_path(
        output_path or config["inference"]["output"]
    )
    write_submission(
        rows,
        destination,
        include_id=bool(config["inference"]["include_id"]),
    )
    report = validate_submission(
        destination,
        samples,
        num_classes=len(assets["classes"]),
        include_id=bool(config["inference"]["include_id"]),
    )
    report.update(
        {
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "weights": (
                "model"
                if raw_weights or checkpoint.get("ema") is None
                else "ema"
            ),
        }
    )
    if not report["valid"]:
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
    return destination, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer the official HOD26 test set")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--raw-weights", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, report = infer(
        load_config(args.config),
        args.checkpoint,
        output_path=args.output,
        device_name=args.device,
        raw_weights=args.raw_weights,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
