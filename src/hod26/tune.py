from __future__ import annotations

import argparse
import copy
import gc
import itertools
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from .checkpoint import load_checkpoint_model
from .config import atomic_json_dump, load_config, resolve_path
from .evaluation import (
    build_coco_ground_truth,
    coco_metrics,
    postprocess_predictions,
)
from .runtime import (
    build_inference_loader,
    load_data_assets,
)
from .data import select_fold_samples


def _validation_ground_truth(
    samples: Sequence[dict[str, Any]],
    classes: Sequence[str],
):
    image_ids = [sample["image_id"] for sample in samples]
    boxes = [
        torch.tensor(
            [obj["bbox"] for obj in sample["objects"]], dtype=torch.float32
        ).reshape(-1, 4)
        for sample in samples
    ]
    labels = [
        torch.tensor(
            [obj["class_id"] for obj in sample["objects"]], dtype=torch.long
        )
        for sample in samples
    ]
    widths = [int(sample["width"]) for sample in samples]
    heights = [int(sample["height"]) for sample in samples]
    return build_coco_ground_truth(
        image_ids, boxes, labels, widths, heights, classes
    )


@torch.inference_mode()
def tune_inference(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    widths: Sequence[int],
    confidences: Sequence[float],
    nms_thresholds: Sequence[float],
    multi_label_modes: Sequence[bool],
    output_path: str | Path,
    device_name: str | None = None,
) -> dict[str, Any]:
    device = torch.device(
        device_name or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    assets = load_data_assets(config)
    _, validation_samples = select_fold_samples(
        assets["manifest"], assets["folds"], int(config["data"]["fold"])
    )
    ground_truth, image_mapping = _validation_ground_truth(
        validation_samples, assets["classes"]
    )
    model, checkpoint = load_checkpoint_model(
        config,
        assets["classes"],
        checkpoint_path,
        device,
        use_ema=True,
    )
    amp = bool(config["train"]["amp"]) and device.type == "cuda"
    max_detections = int(config["inference"]["max_detections"])
    settings = list(
        itertools.product(confidences, nms_thresholds, multi_label_modes)
    )
    results: list[dict[str, Any]] = []
    output_path = resolve_path(output_path)

    for width in widths:
        if width % 32:
            raise ValueError(f"Inference width must be divisible by 32: {width}")
        height = width // 2
        size_config = copy.deepcopy(config)
        size_config["data"]["image_width"] = int(width)
        size_config["data"]["image_height"] = int(height)
        loader = build_inference_loader(
            size_config, assets, list(validation_samples)
        )
        predictions: dict[
            tuple[float, float, bool], list[dict[str, Any]]
        ] = {setting: [] for setting in settings}
        started = time.time()

        for batch in loader:
            images = batch["img"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp):
                output = model(images)
            for confidence, nms_iou, multi_label in settings:
                detections = postprocess_predictions(
                    output,
                    batch["metadata"],
                    num_classes=len(assets["classes"]),
                    confidence=float(confidence),
                    nms_iou=float(nms_iou),
                    max_detections=max_detections,
                    multi_label=bool(multi_label),
                )
                destination = predictions[(confidence, nms_iou, multi_label)]
                for image_id, image_detections in zip(
                    batch["image_id"], detections, strict=True
                ):
                    coco_image_id = image_mapping[image_id]
                    for row in image_detections.cpu().tolist():
                        x1, y1, x2, y2, score, class_id = row
                        destination.append(
                            {
                                "image_id": coco_image_id,
                                "category_id": int(class_id) + 1,
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "score": score,
                            }
                        )

        inference_seconds = time.time() - started
        for confidence, nms_iou, multi_label in settings:
            metrics = coco_metrics(
                ground_truth,
                predictions[(confidence, nms_iou, multi_label)],
            )
            result = {
                "width": int(width),
                "height": int(height),
                "confidence": float(confidence),
                "nms_iou": float(nms_iou),
                "multi_label": bool(multi_label),
                "detections": len(
                    predictions[(confidence, nms_iou, multi_label)]
                ),
                "inference_seconds_for_size_grid": round(
                    inference_seconds, 3
                ),
                **metrics,
            }
            results.append(result)
            print(
                f"size={width} conf={confidence:g} nms={nms_iou:g} "
                f"multi={multi_label} mAP={metrics['map_50_95']:.6f} "
                f"AP50={metrics['map_50']:.6f} AP75={metrics['map_75']:.6f}"
            )

        ranked = sorted(
            results, key=lambda item: item["map_50_95"], reverse=True
        )
        report = {
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "fold": int(config["data"]["fold"]),
            "validation_images": len(validation_samples),
            "results": ranked,
            "best": ranked[0],
        }
        atomic_json_dump(report, output_path)
        del loader, predictions
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "fold": int(config["data"]["fold"]),
        "validation_images": len(validation_samples),
        "results": sorted(
            results, key=lambda item: item["map_50_95"], reverse=True
        ),
    }
    report["best"] = report["results"][0]
    atomic_json_dump(report, output_path)
    print(json.dumps(report["best"], ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune legal single-forward HOD26 inference parameters"
    )
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--widths", nargs="+", type=int, default=[704, 768, 832, 896]
    )
    parser.add_argument(
        "--confidences", nargs="+", type=float, default=[0.001]
    )
    parser.add_argument(
        "--nms", nargs="+", type=float, default=[0.5, 0.6, 0.65, 0.7, 0.75]
    )
    parser.add_argument(
        "--multi-label",
        choices=["true", "false", "both"],
        default="both",
    )
    parser.add_argument(
        "--output", default="storage/runs/inference_tuning.json"
    )
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    modes = {
        "true": [True],
        "false": [False],
        "both": [True, False],
    }[args.multi_label]
    tune_inference(
        load_config(args.config),
        args.checkpoint,
        args.widths,
        args.confidences,
        args.nms,
        modes,
        args.output,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
