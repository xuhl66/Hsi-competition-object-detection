from __future__ import annotations

import contextlib
import io
from typing import Any, Sequence

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.utils.ops import non_max_suppression

from .data import LetterboxMeta, restore_boxes


def postprocess_predictions(
    model_output,
    metadata: Sequence[LetterboxMeta],
    num_classes: int,
    confidence: float,
    nms_iou: float,
    max_detections: int,
) -> list[torch.Tensor]:
    prediction = model_output[0] if isinstance(model_output, tuple) else model_output
    detections = non_max_suppression(
        prediction,
        conf_thres=confidence,
        iou_thres=nms_iou,
        max_det=max_detections,
        nc=num_classes,
        multi_label=True,
    )
    restored: list[torch.Tensor] = []
    for result, image_metadata in zip(detections, metadata, strict=True):
        result = result.clone()
        if len(result):
            result[:, :4] = restore_boxes(result[:, :4], image_metadata)
            keep = (result[:, 2] > result[:, 0]) & (result[:, 3] > result[:, 1])
            result = result[keep]
        restored.append(result)
    return restored


def build_coco_ground_truth(
    image_ids: Sequence[str],
    boxes: Sequence[torch.Tensor],
    labels: Sequence[torch.Tensor],
    widths: Sequence[int],
    heights: Sequence[int],
    classes: Sequence[str],
) -> tuple[COCO, dict[str, int]]:
    mapping = {image_id: index + 1 for index, image_id in enumerate(image_ids)}
    dataset: dict[str, Any] = {
        "info": {"description": "HOD26 local validation"},
        "licenses": [],
        "images": [
            {
                "id": mapping[image_id],
                "file_name": f"{image_id}.png",
                "width": int(width),
                "height": int(height),
            }
            for image_id, width, height in zip(
                image_ids, widths, heights, strict=True
            )
        ],
        "categories": [
            {"id": index + 1, "name": name}
            for index, name in enumerate(classes)
        ],
        "annotations": [],
    }
    annotation_id = 1
    for image_id, image_boxes, image_labels in zip(
        image_ids, boxes, labels, strict=True
    ):
        for box, label in zip(image_boxes.tolist(), image_labels.tolist(), strict=True):
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                continue
            dataset["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": mapping[image_id],
                    "category_id": int(label) + 1,
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    return coco, mapping


def coco_metrics(
    ground_truth: COCO,
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    if predictions:
        with contextlib.redirect_stdout(io.StringIO()):
            detections = ground_truth.loadRes(predictions)
    else:
        detections = COCO()
        detections.dataset = {
            "images": ground_truth.dataset["images"],
            "categories": ground_truth.dataset["categories"],
            "annotations": [],
        }
        detections.createIndex()
    evaluator = COCOeval(ground_truth, detections, "bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats
    precision = evaluator.eval["precision"]
    per_class = {}
    for category_index, category in enumerate(ground_truth.dataset["categories"]):
        values = precision[:, :, category_index, 0, -1]
        valid = values[values > -1]
        per_class[category["name"]] = float(valid.mean()) if valid.size else 0.0
    return {
        "map_50_95": float(stats[0]),
        "map_50": float(stats[1]),
        "map_75": float(stats[2]),
        "map_small": float(stats[3]),
        "map_medium": float(stats[4]),
        "map_large": float(stats[5]),
        "ar_100": float(stats[8]),
        "per_class_map_50_95": per_class,
    }


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    loader,
    classes: Sequence[str],
    device: torch.device,
    confidence: float = 0.001,
    nms_iou: float = 0.65,
    max_detections: int = 300,
    amp: bool = True,
) -> dict[str, Any]:
    model.eval()
    all_image_ids: list[str] = []
    all_gt_boxes: list[torch.Tensor] = []
    all_gt_labels: list[torch.Tensor] = []
    all_widths: list[int] = []
    all_heights: list[int] = []
    raw_predictions: list[tuple[str, torch.Tensor]] = []

    for batch in loader:
        images = batch["img"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            output = model(images)
        detections = postprocess_predictions(
            output,
            batch["metadata"],
            num_classes=len(classes),
            confidence=confidence,
            nms_iou=nms_iou,
            max_detections=max_detections,
        )
        for index, image_id in enumerate(batch["image_id"]):
            all_image_ids.append(image_id)
            all_gt_boxes.append(batch["original_boxes"][index].cpu())
            all_gt_labels.append(batch["original_labels"][index].cpu())
            all_widths.append(batch["metadata"][index].original_width)
            all_heights.append(batch["metadata"][index].original_height)
            raw_predictions.append((image_id, detections[index].cpu()))

    ground_truth, mapping = build_coco_ground_truth(
        all_image_ids,
        all_gt_boxes,
        all_gt_labels,
        all_widths,
        all_heights,
        classes,
    )
    coco_predictions: list[dict[str, Any]] = []
    for image_id, detections in raw_predictions:
        for row in detections.tolist():
            x1, y1, x2, y2, score, class_id = row
            coco_predictions.append(
                {
                    "image_id": mapping[image_id],
                    "category_id": int(class_id) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
            )
    return coco_metrics(ground_truth, coco_predictions)

