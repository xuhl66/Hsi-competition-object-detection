#!/usr/bin/env python3
"""Produce reproducible dataset and experiment evidence for the V6 decision.

The script is intentionally read-only.  It uses only the checked COCO views and
MMDetection text logs, and prints a JSON document to stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
VAL_LINE = re.compile(r"Epoch\(val\) \[(?P<epoch>\d+)\]\[\d+\]\s+(?P<body>.*)$")
METRIC = re.compile(r"(?P<name>[A-Za-z0-9_.-]+): (?P<value>-?\d+(?:\.\d+)?)")
WEAK_CLASSES = frozenset(("car", "e-bike", "people", "stone_block"))
VIS3_WAVELENGTHS_NM = (
    464.5,
    472.8,
    480.2,
    489.3,
    499.0,
    508.2,
    516.3,
    526.1,
    534.7,
    544.3,
    552.3,
    561.8,
    571.2,
    580.5,
    588.1,
    597.2,
)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def rounded(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def dataset_stats(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    categories = {int(row["id"]): str(row["name"]) for row in payload["categories"]}
    image_shapes = {
        int(row["id"]): (float(row["width"]), float(row["height"]))
        for row in payload["images"]
    }
    per_class: dict[int, dict[str, Any]] = {
        category_id: {
            "name": name,
            "images": set(),
            "widths": [],
            "heights": [],
            "areas": [],
            "area_fractions": [],
            "small": 0,
            "medium": 0,
            "large": 0,
        }
        for category_id, name in categories.items()
    }
    objects_per_image: defaultdict[int, int] = defaultdict(int)

    for ann in payload["annotations"]:
        category_id = int(ann["category_id"])
        image_id = int(ann["image_id"])
        width, height = (float(x) for x in ann["bbox"][2:4])
        area = float(ann.get("area", width * height))
        image_width, image_height = image_shapes[image_id]
        row = per_class[category_id]
        row["images"].add(image_id)
        row["widths"].append(width)
        row["heights"].append(height)
        row["areas"].append(area)
        row["area_fractions"].append(area / (image_width * image_height))
        objects_per_image[image_id] += 1
        if area < 32**2:
            row["small"] += 1
        elif area < 96**2:
            row["medium"] += 1
        else:
            row["large"] += 1

    class_rows: list[dict[str, Any]] = []
    all_areas: list[float] = []
    size_counts = {"small": 0, "medium": 0, "large": 0}
    for category_id in sorted(per_class):
        row = per_class[category_id]
        all_areas.extend(row["areas"])
        for size in size_counts:
            size_counts[size] += int(row[size])
        instances = len(row["areas"])
        class_rows.append(
            {
                "id": category_id,
                "name": row["name"],
                "instances": instances,
                "images": len(row["images"]),
                "small_fraction": rounded(row["small"] / instances),
                "medium_fraction": rounded(row["medium"] / instances),
                "large_fraction": rounded(row["large"] / instances),
                "bbox_width_p10_p50_p90": [
                    rounded(percentile(row["widths"], q), 3) for q in (0.1, 0.5, 0.9)
                ],
                "bbox_height_p10_p50_p90": [
                    rounded(percentile(row["heights"], q), 3) for q in (0.1, 0.5, 0.9)
                ],
                "bbox_area_p10_p50_p90": [
                    rounded(percentile(row["areas"], q), 3) for q in (0.1, 0.5, 0.9)
                ],
                "area_fraction_median": rounded(statistics.median(row["area_fractions"]), 8),
            }
        )

    object_counts = list(objects_per_image.values())
    return {
        "path": display_path(path),
        "images": len(payload["images"]),
        "annotations": len(payload["annotations"]),
        "images_with_annotations": len(objects_per_image),
        "objects_per_annotated_image_p10_p50_p90": [
            rounded(percentile(object_counts, q), 3) for q in (0.1, 0.5, 0.9)
        ],
        "bbox_area_p10_p50_p90": [
            rounded(percentile(all_areas, q), 3) for q in (0.1, 0.5, 0.9)
        ],
        "size_distribution": {
            size: {
                "instances": count,
                "fraction": rounded(count / len(all_areas)),
            }
            for size, count in size_counts.items()
        },
        "classes": class_rows,
    }


def parse_validations(paths: Iterable[Path]) -> dict[int, dict[str, float]]:
    validations: dict[int, dict[str, float]] = {}
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = VAL_LINE.search(line)
                if match is None:
                    continue
                epoch = int(match.group("epoch"))
                values = {
                    item.group("name"): float(item.group("value"))
                    for item in METRIC.finditer(match.group("body"))
                }
                if "bbox_mAP" in values:
                    validations[epoch] = values
    return validations


def parse_training_windows(path: Path, epochs: Iterable[int]) -> dict[int, dict[str, float]]:
    requested = set(int(epoch) for epoch in epochs)
    fields = (
        "loss_cls",
        "loss_bbox",
        "loss_iou",
        "fdr_main_matched_iou",
        "fdr_pre_matched_iou",
        "fdr_refinement_l1",
        "loss_salience",
        "salience_foreground_response",
        "salience_background_response",
        "grad_norm",
    )
    values: defaultdict[int, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            epoch = int(row.get("epoch", -1))
            if row.get("mode") != "train" or epoch not in requested:
                continue
            for field in fields:
                value = row.get(field)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    values[epoch][field].append(float(value))
    return {
        epoch: {
            field: rounded(statistics.fmean(samples), 8)
            for field, samples in sorted(per_field.items())
            if samples
        }
        for epoch, per_field in sorted(values.items())
    }


def select_metrics(metrics: dict[str, float]) -> dict[str, float]:
    names = (
        "bbox_mAP",
        "bbox_mAP_50",
        "bbox_mAP_75",
        "bbox_mAP_s",
        "bbox_mAP_m",
        "bbox_mAP_l",
        "bbox_weak4_mAP",
        "bbox_weak4_mAP_75",
    )
    selected = {name: metrics[name] for name in names if name in metrics}
    selected["per_class"] = {
        name.removeprefix("bbox_AP_"): value
        for name, value in metrics.items()
        if name.startswith("bbox_AP_")
    }
    weak = selected["per_class"]
    selected["derived"] = {
        "weak4_mean": rounded(statistics.fmean(weak[name] for name in WEAK_CLASSES)),
        "strong14_mean": rounded(
            statistics.fmean(
                value for name, value in weak.items() if name not in WEAK_CLASSES
            )
        ),
    }
    selected["derived"]["strong14_minus_weak4"] = rounded(
        selected["derived"]["strong14_mean"] - selected["derived"]["weak4_mean"]
    )
    return selected


def checkpoint_delta(
    parent: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    scalar_names = (
        "bbox_mAP",
        "bbox_mAP_50",
        "bbox_mAP_75",
        "bbox_mAP_s",
        "bbox_mAP_m",
        "bbox_mAP_l",
    )
    return {
        "metrics": {
            name: rounded(candidate[name] - parent[name])
            for name in scalar_names
            if name in parent and name in candidate
        },
        "derived": {
            name: rounded(candidate["derived"][name] - parent["derived"][name])
            for name in parent["derived"]
        },
        "per_class": {
            name: rounded(candidate["per_class"][name] - value)
            for name, value in parent["per_class"].items()
        },
    }


def split_drift(train: dict[str, Any], val: dict[str, Any]) -> list[dict[str, Any]]:
    train_by_name = {row["name"]: row for row in train["classes"]}
    val_by_name = {row["name"]: row for row in val["classes"]}
    train_total = train["annotations"]
    val_total = val["annotations"]
    rows = []
    for name in train_by_name:
        train_row = train_by_name[name]
        val_row = val_by_name[name]
        train_share = train_row["instances"] / train_total
        val_share = val_row["instances"] / val_total
        rows.append(
            {
                "name": name,
                "train_instances": train_row["instances"],
                "val_instances": val_row["instances"],
                "train_images": train_row["images"],
                "val_images": val_row["images"],
                "val_to_train_share_ratio": rounded(val_share / train_share),
                "train_val_median_area_ratio": rounded(
                    train_row["bbox_area_p10_p50_p90"][1]
                    / val_row["bbox_area_p10_p50_p90"][1]
                ),
            }
        )
    return rows


def parameter_group(name: str) -> str:
    raw = name.removeprefix("ema_")
    if raw.startswith(("backbone.proxy_", "backbone_proxy_")):
        return "spectral_proxy"
    if raw.startswith(("backbone.spectral_", "backbone_spectral_")):
        return "spectral_encoder"
    if raw.startswith(("backbone.", "backbone_")):
        return "visual_backbone"
    if raw.startswith(("neck.spectral_fusions.", "neck_spectral_fusions_")):
        return "spectral_fusion"
    if raw.startswith(("neck.object_refinements.", "neck_object_refinements_")):
        return "object_refinement"
    if raw.startswith(("neck.salience_high.", "neck_salience_high_")):
        return "multiscale_salience"
    if raw.startswith(("query_head.fdr_branches.", "query_head_fdr_branches_")):
        return "fdr"
    if raw.startswith(("query_head.lqe_layers.", "query_head_lqe_layers_")):
        return "lqe"
    if raw.startswith(("query_head.pre_bbox_head.", "query_head_pre_bbox_head_")):
        return "pre_bbox"
    return "base_detector"


def checkpoint_change(initial_path: Path, trained_path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - diagnostic host has torch
        raise RuntimeError("checkpoint comparison requires torch") from error

    # torch 2.1 accepts mmap only when the filename is a string rather than a
    # pathlib object.
    initial_payload = torch.load(str(initial_path), map_location="cpu", mmap=True)
    trained_payload = torch.load(str(trained_path), map_location="cpu", mmap=True)
    initial = initial_payload["state_dict"]
    trained = trained_payload["state_dict"]
    if initial.keys() != trained.keys():
        missing = sorted(initial.keys() - trained.keys())
        unexpected = sorted(trained.keys() - initial.keys())
        raise RuntimeError(
            f"checkpoint schemas differ: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    aggregate: defaultdict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"tensors": 0, "elements": 0, "initial_sq": 0.0, "delta_sq": 0.0}
    )
    gates: dict[str, dict[str, float]] = {}
    for name, before in initial.items():
        after = trained[name]
        if not getattr(before, "is_floating_point", lambda: False)():
            continue
        coordinate = "ema" if name.startswith("ema_") else "raw"
        group = parameter_group(name)
        row = aggregate[(coordinate, group)]
        row["tensors"] += 1
        row["elements"] += before.numel()
        before_float = before.float()
        after_float = after.float()
        row["initial_sq"] += float(torch.sum(before_float * before_float).item())
        delta = after_float - before_float
        row["delta_sq"] += float(torch.sum(delta * delta).item())
        if name.endswith("spectral_fusions.0.gate") or (
            "spectral_fusions." in name and name.endswith(".gate")
        ):
            effective = torch.tanh(after_float)
            gates[name] = {
                "mean_abs": rounded(effective.abs().mean().item(), 8),
                "max_abs": rounded(effective.abs().max().item(), 8),
                "positive_fraction": rounded((effective > 0).float().mean().item(), 8),
            }

    groups: dict[str, dict[str, Any]] = {"raw": {}, "ema": {}}
    for (coordinate, group), values in sorted(aggregate.items()):
        initial_norm = math.sqrt(values.pop("initial_sq"))
        delta_norm = math.sqrt(values.pop("delta_sq"))
        groups[coordinate][group] = {
            "tensors": int(values["tensors"]),
            "elements": int(values["elements"]),
            "initial_l2": rounded(initial_norm),
            "delta_l2": rounded(delta_norm),
            "relative_l2_change": rounded(delta_norm / max(initial_norm, 1e-12), 8),
        }

    return {
        "initial": display_path(initial_path),
        "trained": display_path(trained_path),
        "groups": groups,
        "effective_spectral_fusion_gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-json", type=Path, default=ROOT / "storage/v2/dataset/fold0/train.json"
    )
    parser.add_argument(
        "--val-json", type=Path, default=ROOT / "storage/v2/dataset/fold0/val.json"
    )
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--trained-checkpoint", type=Path)
    args = parser.parse_args()

    train = dataset_stats(args.train_json)
    val = dataset_stats(args.val_json)
    v4 = parse_validations(
        (ROOT / "storage/v4/runs/co_dino_vitl_hsi_fold0").glob("*.log")
    )
    v5 = parse_validations(
        (ROOT / "storage/v5/runs/co_dino_vitl_fdr_salience_fold0").glob("*.log")
    )
    v5r = parse_validations(
        [ROOT / "storage/v5r/launch_logs/v5r_20260806_222924.console.log"]
    )

    checkpoint_metrics = {
        "v4_epoch30": select_metrics(v4[30]),
        "faulty_v5_epoch18": select_metrics(v5[18]),
        "v5r_epoch40": select_metrics(v5r[40]),
        "v5r_epoch80": select_metrics(v5r[80]),
    }
    report = {
        "sensor": {
            "camera": "XIMEA MQ022HG-IM-SM4X4-VIS3",
            "decoded_band_order": "row-major mosaic indices 0..15",
            "central_wavelengths_nm": VIS3_WAVELENGTHS_NM,
            "source": (
                "XIMEA xiSpec2 TechnicalManual V2.00, section 4.1.2, "
                "table 4-2 and figures 4-4/4-5"
            ),
        },
        "dataset": {
            "train": train,
            "validation": val,
            "split_drift": split_drift(train, val),
        },
        "checkpoints": checkpoint_metrics,
        "checkpoint_deltas_from_v4": {
            name: checkpoint_delta(checkpoint_metrics["v4_epoch30"], metrics)
            for name, metrics in checkpoint_metrics.items()
            if name != "v4_epoch30"
        },
        "v5r_training_windows": parse_training_windows(
            ROOT / "storage/v5r/runs/co_dino_vitl_fdr_prehead_fold0/20260806_223135.log.json",
            (2, 10, 20, 30, 40, 50, 60, 70, 80),
        ),
        "public_board": {
            "v4_epoch30_frozen_protocol": 0.67686,
            "faulty_v5_epoch18_nonfrozen_submission": 0.67112,
            "v5r_epoch40_frozen_protocol": 0.66977,
            "v5r_epoch40_minus_v4": rounded(0.66977 - 0.67686),
        },
    }
    if (args.initial_checkpoint is None) != (args.trained_checkpoint is None):
        parser.error("--initial-checkpoint and --trained-checkpoint must be provided together")
    if args.initial_checkpoint is not None and args.trained_checkpoint is not None:
        report["checkpoint_change"] = checkpoint_change(
            args.initial_checkpoint, args.trained_checkpoint
        )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
