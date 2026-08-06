"""Systematic single-model V4 soup, TTA and box-merge optimization.

The expensive detector forward pass is cached once per geometric view.
Everything after that (view subset selection and NMS/WBF/box-voting search)
is CPU-only and can be reproduced without touching the model or test labels.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import gc
import hashlib
import io
import itertools
import json
import math
import multiprocessing
import os
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from hod26.config import atomic_json_dump
from hod26.submission import validate_submission, write_submission

from .constants import HOD26_CLASSES, WEAK4_IDS
from .data import load_band_limits
from .infer import (
    _flip_cube,
    _invert_boxes,
    _load_cube,
    _resize_pad,
    build_model,
)
from .soup import _primary_ema_state, average_loaded_states


DEFAULT_WIDTHS = (1024, 1152, 1280, 1408)
DEFAULT_FLIPS = ("none", "horizontal", "vertical", "both")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, order=True)
class ViewSpec:
    width: int
    flip: str

    @property
    def key(self) -> str:
        return f"w{self.width}_{self.flip}"

    @classmethod
    def parse(cls, value: str) -> "ViewSpec":
        width, separator, flip = value.partition(":")
        if not separator:
            raise ValueError(f"View must use WIDTH:FLIP syntax: {value!r}")
        spec = cls(int(width), flip)
        if spec.width <= 0 or spec.width % 32:
            raise ValueError(f"Invalid view width: {spec.width}")
        if spec.flip not in DEFAULT_FLIPS:
            raise ValueError(f"Invalid view flip: {spec.flip}")
        return spec


ALL_VIEWS = tuple(
    ViewSpec(width, flip) for width in DEFAULT_WIDTHS for flip in DEFAULT_FLIPS
)

_SEARCH_CACHES: dict[str, "CachedView"] | None = None
_SEARCH_GROUND_TRUTH: COCO | None = None


def _positive_env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _load_samples(
    split: str,
    *,
    test_manifest: Path,
    val_annotations: Path,
) -> list[dict[str, Any]]:
    if split == "test":
        payload = json.loads(test_manifest.read_text(encoding="utf-8"))
        samples = [dict(sample) for sample in payload["samples"]]
    elif split == "val":
        payload = json.loads(val_annotations.read_text(encoding="utf-8"))
        samples = [
            {
                "image_id": str(image["id"]),
                "image": f"data_train/data_train/VIS/{image['file_name']}",
                "width": int(image["width"]),
                "height": int(image["height"]),
            }
            for image in payload["images"]
        ]
    else:
        raise ValueError(f"Unsupported split: {split}")
    return sorted(samples, key=lambda sample: int(sample["image_id"]))


def _cache_path(output_dir: Path, view: ViewSpec) -> Path:
    return output_dir / f"{view.key}.npz"


def _cache_matches(
    path: Path,
    *,
    checkpoint_sha256: str,
    split: str,
    view: ViewSpec,
    expected_images: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            return (
                metadata["checkpoint_sha256"] == checkpoint_sha256
                and metadata["split"] == split
                and metadata["view"] == view.key
                and int(metadata["images"]) == expected_images
                and len(payload["offsets"]) == expected_images + 1
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@torch.inference_mode()
def cache_views(
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    split: str,
    views: Sequence[ViewSpec],
    output_dir: str | Path,
    config_path: str | Path,
    data_root: str | Path,
    stats_path: str | Path,
    upstream_path: str | Path,
    test_manifest: str | Path,
    val_annotations: str | Path,
    device_name: str,
    pre_score: float = 0.0001,
    limit: int | None = None,
) -> list[Path]:
    """Run one checkpoint once for each requested view and cache detections."""

    if not views:
        raise ValueError("At least one TTA view is required")
    destination = Path(output_dir).resolve()
    samples = _load_samples(
        split,
        test_manifest=Path(test_manifest),
        val_annotations=Path(val_annotations),
    )
    if limit is not None:
        samples = samples[:limit]
    outputs = [_cache_path(destination, view) for view in views]
    pending = [
        (view, path)
        for view, path in zip(views, outputs)
        if not _cache_matches(
            path,
            checkpoint_sha256=checkpoint_sha256,
            split=split,
            view=view,
            expected_images=len(samples),
        )
    ]
    if not pending:
        print(
            f"[V4 cache] reusing {len(outputs)} complete {split} view caches",
            flush=True,
        )
        return outputs

    cache_threads = min(
        _positive_env_int("V4_CACHE_CPU_THREADS", 8),
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1),
    )
    torch.set_num_threads(cache_threads)
    try:
        import cv2

        cv2.setNumThreads(_positive_env_int("V4_CACHE_OPENCV_THREADS", 4))
    except ImportError:
        pass
    print(
        f"[V4 cache] CPU threads={cache_threads}, affinity="
        f"{len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 'unknown'}",
        flush=True,
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    low, high = load_band_limits(stats_path)
    model, _, checkpoint_meta = build_model(
        upstream=Path(upstream_path),
        config_path=Path(config_path),
        checkpoint_path=Path(checkpoint_path),
        device=device,
    )
    active_views = [item[0] for item in pending]
    accumulators: dict[str, dict[str, list[Any]]] = {
        view.key: {
            "offsets": [0],
            "boxes": [],
            "scores": [],
            "class_ids": [],
        }
        for view in active_views
    }
    root = Path(data_root)
    started = time.perf_counter()

    for image_index, sample in enumerate(samples):
        image_path = root / sample["image"]
        cube = _load_cube(image_path, low, high)
        original_height, original_width = cube.shape[:2]
        if (original_width, original_height) != (
            int(sample["width"]),
            int(sample["height"]),
        ):
            raise ValueError(f"Manifest/image size mismatch: {sample['image_id']}")

        for view in active_views:
            flipped = _flip_cube(cube, view.flip)
            canvas, sx, sy, resized_width, resized_height = _resize_pad(
                flipped, view.width
            )
            tensor = (
                torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1)))
                .unsqueeze(0)
                .to(device, non_blocking=True)
            )
            meta = {
                "filename": str(image_path.resolve()),
                "ori_filename": image_path.name,
                "ori_shape": (original_height, original_width, 16),
                "img_shape": (resized_height, resized_width, 16),
                "pad_shape": canvas.shape,
                "scale_factor": np.asarray((sx, sy, sx, sy), dtype=np.float32),
                "flip": False,
                "flip_direction": None,
                "img_norm_cfg": {
                    "mean": np.zeros(16, dtype=np.float32),
                    "std": np.ones(16, dtype=np.float32),
                    "to_rgb": False,
                },
            }
            result = model(
                return_loss=False,
                rescale=False,
                img=[tensor],
                img_metas=[[meta]],
            )[0]
            image_boxes: list[np.ndarray] = []
            image_scores: list[np.ndarray] = []
            image_classes: list[np.ndarray] = []
            for class_id, detections in enumerate(result):
                detections = np.asarray(detections, dtype=np.float32)
                if not len(detections):
                    continue
                detections = detections[detections[:, 4] >= pre_score].copy()
                if not len(detections):
                    continue
                detections[:, :4] = _invert_boxes(
                    detections[:, :4],
                    sx=sx,
                    sy=sy,
                    original_width=original_width,
                    original_height=original_height,
                    flip=view.flip,
                )
                valid = (
                    np.isfinite(detections).all(axis=1)
                    & (detections[:, 2] > detections[:, 0])
                    & (detections[:, 3] > detections[:, 1])
                )
                detections = detections[valid]
                if not len(detections):
                    continue
                image_boxes.append(detections[:, :4])
                image_scores.append(detections[:, 4])
                image_classes.append(
                    np.full(len(detections), class_id, dtype=np.int16)
                )

            accumulator = accumulators[view.key]
            if image_boxes:
                boxes = np.concatenate(image_boxes).astype(np.float32, copy=False)
                scores = np.concatenate(image_scores).astype(np.float32, copy=False)
                class_ids = np.concatenate(image_classes)
                accumulator["boxes"].append(boxes)
                accumulator["scores"].append(scores)
                accumulator["class_ids"].append(class_ids)
                count = len(scores)
            else:
                count = 0
            accumulator["offsets"].append(accumulator["offsets"][-1] + count)
            del tensor, result

        if (image_index + 1) % 25 == 0 or image_index + 1 == len(samples):
            elapsed = time.perf_counter() - started
            print(
                f"[V4 cache:{split}] {image_index + 1}/{len(samples)} images, "
                f"{len(active_views)} views, "
                f"{elapsed / (image_index + 1):.2f}s/image",
                flush=True,
            )

    image_ids = np.asarray([sample["image_id"] for sample in samples])
    widths = np.asarray([sample["width"] for sample in samples], dtype=np.int32)
    heights = np.asarray([sample["height"] for sample in samples], dtype=np.int32)
    for view, path in pending:
        accumulator = accumulators[view.key]
        boxes = (
            np.concatenate(accumulator["boxes"])
            if accumulator["boxes"]
            else np.zeros((0, 4), dtype=np.float32)
        )
        scores = (
            np.concatenate(accumulator["scores"])
            if accumulator["scores"]
            else np.zeros(0, dtype=np.float32)
        )
        class_ids = (
            np.concatenate(accumulator["class_ids"])
            if accumulator["class_ids"]
            else np.zeros(0, dtype=np.int16)
        )
        metadata = {
            "schema": 1,
            "split": split,
            "view": view.key,
            "width": view.width,
            "flip": view.flip,
            "images": len(samples),
            "detections": int(len(scores)),
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_epoch": checkpoint_meta.get("epoch"),
            "checkpoint_iter": checkpoint_meta.get("iter"),
            "stats": str(Path(stats_path).resolve()),
            "stats_sha256": _sha256(stats_path),
            "pre_score": pre_score,
            "elapsed_seconds": time.perf_counter() - started,
            "cuda_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
        }
        _atomic_savez(
            path,
            {
                "image_ids": image_ids,
                "widths": widths,
                "heights": heights,
                "offsets": np.asarray(accumulator["offsets"], dtype=np.int64),
                "boxes": boxes,
                "scores": scores,
                "class_ids": class_ids,
                "metadata_json": np.asarray(
                    json.dumps(metadata, ensure_ascii=False)
                ),
            },
        )
        atomic_json_dump(metadata, path.with_suffix(".audit.json"))
        print(f"[V4 cache] wrote {path}", flush=True)
    return outputs


class CachedView:
    """In-memory index over one immutable per-view prediction cache."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as payload:
            self.image_ids = payload["image_ids"].astype(str)
            self.widths = payload["widths"].astype(np.int32)
            self.heights = payload["heights"].astype(np.int32)
            self.offsets = payload["offsets"].astype(np.int64)
            self.boxes = payload["boxes"].astype(np.float32)
            self.scores = payload["scores"].astype(np.float32)
            self.class_ids = payload["class_ids"].astype(np.int16)
            self.metadata = json.loads(str(payload["metadata_json"].item()))
        self.key = str(self.metadata["view"])
        self._classes: list[list[np.ndarray]] = []
        for image_index in range(len(self.image_ids)):
            start, end = self.offsets[image_index : image_index + 2]
            boxes = self.boxes[start:end]
            scores = self.scores[start:end]
            class_ids = self.class_ids[start:end]
            per_class: list[np.ndarray] = []
            for class_id in range(len(HOD26_CLASSES)):
                keep = class_ids == class_id
                if np.any(keep):
                    per_class.append(
                        np.column_stack((boxes[keep], scores[keep])).astype(
                            np.float32, copy=False
                        )
                    )
                else:
                    per_class.append(np.zeros((0, 5), dtype=np.float32))
            self._classes.append(per_class)

    def detections(self, image_index: int, class_id: int) -> np.ndarray:
        return self._classes[image_index][class_id]


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.zeros(0, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, float(box[2] - box[0])) * max(
        0.0, float(box[3] - box[1])
    )
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    return intersection / np.maximum(area_a + area_b - intersection, 1e-12)


def _hard_nms(detections: np.ndarray, iou_threshold: float) -> np.ndarray:
    if not len(detections):
        return np.zeros((0, 5), dtype=np.float32)
    order = np.argsort(-detections[:, 4])
    keep: list[int] = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        ious = _iou_one_to_many(detections[current, :4], detections[order[1:], :4])
        order = order[1:][ious < iou_threshold]
    return np.ascontiguousarray(detections[keep], dtype=np.float32)


def _soft_nms(
    detections: np.ndarray,
    *,
    iou_threshold: float,
    sigma: float,
    min_score: float,
    method: str,
) -> np.ndarray:
    if not len(detections):
        return np.zeros((0, 5), dtype=np.float32)
    from mmcv.ops import soft_nms

    merged, _ = soft_nms(
        detections[:, :4],
        detections[:, 4],
        iou_threshold=float(iou_threshold),
        sigma=float(sigma),
        min_score=float(min_score),
        method=method,
        offset=0,
    )
    return np.ascontiguousarray(
        merged[np.argsort(-merged[:, 4])], dtype=np.float32
    )


def _cluster_score(
    unique_scores: Mapping[int, float],
    *,
    score_mode: str,
    view_count: int,
) -> float:
    values = np.asarray(list(unique_scores.values()), dtype=np.float64)
    if score_mode == "max":
        return float(values.max())
    if score_mode == "avg_present":
        return float(values.mean())
    if score_mode == "avg_all":
        return float(values.sum() / max(view_count, 1))
    if score_mode == "noisy_or":
        return float(1.0 - np.prod(1.0 - np.clip(values, 0.0, 1.0)))
    raise ValueError(f"Unknown score mode: {score_mode}")


def _weighted_box_fusion(
    arrays: Sequence[np.ndarray],
    *,
    iou_threshold: float,
    score_mode: str,
) -> np.ndarray:
    nonempty: list[np.ndarray] = []
    view_ids: list[np.ndarray] = []
    for view_id, array in enumerate(arrays):
        if len(array):
            nonempty.append(array)
            view_ids.append(np.full(len(array), view_id, dtype=np.int16))
    if not nonempty:
        return np.zeros((0, 5), dtype=np.float32)
    candidates = np.concatenate(nonempty).astype(np.float32, copy=False)
    candidate_views = np.concatenate(view_ids)
    # Hard-NMS seeds form a deterministic partition.  Assigning all boxes to
    # those seeds is mathematically the same weighted coordinate vote used by
    # WBF, but avoids the quadratic Python-level dynamic-cluster loop.
    seeds = _hard_nms(candidates, iou_threshold)
    iou_matrix = np.stack(
        [
            _iou_one_to_many(seed[:4], candidates[:, :4])
            for seed in seeds
        ]
    )
    assignments = np.argmax(iou_matrix, axis=0)
    output: list[list[float]] = []
    for seed_index in range(len(seeds)):
        member_indices = np.flatnonzero(assignments == seed_index)
        if not len(member_indices):
            continue
        members = candidates[member_indices]
        member_views = candidate_views[member_indices]
        weights = np.maximum(members[:, 4].astype(np.float64), 1e-8)
        fused_box = np.average(
            members[:, :4].astype(np.float64), axis=0, weights=weights
        )
        unique_scores: dict[int, float] = {}
        for view_id, score in zip(member_views.tolist(), members[:, 4].tolist()):
            unique_scores[view_id] = max(unique_scores.get(view_id, 0.0), score)
        score = _cluster_score(
            unique_scores, score_mode=score_mode, view_count=len(arrays)
        )
        output.append([*fused_box.tolist(), score])
    if not output:
        return np.zeros((0, 5), dtype=np.float32)
    fused = np.asarray(output, dtype=np.float32)
    return np.ascontiguousarray(fused[np.argsort(-fused[:, 4])])


def _box_vote(
    arrays: Sequence[np.ndarray],
    *,
    nms_iou: float,
    vote_iou: float,
    score_mode: str,
) -> np.ndarray:
    nonempty = [array for array in arrays if len(array)]
    if not nonempty:
        return np.zeros((0, 5), dtype=np.float32)
    candidates = np.concatenate(nonempty).astype(np.float32, copy=False)
    seeds = _hard_nms(candidates, nms_iou)
    voted = []
    for seed in seeds:
        ious = _iou_one_to_many(seed[:4], candidates[:, :4])
        members = candidates[ious >= vote_iou]
        weights = np.maximum(members[:, 4].astype(np.float64) ** 2, 1e-12)
        box = np.average(members[:, :4], axis=0, weights=weights)
        if score_mode == "noisy_or":
            score = float(
                1.0 - np.prod(1.0 - np.clip(members[:, 4], 0.0, 1.0))
            )
        else:
            score = float(members[:, 4].max())
        voted.append([*box.tolist(), score])
    result = np.asarray(voted, dtype=np.float32)
    return np.ascontiguousarray(result[np.argsort(-result[:, 4])])


def _merge_class(
    arrays: Sequence[np.ndarray],
    setting: Mapping[str, Any],
) -> np.ndarray:
    pre_score = float(setting.get("pre_score", 0.0001))
    filtered = [
        array[array[:, 4] >= pre_score].astype(np.float32, copy=False)
        for array in arrays
    ]
    nonempty = [array for array in filtered if len(array)]
    if not nonempty:
        return np.zeros((0, 5), dtype=np.float32)
    method = str(setting["method"])
    if method == "identity":
        if len(filtered) != 1:
            raise ValueError("identity merge requires exactly one view")
        merged = filtered[0][np.argsort(-filtered[0][:, 4])]
    elif method == "hard_nms":
        merged = _hard_nms(
            np.concatenate(nonempty), float(setting["iou_threshold"])
        )
    elif method in {"soft_linear", "soft_gaussian"}:
        merged = _soft_nms(
            np.concatenate(nonempty),
            iou_threshold=float(setting["iou_threshold"]),
            sigma=float(setting.get("sigma", 0.5)),
            min_score=pre_score,
            method="linear" if method == "soft_linear" else "gaussian",
        )
    elif method == "wbf":
        merged = _weighted_box_fusion(
            filtered,
            iou_threshold=float(setting["iou_threshold"]),
            score_mode=str(setting.get("score_mode", "avg_all")),
        )
    elif method == "box_vote":
        merged = _box_vote(
            filtered,
            nms_iou=float(setting["iou_threshold"]),
            vote_iou=float(setting.get("vote_iou", 0.6)),
            score_mode=str(setting.get("score_mode", "max")),
        )
    else:
        raise ValueError(f"Unknown merge method: {method}")
    output_score = float(setting.get("output_score", pre_score))
    merged = merged[
        np.isfinite(merged).all(axis=1)
        & (merged[:, 2] > merged[:, 0])
        & (merged[:, 3] > merged[:, 1])
        & (merged[:, 4] >= output_score)
    ]
    return np.ascontiguousarray(merged, dtype=np.float32)


def merge_cached_predictions(
    caches: Sequence[CachedView],
    setting: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not caches:
        raise ValueError("At least one cached view is required")
    reference_ids = caches[0].image_ids.tolist()
    if any(cache.image_ids.tolist() != reference_ids for cache in caches[1:]):
        raise ValueError("TTA caches have different image order")
    max_per_image = int(setting.get("max_per_image", 300))
    rows: list[dict[str, Any]] = []
    for image_index, image_id in enumerate(reference_ids):
        image_rows: list[dict[str, Any]] = []
        for class_id in range(len(HOD26_CLASSES)):
            merged = _merge_class(
                [
                    cache.detections(image_index, class_id)
                    for cache in caches
                ],
                setting,
            )
            for x1, y1, x2, y2, score in merged:
                image_rows.append(
                    {
                        "image_id": str(image_id),
                        "class_id": class_id,
                        "confidence": float(score),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    }
                )
        image_rows.sort(key=lambda row: -row["confidence"])
        rows.extend(image_rows[:max_per_image])
    rows.sort(key=lambda row: (int(row["image_id"]), -row["confidence"]))
    return rows


def _coco_metrics(
    ground_truth: COCO,
    rows: Sequence[Mapping[str, Any]],
    image_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    predictions = [
        {
            "image_id": int(row["image_id"]),
            "category_id": int(row["class_id"]),
            "bbox": [
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]) - float(row["x1"]),
                float(row["y2"]) - float(row["y1"]),
            ],
            "score": float(row["confidence"]),
        }
        for row in rows
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        detections = ground_truth.loadRes(predictions)
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.params.catIds = sorted(ground_truth.getCatIds())
        evaluator.params.imgIds = (
            sorted(int(image_id) for image_id in image_ids)
            if image_ids is not None
            else sorted(ground_truth.getImgIds())
        )
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision = evaluator.eval["precision"]
    iou75 = int(np.argmin(np.abs(evaluator.params.iouThrs - 0.75)))
    per_class: dict[str, float] = {}
    per_class_75: dict[str, float] = {}
    for class_index, name in enumerate(HOD26_CLASSES):
        values = precision[:, :, class_index, 0, -1]
        values = values[values > -1]
        values75 = precision[iou75, :, class_index, 0, -1]
        values75 = values75[values75 > -1]
        per_class[name] = float(values.mean()) if values.size else float("nan")
        per_class_75[name] = (
            float(values75.mean()) if values75.size else float("nan")
        )
    weak = np.asarray(
        [per_class[HOD26_CLASSES[index]] for index in WEAK4_IDS],
        dtype=np.float64,
    )
    weak75 = np.asarray(
        [per_class_75[HOD26_CLASSES[index]] for index in WEAK4_IDS],
        dtype=np.float64,
    )
    stats = evaluator.stats
    return {
        "map_50_95": float(stats[0]),
        "map_50": float(stats[1]),
        "map_75": float(stats[2]),
        "map_small": float(stats[3]),
        "map_medium": float(stats[4]),
        "map_large": float(stats[5]),
        "ar_100": float(stats[8]),
        "weak4_map_50_95": float(np.nanmean(weak)),
        "weak4_map_75": float(np.nanmean(weak75)),
        "per_class_map_50_95": per_class,
    }


def _setting_key(views: Sequence[str], setting: Mapping[str, Any]) -> str:
    payload = {"views": list(views), **dict(setting)}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _rank_key(result: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(result["map_50_95"]),
        float(result["map_75"]),
        float(result["map_small"]),
        float(result["weak4_map_50_95"]),
    )


def _view_sets(
    single_results: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, ...]]:
    ranked = sorted(
        single_results,
        key=lambda key: _rank_key(single_results[key]),
        reverse=True,
    )
    sets: list[tuple[str, ...]] = []

    def add(values: Iterable[str]) -> None:
        value = tuple(dict.fromkeys(values))
        if len(value) >= 2 and value not in sets:
            sets.append(value)

    by_width = {
        width: [view.key for view in ALL_VIEWS if view.width == width]
        for width in DEFAULT_WIDTHS
    }
    by_flip = {
        flip: [view.key for view in ALL_VIEWS if view.flip == flip]
        for flip in DEFAULT_FLIPS
    }
    for width in DEFAULT_WIDTHS:
        add(by_width[width])
        add(
            view.key
            for view in ALL_VIEWS
            if view.width == width and view.flip in {"none", "horizontal"}
        )
    for flip in DEFAULT_FLIPS:
        add(by_flip[flip])
    add(view.key for view in ALL_VIEWS if view.flip in {"none", "horizontal"})
    add(view.key for view in ALL_VIEWS if view.width in {1152, 1280})
    add(
        view.key
        for view in ALL_VIEWS
        if view.width in {1024, 1152, 1280}
    )
    add(view.key for view in ALL_VIEWS)
    for count in (2, 3, 4, 6, 8, 12):
        add(ranked[:count])
    return sets


def _base_setting(method: str, iou: float = 0.65) -> dict[str, Any]:
    return {
        "method": method,
        "iou_threshold": float(iou),
        "sigma": 0.5,
        "score_mode": "max",
        "vote_iou": 0.6,
        "pre_score": 0.0001,
        "output_score": 0.0001,
        "max_per_image": 300,
    }


def _coarse_settings() -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = []
    for iou in (0.55, 0.65, 0.75):
        settings.append(_base_setting("hard_nms", iou))
        settings.append(_base_setting("soft_linear", iou))
        for score_mode in ("max", "avg_all"):
            setting = _base_setting("wbf", iou)
            setting["score_mode"] = score_mode
            settings.append(setting)
    for vote_iou in (0.5, 0.6, 0.7):
        setting = _base_setting("box_vote", 0.65)
        setting["vote_iou"] = vote_iou
        settings.append(setting)
    return settings


def _fine_settings() -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = []
    ious = (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)
    for iou in ious:
        settings.append(_base_setting("hard_nms", iou))
        settings.append(_base_setting("soft_linear", iou))
    for iou, sigma in itertools.product((0.55, 0.65, 0.75), (0.3, 0.5, 0.7)):
        setting = _base_setting("soft_gaussian", iou)
        setting["sigma"] = sigma
        settings.append(setting)
    for iou, score_mode in itertools.product(
        (0.5, 0.55, 0.6, 0.65, 0.7, 0.75),
        ("max", "avg_all", "noisy_or"),
    ):
        setting = _base_setting("wbf", iou)
        setting["score_mode"] = score_mode
        settings.append(setting)
    for iou, vote_iou in itertools.product(
        (0.5, 0.55, 0.6, 0.65, 0.7, 0.75),
        (0.5, 0.6, 0.7),
    ):
        setting = _base_setting("box_vote", iou)
        setting["vote_iou"] = vote_iou
        settings.append(setting)
    return settings


def _search_worker_init() -> None:
    # Each search parent owns many independent configurations.  Keep every
    # worker single-threaded so two NUMA-local pools can saturate the host
    # without BLAS/OpenMP oversubscription.
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        import cv2

        cv2.setNumThreads(0)
    except ImportError:
        pass


def _search_worker(
    job: tuple[tuple[str, ...], dict[str, Any], str],
) -> dict[str, Any]:
    views, setting, stage = job
    if _SEARCH_CACHES is None or _SEARCH_GROUND_TRUTH is None:
        raise RuntimeError("Search worker did not inherit the read-only caches")
    selected = [_SEARCH_CACHES[view] for view in views]
    rows = merge_cached_predictions(selected, setting)
    metrics = _coco_metrics(_SEARCH_GROUND_TRUTH, rows)
    return {
        "_key": _setting_key(views, setting),
        "stage": stage,
        "views": list(views),
        "view_count": len(views),
        "detections": len(rows),
        "setting": dict(setting),
        **metrics,
    }


def search_tta(
    *,
    cache_dir: str | Path,
    val_annotations: str | Path,
    output_path: str | Path,
    candidate_name: str,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    global _SEARCH_CACHES, _SEARCH_GROUND_TRUTH

    cache_root = Path(cache_dir)
    caches = {
        view.key: CachedView(_cache_path(cache_root, view)) for view in ALL_VIEWS
    }
    ground_truth = COCO(str(Path(val_annotations).resolve()))
    _SEARCH_CACHES = caches
    _SEARCH_GROUND_TRUTH = ground_truth
    results: list[dict[str, Any]] = []
    results_by_key: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()

    available_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    worker_count = min(
        _positive_env_int("V4_SEARCH_WORKERS", 32),
        available_cpus,
    )
    print(
        f"[V4 search:{candidate_name}] workers={worker_count}, "
        f"NUMA-local CPUs={available_cpus}",
        flush=True,
    )

    def evaluate_batch(
        executor: concurrent.futures.ProcessPoolExecutor,
        configurations: Sequence[
            tuple[tuple[str, ...], dict[str, Any], str]
        ],
    ) -> list[dict[str, Any]]:
        pending: dict[
            concurrent.futures.Future[dict[str, Any]], str
        ] = {}
        requested_keys: list[str] = []
        scheduled: set[str] = set()
        for views, setting, stage in configurations:
            key = _setting_key(views, setting)
            requested_keys.append(key)
            if key in results_by_key or key in scheduled:
                continue
            scheduled.add(key)
            future = executor.submit(
                _search_worker,
                (tuple(views), dict(setting), stage),
            )
            pending[future] = key
        for future in concurrent.futures.as_completed(pending):
            result = future.result()
            result.update(
                {
                    "candidate": candidate_name,
                    "checkpoint": str(Path(checkpoint_path).resolve()),
                    "checkpoint_sha256": checkpoint_sha256,
                }
            )
            results.append(result)
            results_by_key[result["_key"]] = result
            print(
                f"[V4 search:{candidate_name}] {result['stage']} "
                f"views={result['view_count']:2d} "
                f"{result['setting']['method']:13s} "
                f"mAP={result['map_50_95']:.6f} "
                f"AP75={result['map_75']:.6f}",
                flush=True,
            )
        return [results_by_key[key] for key in requested_keys]

    context = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_search_worker_init,
    ) as executor:
        identity = _base_setting("identity")
        single_configurations = [
            ((view.key,), identity, "single") for view in ALL_VIEWS
        ]
        single_evaluations = evaluate_batch(executor, single_configurations)
        single_results = {
            view.key: result
            for view, result in zip(ALL_VIEWS, single_evaluations)
        }

        coarse_configurations = [
            (views, setting, "coarse")
            for views in _view_sets(single_results)
            for setting in _coarse_settings()
        ]
        evaluate_batch(executor, coarse_configurations)

        coarse_ranked = sorted(
            (result for result in results if result["stage"] == "coarse"),
            key=_rank_key,
            reverse=True,
        )
        top_view_sets: list[tuple[str, ...]] = []
        for result in coarse_ranked:
            views = tuple(result["views"])
            if views not in top_view_sets:
                top_view_sets.append(views)
            if len(top_view_sets) == 4:
                break
        fine_configurations = [
            (views, setting, "fine")
            for views in top_view_sets
            for setting in _fine_settings()
        ]
        evaluate_batch(executor, fine_configurations)

        ranked = sorted(results, key=_rank_key, reverse=True)
        top_pairs: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        for result in ranked:
            pair = (tuple(result["views"]), dict(result["setting"]))
            signature = _setting_key(pair[0], pair[1])
            if any(
                _setting_key(item[0], item[1]) == signature
                for item in top_pairs
            ):
                continue
            top_pairs.append(pair)
            if len(top_pairs) == 5:
                break
        threshold_configurations = []
        for views, original in top_pairs:
            for output_score, max_per_image in itertools.product(
                (0.0001, 0.0003, 0.001, 0.003, 0.01),
                (100, 200, 300, 500),
            ):
                setting = dict(original)
                setting["output_score"] = output_score
                setting["max_per_image"] = max_per_image
                threshold_configurations.append(
                    (views, setting, "threshold")
                )
        evaluate_batch(executor, threshold_configurations)

    ranked = sorted(results, key=_rank_key, reverse=True)
    for result in ranked:
        result.pop("_key", None)
    report = {
        "schema": 1,
        "candidate": candidate_name,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "validation_images": len(caches[ALL_VIEWS[0].key].image_ids),
        "searched_configurations": len(ranked),
        "search_workers": worker_count,
        "selection_metric": "COCO mAP@[.50:.95], maxDets=(1,10,100)",
        "best": ranked[0],
        "results": ranked,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json_dump(report, output_path)
    _SEARCH_CACHES = None
    _SEARCH_GROUND_TRUTH = None
    return report


def _build_soup_candidates(
    *,
    run_dir: Path,
    soup_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    soup_threads = min(
        _positive_env_int("V4_SOUP_CPU_THREADS", 32),
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1),
    )
    torch.set_num_threads(soup_threads)
    print(f"[V4 soup] CPU threads={soup_threads}", flush=True)
    members = OrderedDict(
        [
            ("e30", run_dir / "best_bbox_mAP_epoch_30.pth"),
            ("e42", run_dir / "epoch_42.pth"),
            ("e50", run_dir / "epoch_50.pth"),
            ("e54", run_dir / "epoch_54.pth"),
            ("e64", run_dir / "epoch_64.pth"),
        ]
    )
    missing = [str(path) for path in members.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing soup checkpoints: {missing}")
    hashes: dict[str, str] = {}
    for name, path in members.items():
        print(f"[V4 soup] hashing {name}: {path}", flush=True)
        hashes[name] = _sha256(path)
    candidates = [
        {
            "name": name,
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": hashes[name],
            "kind": "raw_ema",
        }
        for name, path in members.items()
    ]

    definitions: list[tuple[str, tuple[str, ...], tuple[float, ...]]] = []
    tail = tuple(name for name in members if name != "e30")
    for size in range(2, len(members) + 1):
        for addition in itertools.combinations(tail, size - 1):
            names = ("e30", *addition)
            definitions.append(
                ("soup_" + "_".join(names), names, (1.0,) * len(names))
            )
    definitions.extend(
        [
            ("soup_e30x2_e42", ("e30", "e42"), (2.0, 1.0)),
            ("soup_e30x2_e50", ("e30", "e50"), (2.0, 1.0)),
            ("soup_e30x2_e64", ("e30", "e64"), (2.0, 1.0)),
            (
                "soup_e30x2_e42_e50_e64",
                ("e30", "e42", "e50", "e64"),
                (2.0, 1.0, 1.0, 1.0),
            ),
            (
                "soup_e30x3_e42_e64",
                ("e30", "e42", "e64"),
                (3.0, 1.0, 1.0),
            ),
        ]
    )

    soup_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        definition
        for definition in definitions
        if not (soup_dir / f"{definition[0]}.pth").is_file()
    ]
    states: dict[str, OrderedDict[str, torch.Tensor]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    if pending:
        for name, path in members.items():
            print(f"[V4 soup] loading {name}: {path}", flush=True)
            payload = torch.load(path, map_location="cpu")
            states[name] = _primary_ema_state(payload)
            metadata[name] = dict(payload.get("meta", {}))
            del payload
            gc.collect()
        for index, (name, names, weights) in enumerate(pending, 1):
            output = soup_dir / f"{name}.pth"
            print(
                f"[V4 soup] {index}/{len(pending)} {name} "
                f"members={names} weights={weights}",
                flush=True,
            )
            report = average_loaded_states(
                [states[item] for item in names],
                [metadata[item] for item in names],
                [members[item] for item in names],
                output,
                weights,
                [hashes[item] for item in names],
            )
            candidates.append(
                {
                    "name": name,
                    "checkpoint": str(output.resolve()),
                    "checkpoint_sha256": report["output_sha256"],
                    "kind": "soup",
                }
            )
            gc.collect()
        del states, metadata
        gc.collect()

    existing_names = {candidate["name"] for candidate in candidates}
    for name, _, _ in definitions:
        if name in existing_names:
            continue
        output = soup_dir / f"{name}.pth"
        audit = json.loads(
            output.with_suffix(".pth.audit.json").read_text(encoding="utf-8")
        )
        candidates.append(
            {
                "name": name,
                "checkpoint": str(output.resolve()),
                "checkpoint_sha256": audit["output_sha256"],
                "kind": "soup",
            }
        )
    return candidates, hashes


def _run_process_pool(
    jobs: Sequence[tuple[list[str], str]],
    *,
    max_parallel: int = 2,
) -> None:
    # Jobs are submitted in explicit cuda:0/cuda:1 pairs.  Waiting for the
    # complete pair before admitting the next pair prevents an uneven finish
    # time from accidentally placing two large ViT-L processes on one GPU.
    for offset in range(0, len(jobs), max_parallel):
        active: list[tuple[subprocess.Popen, str]] = []
        for command, label in jobs[offset : offset + max_parallel]:
            print(f"[V4 pipeline] starting {label}", flush=True)
            active.append((subprocess.Popen(command), label))
        while active:
            time.sleep(1.0)
            remaining: list[tuple[subprocess.Popen, str]] = []
            failure: tuple[int, str] | None = None
            for process, label in active:
                code = process.poll()
                if code is None:
                    remaining.append((process, label))
                elif code != 0:
                    failure = (code, label)
                else:
                    print(f"[V4 pipeline] completed {label}", flush=True)
            if failure is not None:
                for process, _ in remaining:
                    process.terminate()
                code, label = failure
                raise RuntimeError(f"Background job failed ({code}): {label}")
            active = remaining


def _numa_prefix(node: int) -> list[str]:
    numactl = Path("/usr/bin/numactl")
    if not numactl.is_file():
        return []
    return [
        str(numactl),
        f"--cpunodebind={node}",
        f"--preferred={node}",
    ]


def _cache_command(
    *,
    checkpoint: str,
    checkpoint_sha256: str,
    split: str,
    views: Sequence[ViewSpec],
    output_dir: Path,
    device: str,
    args: argparse.Namespace,
) -> list[str]:
    device_index = int(device.rsplit(":", 1)[-1])
    return [
        *_numa_prefix(device_index),
        sys.executable,
        "-m",
        "hod26.v4.optimize",
        "cache",
        "--checkpoint",
        checkpoint,
        "--checkpoint-sha256",
        checkpoint_sha256,
        "--split",
        split,
        "--views",
        *(f"{view.width}:{view.flip}" for view in views),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--config",
        args.config,
        "--data-root",
        args.data_root,
        "--stats",
        args.stats,
        "--upstream",
        args.upstream,
        "--test-manifest",
        args.test_manifest,
        "--val-annotations",
        args.val_annotations,
    ]


def _candidate_single_view_metrics(
    candidate: Mapping[str, Any],
    cache_dir: Path,
    ground_truth: COCO,
) -> dict[str, Any]:
    cache = CachedView(cache_dir / "w1280_none.npz")
    setting = _base_setting("identity")
    rows = merge_cached_predictions([cache], setting)
    return {
        "name": candidate["name"],
        "checkpoint": candidate["checkpoint"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "kind": candidate["kind"],
        "views": ["w1280_none"],
        "setting": setting,
        **_coco_metrics(ground_truth, rows),
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    run_root = Path(args.run_dir).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    source_run = Path(args.source_run).resolve()
    soup_dir = run_root / "soups"
    candidate_cache = run_root / "candidate_val"
    full_val_root = run_root / "full_val"
    test_root = run_root / "test"
    reports = run_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    candidates, source_hashes = _build_soup_candidates(
        run_dir=source_run,
        soup_dir=soup_dir,
    )
    atomic_json_dump(candidates, reports / "candidate_manifest.json")

    validation_jobs: list[tuple[list[str], str]] = []
    for index, candidate in enumerate(candidates):
        output_dir = candidate_cache / candidate["name"]
        validation_jobs.append(
            (
                _cache_command(
                    checkpoint=candidate["checkpoint"],
                    checkpoint_sha256=candidate["checkpoint_sha256"],
                    split="val",
                    views=(ViewSpec(1280, "none"),),
                    output_dir=output_dir,
                    device=f"cuda:{index % 2}",
                    args=args,
                ),
                f"single-view validation {candidate['name']}",
            )
        )
    _run_process_pool(validation_jobs, max_parallel=2)

    ground_truth = COCO(str(Path(args.val_annotations).resolve()))
    candidate_results = [
        _candidate_single_view_metrics(
            candidate,
            candidate_cache / candidate["name"],
            ground_truth,
        )
        for candidate in candidates
    ]
    candidate_results.sort(key=_rank_key, reverse=True)
    candidate_report = {
        "selection": "single-view 1280/none Fold0 COCO mAP@[.50:.95]",
        "best": candidate_results[0],
        "results": candidate_results,
    }
    atomic_json_dump(candidate_report, reports / "soup_selection.json")
    top_candidates = candidate_results[:2]
    print(
        "[V4 pipeline] full-TTA finalists: "
        + ", ".join(
            f"{item['name']}={item['map_50_95']:.6f}"
            for item in top_candidates
        ),
        flush=True,
    )

    full_jobs: list[tuple[list[str], str]] = []
    for index, candidate in enumerate(top_candidates):
        full_jobs.append(
            (
                _cache_command(
                    checkpoint=candidate["checkpoint"],
                    checkpoint_sha256=candidate["checkpoint_sha256"],
                    split="val",
                    views=ALL_VIEWS,
                    output_dir=full_val_root / candidate["name"],
                    device=f"cuda:{index}",
                    args=args,
                ),
                f"full 16-view validation {candidate['name']}",
            )
        )
    _run_process_pool(full_jobs, max_parallel=2)

    search_jobs: list[tuple[list[str], str]] = []
    for index, candidate in enumerate(top_candidates):
        command = [
            *_numa_prefix(index),
            sys.executable,
            "-m",
            "hod26.v4.optimize",
            "search",
            "--cache-dir",
            str(full_val_root / candidate["name"]),
            "--val-annotations",
            args.val_annotations,
            "--output",
            str(reports / f"tta_search_{candidate['name']}.json"),
            "--candidate-name",
            candidate["name"],
            "--checkpoint",
            candidate["checkpoint"],
            "--checkpoint-sha256",
            candidate["checkpoint_sha256"],
        ]
        search_jobs.append((command, f"TTA/NMS search {candidate['name']}"))
    _run_process_pool(search_jobs, max_parallel=2)

    search_reports = [
        json.loads(
            (reports / f"tta_search_{candidate['name']}.json").read_text(
                encoding="utf-8"
            )
        )
        for candidate in top_candidates
    ]
    best_report = max(search_reports, key=lambda report: _rank_key(report["best"]))
    best = best_report["best"]
    selected_candidate = next(
        candidate
        for candidate in top_candidates
        if candidate["name"] == best_report["candidate"]
    )
    selected_views = [
        next(view for view in ALL_VIEWS if view.key == key)
        for key in best["views"]
    ]
    groups = [selected_views[0::2], selected_views[1::2]]
    test_jobs = [
        (
            _cache_command(
                checkpoint=selected_candidate["checkpoint"],
                checkpoint_sha256=selected_candidate["checkpoint_sha256"],
                split="test",
                views=group,
                output_dir=test_root / selected_candidate["name"],
                device=f"cuda:{index}",
                args=args,
            ),
            f"test TTA shard {index}",
        )
        for index, group in enumerate(groups)
        if group
    ]
    _run_process_pool(test_jobs, max_parallel=2)

    test_caches = [
        CachedView(_cache_path(test_root / selected_candidate["name"], view))
        for view in selected_views
    ]
    test_rows = merge_cached_predictions(test_caches, best["setting"])
    output_csv = Path(args.output).resolve()
    write_submission(test_rows, output_csv, include_id=True)
    test_manifest = json.loads(
        Path(args.test_manifest).read_text(encoding="utf-8")
    )
    validation = validate_submission(
        output_csv,
        test_manifest["samples"],
        num_classes=len(HOD26_CLASSES),
        include_id=True,
    )
    if not validation["valid"]:
        raise RuntimeError(json.dumps(validation, ensure_ascii=False, indent=2))
    final_report = {
        "schema": 1,
        "single_model": True,
        "prediction_ensemble": False,
        "checkpoint": selected_candidate["checkpoint"],
        "checkpoint_sha256": selected_candidate["checkpoint_sha256"],
        "checkpoint_kind": selected_candidate["kind"],
        "source_checkpoint_sha256": source_hashes,
        "validation_selection": best,
        "test_views": [view.key for view in selected_views],
        "test_setting": best["setting"],
        "csv": str(output_csv),
        "csv_sha256": _sha256(output_csv),
        "csv_validation": validation,
        "stats": str(Path(args.stats).resolve()),
        "stats_sha256": _sha256(args.stats),
        "elapsed_seconds": time.perf_counter() - started,
        "kaggle_uploaded": False,
    }
    atomic_json_dump(final_report, output_csv.with_suffix(".csv.audit.json"))
    atomic_json_dump(final_report, reports / "final.json")
    print(
        f"[V4 pipeline] COMPLETE CSV={output_csv} "
        f"val_mAP={best['map_50_95']:.6f}",
        flush=True,
    )
    return final_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize one V4 model with soup, TTA and legal box merging"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache")
    cache.add_argument("--checkpoint", required=True)
    cache.add_argument("--checkpoint-sha256", required=True)
    cache.add_argument("--split", choices=["val", "test"], required=True)
    cache.add_argument("--views", nargs="+", required=True)
    cache.add_argument("--output-dir", required=True)
    cache.add_argument("--device", default="cuda:0")
    cache.add_argument("--config", default="configs/v4/co_dino_vitl_hsi_fold0.py")
    cache.add_argument("--data-root", default="data/raw")
    cache.add_argument("--stats", default="storage/v4/dataset/fold0_band_stats.json")
    cache.add_argument("--upstream", default="storage/upstream/Co-DETR")
    cache.add_argument("--test-manifest", default="storage/cache/test_manifest.json")
    cache.add_argument(
        "--val-annotations", default="storage/v2/dataset/fold0/val.json"
    )
    cache.add_argument("--pre-score", type=float, default=0.0001)
    cache.add_argument("--limit", type=int)

    search = subparsers.add_parser("search")
    search.add_argument("--cache-dir", required=True)
    search.add_argument("--val-annotations", required=True)
    search.add_argument("--output", required=True)
    search.add_argument("--candidate-name", required=True)
    search.add_argument("--checkpoint", required=True)
    search.add_argument("--checkpoint-sha256", required=True)

    pipeline = subparsers.add_parser("pipeline")
    pipeline.add_argument(
        "--source-run",
        default="storage/v4/runs/co_dino_vitl_hsi_fold0",
    )
    pipeline.add_argument(
        "--run-dir",
        default="storage/v4/optimization/fold0_epoch64",
    )
    pipeline.add_argument(
        "--output",
        default="storage/v4/submissions/v4_soup_tta_nms_optimized.csv",
    )
    pipeline.add_argument(
        "--config", default="configs/v4/co_dino_vitl_hsi_fold0.py"
    )
    pipeline.add_argument("--data-root", default="data/raw")
    pipeline.add_argument(
        "--stats", default="storage/v4/dataset/fold0_band_stats.json"
    )
    pipeline.add_argument("--upstream", default="storage/upstream/Co-DETR")
    pipeline.add_argument(
        "--test-manifest", default="storage/cache/test_manifest.json"
    )
    pipeline.add_argument(
        "--val-annotations", default="storage/v2/dataset/fold0/val.json"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "cache":
        outputs = cache_views(
            checkpoint_path=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            split=args.split,
            views=[ViewSpec.parse(value) for value in args.views],
            output_dir=args.output_dir,
            config_path=args.config,
            data_root=args.data_root,
            stats_path=args.stats,
            upstream_path=args.upstream,
            test_manifest=args.test_manifest,
            val_annotations=args.val_annotations,
            device_name=args.device,
            pre_score=args.pre_score,
            limit=args.limit,
        )
        print(json.dumps([str(path) for path in outputs], ensure_ascii=False))
    elif args.command == "search":
        report = search_tta(
            cache_dir=args.cache_dir,
            val_annotations=args.val_annotations,
            output_path=args.output,
            candidate_name=args.candidate_name,
            checkpoint_path=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
        )
        print(json.dumps(report["best"], ensure_ascii=False, indent=2))
    elif args.command == "pipeline":
        report = run_pipeline(args)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
