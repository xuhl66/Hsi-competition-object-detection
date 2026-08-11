"""V6-specific, single-checkpoint class calibration.

The native Co-DINO decoder ranks ``num_queries * num_classes`` sigmoid scores
globally before class-wise NMS.  V6's IA-BCE score scales are intentionally
quality-aware but are not guaranteed to be comparable between classes.  This
module keeps the converged checkpoint, input view, boxes and within-class score
ordering fixed, while making the cross-class top-k competition observable and
calibratable on Fold0.

This file is deliberately outside the V6 checkpoint runtime source lock.  It
does not alter training or the source-locked native/champion inference paths.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from hod26.config import atomic_json_dump
from hod26.submission import validate_submission, write_submission

from .constants import HOD26_CLASSES
from .data import load_band_limits
from .infer import _load_cube, _resize_pad, _restore_boxes, _sha256, build_model


NUM_CLASSES = len(HOD26_CLASSES)
NATIVE_MAX_PER_IMAGE = 300
NATIVE_PRE_NMS_TOPK = 1500
NATIVE_NMS_IOU = 0.75
NATIVE_NMS_MIN_SCORE = 0.0001


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_records(path: Path, kind: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if kind == "annotation":
        return [
            {
                "image_id": str(row["id"]),
                "image": str(row["file_name"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
            }
            for row in payload["images"]
        ]
    if kind == "manifest":
        return [dict(row) for row in payload["samples"]]
    raise ValueError(f"Unknown record kind: {kind}")


def _restore_query_boxes(
    normalized_cxcywh: np.ndarray,
    *,
    resized_width: int,
    resized_height: int,
    scale_x: float,
    scale_y: float,
    original_width: int,
    original_height: int,
) -> np.ndarray:
    value = np.asarray(normalized_cxcywh, np.float32)
    boxes = np.empty_like(value)
    boxes[:, 0] = (value[:, 0] - value[:, 2] * 0.5) * resized_width / scale_x
    boxes[:, 1] = (value[:, 1] - value[:, 3] * 0.5) * resized_height / scale_y
    boxes[:, 2] = (value[:, 0] + value[:, 2] * 0.5) * resized_width / scale_x
    boxes[:, 3] = (value[:, 1] + value[:, 3] * 0.5) * resized_height / scale_y
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, original_width)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, original_height)
    return boxes


@torch.inference_mode()
def extract_query_cache(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    records_path: str | Path,
    records_kind: str,
    image_root: str | Path,
    stats_path: str | Path,
    output_path: str | Path,
    device_name: str = "cuda:0",
    limit: int | None = None,
) -> dict[str, Any]:
    """Persist the last decoder's 1,500 boxes and all 18 sigmoid scores."""

    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    records_path = Path(records_path)
    output_path = Path(output_path)
    records = _load_records(records_path, records_kind)
    if limit is not None:
        records = records[: int(limit)]
    if not records:
        raise ValueError("V6 calibration cache needs at least one image")

    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    model, _, metadata = build_model(config_path, checkpoint_path, device)
    low, high = load_band_limits(stats_path)
    root = Path(image_root)

    score_rows: list[np.ndarray] = []
    box_rows: list[np.ndarray] = []
    native_rows: list[np.ndarray] = []
    native_counts: list[int] = []
    image_ids: list[int] = []
    widths: list[int] = []
    heights: list[int] = []
    captured_query_outputs: list[Any] = []
    original_query_forward = model.query_head.forward

    def observed_query_forward(*args, **kwargs):
        output = original_query_forward(*args, **kwargs)
        captured_query_outputs.append(output)
        return output

    # Upstream ``simple_test_bboxes`` calls ``self.forward`` directly rather
    # than ``self(...)``, so a standard nn.Module forward hook is bypassed.
    # This instance-local wrapper observes the same call and is restored in
    # the finally block below.
    model.query_head.forward = observed_query_forward
    started = time.perf_counter()
    try:
        for index, record in enumerate(records):
            filename = root / record["image"]
            cube = _load_cube(filename, low, high)
            original_height, original_width = cube.shape[:2]
            if (original_width, original_height) != (
                int(record["width"]),
                int(record["height"]),
            ):
                raise RuntimeError(f"Image geometry mismatch for {filename}")
            canvas, sx, sy, resized_width, resized_height = _resize_pad(cube, 1280)
            tensor = (
                torch.from_numpy(canvas.transpose(2, 0, 1).copy())
                .unsqueeze(0)
                .to(device)
            )
            meta = {
                "filename": str(filename.resolve()),
                "ori_filename": filename.name,
                "ori_shape": (original_height, original_width, 16),
                "img_shape": (resized_height, resized_width, 16),
                "pad_shape": canvas.shape,
                "scale_factor": np.asarray((sx, sy, sx, sy), np.float32),
                "flip": False,
                "flip_direction": None,
                "img_norm_cfg": {
                    "mean": np.zeros(16, np.float32),
                    "std": np.ones(16, np.float32),
                    "to_rgb": False,
                },
            }
            # Run the exact public/native detector entry point and observe the
            # query head with a read-only hook.  This retains BaseDetector's
            # auto_fp16 casting/autocast and all source-locked runtime details.
            captured_query_outputs.clear()
            native_result = model(
                return_loss=False,
                rescale=False,
                img=[tensor],
                img_metas=[[meta]],
            )[0]
            if len(captured_query_outputs) != 1:
                raise RuntimeError(
                    "V6 native inference did not execute exactly one query-head forward"
                )
            outputs = captured_query_outputs[0]
            scores = outputs[0][-1, 0].float().sigmoid().cpu().numpy()
            normalized_boxes = outputs[1][-1, 0].float().cpu().numpy()
            if scores.shape != (NATIVE_PRE_NMS_TOPK, NUM_CLASSES):
                raise RuntimeError(f"Unexpected V6 query score shape: {scores.shape}")
            if normalized_boxes.shape != (NATIVE_PRE_NMS_TOPK, 4):
                raise RuntimeError(
                    f"Unexpected V6 query box shape: {normalized_boxes.shape}"
                )
            boxes = _restore_query_boxes(
                normalized_boxes,
                resized_width=resized_width,
                resized_height=resized_height,
                scale_x=sx,
                scale_y=sy,
                original_width=original_width,
                original_height=original_height,
            )
            restored_native = []
            for class_id, rows in enumerate(native_result):
                value = np.asarray(rows, np.float32).reshape(-1, 5).copy()
                if len(value):
                    value[:, :4] = _restore_boxes(
                        value[:, :4],
                        sx,
                        sy,
                        original_width,
                        original_height,
                        False,
                    )
                    restored_native.append(
                        np.column_stack(
                            (
                                np.full(len(value), class_id, np.float32),
                                value,
                            )
                        )
                    )
            native = (
                np.concatenate(restored_native).astype(np.float32, copy=False)
                if restored_native
                else np.zeros((0, 6), np.float32)
            )
            if len(native) > NATIVE_MAX_PER_IMAGE:
                raise RuntimeError("V6 native inference exceeded global max300")
            dense_native = np.zeros((NATIVE_MAX_PER_IMAGE, 6), np.float32)
            dense_native[: len(native)] = native
            finite = np.isfinite(scores).all() and np.isfinite(boxes).all()
            if not finite or np.any(scores < 0) or np.any(scores > 1):
                raise RuntimeError(f"Non-finite V6 query output for {filename}")
            score_rows.append(np.asarray(scores, np.float32))
            box_rows.append(np.asarray(boxes, np.float32))
            native_rows.append(dense_native)
            native_counts.append(len(native))
            image_ids.append(int(record["image_id"]))
            widths.append(original_width)
            heights.append(original_height)
            del tensor, native_result, outputs
            if (index + 1) % 25 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"[V6 calibration cache] {index + 1}/{len(records)} "
                    f"{elapsed / (index + 1):.2f}s/image",
                    flush=True,
                )
    finally:
        model.query_head.forward = original_query_forward

    _atomic_npz(
        output_path,
        scores=np.stack(score_rows).astype(np.float32, copy=False),
        boxes=np.stack(box_rows).astype(np.float32, copy=False),
        native=np.stack(native_rows).astype(np.float32, copy=False),
        native_counts=np.asarray(native_counts, np.int16),
        image_ids=np.asarray(image_ids, np.int64),
        widths=np.asarray(widths, np.int32),
        heights=np.asarray(heights, np.int32),
    )
    report = {
        "schema_version": 1,
        "kind": "v6_query_cache",
        "path": str(output_path.resolve()),
        "sha256": _sha256(output_path),
        "images": len(records),
        "queries_per_image": NATIVE_PRE_NMS_TOPK,
        "classes": NUM_CLASSES,
        "records": str(records_path.resolve()),
        "records_sha256": _sha256(records_path),
        "records_kind": records_kind,
        "image_root": str(root.resolve()),
        "stats": str(Path(stats_path).resolve()),
        "stats_sha256": _sha256(stats_path),
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": metadata.get("epoch"),
        "checkpoint_attempted_updates": metadata.get("iter"),
        "checkpoint_successful_updates": metadata.get("hod26_v6_successful_updates"),
        "elapsed_seconds": time.perf_counter() - started,
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "source": str(Path(__file__).resolve()),
        "source_sha256": _sha256(__file__),
    }
    atomic_json_dump(report, output_path.with_suffix(output_path.suffix + ".audit.json"))
    del model
    gc.collect()
    return report


def _load_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        result = {name: payload[name] for name in payload.files}
    scores, boxes = result.get("scores"), result.get("boxes")
    native, native_counts = result.get("native"), result.get("native_counts")
    if scores is None or boxes is None or native is None or native_counts is None:
        raise ValueError("V6 calibration cache is missing query or native outputs")
    if scores.ndim != 3 or scores.shape[1:] != (NATIVE_PRE_NMS_TOPK, NUM_CLASSES):
        raise ValueError(f"Invalid V6 cached score shape: {scores.shape}")
    if boxes.shape != (scores.shape[0], NATIVE_PRE_NMS_TOPK, 4):
        raise ValueError(f"Invalid V6 cached box shape: {boxes.shape}")
    if native.shape != (scores.shape[0], NATIVE_MAX_PER_IMAGE, 6):
        raise ValueError(f"Invalid V6 cached native shape: {native.shape}")
    if native_counts.shape != (scores.shape[0],):
        raise ValueError(f"Invalid V6 cached native-count shape: {native_counts.shape}")
    return result


def decode_cached_native(cache: dict[str, np.ndarray]) -> list[list[np.ndarray]]:
    """Read the exact native result captured from the formal detector call."""

    predictions: list[list[np.ndarray]] = []
    for dense, count_value in zip(
        cache["native"], cache["native_counts"], strict=True
    ):
        count = int(count_value)
        rows = dense[:count]
        per_class = []
        for class_id in range(NUM_CLASSES):
            selected = rows[rows[:, 0].astype(np.int64) == class_id, 1:]
            per_class.append(np.asarray(selected, np.float32).reshape(-1, 5).copy())
        predictions.append(per_class)
    return predictions


def match_native_queries(
    cache: dict[str, np.ndarray],
    *,
    score_tolerance: float = 1e-3,
    max_box_error: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover the decoder query behind every retained native detection.

    Soft-NMS changes scores but not coordinates.  The cached native boxes were
    restored by the framework while the query boxes were restored from the
    observed decoder tensor, so normal fp16/fp32 conversion can leave a small
    sub-pixel difference.  Matching is therefore performed jointly per image
    and class, restricted to the original global top-1500 score support, with
    a one-to-one linear assignment and fail-closed geometry/score checks.
    """

    from scipy.optimize import linear_sum_assignment

    if score_tolerance < 0 or max_box_error <= 0:
        raise ValueError("Invalid V6 native-query matching tolerances")
    image_count = int(cache["scores"].shape[0])
    query_indexes = np.full(
        (image_count, NATIVE_MAX_PER_IMAGE), -1, dtype=np.int16
    )
    selected_errors: list[float] = []
    selected_score_deficits: list[float] = []
    ambiguous_rows = 0
    topk_tie_candidates = 0

    for image_index in range(image_count):
        scores = cache["scores"][image_index]
        boxes = cache["boxes"][image_index]
        count = int(cache["native_counts"][image_index])
        native = cache["native"][image_index, :count]
        flat_scores = scores.reshape(-1)
        cutoff = float(
            np.partition(flat_scores, -NATIVE_PRE_NMS_TOPK)[
                -NATIVE_PRE_NMS_TOPK
            ]
        )
        topk_tie_candidates += max(
            0,
            int(np.count_nonzero(flat_scores >= cutoff - 1e-7))
            - NATIVE_PRE_NMS_TOPK,
        )

        for class_id in range(NUM_CLASSES):
            row_indexes = np.flatnonzero(
                native[:, 0].astype(np.int64) == class_id
            )
            if not len(row_indexes):
                continue
            # Include exact cutoff ties rather than depending on an arbitrary
            # CPU top-k tie order. The later assignment remains one-to-one.
            candidates = np.flatnonzero(scores[:, class_id] >= cutoff - 1e-7)
            if len(candidates) < len(row_indexes):
                raise RuntimeError(
                    "V6 native-query mapping has fewer top-k candidates than "
                    f"native detections for image={image_index}, class={class_id}"
                )
            rows = native[row_indexes]
            box_error = np.max(
                np.abs(rows[:, None, 1:5] - boxes[candidates][None, :, :]),
                axis=2,
            ).astype(np.float64)
            pre_scores = scores[candidates, class_id].astype(np.float64)
            post_scores = rows[:, 5].astype(np.float64)
            score_deficit = post_scores[:, None] - pre_scores[None, :]
            invalid_score = score_deficit > score_tolerance

            # Geometry dominates. Score distance is only a deterministic
            # tie-breaker for duplicate decoder boxes; a score that could not
            # have produced the post-Soft-NMS value receives a hard penalty.
            cost = box_error.copy()
            cost += invalid_score.astype(np.float64) * 1e6
            cost += np.abs(score_deficit) * 1e-7
            cost += candidates[None, :].astype(np.float64) * 1e-12
            assigned_rows, assigned_columns = linear_sum_assignment(cost)
            if len(assigned_rows) != len(row_indexes):
                raise RuntimeError("Incomplete V6 native-query linear assignment")
            order = np.argsort(assigned_rows)
            assigned_columns = assigned_columns[order]
            chosen_queries = candidates[assigned_columns]
            chosen_errors = box_error[np.arange(len(row_indexes)), assigned_columns]
            chosen_deficits = score_deficit[
                np.arange(len(row_indexes)), assigned_columns
            ]
            if np.max(chosen_errors, initial=0.0) > max_box_error:
                raise RuntimeError(
                    "V6 native-query geometry gate failed: "
                    f"max error={float(chosen_errors.max()):.6f}px"
                )
            if np.max(chosen_deficits, initial=-np.inf) > score_tolerance:
                raise RuntimeError(
                    "V6 native-query score gate failed: native Soft-NMS score "
                    "exceeds its matched pre-NMS score"
                )
            if len(np.unique(chosen_queries)) != len(chosen_queries):
                raise RuntimeError("V6 native-query mapping reused a class/query pair")

            if box_error.shape[1] > 1:
                two_best = np.partition(box_error, 1, axis=1)[:, :2]
                ambiguous_rows += int(
                    np.sum(np.abs(two_best[:, 1] - two_best[:, 0]) <= 1e-4)
                )
            query_indexes[image_index, row_indexes] = chosen_queries.astype(np.int16)
            selected_errors.extend(chosen_errors.tolist())
            selected_score_deficits.extend(chosen_deficits.tolist())

        if np.any(query_indexes[image_index, :count] < 0):
            raise RuntimeError(f"Unmapped V6 native detection in image {image_index}")

    errors = np.asarray(selected_errors, np.float64)
    deficits = np.asarray(selected_score_deficits, np.float64)
    report = {
        "method": "per-image/per-class top1500-constrained linear assignment",
        "mapped_detections": int(len(errors)),
        "max_box_error_px": float(errors.max(initial=0.0)),
        "p99_box_error_px": float(np.quantile(errors, 0.99)),
        "max_native_minus_query_score": float(deficits.max(initial=-np.inf)),
        "ambiguous_geometry_rows": int(ambiguous_rows),
        "topk_cutoff_tie_candidates": int(topk_tie_candidates),
        "score_tolerance": float(score_tolerance),
        "max_box_error_gate_px": float(max_box_error),
    }
    return query_indexes, report


def native_query_purities(
    cache: dict[str, np.ndarray], query_indexes: np.ndarray
) -> np.ndarray:
    """Return ``p(class)/(p(class)+max_other)`` for retained detections."""

    if query_indexes.shape != (cache["scores"].shape[0], NATIVE_MAX_PER_IMAGE):
        raise ValueError("Invalid V6 native-query index shape")
    purities = np.ones(query_indexes.shape, np.float32)
    for image_index in range(len(query_indexes)):
        count = int(cache["native_counts"][image_index])
        if not count:
            continue
        native = cache["native"][image_index, :count]
        classes = native[:, 0].astype(np.int64)
        queries = query_indexes[image_index, :count].astype(np.int64)
        probabilities = cache["scores"][image_index, queries]
        selected = probabilities[np.arange(count), classes]
        competing = probabilities.copy()
        competing[np.arange(count), classes] = -np.inf
        other = competing.max(axis=1)
        purity = selected / np.clip(selected + other, 1e-8, None)
        if not np.isfinite(purity).all() or np.any(purity <= 0) or np.any(purity > 1):
            raise RuntimeError("Invalid V6 fixed-candidate query purity")
        purities[image_index, :count] = purity.astype(np.float32)
    return purities


def decode_fixed_selected_margin(
    cache: dict[str, np.ndarray],
    exponents: Sequence[float],
    *,
    query_indexes: np.ndarray | None = None,
    purities: np.ndarray | None = None,
) -> list[list[np.ndarray]]:
    """Re-rank only already-retained native detections within each class.

    Boxes, labels and the per-image detection count are immutable.  This avoids
    the test-distribution risk observed when pre-top-k calibration allowed the
    dominant ``car`` class to take additional slots from other classes.
    """

    exponent = np.asarray(exponents, np.float64)
    if exponent.shape != (NUM_CLASSES,) or not np.isfinite(exponent).all():
        raise ValueError("V6 fixed-candidate calibration needs 18 exponents")
    if np.any(exponent < 0) or np.any(exponent > 4):
        raise ValueError("V6 fixed-candidate exponents must be in [0, 4]")
    if query_indexes is None:
        query_indexes, _ = match_native_queries(cache)
    if purities is None:
        purities = native_query_purities(cache, query_indexes)

    predictions: list[list[np.ndarray]] = []
    for image_index in range(cache["scores"].shape[0]):
        count = int(cache["native_counts"][image_index])
        rows = cache["native"][image_index, :count].copy()
        classes = rows[:, 0].astype(np.int64)
        if count and np.any(exponent[classes] != 0):
            factors = np.power(
                purities[image_index, :count].astype(np.float64),
                exponent[classes],
            ).astype(np.float32)
            rows[:, 5] *= factors
        per_class = []
        for class_id in range(NUM_CLASSES):
            selected = rows[classes == class_id, 1:]
            per_class.append(np.asarray(selected, np.float32).reshape(-1, 5))
        predictions.append(per_class)
    return predictions


def decode_native(scores: np.ndarray, boxes: np.ndarray) -> list[list[np.ndarray]]:
    """Reproduce the source-locked V6 native global-top1500/Soft-NMS path."""

    from mmcv.ops import batched_nms

    predictions: list[list[np.ndarray]] = []
    for image_scores, image_boxes in zip(scores, boxes, strict=True):
        flat_scores = torch.from_numpy(np.ascontiguousarray(image_scores.reshape(-1)))
        selected_scores, indexes = flat_scores.topk(NATIVE_PRE_NMS_TOPK)
        labels = indexes.remainder(NUM_CLASSES).long()
        query_indexes = indexes.div(NUM_CLASSES, rounding_mode="floor").long()
        selected_boxes = torch.from_numpy(np.ascontiguousarray(image_boxes))[query_indexes]
        detections, keep = batched_nms(
            selected_boxes,
            selected_scores,
            labels,
            {
                "type": "soft_nms",
                "iou_threshold": NATIVE_NMS_IOU,
                "min_score": NATIVE_NMS_MIN_SCORE,
            },
        )
        detections = detections[:NATIVE_MAX_PER_IMAGE].cpu().numpy().astype(np.float32)
        kept_labels = labels[keep][:NATIVE_MAX_PER_IMAGE].cpu().numpy()
        per_class = []
        for class_id in range(NUM_CLASSES):
            per_class.append(detections[kept_labels == class_id].copy())
        predictions.append(per_class)
    return predictions


def build_expanded_pool(
    scores: np.ndarray,
    boxes: np.ndarray,
    *,
    pre_nms_per_class: int = 100,
    keep_per_class: int = 60,
    nms_iou: float = NATIVE_NMS_IOU,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover strong candidates per class before any cross-class truncation."""

    from mmcv.ops import soft_nms

    if not 1 <= keep_per_class <= pre_nms_per_class <= scores.shape[1]:
        raise ValueError("Invalid expanded-pool top-k contract")
    pool = np.zeros(
        (scores.shape[0], NUM_CLASSES, keep_per_class, 5), dtype=np.float32
    )
    counts = np.zeros((scores.shape[0], NUM_CLASSES), dtype=np.int16)
    started = time.perf_counter()
    for image_index, (image_scores, image_boxes) in enumerate(
        zip(scores, boxes, strict=True)
    ):
        for class_id in range(NUM_CLASSES):
            class_scores = torch.from_numpy(
                np.ascontiguousarray(image_scores[:, class_id])
            )
            selected_scores, indexes = class_scores.topk(pre_nms_per_class)
            detections, _ = soft_nms(
                np.ascontiguousarray(image_boxes[indexes.numpy()], np.float32),
                selected_scores.numpy(),
                iou_threshold=float(nms_iou),
                min_score=NATIVE_NMS_MIN_SCORE,
                method="linear",
                offset=0,
            )
            detections = np.asarray(detections, np.float32)[:keep_per_class]
            count = len(detections)
            pool[image_index, class_id, :count] = detections
            counts[image_index, class_id] = count
        if (image_index + 1) % 50 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[V6 calibration pool] {image_index + 1}/{len(scores)} "
                f"{elapsed / (image_index + 1):.3f}s/image",
                flush=True,
            )
    return pool, counts


def save_expanded_pool(
    path: str | Path,
    pool: np.ndarray,
    counts: np.ndarray,
    *,
    cache_path: str | Path,
    pre_nms_per_class: int,
    keep_per_class: int,
    nms_iou: float,
) -> dict[str, Any]:
    path = Path(path)
    _atomic_npz(path, pool=pool, counts=counts)
    report = {
        "schema_version": 1,
        "kind": "v6_expanded_class_pool",
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "query_cache": str(Path(cache_path).resolve()),
        "query_cache_sha256": _sha256(cache_path),
        "images": int(pool.shape[0]),
        "classes": NUM_CLASSES,
        "pre_nms_per_class": int(pre_nms_per_class),
        "keep_per_class": int(keep_per_class),
        "nms": {
            "type": "soft_nms",
            "method": "linear",
            "iou_threshold": float(nms_iou),
            "min_score": NATIVE_NMS_MIN_SCORE,
        },
        "source": str(Path(__file__).resolve()),
        "source_sha256": _sha256(__file__),
    }
    atomic_json_dump(report, path.with_suffix(path.suffix + ".audit.json"))
    return report


def load_or_build_pool(
    *,
    cache_path: str | Path,
    pool_path: str | Path,
    pre_nms_per_class: int = 100,
    keep_per_class: int = 60,
    nms_iou: float = NATIVE_NMS_IOU,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    cache = _load_cache(cache_path)
    pool_path = Path(pool_path)
    if pool_path.is_file():
        audit_path = pool_path.with_suffix(pool_path.suffix + ".audit.json")
        if not audit_path.is_file():
            raise RuntimeError("V6 expanded pool exists without an audit")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        expected = {
            "query_cache_sha256": _sha256(cache_path),
            "pre_nms_per_class": int(pre_nms_per_class),
            "keep_per_class": int(keep_per_class),
        }
        for name, value in expected.items():
            if audit.get(name) != value:
                raise RuntimeError(f"V6 expanded pool {name} mismatch")
        with np.load(pool_path, allow_pickle=False) as payload:
            pool, counts = payload["pool"], payload["counts"]
    else:
        pool, counts = build_expanded_pool(
            cache["scores"],
            cache["boxes"],
            pre_nms_per_class=pre_nms_per_class,
            keep_per_class=keep_per_class,
            nms_iou=nms_iou,
        )
        save_expanded_pool(
            pool_path,
            pool,
            counts,
            cache_path=cache_path,
            pre_nms_per_class=pre_nms_per_class,
            keep_per_class=keep_per_class,
            nms_iou=nms_iou,
        )
    if pool.shape[:2] != (cache["scores"].shape[0], NUM_CLASSES):
        raise RuntimeError("V6 expanded pool does not match its query cache")
    return cache, pool, counts


def decode_calibrated(
    pool: np.ndarray,
    counts: np.ndarray,
    *,
    factors: Sequence[float],
    caps: int | Sequence[int],
    max_per_image: int = NATIVE_MAX_PER_IMAGE,
) -> list[list[np.ndarray]]:
    """Apply constant per-class calibration before the final global top-k."""

    factor_array = np.asarray(factors, np.float64)
    cap_array = (
        np.full(NUM_CLASSES, int(caps), np.int64)
        if np.isscalar(caps)
        else np.asarray(caps, np.int64)
    )
    if factor_array.shape != (NUM_CLASSES,) or not np.isfinite(factor_array).all():
        raise ValueError("V6 calibration needs 18 finite class factors")
    if np.any(factor_array <= 0) or np.any(factor_array > 1):
        raise ValueError("V6 calibration factors must be in (0, 1]")
    if cap_array.shape != (NUM_CLASSES,) or np.any(cap_array < 1):
        raise ValueError("V6 calibration needs 18 positive class caps")
    predictions: list[list[np.ndarray]] = []
    for image_index in range(pool.shape[0]):
        candidates, labels = [], []
        for class_id in range(NUM_CLASSES):
            count = min(int(counts[image_index, class_id]), int(cap_array[class_id]))
            if not count:
                continue
            value = pool[image_index, class_id, :count].copy()
            value[:, 4] *= np.float32(factor_array[class_id])
            candidates.append(value)
            labels.append(np.full(count, class_id, np.int64))
        if not candidates:
            predictions.append([np.zeros((0, 5), np.float32) for _ in range(NUM_CLASSES)])
            continue
        merged = np.concatenate(candidates)
        merged_labels = np.concatenate(labels)
        order = np.argsort(-merged[:, 4], kind="stable")[: int(max_per_image)]
        merged, merged_labels = merged[order], merged_labels[order]
        predictions.append(
            [merged[merged_labels == class_id].copy() for class_id in range(NUM_CLASSES)]
        )
    return predictions


def _box_iou_against(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.zeros(0, np.float32)
    left = np.maximum(box[:2], boxes[:, :2])
    right = np.minimum(box[2:4], boxes[:, 2:4])
    intersection = np.prod(np.clip(right - left, 0, None), axis=1)
    box_area = max(0.0, float(box[2] - box[0])) * max(
        0.0, float(box[3] - box[1])
    )
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    return intersection / np.clip(box_area + areas - intersection, 1e-12, None)


def decode_hybrid_rescue(
    cache: dict[str, np.ndarray],
    pool: np.ndarray,
    counts: np.ndarray,
    *,
    factors: Sequence[float],
    caps: int | Sequence[int],
    rescue_classes: Sequence[bool] | None = None,
    dedup_iou: float = NATIVE_NMS_IOU,
    max_per_image: int = NATIVE_MAX_PER_IMAGE,
) -> list[list[np.ndarray]]:
    """Preserve native detections, then fill freed class slots from queries.

    Unlike a full re-decode, this path cannot perturb a native high-confidence
    box.  Expanded candidates are admitted only when they do not duplicate a
    retained native/candidate box of the same class.
    """

    factor_array = np.asarray(factors, np.float64)
    cap_array = (
        np.full(NUM_CLASSES, int(caps), np.int64)
        if np.isscalar(caps)
        else np.asarray(caps, np.int64)
    )
    if factor_array.shape != (NUM_CLASSES,) or np.any(factor_array <= 0):
        raise ValueError("V6 hybrid rescue needs 18 positive factors")
    if np.any(factor_array > 1) or cap_array.shape != (NUM_CLASSES,):
        raise ValueError("Invalid V6 hybrid rescue calibration")
    rescue_mask = (
        np.ones(NUM_CLASSES, dtype=bool)
        if rescue_classes is None
        else np.asarray(rescue_classes, dtype=bool)
    )
    if rescue_mask.shape != (NUM_CLASSES,):
        raise ValueError("V6 hybrid rescue needs an 18-class enable mask")
    if not 0 < dedup_iou <= 1:
        raise ValueError("Invalid V6 hybrid rescue dedup IoU")
    native_predictions = decode_cached_native(cache)
    predictions: list[list[np.ndarray]] = []
    for image_index, native_per_class in enumerate(native_predictions):
        calibrated_classes: list[np.ndarray] = []
        candidates, labels = [], []
        for class_id, native in enumerate(native_per_class):
            if not rescue_mask[class_id]:
                value = native.copy()
                if len(value):
                    value[:, 4] *= np.float32(factor_array[class_id])
                    candidates.append(value)
                    labels.append(np.full(len(value), class_id, np.int64))
                calibrated_classes.append(value)
                continue
            cap = int(cap_array[class_id])
            retained = [row.copy() for row in native[:cap]]
            existing = (
                np.asarray(retained, np.float32).reshape(-1, 5)
                if retained
                else np.zeros((0, 5), np.float32)
            )
            for row in pool[
                image_index,
                class_id,
                : int(counts[image_index, class_id]),
            ]:
                if len(retained) >= cap:
                    break
                if len(existing) and np.max(
                    _box_iou_against(row[:4], existing[:, :4]), initial=0.0
                ) >= dedup_iou:
                    continue
                retained.append(row.copy())
                existing = np.asarray(retained, np.float32).reshape(-1, 5)
            value = (
                np.asarray(retained, np.float32).reshape(-1, 5)
                if retained
                else np.zeros((0, 5), np.float32)
            )
            if len(value):
                value = value[np.argsort(-value[:, 4], kind="stable")]
                value[:, 4] *= np.float32(factor_array[class_id])
                candidates.append(value)
                labels.append(np.full(len(value), class_id, np.int64))
            calibrated_classes.append(value)
        if candidates:
            merged = np.concatenate(candidates)
            merged_labels = np.concatenate(labels)
            order = np.argsort(-merged[:, 4], kind="stable")[: int(max_per_image)]
            merged, merged_labels = merged[order], merged_labels[order]
            calibrated_classes = [
                merged[merged_labels == class_id].copy()
                for class_id in range(NUM_CLASSES)
            ]
        predictions.append(calibrated_classes)
    return predictions


def prediction_counts(predictions: Sequence[Sequence[np.ndarray]]) -> list[int]:
    return [
        int(sum(len(image[class_id]) for image in predictions))
        for class_id in range(NUM_CLASSES)
    ]


def _mean_valid(values: np.ndarray) -> float:
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else float("nan")


def evaluate_coco(
    annotation_path: str | Path,
    predictions: Sequence[Sequence[np.ndarray]],
    *,
    image_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Evaluate exactly the COCO mAP tensor without duplicated console tables."""

    from mmdet.datasets.api_wrappers import COCO, COCOeval

    annotation_path = Path(annotation_path)
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotation_path))
    ordered_ids = [int(row["id"]) for row in ground_truth.dataset["images"]]
    if len(ordered_ids) != len(predictions):
        raise ValueError("Prediction/image count mismatch in V6 calibration evaluation")
    category_ids = sorted(int(row["id"]) for row in ground_truth.dataset["categories"])
    if category_ids != list(range(NUM_CLASSES)):
        raise RuntimeError("V6 calibration expects contiguous official category IDs")
    detections = []
    for image_id, per_class in zip(ordered_ids, predictions, strict=True):
        if len(per_class) != NUM_CLASSES:
            raise ValueError("V6 calibration prediction has the wrong class count")
        for class_id, rows in enumerate(per_class):
            for x1, y1, x2, y2, score in np.asarray(rows, np.float32).reshape(-1, 5):
                detections.append(
                    {
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score),
                    }
                )
    target_ids = ordered_ids if image_ids is None else [int(value) for value in image_ids]
    with contextlib.redirect_stdout(io.StringIO()):
        result = ground_truth.loadRes(detections)
        evaluator = COCOeval(ground_truth, result, "bbox")
        evaluator.params.catIds = category_ids
        evaluator.params.imgIds = target_ids
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"]
    iou50 = int(np.argmin(np.abs(evaluator.params.iouThrs - 0.50)))
    iou75 = int(np.argmin(np.abs(evaluator.params.iouThrs - 0.75)))
    metrics: dict[str, Any] = {
        "mAP": _mean_valid(precision[:, :, :, 0, -1]),
        "AP50": _mean_valid(precision[iou50, :, :, 0, -1]),
        "AP75": _mean_valid(precision[iou75, :, :, 0, -1]),
        "APs": _mean_valid(precision[:, :, :, 1, -1]),
        "APm": _mean_valid(precision[:, :, :, 2, -1]),
        "APl": _mean_valid(precision[:, :, :, 3, -1]),
        "per_class": {
            name: _mean_valid(precision[:, :, class_id, 0, -1])
            for class_id, name in enumerate(HOD26_CLASSES)
        },
        "images": len(target_ids),
    }
    return metrics


def _single_class_ap_bundle(
    ground_truth,
    ordered_ids: Sequence[int],
    predictions: Sequence[Sequence[np.ndarray]],
    *,
    class_id: int,
    folds: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Evaluate one class once on full Fold0 and its fixed stability folds."""

    from mmdet.datasets.api_wrappers import COCOeval

    detections = []
    for image_id, per_class in zip(ordered_ids, predictions, strict=True):
        for x1, y1, x2, y2, score in np.asarray(
            per_class[class_id], np.float32
        ).reshape(-1, 5):
            detections.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(class_id),
                    "bbox": [
                        float(x1),
                        float(y1),
                        float(x2 - x1),
                        float(y2 - y1),
                    ],
                    "score": float(score),
                }
            )
    if not detections:
        raise RuntimeError(f"V6 fixed calibration has no class-{class_id} detections")
    with contextlib.redirect_stdout(io.StringIO()):
        result = ground_truth.loadRes(detections)

    def evaluate_ids(image_ids: Sequence[int]) -> float:
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator = COCOeval(ground_truth, result, "bbox")
            evaluator.params.catIds = [int(class_id)]
            evaluator.params.imgIds = [int(value) for value in image_ids]
            evaluator.params.maxDets = [1, 10, 100]
            evaluator.evaluate()
            evaluator.accumulate()
        return _mean_valid(evaluator.eval["precision"][:, :, 0, 0, -1])

    return {
        "full": evaluate_ids(ordered_ids),
        "folds": [evaluate_ids(image_ids) for image_ids in folds],
    }


def _fixed_candidate_invariants(
    native: Sequence[Sequence[np.ndarray]],
    calibrated: Sequence[Sequence[np.ndarray]],
) -> dict[str, Any]:
    if len(native) != len(calibrated):
        raise RuntimeError("V6 fixed calibration changed the image count")
    changed_scores = 0
    total = 0
    for native_image, calibrated_image in zip(native, calibrated, strict=True):
        if len(native_image) != NUM_CLASSES or len(calibrated_image) != NUM_CLASSES:
            raise RuntimeError("V6 fixed calibration changed the class schema")
        for base, candidate in zip(native_image, calibrated_image, strict=True):
            if base.shape != candidate.shape:
                raise RuntimeError("V6 fixed calibration changed a class count")
            if not np.array_equal(base[:, :4], candidate[:, :4]):
                raise RuntimeError("V6 fixed calibration changed a retained box")
            if np.any(candidate[:, 4] > base[:, 4] + 1e-7):
                raise RuntimeError("V6 fixed calibration unexpectedly increased a score")
            changed_scores += int(np.count_nonzero(candidate[:, 4] != base[:, 4]))
            total += len(base)
    native_counts = prediction_counts(native)
    calibrated_counts = prediction_counts(calibrated)
    if native_counts != calibrated_counts:
        raise RuntimeError("V6 fixed calibration changed class allocation")
    return {
        "boxes_bitwise_identical": True,
        "labels_and_class_counts_identical": True,
        "native_class_detection_counts": native_counts,
        "calibrated_class_detection_counts": calibrated_counts,
        "total_detections": int(total),
        "changed_scores": int(changed_scores),
    }


def quantile_factors(
    scores: np.ndarray,
    *,
    quantile: float,
    exponent: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < quantile < 1 or not 0 <= exponent <= 1:
        raise ValueError("Invalid V6 quantile calibration")
    anchors = np.quantile(scores, quantile, axis=(0, 1)).astype(np.float64)
    anchors = np.clip(anchors, 1e-8, None)
    factors = np.power(anchors.min() / anchors, float(exponent))
    factors /= factors.max()
    return factors, anchors


def margin_calibrated_probabilities(
    probabilities: torch.Tensor,
    exponents: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Suppress class-ambiguous query scores without changing any box."""

    if probabilities.shape[-1] != NUM_CLASSES:
        raise ValueError("V6 margin calibration expects 18 class probabilities")
    exponent = torch.as_tensor(
        exponents,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    if exponent.shape != (NUM_CLASSES,) or not torch.isfinite(exponent).all():
        raise ValueError("V6 margin calibration needs 18 finite exponents")
    if torch.any(exponent < 0) or torch.any(exponent > 4):
        raise ValueError("V6 margin exponents must be in [0, 4]")
    top_values, top_indices = probabilities.topk(2, dim=-1)
    class_ids = torch.arange(NUM_CLASSES, device=probabilities.device)
    view_shape = (1,) * (probabilities.ndim - 1) + (NUM_CLASSES,)
    class_ids = class_ids.view(view_shape)
    other = torch.where(
        class_ids == top_indices[..., :1],
        top_values[..., 1:2],
        top_values[..., :1],
    )
    purity = probabilities / (probabilities + other).clamp_min(1e-8)
    return probabilities * purity.pow(exponent.view(view_shape))


@contextlib.contextmanager
def exact_query_margin_calibration(model, exponents: Sequence[float]):
    """Patch only final query logits, then restore the model unconditionally."""

    exponent_values = [float(value) for value in exponents]
    if len(exponent_values) != NUM_CLASSES:
        raise ValueError("V6 exact margin calibration needs 18 exponents")
    original_forward = model.query_head.forward

    def calibrated_forward(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        class_logits = outputs[0].clone()
        probabilities = class_logits[-1].float().sigmoid()
        calibrated = margin_calibrated_probabilities(
            probabilities, exponent_values
        ).clamp(1e-6, 1 - 1e-6)
        class_logits[-1] = torch.logit(calibrated).to(class_logits.dtype)
        return (class_logits,) + tuple(outputs[1:])

    model.query_head.forward = calibrated_forward
    try:
        yield
    finally:
        model.query_head.forward = original_forward


@torch.inference_mode()
def exact_margin_predictions(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    records_path: str | Path,
    records_kind: str,
    image_root: str | Path,
    stats_path: str | Path,
    exponents: Sequence[float],
    device_name: str,
) -> tuple[list[list[np.ndarray]], list[dict[str, Any]], dict[str, Any]]:
    """Run margin calibration through the framework-native postprocessor."""

    from .raw_infer import predict_cube_native

    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    records_path = Path(records_path)
    records = _load_records(records_path, records_kind)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    model, _, metadata = build_model(config_path, checkpoint_path, device)
    low, high = load_band_limits(stats_path)
    root = Path(image_root)
    predictions: list[list[np.ndarray]] = []
    started = time.perf_counter()
    with exact_query_margin_calibration(model, exponents):
        for index, record in enumerate(records):
            filename = root / record["image"]
            cube = _load_cube(filename, low, high)
            if cube.shape[:2] != (int(record["height"]), int(record["width"])):
                raise RuntimeError(f"Image geometry mismatch for {filename}")
            predictions.append(
                predict_cube_native(
                    model,
                    cube,
                    filename=filename,
                    device=device,
                )
            )
            if (index + 1) % 25 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"[V6 exact margin] {index + 1}/{len(records)} "
                    f"{elapsed / (index + 1):.2f}s/image",
                    flush=True,
                )
    report = {
        "schema_version": 1,
        "kind": "v6_exact_query_margin_predictions",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": metadata.get("epoch"),
        "checkpoint_attempted_updates": metadata.get("iter"),
        "checkpoint_successful_updates": metadata.get("hod26_v6_successful_updates"),
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "stats": str(Path(stats_path).resolve()),
        "stats_sha256": _sha256(stats_path),
        "records": str(records_path.resolve()),
        "records_sha256": _sha256(records_path),
        "exponents": [float(value) for value in exponents],
        "protocol": (
            "1280 identity single-view; score=p*(p/(p+max_other))^alpha; "
            "framework-native global top1500 + linear Soft-NMS iou=.75 + max300"
        ),
        "images": len(records),
        "elapsed_seconds": time.perf_counter() - started,
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "source": str(Path(__file__).resolve()),
        "source_sha256": _sha256(__file__),
    }
    del model
    gc.collect()
    return predictions, records, report


def evaluate_exact_margin(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    annotation_path: str | Path,
    image_root: str | Path,
    stats_path: str | Path,
    exponents: Sequence[float],
    output_path: str | Path,
    device_name: str,
) -> dict[str, Any]:
    predictions, _, report = exact_margin_predictions(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        records_path=annotation_path,
        records_kind="annotation",
        image_root=image_root,
        stats_path=stats_path,
        exponents=exponents,
        device_name=device_name,
    )
    report["metrics"] = evaluate_coco(annotation_path, predictions)
    report["class_detection_counts"] = prediction_counts(predictions)
    annotation = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    image_ids = [int(row["id"]) for row in annotation["images"]]
    report["fold_metrics"] = [
        evaluate_coco(
            annotation_path,
            predictions,
            image_ids=[image_id for image_id in image_ids if image_id % 3 == fold],
        )
        for fold in range(3)
    ]
    atomic_json_dump(report, output_path)
    return report


def emit_exact_margin_submission(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    image_root: str | Path,
    stats_path: str | Path,
    exponents: Sequence[float],
    calibration_path: str | Path,
    output_path: str | Path,
    device_name: str,
) -> dict[str, Any]:
    predictions, samples, report = exact_margin_predictions(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        records_path=manifest_path,
        records_kind="manifest",
        image_root=image_root,
        stats_path=stats_path,
        exponents=exponents,
        device_name=device_name,
    )
    rows = []
    per_image_counts = []
    for sample, per_class in zip(samples, predictions, strict=True):
        image_rows = []
        for class_id, detections in enumerate(per_class):
            for x1, y1, x2, y2, score in detections:
                image_rows.append(
                    {
                        "image_id": str(sample["image_id"]),
                        "class_id": class_id,
                        "confidence": float(score),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    }
                )
        per_image_counts.append(len(image_rows))
        rows.extend(image_rows)
    rows.sort(key=lambda row: (int(row["image_id"]), -row["confidence"]))
    output_path = Path(output_path)
    write_submission(rows, output_path, include_id=True)
    audit = validate_submission(
        output_path,
        samples,
        num_classes=NUM_CLASSES,
        include_id=True,
    )
    audit.update(report)
    audit.update(
        architecture="CoSpec-DINO ViT-L V6",
        single_model=True,
        prediction_ensemble=False,
        calibration=str(Path(calibration_path).resolve()),
        calibration_sha256=_sha256(calibration_path),
        max_detections_per_image=max(per_image_counts),
        min_detections_per_image=min(per_image_counts),
    )
    audit_path = output_path.with_suffix(output_path.suffix + ".audit.json")
    audit["audit"] = str(audit_path.resolve())
    atomic_json_dump(audit, audit_path)
    if not audit["valid"]:
        raise RuntimeError(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


def diagnose_fixed_selected_margin(
    *,
    cache_path: str | Path,
    annotation_path: str | Path,
    output_path: str | Path,
    alpha_grid: Sequence[float] = (
        0.0,
        0.10,
        0.20,
        0.35,
        0.50,
        0.70,
        0.90,
        1.20,
        1.60,
        2.20,
        3.00,
        3.50,
        4.00,
    ),
) -> dict[str, Any]:
    """Fit V6 class-specific ranking calibration on immutable detections."""

    from mmdet.datasets.api_wrappers import COCO

    cache_path = Path(cache_path)
    annotation_path = Path(annotation_path)
    output_path = Path(output_path)
    cache = _load_cache(cache_path)
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    ordered_ids = [int(row["id"]) for row in annotation["images"]]
    if ordered_ids != cache["image_ids"].astype(np.int64).tolist():
        raise RuntimeError("V6 fixed calibration cache order differs from Fold0")
    grid = sorted({float(value) for value in alpha_grid})
    if not grid or grid[0] != 0 or any(value < 0 or value > 4 for value in grid):
        raise ValueError("V6 fixed calibration grid must include zero and lie in [0, 4]")

    started = time.perf_counter()
    query_indexes, mapping_report = match_native_queries(cache)
    purities = native_query_purities(cache, query_indexes)
    native_predictions = decode_cached_native(cache)
    zero_predictions = decode_fixed_selected_margin(
        cache,
        [0.0] * NUM_CLASSES,
        query_indexes=query_indexes,
        purities=purities,
    )
    zero_invariants = _fixed_candidate_invariants(native_predictions, zero_predictions)
    if zero_invariants["changed_scores"] != 0:
        raise RuntimeError("V6 alpha=0 failed exact native-score identity")

    folds = [
        [image_id for image_id in ordered_ids if image_id % 3 == fold]
        for fold in range(3)
    ]
    native_metrics = evaluate_coco(annotation_path, native_predictions)
    native_fold_metrics = [
        evaluate_coco(annotation_path, native_predictions, image_ids=image_ids)
        for image_ids in folds
    ]
    gt_counts = np.zeros(NUM_CLASSES, np.int64)
    gt_fold_counts = np.zeros((3, NUM_CLASSES), np.int64)
    gt_image_sets = [set() for _ in range(NUM_CLASSES)]
    gt_fold_image_sets = [
        [set() for _ in range(NUM_CLASSES)] for _ in range(3)
    ]
    image_fold = {image_id: image_id % 3 for image_id in ordered_ids}
    for row in annotation["annotations"]:
        class_id = int(row["category_id"])
        image_id = int(row["image_id"])
        fold = image_fold[image_id]
        gt_counts[class_id] += 1
        gt_fold_counts[fold, class_id] += 1
        gt_image_sets[class_id].add(image_id)
        gt_fold_image_sets[fold][class_id].add(image_id)
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotation_path))

    class_scans = []
    selected_exponents = np.zeros(NUM_CLASSES, np.float64)
    for class_id, class_name in enumerate(HOD26_CLASSES):
        baseline_full = float(native_metrics["per_class"][class_name])
        baseline_folds = [
            float(metrics["per_class"][class_name])
            for metrics in native_fold_metrics
        ]
        candidates = []
        for alpha in grid:
            if alpha == 0:
                bundle = {"full": baseline_full, "folds": baseline_folds}
            else:
                exponents = np.zeros(NUM_CLASSES, np.float64)
                exponents[class_id] = alpha
                predictions = decode_fixed_selected_margin(
                    cache,
                    exponents,
                    query_indexes=query_indexes,
                    purities=purities,
                )
                bundle = _single_class_ap_bundle(
                    ground_truth,
                    ordered_ids,
                    predictions,
                    class_id=class_id,
                    folds=folds,
                )
            full_delta = float(bundle["full"] - baseline_full)
            fold_deltas = [
                float(value - baseline)
                for value, baseline in zip(
                    bundle["folds"], baseline_folds, strict=True
                )
            ]
            finite_fold_deltas = [
                value for value in fold_deltas if math.isfinite(value)
            ]
            positive_folds = int(sum(value > 0 for value in finite_fold_deltas))
            median_fold = float(np.median(finite_fold_deltas))
            worst_fold = float(min(finite_fold_deltas))
            robust_score = float(
                full_delta + 0.25 * median_fold + 0.10 * worst_fold
            )
            candidates.append(
                {
                    "alpha": alpha,
                    "full_ap": float(bundle["full"]),
                    "full_delta_ap": full_delta,
                    "fold_ap": [float(value) for value in bundle["folds"]],
                    "fold_delta_ap": fold_deltas,
                    "positive_folds": positive_folds,
                    "robust_score": robust_score,
                }
            )

        distinct_images = len(gt_image_sets[class_id])
        fold_distinct_images = [
            len(gt_fold_image_sets[fold][class_id]) for fold in range(3)
        ]
        # Independent scenes, not box count, determine calibration support.
        # This explicitly freezes stone_block (54 boxes but only 8 images).
        rare = distinct_images < 24 or min(fold_distinct_images) < 5
        eligible = []
        for index, candidate in enumerate(candidates):
            if candidate["alpha"] == 0 or rare:
                continue
            neighbor_positive = any(
                candidates[neighbor]["full_delta_ap"] > 0
                for neighbor in (index - 1, index + 1)
                if 0 <= neighbor < len(candidates)
            )
            stable = (
                candidate["full_delta_ap"] > 1e-4
                and candidate["positive_folds"] == 3
                and min(candidate["fold_delta_ap"]) > 0
                and neighbor_positive
            )
            if stable:
                eligible.append(candidate)
        selected = (
            max(eligible, key=lambda row: row["robust_score"])
            if eligible
            else candidates[0]
        )
        selected_exponents[class_id] = float(selected["alpha"])
        class_scans.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "ground_truth_count": int(gt_counts[class_id]),
                "fold_ground_truth_counts": gt_fold_counts[:, class_id].tolist(),
                "ground_truth_image_count": int(distinct_images),
                "fold_ground_truth_image_counts": fold_distinct_images,
                "rare_class_frozen": bool(rare),
                "baseline_full_ap": baseline_full,
                "baseline_fold_ap": baseline_folds,
                "selected": selected,
                "candidates": candidates,
            }
        )
        print(
            f"[V6 fixed calibration] {class_name:16s} alpha={selected['alpha']:.2f} "
            f"deltaAP={selected['full_delta_ap']:+.6f} "
            f"folds={selected['positive_folds']}/3",
            flush=True,
        )

    selected_predictions = decode_fixed_selected_margin(
        cache,
        selected_exponents,
        query_indexes=query_indexes,
        purities=purities,
    )
    invariants = _fixed_candidate_invariants(native_predictions, selected_predictions)
    selected_metrics = evaluate_coco(annotation_path, selected_predictions)
    selected_fold_metrics = [
        evaluate_coco(annotation_path, selected_predictions, image_ids=image_ids)
        for image_ids in folds
    ]
    fold_deltas = [
        float(candidate["mAP"] - baseline["mAP"])
        for candidate, baseline in zip(
            selected_fold_metrics, native_fold_metrics, strict=True
        )
    ]
    delta_map = float(selected_metrics["mAP"] - native_metrics["mAP"])
    if delta_map <= 0 or sum(value > 0 for value in fold_deltas) < 2:
        raise RuntimeError(
            "V6 fixed-candidate class calibration failed the combined stability gate"
        )

    cache_audit_path = cache_path.with_suffix(cache_path.suffix + ".audit.json")
    cache_audit = (
        json.loads(cache_audit_path.read_text(encoding="utf-8"))
        if cache_audit_path.is_file()
        else {}
    )
    report = {
        "schema_version": 1,
        "kind": "v6_fixed_native_candidate_class_margin_calibration",
        "checkpoint_lineage": "single V6 e34 EMA checkpoint",
        "checkpoint": cache_audit.get("checkpoint"),
        "checkpoint_sha256": cache_audit.get("checkpoint_sha256"),
        "query_cache": str(cache_path.resolve()),
        "query_cache_sha256": _sha256(cache_path),
        "annotation": str(annotation_path.resolve()),
        "annotation_sha256": _sha256(annotation_path),
        "protocol": (
            "retain exact native e34 boxes, labels and counts; multiply only each "
            "retained score by (p_class/(p_class+max_other))^class_alpha"
        ),
        "alpha_grid": grid,
        "selection_gate": {
            "minimum_full_class_ap_gain": 1e-4,
            "minimum_positive_folds": 3,
            "minimum_worst_fold_class_ap_gain": "strictly positive",
            "neighboring_alpha_must_also_improve_full_ap": True,
            "rare_class_freeze": (
                "distinct GT images <24 or any fold distinct GT images <5"
            ),
        },
        "mapping": mapping_report,
        "native": {
            "metrics": native_metrics,
            "fold_metrics": native_fold_metrics,
            "class_detection_counts": prediction_counts(native_predictions),
        },
        "selected": {
            "exponents": selected_exponents.tolist(),
            "metrics": selected_metrics,
            "fold_metrics": selected_fold_metrics,
            "delta_mAP_vs_native": delta_map,
            "fold_delta_mAP_vs_native": fold_deltas,
            "positive_folds": int(sum(value > 0 for value in fold_deltas)),
        },
        "class_scans": class_scans,
        "invariants": invariants,
        "elapsed_seconds": time.perf_counter() - started,
        "source": str(Path(__file__).resolve()),
        "source_sha256": _sha256(__file__),
    }
    atomic_json_dump(report, output_path)
    print(
        f"[V6 fixed calibration] selected mAP={selected_metrics['mAP']:.8f} "
        f"delta={delta_map:+.8f} folds={fold_deltas}",
        flush=True,
    )
    return report


def emit_fixed_selected_margin_submission(
    *,
    cache_path: str | Path,
    calibration_path: str | Path,
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path,
    stats_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Emit the stability-gated fixed-candidate V6 calibration submission."""

    cache_path = Path(cache_path)
    calibration_path = Path(calibration_path)
    manifest_path = Path(manifest_path)
    checkpoint_path = Path(checkpoint_path)
    cache = _load_cache(cache_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("kind") != "v6_fixed_native_candidate_class_margin_calibration":
        raise RuntimeError("Wrong calibration family for V6 fixed-candidate emit")
    exponents = [float(value) for value in calibration["selected"]["exponents"]]
    if len(exponents) != NUM_CLASSES:
        raise RuntimeError("V6 fixed-candidate calibration has wrong class count")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    expected_ids = [int(row["image_id"]) for row in samples]
    if expected_ids != cache["image_ids"].astype(np.int64).tolist():
        raise RuntimeError("V6 fixed test cache order differs from the manifest")

    checkpoint_sha = _sha256(checkpoint_path)
    if calibration.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("V6 fixed calibration/checkpoint lineage mismatch")
    cache_audit_path = cache_path.with_suffix(cache_path.suffix + ".audit.json")
    if not cache_audit_path.is_file():
        raise RuntimeError("V6 fixed test cache audit is missing")
    cache_audit = json.loads(cache_audit_path.read_text(encoding="utf-8"))
    if cache_audit.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("V6 fixed test cache/checkpoint lineage mismatch")

    query_indexes, mapping_report = match_native_queries(cache)
    purities = native_query_purities(cache, query_indexes)
    native_predictions = decode_cached_native(cache)
    predictions = decode_fixed_selected_margin(
        cache,
        exponents,
        query_indexes=query_indexes,
        purities=purities,
    )
    invariants = _fixed_candidate_invariants(native_predictions, predictions)
    rows = []
    per_image_counts = []
    for sample, per_class in zip(samples, predictions, strict=True):
        image_rows = []
        for class_id, detections in enumerate(per_class):
            for x1, y1, x2, y2, score in detections:
                image_rows.append(
                    {
                        "image_id": str(sample["image_id"]),
                        "class_id": class_id,
                        "confidence": float(score),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    }
                )
        if len(image_rows) > NATIVE_MAX_PER_IMAGE:
            raise RuntimeError("V6 fixed calibration exceeded native max300")
        per_image_counts.append(len(image_rows))
        rows.extend(image_rows)
    rows.sort(key=lambda row: (int(row["image_id"]), -row["confidence"]))
    output_path = Path(output_path)
    write_submission(rows, output_path, include_id=True)
    report = validate_submission(
        output_path,
        samples,
        num_classes=NUM_CLASSES,
        include_id=True,
    )
    report.update(
        architecture="CoSpec-DINO ViT-L V6",
        single_model=True,
        prediction_ensemble=False,
        checkpoint=str(checkpoint_path.resolve()),
        checkpoint_sha256=checkpoint_sha,
        config=str(Path(config_path).resolve()),
        config_sha256=_sha256(config_path),
        stats=str(Path(stats_path).resolve()),
        stats_sha256=_sha256(stats_path),
        query_cache=str(cache_path.resolve()),
        query_cache_sha256=_sha256(cache_path),
        calibration=str(calibration_path.resolve()),
        calibration_sha256=_sha256(calibration_path),
        selected_exponents=exponents,
        mapping=mapping_report,
        invariants=invariants,
        class_detection_counts=prediction_counts(predictions),
        inference_protocol=(
            "V6 e34 native 1280 identity single-view; exact native boxes/labels/"
            "counts; class-specific retained-query margin score re-ranking only"
        ),
        max_detections_per_image=max(per_image_counts),
        min_detections_per_image=min(per_image_counts),
        source=str(Path(__file__).resolve()),
        source_sha256=_sha256(__file__),
    )
    audit_path = output_path.with_suffix(output_path.suffix + ".audit.json")
    report["audit"] = str(audit_path.resolve())
    atomic_json_dump(report, audit_path)
    if not report["valid"]:
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _parse_exponents(value: str) -> list[float]:
    if Path(value).is_file():
        payload = json.loads(Path(value).read_text(encoding="utf-8"))
        values = payload.get("selected", payload).get("exponents")
    else:
        parts = [item.strip() for item in value.split(",") if item.strip()]
        values = [float(parts[0])] * NUM_CLASSES if len(parts) == 1 else parts
    result = [float(item) for item in values]
    if len(result) != NUM_CLASSES:
        raise ValueError("V6 exact margin needs one or 18 exponents")
    return result


def _variant(
    name: str,
    factors: Sequence[float],
    caps: int | Sequence[int],
) -> dict[str, Any]:
    cap_values = (
        [int(caps)] * NUM_CLASSES
        if np.isscalar(caps)
        else [int(value) for value in caps]
    )
    return {
        "name": name,
        "factors": [float(value) for value in factors],
        "caps": cap_values,
    }


def diagnose(
    *,
    cache_path: str | Path,
    pool_path: str | Path,
    annotation_path: str | Path,
    output_path: str | Path,
    pre_nms_per_class: int = 100,
    keep_per_class: int = 60,
) -> dict[str, Any]:
    """Run a low-dimensional, stability-checked Fold0 calibration search."""

    cache, pool, counts = load_or_build_pool(
        cache_path=cache_path,
        pool_path=pool_path,
        pre_nms_per_class=pre_nms_per_class,
        keep_per_class=keep_per_class,
    )
    annotation = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    ordered_ids = [int(row["id"]) for row in annotation["images"]]
    if ordered_ids != cache["image_ids"].astype(np.int64).tolist():
        raise RuntimeError("V6 calibration cache order differs from Fold0 annotations")

    started = time.perf_counter()
    native_predictions = decode_cached_native(cache)
    native_metrics = evaluate_coco(annotation_path, native_predictions)
    native_counts = prediction_counts(native_predictions)
    print(
        f"[V6 calibration] native mAP={native_metrics['mAP']:.8f} "
        f"car={native_counts[5]}",
        flush=True,
    )

    identity = np.ones(NUM_CLASSES, np.float64)
    variants: list[dict[str, Any]] = []
    for cap in (20, 24, 30, 36, 48, 60):
        variants.append(_variant(f"uniform_cap{cap}", identity, cap))
    for cap in (24, 30, 36, 48):
        for car_factor in (0.75, 0.50, 0.33, 0.20, 0.10, 0.05):
            factors = identity.copy()
            factors[5] = car_factor
            variants.append(
                _variant(f"cap{cap}_car{car_factor:g}", factors, cap)
            )
    quantile_anchors: dict[str, list[float]] = {}
    for quantile in (0.98, 0.99, 0.995):
        for exponent in (0.25, 0.50, 0.75, 1.00):
            factors, anchors = quantile_factors(
                cache["scores"], quantile=quantile, exponent=exponent
            )
            quantile_anchors[f"q{quantile:g}"] = anchors.tolist()
            for cap in (24, 30, 36, 48):
                variants.append(
                    _variant(
                        f"q{quantile:g}_a{exponent:g}_cap{cap}",
                        factors,
                        cap,
                    )
                )

    results = []
    for index, candidate in enumerate(variants):
        predictions = decode_calibrated(
            pool,
            counts,
            factors=candidate["factors"],
            caps=candidate["caps"],
        )
        metrics = evaluate_coco(annotation_path, predictions)
        row = {
            **candidate,
            "metrics": metrics,
            "class_detection_counts": prediction_counts(predictions),
            "delta_mAP_vs_native": metrics["mAP"] - native_metrics["mAP"],
        }
        results.append(row)
        print(
            f"[V6 calibration] {index + 1}/{len(variants)} {candidate['name']} "
            f"mAP={metrics['mAP']:.8f} delta={row['delta_mAP_vs_native']:+.8f}",
            flush=True,
        )

    results.sort(key=lambda row: row["metrics"]["mAP"], reverse=True)
    folds = [[image_id for image_id in ordered_ids if image_id % 3 == fold] for fold in range(3)]
    native_fold = [
        evaluate_coco(annotation_path, native_predictions, image_ids=image_ids)
        for image_ids in folds
    ]
    finalists = results[: min(10, len(results))]
    for candidate in finalists:
        predictions = decode_calibrated(
            pool,
            counts,
            factors=candidate["factors"],
            caps=candidate["caps"],
        )
        candidate["fold_metrics"] = [
            evaluate_coco(annotation_path, predictions, image_ids=image_ids)
            for image_ids in folds
        ]
        deltas = [
            value["mAP"] - baseline["mAP"]
            for value, baseline in zip(candidate["fold_metrics"], native_fold, strict=True)
        ]
        candidate["fold_delta_mAP_vs_native"] = deltas
        candidate["positive_folds"] = int(sum(value > 0 for value in deltas))
        candidate["robust_score"] = float(
            candidate["delta_mAP_vs_native"]
            + 0.25 * float(np.median(deltas))
            + 0.10 * min(deltas)
        )

    eligible = [
        candidate
        for candidate in finalists
        if candidate["delta_mAP_vs_native"] > 0
        and candidate.get("positive_folds", 0) >= 2
        and min(candidate["caps"]) >= 20
    ]
    if not eligible:
        raise RuntimeError("No V6 calibration candidate passed the stability gate")
    selected = max(eligible, key=lambda row: row["robust_score"])
    report = {
        "schema_version": 1,
        "kind": "v6_fold0_class_calibration",
        "checkpoint_lineage": "single V6 e34 EMA checkpoint",
        "query_cache": str(Path(cache_path).resolve()),
        "query_cache_sha256": _sha256(cache_path),
        "expanded_pool": str(Path(pool_path).resolve()),
        "expanded_pool_sha256": _sha256(pool_path),
        "annotation": str(Path(annotation_path).resolve()),
        "annotation_sha256": _sha256(annotation_path),
        "native": {
            "metrics": native_metrics,
            "class_detection_counts": native_counts,
            "fold_metrics": native_fold,
        },
        "pool_contract": {
            "pre_nms_per_class": pre_nms_per_class,
            "keep_per_class": keep_per_class,
            "nms_type": "linear Soft-NMS",
            "nms_iou": NATIVE_NMS_IOU,
            "nms_min_score": NATIVE_NMS_MIN_SCORE,
            "global_max_per_image": NATIVE_MAX_PER_IMAGE,
        },
        "quantile_anchors": quantile_anchors,
        "selected": selected,
        "top_candidates": finalists,
        "all_candidates": results,
        "elapsed_seconds": time.perf_counter() - started,
        "source": str(Path(__file__).resolve()),
        "source_sha256": _sha256(__file__),
    }
    atomic_json_dump(report, output_path)
    print(
        f"[V6 calibration] selected={selected['name']} "
        f"mAP={selected['metrics']['mAP']:.8f} "
        f"delta={selected['delta_mAP_vs_native']:+.8f}",
        flush=True,
    )
    return report


def diagnose_hybrid(
    *,
    cache_path: str | Path,
    pool_path: str | Path,
    annotation_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Search the conservative native-preserving rescue calibration."""

    cache, pool, counts = load_or_build_pool(
        cache_path=cache_path,
        pool_path=pool_path,
    )
    annotation = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    ordered_ids = [int(row["id"]) for row in annotation["images"]]
    if ordered_ids != cache["image_ids"].astype(np.int64).tolist():
        raise RuntimeError("V6 hybrid cache order differs from Fold0 annotations")
    native_predictions = decode_cached_native(cache)
    native_metrics = evaluate_coco(annotation_path, native_predictions)
    native_counts = prediction_counts(native_predictions)
    identity = np.ones(NUM_CLASSES, np.float64)

    variants: list[dict[str, Any]] = []
    for noncar_cap in (30, 45, 60):
        for car_cap in (20, 30, 45, 60, 90):
            caps = [noncar_cap] * NUM_CLASSES
            caps[5] = car_cap
            candidate = _variant(
                f"hybrid_noncar{noncar_cap}_car{car_cap}", identity, caps
            )
            candidate.update(decoder="hybrid_rescue", dedup_iou=NATIVE_NMS_IOU)
            variants.append(candidate)

    cap_results = []
    for index, candidate in enumerate(variants):
        predictions = decode_hybrid_rescue(
            cache,
            pool,
            counts,
            factors=candidate["factors"],
            caps=candidate["caps"],
            dedup_iou=candidate["dedup_iou"],
        )
        metrics = evaluate_coco(annotation_path, predictions)
        row = {
            **candidate,
            "metrics": metrics,
            "class_detection_counts": prediction_counts(predictions),
            "delta_mAP_vs_native": metrics["mAP"] - native_metrics["mAP"],
        }
        cap_results.append(row)
        print(
            f"[V6 hybrid] {index + 1}/{len(variants)} {candidate['name']} "
            f"mAP={metrics['mAP']:.8f} delta={row['delta_mAP_vs_native']:+.8f}",
            flush=True,
        )
    cap_results.sort(key=lambda row: row["metrics"]["mAP"], reverse=True)

    # Score-normalization is searched only around the strongest structural
    # allocation, keeping the overall family low-dimensional.
    base_caps = cap_results[0]["caps"]
    calibrated_results = []
    for quantile in (0.98, 0.99, 0.995):
        for exponent in (0.25, 0.50, 0.75, 1.00):
            factors, anchors = quantile_factors(
                cache["scores"], quantile=quantile, exponent=exponent
            )
            candidate = _variant(
                f"hybrid_q{quantile:g}_a{exponent:g}", factors, base_caps
            )
            candidate.update(
                decoder="hybrid_rescue",
                dedup_iou=NATIVE_NMS_IOU,
                quantile_anchors=anchors.tolist(),
            )
            predictions = decode_hybrid_rescue(
                cache,
                pool,
                counts,
                factors=factors,
                caps=base_caps,
                dedup_iou=NATIVE_NMS_IOU,
            )
            metrics = evaluate_coco(annotation_path, predictions)
            row = {
                **candidate,
                "metrics": metrics,
                "class_detection_counts": prediction_counts(predictions),
                "delta_mAP_vs_native": metrics["mAP"] - native_metrics["mAP"],
            }
            calibrated_results.append(row)
            print(
                f"[V6 hybrid] {candidate['name']} mAP={metrics['mAP']:.8f} "
                f"delta={row['delta_mAP_vs_native']:+.8f}",
                flush=True,
            )

    results = sorted(
        cap_results + calibrated_results,
        key=lambda row: row["metrics"]["mAP"],
        reverse=True,
    )
    folds = [[image_id for image_id in ordered_ids if image_id % 3 == fold] for fold in range(3)]
    native_fold = [
        evaluate_coco(annotation_path, native_predictions, image_ids=image_ids)
        for image_ids in folds
    ]
    finalists = results[: min(10, len(results))]
    for candidate in finalists:
        predictions = decode_hybrid_rescue(
            cache,
            pool,
            counts,
            factors=candidate["factors"],
            caps=candidate["caps"],
            dedup_iou=candidate["dedup_iou"],
        )
        candidate["fold_metrics"] = [
            evaluate_coco(annotation_path, predictions, image_ids=image_ids)
            for image_ids in folds
        ]
        deltas = [
            value["mAP"] - baseline["mAP"]
            for value, baseline in zip(candidate["fold_metrics"], native_fold, strict=True)
        ]
        candidate["fold_delta_mAP_vs_native"] = deltas
        candidate["positive_folds"] = int(sum(value > 0 for value in deltas))
        candidate["robust_score"] = float(
            candidate["delta_mAP_vs_native"]
            + 0.25 * float(np.median(deltas))
            + 0.10 * min(deltas)
        )
    eligible = [
        candidate
        for candidate in finalists
        if candidate["delta_mAP_vs_native"] > 0
        and candidate.get("positive_folds", 0) >= 2
    ]
    if not eligible:
        raise RuntimeError("No V6 hybrid candidate passed the stability gate")
    selected = max(eligible, key=lambda row: row["robust_score"])
    report = {
        "schema_version": 1,
        "kind": "v6_fold0_native_preserving_class_calibration",
        "checkpoint_lineage": "single V6 e34 EMA checkpoint",
        "query_cache": str(Path(cache_path).resolve()),
        "query_cache_sha256": _sha256(cache_path),
        "expanded_pool": str(Path(pool_path).resolve()),
        "expanded_pool_sha256": _sha256(pool_path),
        "annotation": str(Path(annotation_path).resolve()),
        "annotation_sha256": _sha256(annotation_path),
        "native": {
            "metrics": native_metrics,
            "class_detection_counts": native_counts,
            "fold_metrics": native_fold,
        },
        "selected": selected,
        "top_candidates": finalists,
        "all_candidates": results,
        "source": str(Path(__file__).resolve()),
        "source_sha256": _sha256(__file__),
    }
    atomic_json_dump(report, output_path)
    print(
        f"[V6 hybrid] selected={selected['name']} "
        f"mAP={selected['metrics']['mAP']:.8f} "
        f"delta={selected['delta_mAP_vs_native']:+.8f}",
        flush=True,
    )
    return report


def emit_submission(
    *,
    cache_path: str | Path,
    pool_path: str | Path,
    calibration_path: str | Path,
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path,
    stats_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    cache, pool, counts = load_or_build_pool(
        cache_path=cache_path,
        pool_path=pool_path,
    )
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    selected = calibration["selected"]
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    samples = manifest["samples"]
    expected_ids = [int(row["image_id"]) for row in samples]
    if expected_ids != cache["image_ids"].astype(np.int64).tolist():
        raise RuntimeError("V6 test query cache order differs from the manifest")
    if selected.get("decoder") == "hybrid_rescue":
        predictions = decode_hybrid_rescue(
            cache,
            pool,
            counts,
            factors=selected["factors"],
            caps=selected["caps"],
            dedup_iou=float(selected["dedup_iou"]),
            rescue_classes=selected.get("rescue_classes"),
        )
    else:
        predictions = decode_calibrated(
            pool,
            counts,
            factors=selected["factors"],
            caps=selected["caps"],
        )
    rows = []
    per_image_counts = []
    for sample, per_class in zip(samples, predictions, strict=True):
        image_rows = []
        for class_id, detections in enumerate(per_class):
            for x1, y1, x2, y2, score in detections:
                image_rows.append(
                    {
                        "image_id": str(sample["image_id"]),
                        "class_id": class_id,
                        "confidence": float(score),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    }
                )
        if len(image_rows) > NATIVE_MAX_PER_IMAGE:
            raise RuntimeError("V6 calibrated inference exceeded global max300")
        per_image_counts.append(len(image_rows))
        rows.extend(image_rows)
    rows.sort(key=lambda row: (int(row["image_id"]), -row["confidence"]))
    output_path = Path(output_path)
    write_submission(rows, output_path, include_id=True)
    report = validate_submission(
        output_path,
        samples,
        num_classes=NUM_CLASSES,
        include_id=True,
    )
    report.update(
        architecture="CoSpec-DINO ViT-L V6",
        single_model=True,
        prediction_ensemble=False,
        checkpoint=str(Path(checkpoint_path).resolve()),
        checkpoint_sha256=_sha256(checkpoint_path),
        config=str(Path(config_path).resolve()),
        config_sha256=_sha256(config_path),
        stats=str(Path(stats_path).resolve()),
        stats_sha256=_sha256(stats_path),
        query_cache=str(Path(cache_path).resolve()),
        query_cache_sha256=_sha256(cache_path),
        expanded_pool=str(Path(pool_path).resolve()),
        expanded_pool_sha256=_sha256(pool_path),
        calibration=str(Path(calibration_path).resolve()),
        calibration_sha256=_sha256(calibration_path),
        selected_calibration=selected,
        inference_protocol=(
            "1280 identity single-view; complete 1500-query/18-class score export; "
            "Fold0-locked constant class calibration; per-class linear Soft-NMS "
            "iou=.75; calibrated global max300"
        ),
        max_detections_per_image=max(per_image_counts),
        min_detections_per_image=min(per_image_counts),
        source=str(Path(__file__).resolve()),
        source_sha256=_sha256(__file__),
    )
    audit_path = output_path.with_suffix(output_path.suffix + ".audit.json")
    report["audit"] = str(audit_path.resolve())
    atomic_json_dump(report, audit_path)
    if not report["valid"]:
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--config", default="configs/v6/co_spec_dino_vitl_fold0.py")
    extract.add_argument("--checkpoint", required=True)
    extract.add_argument("--records", required=True)
    extract.add_argument("--records-kind", choices=("annotation", "manifest"), required=True)
    extract.add_argument("--image-root", required=True)
    extract.add_argument("--stats", default="storage/v6/dataset/fold0_band_stats.json")
    extract.add_argument("--output", required=True)
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--limit", type=int)

    diagnose_parser = subparsers.add_parser("diagnose")
    diagnose_parser.add_argument("--cache", required=True)
    diagnose_parser.add_argument("--pool", required=True)
    diagnose_parser.add_argument("--annotation", required=True)
    diagnose_parser.add_argument("--output", required=True)
    diagnose_parser.add_argument("--pre-nms-per-class", type=int, default=100)
    diagnose_parser.add_argument("--keep-per-class", type=int, default=60)

    hybrid_parser = subparsers.add_parser("diagnose-hybrid")
    hybrid_parser.add_argument("--cache", required=True)
    hybrid_parser.add_argument("--pool", required=True)
    hybrid_parser.add_argument("--annotation", required=True)
    hybrid_parser.add_argument("--output", required=True)

    exact_eval = subparsers.add_parser("exact-eval")
    exact_eval.add_argument("--config", default="configs/v6/co_spec_dino_vitl_fold0.py")
    exact_eval.add_argument("--checkpoint", required=True)
    exact_eval.add_argument("--annotation", required=True)
    exact_eval.add_argument("--image-root", required=True)
    exact_eval.add_argument("--stats", default="storage/v6/dataset/fold0_band_stats.json")
    exact_eval.add_argument("--exponents", required=True)
    exact_eval.add_argument("--output", required=True)
    exact_eval.add_argument("--device", default="cuda:0")

    exact_emit = subparsers.add_parser("exact-emit")
    exact_emit.add_argument("--config", default="configs/v6/co_spec_dino_vitl_fold0.py")
    exact_emit.add_argument("--checkpoint", required=True)
    exact_emit.add_argument("--manifest", required=True)
    exact_emit.add_argument("--image-root", default="data/raw")
    exact_emit.add_argument("--stats", default="storage/v6/dataset/fold0_band_stats.json")
    exact_emit.add_argument("--exponents", required=True)
    exact_emit.add_argument("--calibration", required=True)
    exact_emit.add_argument("--output", required=True)
    exact_emit.add_argument("--device", default="cuda:0")

    fixed_diagnose = subparsers.add_parser("fixed-diagnose")
    fixed_diagnose.add_argument("--cache", required=True)
    fixed_diagnose.add_argument("--annotation", required=True)
    fixed_diagnose.add_argument("--output", required=True)

    fixed_emit = subparsers.add_parser("fixed-emit")
    fixed_emit.add_argument("--cache", required=True)
    fixed_emit.add_argument("--calibration", required=True)
    fixed_emit.add_argument("--manifest", required=True)
    fixed_emit.add_argument("--checkpoint", required=True)
    fixed_emit.add_argument("--config", default="configs/v6/co_spec_dino_vitl_fold0.py")
    fixed_emit.add_argument("--stats", default="storage/v6/dataset/fold0_band_stats.json")
    fixed_emit.add_argument("--output", required=True)

    emit = subparsers.add_parser("emit")
    emit.add_argument("--cache", required=True)
    emit.add_argument("--pool", required=True)
    emit.add_argument("--calibration", required=True)
    emit.add_argument("--manifest", required=True)
    emit.add_argument("--checkpoint", required=True)
    emit.add_argument("--config", default="configs/v6/co_spec_dino_vitl_fold0.py")
    emit.add_argument("--stats", default="storage/v6/dataset/fold0_band_stats.json")
    emit.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "extract":
        report = extract_query_cache(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            records_path=args.records,
            records_kind=args.records_kind,
            image_root=args.image_root,
            stats_path=args.stats,
            output_path=args.output,
            device_name=args.device,
            limit=args.limit,
        )
    elif args.command == "diagnose":
        report = diagnose(
            cache_path=args.cache,
            pool_path=args.pool,
            annotation_path=args.annotation,
            output_path=args.output,
            pre_nms_per_class=args.pre_nms_per_class,
            keep_per_class=args.keep_per_class,
        )
    elif args.command == "diagnose-hybrid":
        report = diagnose_hybrid(
            cache_path=args.cache,
            pool_path=args.pool,
            annotation_path=args.annotation,
            output_path=args.output,
        )
    elif args.command == "exact-eval":
        report = evaluate_exact_margin(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            annotation_path=args.annotation,
            image_root=args.image_root,
            stats_path=args.stats,
            exponents=_parse_exponents(args.exponents),
            output_path=args.output,
            device_name=args.device,
        )
    elif args.command == "exact-emit":
        report = emit_exact_margin_submission(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            image_root=args.image_root,
            stats_path=args.stats,
            exponents=_parse_exponents(args.exponents),
            calibration_path=args.calibration,
            output_path=args.output,
            device_name=args.device,
        )
    elif args.command == "fixed-diagnose":
        report = diagnose_fixed_selected_margin(
            cache_path=args.cache,
            annotation_path=args.annotation,
            output_path=args.output,
        )
    elif args.command == "fixed-emit":
        report = emit_fixed_selected_margin_submission(
            cache_path=args.cache,
            calibration_path=args.calibration,
            manifest_path=args.manifest,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            stats_path=args.stats,
            output_path=args.output,
        )
    else:
        report = emit_submission(
            cache_path=args.cache,
            pool_path=args.pool,
            calibration_path=args.calibration,
            manifest_path=args.manifest,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            stats_path=args.stats,
            output_path=args.output,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
