"""Four-corner sliced inference with validation-locked consistency fusion.

This module keeps the public-winning V4 epoch-30 two-view prediction as an
immutable anchor.  Four overlapping 60% corner crops add higher-resolution
localization evidence.  Validation labels are used only to select a compact,
predeclared family of fusion/calibration settings; test predictions are never
used for selection.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from pycocotools.coco import COCO

from hod26.config import atomic_json_dump
from hod26.data import make_balanced_folds
from hod26.submission import validate_submission, write_submission

from .constants import HOD26_CLASSES
from .data import load_band_limits
from .infer import _invert_boxes, _load_cube, _resize_pad, build_model
from .optimize import (
    CachedView,
    _atomic_savez,
    _base_setting,
    _coco_metrics,
    _iou_one_to_many,
    _load_samples,
    _merge_class,
    _sha256,
    _soft_nms,
)


CHECKPOINT_SHA256 = (
    "2b96f909af97c93c01c2141d032eb391428b18aac76be5e38f4f3ae5f9198932"
)
BASELINE_SETTING = {
    **_base_setting("soft_gaussian", 0.55),
    "sigma": 0.5,
    "max_per_image": 300,
}
SLICE_FRACTION = 0.60
SLICE_WIDTH = 1280
SLICE_NAMES = ("tl", "tr", "bl", "br")


@dataclass(frozen=True, order=True)
class SliceSpec:
    name: str
    width: int = SLICE_WIDTH
    fraction: float = SLICE_FRACTION

    def __post_init__(self) -> None:
        if self.name not in SLICE_NAMES:
            raise ValueError(f"Unknown slice: {self.name}")
        if self.width <= 0 or self.width % 32:
            raise ValueError(f"Slice width must be a positive multiple of 32")
        if not 0.5 < self.fraction < 1.0:
            raise ValueError("Slice fraction must be strictly between 0.5 and 1.0")

    @property
    def key(self) -> str:
        fraction = int(round(self.fraction * 100))
        return f"slice_{self.name}_f{fraction:02d}_w{self.width}"

    def bounds(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        crop_width = min(image_width, max(1, int(round(image_width * self.fraction))))
        crop_height = min(
            image_height, max(1, int(round(image_height * self.fraction)))
        )
        x0 = 0 if self.name[1] == "l" else image_width - crop_width
        y0 = 0 if self.name[0] == "t" else image_height - crop_height
        return x0, y0, x0 + crop_width, y0 + crop_height


ALL_SLICES = tuple(SliceSpec(name) for name in SLICE_NAMES)


@dataclass(frozen=True)
class FusionConfig:
    """Small, deliberately bounded calibration family."""

    name: str
    match_iou: float = 0.45
    match_score: float = 0.001
    refine_weight: float = 0.50
    adaptive_refine: bool = False
    support_slope: float = 0.0
    iou_slope: float = 0.0
    tile_enabled: bool = False
    tile_cluster_iou: float = 0.50
    tile_score_min: float = 0.03
    tile_min_support: int = 2
    tile_score_scale: float = 0.65
    border_margin: float = 0.015
    addition_nms_sigma: float = 0.50
    max_per_image: int = 300

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FusionConfig":
        return cls(**{field: payload[field] for field in cls.__dataclass_fields__})


class SliceCache(CachedView):
    """A regular detection cache with one crop rectangle per source image."""

    def __init__(self, path: str | Path):
        super().__init__(path)
        with np.load(self.path, allow_pickle=False) as payload:
            self.crop_boxes = payload["crop_boxes"].astype(np.float32)
        if self.crop_boxes.shape != (len(self.image_ids), 4):
            raise ValueError(f"Invalid crop_boxes in {self.path}")


def _slice_cache_path(output_dir: Path, spec: SliceSpec) -> Path:
    return output_dir / f"{spec.key}.npz"


def _cache_matches(
    path: Path,
    *,
    checkpoint_sha256: str,
    split: str,
    spec: SliceSpec,
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
                and metadata["view"] == spec.key
                and int(metadata["images"]) == expected_images
                and len(payload["offsets"]) == expected_images + 1
                and payload["crop_boxes"].shape == (expected_images, 4)
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _positive_env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


@torch.inference_mode()
def cache_slices(
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    split: str,
    specs: Sequence[SliceSpec],
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
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("Sliced pipeline is locked to the winning V4 e30 checkpoint")
    if not specs:
        raise ValueError("At least one slice is required")
    samples = _load_samples(
        split,
        test_manifest=Path(test_manifest),
        val_annotations=Path(val_annotations),
    )
    if limit is not None:
        samples = samples[:limit]
    destination = Path(output_dir).resolve()
    outputs = [_slice_cache_path(destination, spec) for spec in specs]
    pending = [
        (spec, path)
        for spec, path in zip(specs, outputs)
        if not _cache_matches(
            path,
            checkpoint_sha256=checkpoint_sha256,
            split=split,
            spec=spec,
            expected_images=len(samples),
        )
    ]
    if not pending:
        print(f"[V4 sliced] reusing {len(outputs)} {split} slice caches", flush=True)
        return outputs

    cpu_threads = min(
        _positive_env_int("V4_SLICE_CPU_THREADS", 8),
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1),
    )
    torch.set_num_threads(cpu_threads)
    try:
        import cv2

        cv2.setNumThreads(_positive_env_int("V4_SLICE_OPENCV_THREADS", 4))
    except ImportError:
        pass
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
    active_specs = [spec for spec, _ in pending]
    accumulators: dict[str, dict[str, list[Any]]] = {
        spec.key: {
            "offsets": [0],
            "boxes": [],
            "scores": [],
            "class_ids": [],
            "crop_boxes": [],
        }
        for spec in active_specs
    }
    root = Path(data_root)
    started = time.perf_counter()
    print(
        f"[V4 sliced:{split}] device={device}, slices="
        f"{','.join(spec.name for spec in active_specs)}, images={len(samples)}",
        flush=True,
    )

    for image_index, sample in enumerate(samples):
        image_path = root / sample["image"]
        cube = _load_cube(image_path, low, high)
        image_height, image_width = cube.shape[:2]
        expected_shape = (int(sample["height"]), int(sample["width"]))
        if (image_height, image_width) != expected_shape:
            raise ValueError(
                f"Manifest/image size mismatch for {sample['image_id']}: "
                f"{(image_height, image_width)} != {expected_shape}"
            )

        for spec in active_specs:
            x0, y0, x1, y1 = spec.bounds(image_width, image_height)
            crop = np.ascontiguousarray(cube[y0:y1, x0:x1])
            crop_height, crop_width = crop.shape[:2]
            canvas, sx, sy, resized_width, resized_height = _resize_pad(
                crop, spec.width
            )
            tensor = (
                torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1)))
                .unsqueeze(0)
                .to(device, non_blocking=True)
            )
            meta = {
                "filename": str(image_path.resolve()),
                "ori_filename": image_path.name,
                "ori_shape": (crop_height, crop_width, 16),
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
                    original_width=crop_width,
                    original_height=crop_height,
                    flip="none",
                )
                detections[:, [0, 2]] += x0
                detections[:, [1, 3]] += y0
                detections[:, [0, 2]] = detections[:, [0, 2]].clip(
                    0, image_width
                )
                detections[:, [1, 3]] = detections[:, [1, 3]].clip(
                    0, image_height
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

            accumulator = accumulators[spec.key]
            accumulator["crop_boxes"].append((x0, y0, x1, y1))
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
                f"[V4 sliced:{split}] {image_index + 1}/{len(samples)}, "
                f"{len(active_specs)} slices, "
                f"{elapsed / (image_index + 1):.2f}s/image",
                flush=True,
            )

    image_ids = np.asarray([sample["image_id"] for sample in samples])
    widths = np.asarray([sample["width"] for sample in samples], dtype=np.int32)
    heights = np.asarray([sample["height"] for sample in samples], dtype=np.int32)
    for spec, path in pending:
        accumulator = accumulators[spec.key]
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
            "kind": "four_corner_slice",
            "split": split,
            "view": spec.key,
            "slice_name": spec.name,
            "slice_fraction": spec.fraction,
            "network_width": spec.width,
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
                "crop_boxes": np.asarray(
                    accumulator["crop_boxes"], dtype=np.float32
                ),
                "metadata_json": np.asarray(
                    json.dumps(metadata, ensure_ascii=False)
                ),
            },
        )
        atomic_json_dump(metadata, path.with_suffix(".audit.json"))
        print(f"[V4 sliced] wrote {path}", flush=True)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs


def _box_is_slice_safe(
    box: np.ndarray,
    crop: np.ndarray,
    image_width: int,
    image_height: int,
    margin_fraction: float,
) -> bool:
    """Reject boxes clipped by an artificial crop edge, not a real image edge."""

    x0, y0, x1, y1 = (float(value) for value in crop)
    margin_x = max(2.0, (x1 - x0) * margin_fraction)
    margin_y = max(2.0, (y1 - y0) * margin_fraction)
    if x0 > 0 and float(box[0]) <= x0 + margin_x:
        return False
    if y0 > 0 and float(box[1]) <= y0 + margin_y:
        return False
    if x1 < image_width and float(box[2]) >= x1 - margin_x:
        return False
    if y1 < image_height and float(box[3]) >= y1 - margin_y:
        return False
    return True


def _pairwise_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float32)
    return np.stack([_iou_one_to_many(box, right) for box in left]).astype(
        np.float32, copy=False
    )


def _greedy_one_to_one(
    base_boxes: np.ndarray,
    candidates: np.ndarray,
    *,
    eligible_base: np.ndarray,
    candidate_mask: np.ndarray,
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], set[int]]:
    candidate_indices = np.flatnonzero(candidate_mask)
    base_indices = np.flatnonzero(eligible_base)
    if not len(candidate_indices) or not len(base_indices):
        return [], set()
    ious = _pairwise_iou(
        base_boxes[base_indices],
        candidates[candidate_indices, :4],
    )
    pair_positions = np.argwhere(ious >= iou_threshold)
    pairs = sorted(
        (
            (
                int(base_indices[left]),
                int(candidate_indices[right]),
                float(ious[left, right]),
            )
            for left, right in pair_positions
        ),
        key=lambda item: (-item[2], item[0], item[1]),
    )
    used_base: set[int] = set()
    used_candidates: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for base_index, candidate_index, iou in pairs:
        if base_index in used_base or candidate_index in used_candidates:
            continue
        used_base.add(base_index)
        used_candidates.add(candidate_index)
        matches.append((base_index, candidate_index, iou))
    return matches, used_candidates


def _cluster_tile_candidates(
    candidates: Sequence[tuple[int, np.ndarray]],
    *,
    iou_threshold: float,
) -> list[list[tuple[int, np.ndarray]]]:
    """Greedy clusters with at most one detection from each crop."""

    ordered = sorted(candidates, key=lambda item: -float(item[1][4]))
    consumed: set[int] = set()
    clusters: list[list[tuple[int, np.ndarray]]] = []
    for seed_index, (seed_tile, seed) in enumerate(ordered):
        if seed_index in consumed:
            continue
        consumed.add(seed_index)
        cluster = [(seed_tile, seed)]
        used_tiles = {seed_tile}
        matches: list[tuple[float, int, int, np.ndarray]] = []
        for index, (tile_id, detection) in enumerate(ordered):
            if index in consumed or tile_id in used_tiles:
                continue
            iou = float(_iou_one_to_many(seed[:4], detection[None, :4])[0])
            if iou >= iou_threshold:
                matches.append((iou, index, tile_id, detection))
        for _, index, tile_id, detection in sorted(
            matches, key=lambda item: (-item[0], item[1])
        ):
            if index in consumed or tile_id in used_tiles:
                continue
            consumed.add(index)
            used_tiles.add(tile_id)
            cluster.append((tile_id, detection))
        clusters.append(cluster)
    return clusters


def _calibrated_score(
    score: float,
    *,
    support_ratio: float | None,
    mean_iou: float | None,
    config: FusionConfig,
) -> float:
    log_scale = 0.0
    if support_ratio is not None:
        log_scale += config.support_slope * (support_ratio - 0.75)
    if mean_iou is not None:
        log_scale += config.iou_slope * (mean_iou - 0.65)
    return float(np.clip(score * math.exp(log_scale), 0.0, 1.0))


def _fuse_class(
    base: np.ndarray,
    tiles: Sequence[np.ndarray],
    crops: Sequence[np.ndarray],
    *,
    image_width: int,
    image_height: int,
    config: FusionConfig,
) -> tuple[np.ndarray, dict[str, int]]:
    if not len(base) and not config.tile_enabled:
        return np.zeros((0, 5), dtype=np.float32), {
            "base": 0,
            "refined": 0,
            "tile_added": 0,
        }

    base = np.asarray(base, dtype=np.float32).copy()
    match_lists: list[list[tuple[np.ndarray, float]]] = [
        [] for _ in range(len(base))
    ]
    eligible_counts = np.zeros(len(base), dtype=np.int16)
    used_by_tile: list[set[int]] = []
    safe_masks: list[np.ndarray] = []

    for tile, crop in zip(tiles, crops):
        safe = np.asarray(
            [
                _box_is_slice_safe(
                    detection[:4],
                    crop,
                    image_width,
                    image_height,
                    config.border_margin,
                )
                for detection in tile
            ],
            dtype=bool,
        )
        safe_masks.append(safe)
        eligible_base = np.asarray(
            [
                _box_is_slice_safe(
                    detection[:4],
                    crop,
                    image_width,
                    image_height,
                    config.border_margin,
                )
                for detection in base
            ],
            dtype=bool,
        )
        eligible_counts += eligible_base.astype(np.int16)
        matches, used = _greedy_one_to_one(
            base[:, :4],
            tile,
            eligible_base=eligible_base,
            candidate_mask=safe & (tile[:, 4] >= config.match_score),
            iou_threshold=config.match_iou,
        )
        used_by_tile.append(used)
        for base_index, tile_index, iou in matches:
            match_lists[base_index].append((tile[tile_index], iou))

    refined = 0
    for index, matches in enumerate(match_lists):
        support = len(matches)
        eligible = int(eligible_counts[index])
        support_ratio = support / eligible if eligible else None
        mean_iou = (
            float(np.mean([item[1] for item in matches])) if matches else None
        )
        original_score = float(base[index, 4])
        base[index, 4] = _calibrated_score(
            original_score,
            support_ratio=support_ratio,
            mean_iou=mean_iou,
            config=config,
        )
        if not matches or config.refine_weight <= 0:
            continue
        member_boxes = np.asarray([item[0][:4] for item in matches], np.float64)
        weights = np.maximum(
            np.asarray([item[0][4] for item in matches], np.float64) ** 2,
            1e-12,
        )
        slice_box = np.average(member_boxes, axis=0, weights=weights)
        effective_weight = config.refine_weight
        if config.adaptive_refine:
            ratio_quality = support_ratio if support_ratio is not None else 0.0
            iou_quality = np.clip(
                (float(mean_iou) - config.match_iou)
                / max(1.0 - config.match_iou, 1e-6),
                0.0,
                1.0,
            )
            # The square root avoids collapsing useful high-resolution
            # evidence while still shrinking uncertain localization votes.
            effective_weight *= math.sqrt(max(ratio_quality * iou_quality, 0.0))
        base[index, :4] = (
            (1.0 - effective_weight) * base[index, :4]
            + effective_weight * slice_box
        )
        refined += 1

    additions: list[list[float]] = []
    if config.tile_enabled:
        candidates: list[tuple[int, np.ndarray]] = []
        for tile_id, (tile, safe, used) in enumerate(
            zip(tiles, safe_masks, used_by_tile)
        ):
            for index, detection in enumerate(tile):
                if (
                    index not in used
                    and safe[index]
                    and float(detection[4]) >= config.tile_score_min
                ):
                    candidates.append((tile_id, detection))
        clusters = _cluster_tile_candidates(
            candidates, iou_threshold=config.tile_cluster_iou
        )
        for cluster in clusters:
            support = len({tile_id for tile_id, _ in cluster})
            if support < config.tile_min_support:
                continue
            boxes = np.asarray([item[1][:4] for item in cluster], np.float64)
            scores = np.asarray([item[1][4] for item in cluster], np.float64)
            weights = np.maximum(scores**2, 1e-12)
            fused_box = np.average(boxes, axis=0, weights=weights)
            eligible = sum(
                _box_is_slice_safe(
                    fused_box,
                    crop,
                    image_width,
                    image_height,
                    config.border_margin,
                )
                for crop in crops
            )
            ratio = min(1.0, support / eligible) if eligible else None
            cluster_ious = [
                float(_iou_one_to_many(fused_box, box[None, :])[0])
                for box in boxes
            ]
            score = _calibrated_score(
                float(scores.max()) * config.tile_score_scale,
                support_ratio=ratio,
                mean_iou=float(np.mean(cluster_ious)),
                config=config,
            )
            if len(base):
                overlaps = _iou_one_to_many(fused_box, base[:, :4])
                # Anchor preservation: additions are decayed against the
                # already validated global predictions, never vice versa.
                score *= float(
                    np.exp(
                        -np.sum(overlaps.astype(np.float64) ** 2)
                        / max(config.addition_nms_sigma, 1e-6)
                    )
                )
            if score >= float(BASELINE_SETTING["output_score"]):
                additions.append([*fused_box.tolist(), score])

    if additions:
        added = _soft_nms(
            np.asarray(additions, dtype=np.float32),
            iou_threshold=0.55,
            sigma=config.addition_nms_sigma,
            min_score=float(BASELINE_SETTING["output_score"]),
            method="gaussian",
        )
        output = np.concatenate((base, added)).astype(np.float32, copy=False)
    else:
        added = np.zeros((0, 5), dtype=np.float32)
        output = base
    if len(output):
        output[:, [0, 2]] = output[:, [0, 2]].clip(0, image_width)
        output[:, [1, 3]] = output[:, [1, 3]].clip(0, image_height)
        valid = (
            np.isfinite(output).all(axis=1)
            & (output[:, 2] > output[:, 0])
            & (output[:, 3] > output[:, 1])
            & (output[:, 4] >= float(BASELINE_SETTING["output_score"]))
        )
        output = output[valid]
        output = output[np.argsort(-output[:, 4])]
    return np.ascontiguousarray(output, dtype=np.float32), {
        "base": len(base),
        "refined": refined,
        "tile_added": len(added),
    }


def fuse_predictions(
    global_caches: Sequence[CachedView],
    slice_caches: Sequence[SliceCache],
    config: FusionConfig,
    *,
    selected_image_ids: set[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if len(global_caches) != 2:
        raise ValueError("Exactly two locked global baseline views are required")
    if len(slice_caches) != 4:
        raise ValueError("Exactly four slice views are required")
    reference_ids = global_caches[0].image_ids.tolist()
    all_caches: Sequence[CachedView] = [*global_caches, *slice_caches]
    if any(cache.image_ids.tolist() != reference_ids for cache in all_caches[1:]):
        raise ValueError("Global and sliced caches have different image order")
    if any(
        cache.metadata.get("checkpoint_sha256") != CHECKPOINT_SHA256
        for cache in all_caches
    ):
        raise ValueError("A cache does not belong to the locked e30 checkpoint")

    rows: list[dict[str, Any]] = []
    diagnostics = {
        "images": 0,
        "base_detections": 0,
        "refined_detections": 0,
        "tile_additions": 0,
    }
    for image_index, image_id_string in enumerate(reference_ids):
        image_id = int(image_id_string)
        if selected_image_ids is not None and image_id not in selected_image_ids:
            continue
        diagnostics["images"] += 1
        width = int(global_caches[0].widths[image_index])
        height = int(global_caches[0].heights[image_index])
        crops = [cache.crop_boxes[image_index] for cache in slice_caches]
        image_rows: list[dict[str, Any]] = []
        for class_id in range(len(HOD26_CLASSES)):
            base = _merge_class(
                [
                    cache.detections(image_index, class_id)
                    for cache in global_caches
                ],
                BASELINE_SETTING,
            )
            fused, stats = _fuse_class(
                base,
                [
                    cache.detections(image_index, class_id)
                    for cache in slice_caches
                ],
                crops,
                image_width=width,
                image_height=height,
                config=config,
            )
            diagnostics["base_detections"] += stats["base"]
            diagnostics["refined_detections"] += stats["refined"]
            diagnostics["tile_additions"] += stats["tile_added"]
            for x1, y1, x2, y2, score in fused:
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
        rows.extend(image_rows[: config.max_per_image])
    rows.sort(key=lambda row: (int(row["image_id"]), -row["confidence"]))
    return rows, diagnostics


def _load_cache_set(
    global_cache_dir: str | Path,
    slice_cache_dir: str | Path,
) -> tuple[list[CachedView], list[SliceCache]]:
    global_root = Path(global_cache_dir)
    slice_root = Path(slice_cache_dir)
    global_caches = [
        CachedView(global_root / "w1280_none.npz"),
        CachedView(global_root / "w1280_horizontal.npz"),
    ]
    slice_caches = [
        SliceCache(_slice_cache_path(slice_root, spec)) for spec in ALL_SLICES
    ]
    return global_caches, slice_caches


def _validation_split(
    *,
    train_manifest: str | Path,
    val_annotations: str | Path,
    seed: int,
) -> dict[str, Any]:
    ground_truth_payload = json.loads(
        Path(val_annotations).read_text(encoding="utf-8")
    )
    validation_ids = {str(image["id"]) for image in ground_truth_payload["images"]}
    manifest = json.loads(Path(train_manifest).read_text(encoding="utf-8"))
    samples = [
        sample for sample in manifest["samples"] if sample["image_id"] in validation_ids
    ]
    if len(samples) != len(validation_ids):
        raise ValueError("Validation IDs do not map one-to-one to train manifest")
    folds = make_balanced_folds(
        samples,
        num_folds=3,
        num_classes=len(HOD26_CLASSES),
        seed=seed,
    )
    gate_fold = min(
        range(3),
        key=lambda index: (
            abs(folds["fold_image_counts"][index] - len(samples) / 3),
            index,
        ),
    )
    gate_ids = {
        int(image_id)
        for image_id, fold in folds["fold_by_image_id"].items()
        if fold == gate_fold
    }
    tune_ids = {int(image_id) for image_id in validation_ids} - gate_ids
    return {
        "method": "near-duplicate-grouped, multilabel-balanced 3-way split",
        "seed": seed,
        "gate_fold": gate_fold,
        "tune_ids": sorted(tune_ids),
        "gate_ids": sorted(gate_ids),
        "fold_image_counts": folds["fold_image_counts"],
        "fold_class_counts": folds["fold_class_counts"],
    }


def _rank(metrics: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics["map_50_95"]),
        float(metrics["map_75"]),
        float(metrics["map_small"]),
        float(metrics["weak4_map_50_95"]),
    )


def search_calibration(
    *,
    global_cache_dir: str | Path,
    slice_cache_dir: str | Path,
    val_annotations: str | Path,
    train_manifest: str | Path,
    output_path: str | Path,
    seed: int = 260730,
) -> dict[str, Any]:
    started = time.perf_counter()
    global_caches, slice_caches = _load_cache_set(
        global_cache_dir, slice_cache_dir
    )
    ground_truth = COCO(str(Path(val_annotations).resolve()))
    split = _validation_split(
        train_manifest=train_manifest,
        val_annotations=val_annotations,
        seed=seed,
    )
    tune_ids = set(split["tune_ids"])
    gate_ids = set(split["gate_ids"])

    baseline = FusionConfig(
        name="locked_global_baseline",
        refine_weight=0.0,
        support_slope=0.0,
        iou_slope=0.0,
        tile_enabled=False,
    )
    baseline_rows, baseline_diagnostics = fuse_predictions(
        global_caches, slice_caches, baseline
    )
    baseline_metrics = {
        "tune": _coco_metrics(ground_truth, baseline_rows, sorted(tune_ids)),
        "gate": _coco_metrics(ground_truth, baseline_rows, sorted(gate_ids)),
        "full": _coco_metrics(ground_truth, baseline_rows),
        "diagnostics": baseline_diagnostics,
    }
    print(
        "[V4 sliced search] locked baseline "
        f"tune={baseline_metrics['tune']['map_50_95']:.6f}, "
        f"gate={baseline_metrics['gate']['map_50_95']:.6f}, "
        f"full={baseline_metrics['full']['map_50_95']:.6f}",
        flush=True,
    )
    if abs(baseline_metrics["full"]["map_50_95"] - 0.7171174791914317) > 1e-9:
        raise RuntimeError("Locked baseline reproduction failed")

    results: list[dict[str, Any]] = []

    def evaluate(config: FusionConfig, stage: str) -> dict[str, Any]:
        rows, diagnostics = fuse_predictions(
            global_caches,
            slice_caches,
            config,
            selected_image_ids=tune_ids,
        )
        metrics = _coco_metrics(ground_truth, rows, sorted(tune_ids))
        result = {
            "stage": stage,
            "config": asdict(config),
            "diagnostics": diagnostics,
            **metrics,
        }
        results.append(result)
        print(
            f"[V4 sliced search] {stage:11s} {config.name:36s} "
            f"mAP={metrics['map_50_95']:.6f} "
            f"AP75={metrics['map_75']:.6f} "
            f"APs={metrics['map_small']:.6f}",
            flush=True,
        )
        return result

    geometry_results = []
    # First resolve the only unresolved scale question from the diagnostic
    # pass: how far a trusted sliced box should move the global anchor.
    for adaptive_refine in (False, True):
        for refine_weight in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
            geometry_results.append(
                evaluate(
                    FusionConfig(
                        name=(
                            f"step_w{refine_weight:.2f}"
                            f"_adaptive{int(adaptive_refine)}"
                        ),
                        match_iou=0.55,
                        match_score=0.01,
                        refine_weight=refine_weight,
                        adaptive_refine=adaptive_refine,
                    ),
                    "step_size",
                )
            )
    best_step = max(geometry_results, key=_rank)
    step_config = FusionConfig.from_dict(best_step["config"])

    threshold_results = []
    for match_iou in (0.45, 0.50, 0.55, 0.60, 0.65):
        for match_score in (0.005, 0.01, 0.02, 0.05):
            threshold_results.append(
                evaluate(
                    replace(
                        step_config,
                        name=(
                            f"gate_i{match_iou:.2f}_s{match_score:g}"
                        ),
                        match_iou=match_iou,
                        match_score=match_score,
                    ),
                    "match_gate",
                )
            )
    best_geometry = max(
        [*geometry_results, *threshold_results],
        key=_rank,
    )
    geometry_config = FusionConfig.from_dict(best_geometry["config"])

    calibration_results = []
    for support_slope in (-0.05, 0.0, 0.05, 0.15):
        for iou_slope in (-0.25, 0.0, 0.25):
            calibration_results.append(
                evaluate(
                    replace(
                        geometry_config,
                        name=(
                            f"cal_sup{support_slope:+.2f}"
                            f"_iou{iou_slope:+.2f}"
                        ),
                        support_slope=support_slope,
                        iou_slope=iou_slope,
                    ),
                    "calibration",
                )
            )
    best_calibration = max(calibration_results, key=_rank)
    calibrated_config = FusionConfig.from_dict(best_calibration["config"])

    addition_results = [best_calibration]
    for minimum, scale in (
        (0.01, 0.40),
        (0.01, 0.70),
        (0.03, 0.40),
        (0.03, 0.70),
        (0.10, 0.40),
        (0.10, 0.70),
    ):
        addition_results.append(
            evaluate(
                replace(
                    calibrated_config,
                    name=f"add2_min{minimum:.2f}_scale{scale:.2f}",
                    tile_enabled=True,
                    tile_score_min=minimum,
                    tile_min_support=2,
                    tile_score_scale=scale,
                ),
                "additions",
            )
        )
    for minimum, scale in ((0.20, 0.40), (0.40, 0.40)):
        addition_results.append(
            evaluate(
                replace(
                    calibrated_config,
                    name=f"add1_min{minimum:.2f}_scale{scale:.2f}",
                    tile_enabled=True,
                    tile_score_min=minimum,
                    tile_min_support=1,
                    tile_score_scale=scale,
                ),
                "additions",
            )
        )

    selected_tune = max(addition_results, key=_rank)
    selected_config = FusionConfig.from_dict(selected_tune["config"])
    gate_rows, gate_diagnostics = fuse_predictions(
        global_caches,
        slice_caches,
        selected_config,
        selected_image_ids=gate_ids,
    )
    gate_metrics = _coco_metrics(ground_truth, gate_rows, sorted(gate_ids))
    full_rows, full_diagnostics = fuse_predictions(
        global_caches, slice_caches, selected_config
    )
    full_metrics = _coco_metrics(ground_truth, full_rows)
    gate_delta = (
        gate_metrics["map_50_95"] - baseline_metrics["gate"]["map_50_95"]
    )
    report = {
        "schema": 1,
        "method": "four 60% corner slices + consistency fusion + calibration",
        "single_model": True,
        "prediction_ensemble": False,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "global_baseline_views": ["w1280_none", "w1280_horizontal"],
        "global_baseline_setting": BASELINE_SETTING,
        "slice_views": [spec.key for spec in ALL_SLICES],
        "split": split,
        "selection_rule": (
            "maximize tune mAP@[.50:.95], then AP75, small AP, weak4 AP; "
            "the original coarse diagnostic observed the gate once before this "
            "fine-step revision, so both passes are retained in the audit"
        ),
        "gate_status": "reused_after_retained_coarse_diagnostic_pass",
        "searched_configurations": len(results),
        "baseline": baseline_metrics,
        "selected": {
            "tune": selected_tune,
            "gate": gate_metrics,
            "gate_diagnostics": gate_diagnostics,
            "full": full_metrics,
            "full_diagnostics": full_diagnostics,
            "gate_delta_map_50_95": gate_delta,
            "recommended_for_submission": gate_delta > 0.0,
        },
        "results": sorted(results, key=_rank, reverse=True),
        "elapsed_seconds": time.perf_counter() - started,
        "test_labels_used": False,
        "kaggle_uploaded": False,
    }
    atomic_json_dump(report, output_path)
    print(
        "[V4 sliced search] SELECTED "
        f"{selected_config.name}: tune={selected_tune['map_50_95']:.6f}, "
        f"gate={gate_metrics['map_50_95']:.6f} "
        f"(delta={gate_delta:+.6f}), full={full_metrics['map_50_95']:.6f}",
        flush=True,
    )
    return report


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _deployment_config(
    calibration: Mapping[str, Any],
    *,
    minimum_complexity_gain: float = 0.0001,
) -> tuple[FusionConfig, dict[str, Any]]:
    """Reject an optional branch whose tune gain is below a noise floor."""

    selected = calibration["selected"]["tune"]
    selected_config = FusionConfig.from_dict(selected["config"])
    calibration_results = [
        result
        for result in calibration["results"]
        if result["stage"] == "calibration"
    ]
    best_core = max(calibration_results, key=_rank)
    addition_gain = float(selected["map_50_95"]) - float(
        best_core["map_50_95"]
    )
    reject_addition = (
        selected_config.tile_enabled
        and addition_gain < minimum_complexity_gain
    )
    deployed = best_core if reject_addition else selected
    decision = {
        "search_selected_name": selected_config.name,
        "deployed_name": deployed["config"]["name"],
        "tile_addition_tune_gain": addition_gain,
        "minimum_complexity_gain": minimum_complexity_gain,
        "tile_addition_rejected_as_noise": reject_addition,
        "rule": (
            "an optional tile-only proposal branch must add at least 0.0001 "
            "tune mAP; high-resolution coordinate fusion is retained"
        ),
    }
    return FusionConfig.from_dict(deployed["config"]), decision


def generate_submission(
    *,
    global_cache_dir: str | Path,
    slice_cache_dir: str | Path,
    calibration_report: str | Path,
    test_manifest: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    report = json.loads(Path(calibration_report).read_text(encoding="utf-8"))
    config, deployment_decision = _deployment_config(report)
    global_caches, slice_caches = _load_cache_set(
        global_cache_dir, slice_cache_dir
    )
    rows, diagnostics = fuse_predictions(global_caches, slice_caches, config)
    output = Path(output_path).resolve()
    write_submission(rows, output, include_id=True)
    manifest = json.loads(Path(test_manifest).read_text(encoding="utf-8"))
    validation = validate_submission(
        output,
        manifest["samples"],
        num_classes=len(HOD26_CLASSES),
        include_id=True,
    )
    if not validation["valid"]:
        raise RuntimeError(json.dumps(validation, ensure_ascii=False, indent=2))
    cache_paths = [
        cache.path.resolve() for cache in [*global_caches, *slice_caches]
    ]
    audit = {
        "schema": 1,
        "method": report["method"],
        "single_model": True,
        "prediction_ensemble": False,
        "checkpoint": str(
            Path(
                "storage/v4/runs/co_dino_vitl_hsi_fold0/"
                "best_bbox_mAP_epoch_30.pth"
            ).resolve()
        ),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "selected_config": asdict(config),
        "validation_search_selection": report["selected"],
        "deployment_decision": deployment_decision,
        "test_diagnostics": diagnostics,
        "prediction_caches": {
            str(path): _file_sha256(path) for path in cache_paths
        },
        "csv": str(output),
        "csv_sha256": _file_sha256(output),
        "csv_validation": validation,
        "elapsed_seconds": time.perf_counter() - started,
        "test_labels_used": False,
        "kaggle_uploaded": False,
    }
    atomic_json_dump(audit, output.with_suffix(".csv.audit.json"))
    print(
        f"[V4 sliced] COMPLETE CSV={output}, rows={len(rows)}, "
        f"sha256={audit['csv_sha256']}, upload=DISABLED",
        flush=True,
    )
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V4 e30 four-slice consistency fusion and calibration"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache")
    cache.add_argument("--checkpoint", required=True)
    cache.add_argument("--checkpoint-sha256", required=True)
    cache.add_argument("--split", choices=("val", "test"), required=True)
    cache.add_argument("--slices", nargs="+", choices=SLICE_NAMES, required=True)
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
    search.add_argument("--global-cache-dir", required=True)
    search.add_argument("--slice-cache-dir", required=True)
    search.add_argument("--val-annotations", required=True)
    search.add_argument("--train-manifest", required=True)
    search.add_argument("--output", required=True)
    search.add_argument("--seed", type=int, default=260730)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--global-cache-dir", required=True)
    generate.add_argument("--slice-cache-dir", required=True)
    generate.add_argument("--calibration-report", required=True)
    generate.add_argument("--test-manifest", required=True)
    generate.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "cache":
        paths = cache_slices(
            checkpoint_path=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            split=args.split,
            specs=[SliceSpec(name) for name in args.slices],
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
        print(json.dumps([str(path) for path in paths], ensure_ascii=False))
    elif args.command == "search":
        report = search_calibration(
            global_cache_dir=args.global_cache_dir,
            slice_cache_dir=args.slice_cache_dir,
            val_annotations=args.val_annotations,
            train_manifest=args.train_manifest,
            output_path=args.output,
            seed=args.seed,
        )
        print(json.dumps(report["selected"], ensure_ascii=False, indent=2))
    elif args.command == "generate":
        audit = generate_submission(
            global_cache_dir=args.global_cache_dir,
            slice_cache_dir=args.slice_cache_dir,
            calibration_report=args.calibration_report,
            test_manifest=args.test_manifest,
            output_path=args.output,
        )
        print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
