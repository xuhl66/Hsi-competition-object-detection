"""Leakage-safe 16-band loading and wavelength-physical V6 augmentation."""

from __future__ import annotations

import json
import os.path as osp
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import mmcv
import numpy as np
from mmdet.datasets import CocoDataset
from mmdet.datasets.builder import DATASETS, PIPELINES

from hod26.data import decode_x2cube

from .constants import HOD26_CLASSES, WAVELENGTHS_NM, WEAK4_IDS


def load_band_limits(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    low = np.asarray(payload["low"], dtype=np.float32)
    high = np.asarray(payload["high"], dtype=np.float32)
    if low.shape != (16,) or high.shape != (16,) or np.any(high <= low):
        raise ValueError(f"Invalid 16-band robust limits: {path}")
    return low, high


def normalize_cube(cube: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    value = cube.astype(np.float32, copy=False)
    value = (value - low.reshape(1, 1, 16)) / (high - low).reshape(1, 1, 16)
    return np.ascontiguousarray(np.clip(value, 0.0, 1.0))


def _load_cube(path: str | Path, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    mosaic = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mosaic is None:
        raise FileNotFoundError(path)
    if mosaic.dtype != np.uint16 or mosaic.ndim != 2:
        raise ValueError(f"Expected single-channel uint16 X2Cube mosaic: {path}")
    return normalize_cube(decode_x2cube(mosaic), low, high)


@PIPELINES.register_module()
class LoadHOD26V6HSIFromFile:
    def __init__(self, stats_file: str) -> None:
        self.stats_file = str(stats_file)
        self.low, self.high = load_band_limits(stats_file)

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        filename = results["img_info"]["filename"]
        if results.get("img_prefix"):
            filename = osp.join(results["img_prefix"], filename)
        cube = _load_cube(filename, self.low, self.high)
        results.update(
            filename=filename,
            ori_filename=results["img_info"]["filename"],
            img=cube,
            img_shape=cube.shape,
            ori_shape=cube.shape,
            pad_shape=cube.shape,
            scale_factor=1.0,
            img_fields=["img"],
            img_norm_cfg=dict(
                mean=np.zeros(16, np.float32),
                std=np.ones(16, np.float32),
                to_rgb=False,
            ),
            hod26_band_low=self.low,
            hod26_band_high=self.high,
        )
        return results


def _resize_pad(
    image: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, float, float, int, int]:
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = mmcv.imresize(
        image, (resized_w, resized_h), interpolation="bilinear", backend="cv2"
    )
    canvas = np.zeros((height, width, 16), dtype=np.float32)
    canvas[:resized_h, :resized_w] = resized
    return canvas, resized_w / source_w, resized_h / source_h, resized_w, resized_h


def _flip_boxes(
    boxes: np.ndarray, width: int, height: int, horizontal: bool, vertical: bool
) -> np.ndarray:
    boxes = boxes.copy()
    if horizontal and len(boxes):
        x1 = boxes[:, 0].copy()
        boxes[:, 0] = width - boxes[:, 2]
        boxes[:, 2] = width - x1
    if vertical and len(boxes):
        y1 = boxes[:, 1].copy()
        boxes[:, 1] = height - boxes[:, 3]
        boxes[:, 3] = height - y1
    return boxes


def wavelength_sensor_augment(
    image: np.ndarray,
    *,
    exposure: float,
    slope: float,
    band_gain: float,
    offset: float,
    shot_noise: float,
    read_noise: float,
    dropout_probability: float,
    blur_probability: float,
) -> np.ndarray:
    """Apply one physically coherent response perturbation to all pixels."""

    wavelengths = np.asarray(WAVELENGTHS_NM, dtype=np.float32)
    axis = (wavelengths - wavelengths.mean()) / np.ptp(wavelengths)
    global_gain = np.float32(np.exp(np.random.uniform(-exposure, exposure)))
    illumination = np.exp(np.random.uniform(-slope, slope) * axis).astype(np.float32)
    knots_nm = np.linspace(wavelengths[0], wavelengths[-1], 5, dtype=np.float32)
    knot_gain = np.random.normal(0.0, band_gain, 5).astype(np.float32)
    smooth_gain = np.interp(wavelengths, knots_nm, knot_gain).astype(np.float32)
    band_offset = np.random.normal(0.0, offset, 16).astype(np.float32)
    augmented = image * (global_gain * illumination + smooth_gain).reshape(1, 1, 16)
    augmented += band_offset.reshape(1, 1, 16)
    if shot_noise > 0:
        sigma = shot_noise * np.sqrt(np.clip(augmented, 0.0, 1.0) + 1e-3)
        augmented += np.random.normal(0.0, 1.0, image.shape).astype(np.float32) * sigma
    if read_noise > 0:
        augmented += np.random.normal(0.0, read_noise, image.shape).astype(np.float32)
    if dropout_probability > 0 and np.random.random() < dropout_probability:
        band = int(np.random.randint(16))
        if band == 0:
            replacement = augmented[..., 1]
        elif band == 15:
            replacement = augmented[..., 14]
        else:
            left = wavelengths[band] - wavelengths[band - 1]
            right = wavelengths[band + 1] - wavelengths[band]
            replacement = (
                right * augmented[..., band - 1] + left * augmented[..., band + 1]
            ) / (left + right)
        augmented[..., band] = replacement
    if blur_probability > 0 and np.random.random() < blur_probability:
        sigma = float(np.random.uniform(0.2, 0.7))
        augmented = cv2.GaussianBlur(augmented, (3, 3), sigmaX=sigma, sigmaY=sigma)
        if augmented.ndim == 2:
            augmented = augmented[..., None]
    return np.ascontiguousarray(np.clip(augmented, 0.0, 1.0), dtype=np.float32)


def _object_crop(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    min_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(boxes):
        return image, boxes, labels
    height, width = image.shape[:2]
    anchor = boxes[int(np.random.randint(len(boxes)))]
    cx, cy = float((anchor[0] + anchor[2]) / 2), float((anchor[1] + anchor[3]) / 2)
    scale = float(np.random.uniform(min_scale, 1.0))
    crop_w = max(32, min(width, int(round(width * scale))))
    crop_h = max(32, min(height, int(round(height * scale))))
    left_lo, left_hi = max(0, int(cx - crop_w)), min(int(cx), width - crop_w)
    top_lo, top_hi = max(0, int(cy - crop_h)), min(int(cy), height - crop_h)
    left = int(np.random.randint(left_lo, left_hi + 1)) if left_hi >= left_lo else 0
    top = int(np.random.randint(top_lo, top_hi + 1)) if top_hi >= top_lo else 0
    centres_x = (boxes[:, 0] + boxes[:, 2]) / 2
    centres_y = (boxes[:, 1] + boxes[:, 3]) / 2
    keep = (
        (centres_x >= left) & (centres_x < left + crop_w)
        & (centres_y >= top) & (centres_y < top + crop_h)
    )
    if not np.any(keep):
        return image, boxes, labels
    target = boxes[keep].copy()
    target[:, 0::2] = np.clip(target[:, 0::2] - left, 0, crop_w)
    target[:, 1::2] = np.clip(target[:, 1::2] - top, 0, crop_h)
    valid = (target[:, 2] - target[:, 0] >= 2) & (target[:, 3] - target[:, 1] >= 2)
    if not np.any(valid):
        return image, boxes, labels
    return (
        np.ascontiguousarray(image[top : top + crop_h, left : left + crop_w]),
        target[valid],
        labels[keep][valid],
    )


def four_scene_mosaic(
    samples: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(samples) != 4:
        raise ValueError("V6 Mosaic needs four distinct official scenes")
    samples = list(samples)
    np.random.shuffle(samples)
    tile_h, tile_w, bands = samples[0][0].shape
    if bands != 16:
        raise ValueError("V6 Mosaic requires native 16-band inputs")
    canvas = np.zeros((tile_h * 2, tile_w * 2, 16), np.float32)
    offsets = ((0, 0), (tile_w, 0), (0, tile_h), (tile_w, tile_h))
    merged_boxes, merged_labels = [], []
    for (image, boxes, labels), (ox, oy) in zip(samples, offsets, strict=True):
        h, w = image.shape[:2]
        canvas[oy : oy + tile_h, ox : ox + tile_w] = mmcv.imresize(
            image, (tile_w, tile_h), interpolation="bilinear", backend="cv2"
        )
        target = np.asarray(boxes, np.float32).copy()
        if len(target):
            target *= np.asarray((tile_w / w, tile_h / h, tile_w / w, tile_h / h))
            target += np.asarray((ox, oy, ox, oy), np.float32)
            merged_boxes.append(target)
            merged_labels.append(np.asarray(labels, np.int64))
    return (
        np.ascontiguousarray(canvas),
        np.ascontiguousarray(np.concatenate(merged_boxes), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(merged_labels), dtype=np.int64),
    )


@PIPELINES.register_module()
class HOD26V6StagedAugment:
    """Four stages locked to Fold0 update-aligned epoch boundaries."""

    STAGES = ("stabilize", "capacity", "clean", "polish")

    def __init__(
        self,
        stage_updates: Sequence[int] = (6000, 56400, 80400),
        widths: Mapping[str, Sequence[int]] | None = None,
        mosaic_probability: float = 0.25,
    ) -> None:
        if tuple(stage_updates) != tuple(sorted(int(x) for x in stage_updates)) or len(stage_updates) != 3:
            raise ValueError("stage_updates must be three ordered boundaries")
        self.stage_updates = tuple(int(x) for x in stage_updates)
        self.mosaic_probability = float(mosaic_probability)
        if not 0 <= self.mosaic_probability <= 1:
            raise ValueError("mosaic_probability must be in [0,1]")
        configured = widths or {
            "stabilize": (896, 1024, 1152),
            "capacity": (896, 960, 1024, 1088, 1152, 1216, 1280),
            "clean": (1152, 1280),
            "polish": (1280,),
        }
        self.widths = {name: tuple(int(v) for v in configured[name]) for name in self.STAGES}
        if any(v <= 0 or v % 32 for values in self.widths.values() for v in values):
            raise ValueError("Every V6 width must be a positive multiple of 32")
        self.epoch = 0
        self.successful_updates = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.successful_updates = self.epoch * 1200

    def set_successful_updates(self, updates: int) -> None:
        self.successful_updates = int(updates)
        self.epoch = self.successful_updates // 1200

    @property
    def stage(self) -> str:
        if self.successful_updates < self.stage_updates[0]:
            return "stabilize"
        if self.successful_updates < self.stage_updates[1]:
            return "capacity"
        if self.successful_updates < self.stage_updates[2]:
            return "clean"
        return "polish"

    def __call__(self, results: dict[str, Any]) -> dict[str, Any] | None:
        stage = self.stage
        image = results["img"]
        boxes = results.get("gt_bboxes", np.zeros((0, 4), np.float32))
        labels = results.get("gt_labels", np.zeros(0, np.int64))
        settings = {
            "stabilize": (0.35, (0.04, 0.025, 0.012, 0.004, 0.003, 0.002, 0.01, 0.04), 0.00, 0.02),
            "capacity": (0.85, (0.14, 0.080, 0.035, 0.012, 0.010, 0.005, 0.06, 0.12), 0.32, 0.08),
            "clean": (0.50, (0.07, 0.040, 0.018, 0.006, 0.005, 0.003, 0.02, 0.06), 0.10, 0.04),
            "polish": (0.16, (0.02, 0.012, 0.006, 0.002, 0.002, 0.001, 0.00, 0.02), 0.00, 0.01),
        }
        sensor_prob, sensor_args, crop_prob, vertical_prob = settings[stage]
        mosaic = False
        if stage == "capacity" and np.random.random() < self.mosaic_probability:
            dataset = results.get("hod26_dataset")
            sample_index = results.get("hod26_sample_idx")
            low, high = results.get("hod26_band_low"), results.get("hod26_band_high")
            if dataset is None or sample_index is None or low is None or high is None:
                raise RuntimeError("V6 Mosaic dataset context is missing")
            samples = [(image, boxes, labels)]
            for index in dataset.sample_mosaic_indices(int(sample_index), 3):
                samples.append(dataset.load_raw_sample(index, low=low, high=high))
            image, boxes, labels = four_scene_mosaic(samples)
            mosaic = True
        if np.random.random() < sensor_prob:
            image = wavelength_sensor_augment(
                image,
                exposure=sensor_args[0], slope=sensor_args[1], band_gain=sensor_args[2],
                offset=sensor_args[3], shot_noise=sensor_args[4], read_noise=sensor_args[5],
                dropout_probability=sensor_args[6], blur_probability=sensor_args[7],
            )
        if not mosaic and np.random.random() < crop_prob:
            image, boxes, labels = _object_crop(image, boxes, labels, 0.62)
        horizontal = bool(np.random.random() < 0.5)
        vertical = bool(np.random.random() < vertical_prob)
        height, width = image.shape[:2]
        if horizontal:
            image = np.ascontiguousarray(image[:, ::-1])
        if vertical:
            image = np.ascontiguousarray(image[::-1])
        boxes = _flip_boxes(boxes, width, height, horizontal, vertical)
        target_width = int(np.random.choice(self.widths[stage]))
        canvas, sx, sy, resized_w, resized_h = _resize_pad(image, target_width, target_width // 2)
        if len(boxes):
            boxes = boxes * np.asarray((sx, sy, sx, sy), np.float32)
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, resized_w)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, resized_h)
            valid = (boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2)
            boxes, labels = boxes[valid], labels[valid]
        if not len(boxes):
            return None
        results.update(
            img=canvas,
            gt_bboxes=np.ascontiguousarray(boxes, dtype=np.float32),
            gt_labels=np.ascontiguousarray(labels, dtype=np.int64),
            img_shape=(resized_h, resized_w, 16),
            pad_shape=canvas.shape,
            scale_factor=np.asarray((sx, sy, sx, sy), np.float32),
            flip=horizontal or vertical,
            flip_direction=("diagonal" if horizontal and vertical else "horizontal" if horizontal else "vertical" if vertical else None),
            hod26_v6_stage=stage,
            hod26_v6_mosaic=mosaic,
            hod26_v6_mixup=False,
        )
        return results


@PIPELINES.register_module()
class HOD26V6ResizePad:
    def __init__(self, width: int = 1280, height: int = 640) -> None:
        if width % 32 or height % 32:
            raise ValueError("V6 canvas must be divisible by 32")
        self.width, self.height = int(width), int(height)

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        canvas, sx, sy, resized_w, resized_h = _resize_pad(
            results["img"], self.width, self.height
        )
        results.update(
            img=canvas,
            img_shape=(resized_h, resized_w, 16),
            pad_shape=canvas.shape,
            scale_factor=np.asarray((sx, sy, sx, sy), np.float32),
        )
        results.setdefault("flip", False)
        results.setdefault("flip_direction", None)
        return results


@PIPELINES.register_module()
class HOD26V6FixedSensorShift:
    """Deterministic promotion-gate shifts; never used for model selection fit."""

    MODES = ("exposure_low", "spectral_slope", "band_response", "noise_blur")

    def __init__(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unknown fixed sensor shift: {mode}")
        self.mode = mode

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        results["img"] = apply_fixed_sensor_shift(
            results["img"], self.mode, image_id=int(results["img_info"].get("id", 0))
        )
        results["hod26_v6_sensor_shift"] = self.mode
        return results


def apply_fixed_sensor_shift(
    image: np.ndarray, mode: str, *, image_id: int
) -> np.ndarray:
    """Apply one immutable sensor shift for promotion diagnostics."""

    if mode not in HOD26V6FixedSensorShift.MODES:
        raise ValueError(f"Unknown fixed sensor shift: {mode}")
    value = image.astype(np.float32, copy=True)
    wavelengths = np.asarray(WAVELENGTHS_NM, np.float32)
    axis = (wavelengths - wavelengths.mean()) / np.ptp(wavelengths)
    if mode == "exposure_low":
        value *= 0.88
    elif mode == "spectral_slope":
        value *= np.exp(0.08 * axis).reshape(1, 1, 16)
    elif mode == "band_response":
        response = 1.0 + 0.025 * np.sin(np.linspace(0, 2 * np.pi, 16))
        value *= response.reshape(1, 1, 16)
    else:
        rng = np.random.RandomState(int(image_id) + 2608)
        value += rng.normal(0.0, 0.004, value.shape).astype(np.float32)
        value = cv2.GaussianBlur(value, (3, 3), 0.35)
    return np.ascontiguousarray(np.clip(value, 0.0, 1.0), np.float32)


@DATASETS.register_module()
class HOD26V6CocoDataset(CocoDataset):
    CLASSES = HOD26_CLASSES

    def __init__(self, *args, max_images: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if max_images is not None:
            limit = int(max_images)
            if limit < 1:
                raise ValueError("max_images must be positive")
            self.data_infos, self.img_ids = self.data_infos[:limit], self.img_ids[:limit]
            if not self.test_mode:
                self._set_group_flag()
        self.base_image_count = len(self.data_infos)

    def prepare_train_img(self, idx: int):
        results = dict(
            img_info=self.data_infos[idx],
            ann_info=self.get_ann_info(idx),
            hod26_dataset=self,
            hod26_sample_idx=int(idx),
        )
        if self.proposals is not None:
            results["proposals"] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def sample_mosaic_indices(self, current_index: int, count: int) -> list[int]:
        selected, used = [], {int(self.img_ids[current_index])}
        for _ in range(max(64, count * 32)):
            if len(selected) == count:
                break
            candidate = int(np.random.randint(len(self)))
            image_id = int(self.img_ids[candidate])
            if image_id not in used:
                used.add(image_id)
                selected.append(candidate)
        if len(selected) != count:
            raise RuntimeError("Not enough distinct V6 Mosaic scenes")
        return selected

    def load_raw_sample(
        self, index: int, *, low: np.ndarray, high: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        info, image_id = self.data_infos[index], self.img_ids[index]
        filename = osp.join(self.img_prefix, info["filename"]) if self.img_prefix else info["filename"]
        image = _load_cube(filename, np.asarray(low), np.asarray(high))
        parsed = self._parse_ann_info(info, self.coco.load_anns(self.coco.get_ann_ids(img_ids=[image_id])))
        return image, parsed["bboxes"], parsed["labels"]

    def evaluate(self, results, *args, **kwargs):
        metrics = super().evaluate(results, *args, **kwargs)
        metric_arg = kwargs.get("metric", args[0] if args else "bbox")
        if "bbox" not in ([metric_arg] if isinstance(metric_arg, str) else metric_arg):
            return metrics
        import tempfile
        from mmdet.datasets.api_wrappers import COCOeval

        with tempfile.TemporaryDirectory() as directory:
            result_files, _ = self.format_results(results, str(Path(directory) / "v6"))
            evaluator = COCOeval(self.coco, self.coco.loadRes(mmcv.load(result_files["bbox"])), "bbox")
            evaluator.params.catIds, evaluator.params.imgIds = self.cat_ids, self.img_ids
            evaluator.params.maxDets = [1, 10, 100]
            evaluator.evaluate(); evaluator.accumulate()
            precision = evaluator.eval["precision"]
            iou75 = int(np.argmin(np.abs(evaluator.params.iouThrs - 0.75)))
            per_class, per_class75 = {}, {}
            for index, name in enumerate(self.CLASSES):
                values = precision[:, :, index, 0, -1]; values = values[values > -1]
                values75 = precision[iou75, :, index, 0, -1]; values75 = values75[values75 > -1]
                per_class[name] = float(values.mean()) if values.size else float("nan")
                per_class75[name] = float(values75.mean()) if values75.size else float("nan")
                metrics[f"bbox_AP_{name}"] = per_class[name]
                metrics[f"bbox_AP75_{name}"] = per_class75[name]
            weak = np.asarray([per_class[self.CLASSES[i]] for i in WEAK4_IDS], np.float64)
            weak75 = np.asarray([per_class75[self.CLASSES[i]] for i in WEAK4_IDS], np.float64)
            strong = np.asarray([v for i, v in enumerate(per_class.values()) if i not in WEAK4_IDS], np.float64)
            metrics["bbox_weak4_mAP"] = float(np.nanmean(weak))
            metrics["bbox_weak4_mAP_75"] = float(np.nanmean(weak75))
            metrics["bbox_strong14_mAP"] = float(np.nanmean(strong))
        return metrics
