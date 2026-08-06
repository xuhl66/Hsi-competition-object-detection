"""MMDetection dataset and staged 16-band augmentation for HOD26 V4."""

from __future__ import annotations

import json
import os.path as osp
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import mmcv
import numpy as np
from mmdet.datasets import CocoDataset
from mmdet.datasets.builder import DATASETS, PIPELINES

from hod26.data import decode_x2cube

from .constants import HOD26_CLASSES, WEAK4_IDS


def load_band_limits(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    low = np.asarray(payload["low"], dtype=np.float32)
    high = np.asarray(payload["high"], dtype=np.float32)
    if low.shape != (16,) or high.shape != (16,):
        raise ValueError(f"Expected 16 band limits in {path}")
    if np.any(high <= low):
        raise ValueError(f"Non-positive band range in {path}")
    return low, high


def normalize_cube(cube: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    cube = cube.astype(np.float32, copy=False)
    cube = (cube - low.reshape(1, 1, 16)) / (high - low).reshape(1, 1, 16)
    return np.ascontiguousarray(np.clip(cube, 0.0, 1.0))


def _load_normalized_cube(
    filename: str | Path,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    mosaic = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
    if mosaic is None:
        raise FileNotFoundError(filename)
    if mosaic.dtype != np.uint16 or mosaic.ndim != 2:
        raise ValueError(
            f"Expected a single-channel uint16 PNG, got "
            f"{mosaic.dtype} {mosaic.shape} at {filename}"
        )
    return normalize_cube(decode_x2cube(mosaic), low, high)


@PIPELINES.register_module()
class LoadHOD26HSIFromFile:
    """Load the uint16 snapshot mosaic and apply the official X2Cube decode."""

    def __init__(self, stats_file: str) -> None:
        self.stats_file = str(stats_file)
        self.low, self.high = load_band_limits(self.stats_file)

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        filename = results["img_info"]["filename"]
        if results.get("img_prefix"):
            filename = osp.join(results["img_prefix"], filename)
        cube = _load_normalized_cube(filename, self.low, self.high)
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
                mean=np.zeros(16, dtype=np.float32),
                std=np.ones(16, dtype=np.float32),
                to_rgb=False,
            ),
            hod26_band_low=self.low,
            hod26_band_high=self.high,
        )
        return results


def _resize_to_canvas(
    image: np.ndarray,
    target_width: int,
    target_height: int,
) -> tuple[np.ndarray, float, float, int, int]:
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = mmcv.imresize(
        image,
        (new_width, new_height),
        interpolation="bilinear",
        backend="cv2",
    )
    canvas = np.zeros((target_height, target_width, image.shape[2]), dtype=np.float32)
    canvas[:new_height, :new_width] = resized
    return canvas, new_width / width, new_height / height, new_width, new_height


def _flip_boxes(
    boxes: np.ndarray, width: int, height: int, horizontal: bool, vertical: bool
) -> np.ndarray:
    boxes = boxes.copy()
    if horizontal and len(boxes):
        old_x1 = boxes[:, 0].copy()
        boxes[:, 0] = width - boxes[:, 2]
        boxes[:, 2] = width - old_x1
    if vertical and len(boxes):
        old_y1 = boxes[:, 1].copy()
        boxes[:, 1] = height - boxes[:, 3]
        boxes[:, 3] = height - old_y1
    return boxes


def _spectral_augment(
    image: np.ndarray,
    *,
    gain: float,
    slope: float,
    noise: float,
    dropout: float,
) -> np.ndarray:
    bands = image.shape[2]
    axis = np.linspace(-1.0, 1.0, bands, dtype=np.float32)
    global_gain = np.float32(1.0 + np.random.uniform(-gain, gain))
    spectral_slope = np.float32(np.random.uniform(-slope, slope))
    knots = np.random.normal(0.0, gain * 0.25, size=5).astype(np.float32)
    smooth = np.interp(
        np.linspace(0.0, 1.0, bands),
        np.linspace(0.0, 1.0, len(knots)),
        knots,
    ).astype(np.float32)
    factors = global_gain + spectral_slope * axis + smooth
    augmented = image * factors.reshape(1, 1, bands)
    if noise > 0:
        augmented += np.random.normal(0.0, noise, size=image.shape).astype(np.float32)
    if dropout > 0 and np.random.random() < dropout:
        band = int(np.random.randint(bands))
        neighbour = max(0, min(bands - 1, band + (-1 if band else 1)))
        augmented[..., band] = augmented[..., neighbour]
    return np.ascontiguousarray(np.clip(augmented, 0.0, 1.0))


def _object_preserving_crop(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    min_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return image, boxes, labels
    height, width = image.shape[:2]
    anchor = boxes[int(np.random.randint(len(boxes)))]
    anchor_x = float((anchor[0] + anchor[2]) * 0.5)
    anchor_y = float((anchor[1] + anchor[3]) * 0.5)
    scale = float(np.random.uniform(min_scale, 1.0))
    crop_width = max(32, min(width, int(round(width * scale))))
    crop_height = max(32, min(height, int(round(height * scale))))
    left_min = max(0, int(np.floor(anchor_x - crop_width)))
    left_max = min(int(np.floor(anchor_x)), width - crop_width)
    top_min = max(0, int(np.floor(anchor_y - crop_height)))
    top_max = min(int(np.floor(anchor_y)), height - crop_height)
    left = (
        int(np.random.randint(left_min, left_max + 1))
        if left_max >= left_min
        else max(0, min(width - crop_width, int(anchor_x - crop_width / 2)))
    )
    top = (
        int(np.random.randint(top_min, top_max + 1))
        if top_max >= top_min
        else max(0, min(height - crop_height, int(anchor_y - crop_height / 2)))
    )
    centres_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
    centres_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
    keep = (
        (centres_x >= left)
        & (centres_x < left + crop_width)
        & (centres_y >= top)
        & (centres_y < top + crop_height)
    )
    if not np.any(keep):
        return image, boxes, labels
    cropped_boxes = boxes[keep].copy()
    cropped_boxes[:, 0::2] -= left
    cropped_boxes[:, 1::2] -= top
    cropped_boxes[:, 0::2] = np.clip(cropped_boxes[:, 0::2], 0, crop_width)
    cropped_boxes[:, 1::2] = np.clip(cropped_boxes[:, 1::2], 0, crop_height)
    valid = (cropped_boxes[:, 2] - cropped_boxes[:, 0] >= 2) & (
        cropped_boxes[:, 3] - cropped_boxes[:, 1] >= 2
    )
    if not np.any(valid):
        return image, boxes, labels
    return (
        np.ascontiguousarray(image[top : top + crop_height, left : left + crop_width]),
        cropped_boxes[valid],
        labels[keep][valid],
    )


def _four_image_mosaic(
    samples: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tile four full 16-band scenes without dropping or interpolating labels."""

    if len(samples) != 4:
        raise ValueError("HOD26 Mosaic requires exactly four samples")
    samples = list(samples)
    np.random.shuffle(samples)
    tile_height, tile_width, bands = samples[0][0].shape
    if bands != 16:
        raise ValueError("HOD26 Mosaic requires native 16-band cubes")
    canvas = np.zeros(
        (tile_height * 2, tile_width * 2, bands),
        dtype=np.float32,
    )
    offsets = (
        (0, 0),
        (tile_width, 0),
        (0, tile_height),
        (tile_width, tile_height),
    )
    merged_boxes: list[np.ndarray] = []
    merged_labels: list[np.ndarray] = []
    for (image, boxes, labels), (offset_x, offset_y) in zip(
        samples, offsets, strict=True
    ):
        height, width = image.shape[:2]
        if image.ndim != 3 or image.shape[2] != bands:
            raise ValueError("Every Mosaic source must have the same band count")
        resized = mmcv.imresize(
            image,
            (tile_width, tile_height),
            interpolation="bilinear",
            backend="cv2",
        )
        canvas[
            offset_y : offset_y + tile_height,
            offset_x : offset_x + tile_width,
        ] = resized
        scaled_boxes = np.asarray(boxes, dtype=np.float32).copy()
        if len(scaled_boxes):
            scaled_boxes *= np.asarray(
                (
                    tile_width / width,
                    tile_height / height,
                    tile_width / width,
                    tile_height / height,
                ),
                dtype=np.float32,
            )
            scaled_boxes += np.asarray(
                (offset_x, offset_y, offset_x, offset_y),
                dtype=np.float32,
            )
            merged_boxes.append(scaled_boxes)
            merged_labels.append(np.asarray(labels, dtype=np.int64))
    if not merged_boxes:
        return (
            np.ascontiguousarray(canvas),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
        )
    return (
        np.ascontiguousarray(canvas),
        np.ascontiguousarray(np.concatenate(merged_boxes), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(merged_labels), dtype=np.int64),
    )


@PIPELINES.register_module()
class HOD26StagedAugment:
    """One continuous four-stage schedule addressed by the current epoch."""

    STAGES = ("stabilize", "strong", "refine", "localize")

    def __init__(
        self,
        stage_epochs: Sequence[int] = (4, 40, 56),
        widths: Mapping[str, Sequence[int]] | None = None,
        mosaic_probability: float = 0.25,
        mixup_probability: float = 0.0,
    ) -> None:
        if len(stage_epochs) != 3 or list(stage_epochs) != sorted(stage_epochs):
            raise ValueError("stage_epochs must be three increasing boundaries")
        self.stage_epochs = tuple(int(value) for value in stage_epochs)
        self.mosaic_probability = float(mosaic_probability)
        self.mixup_probability = float(mixup_probability)
        if not 0.0 <= self.mosaic_probability <= 1.0:
            raise ValueError("mosaic_probability must be in [0, 1]")
        if self.mixup_probability != 0.0:
            raise ValueError(
                "V4 intentionally disables MixUp because hard detection labels "
                "do not represent full-scene spectral mixtures"
            )
        configured = widths or {
            "stabilize": (896, 1024),
            "strong": (896, 960, 1024, 1088, 1152, 1216, 1280),
            "refine": (1024, 1152, 1280),
            "localize": (1152, 1280),
        }
        self.widths = {
            stage: tuple(int(value) for value in configured[stage])
            for stage in self.STAGES
        }
        if any(
            width <= 0 or width % 32
            for values in self.widths.values()
            for width in values
        ):
            raise ValueError("Every training width must be a positive multiple of 32")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def stage(self) -> str:
        if self.epoch < self.stage_epochs[0]:
            return "stabilize"
        if self.epoch < self.stage_epochs[1]:
            return "strong"
        if self.epoch < self.stage_epochs[2]:
            return "refine"
        return "localize"

    def __call__(self, results: dict[str, Any]) -> dict[str, Any] | None:
        stage = self.stage
        image = results["img"]
        boxes = results.get("gt_bboxes", np.zeros((0, 4), dtype=np.float32))
        labels = results.get("gt_labels", np.zeros(0, dtype=np.int64))
        mosaic_applied = False

        if stage == "stabilize":
            spectral_probability, spectral = 0.35, (0.05, 0.025, 0.003, 0.01)
            crop_probability = 0.0
            vertical_probability = 0.04
        elif stage == "strong":
            spectral_probability, spectral = 0.85, (0.14, 0.07, 0.010, 0.08)
            crop_probability = 0.38
            vertical_probability = 0.12
        elif stage == "refine":
            spectral_probability, spectral = 0.65, (0.08, 0.04, 0.005, 0.03)
            crop_probability = 0.16
            vertical_probability = 0.08
        else:
            spectral_probability, spectral = 0.18, (0.025, 0.012, 0.002, 0.0)
            crop_probability = 0.0
            vertical_probability = 0.02

        if (
            stage == "strong"
            and self.mosaic_probability > 0
            and np.random.random() < self.mosaic_probability
        ):
            dataset = results.get("hod26_dataset")
            sample_index = results.get("hod26_sample_idx")
            low = results.get("hod26_band_low")
            high = results.get("hod26_band_high")
            if dataset is None or sample_index is None or low is None or high is None:
                raise RuntimeError(
                    "Mosaic requires HOD26CocoDataset sample context and band limits"
                )
            samples = [(image, boxes, labels)]
            for extra_index in dataset.sample_mosaic_indices(
                int(sample_index), count=3
            ):
                samples.append(
                    dataset.load_raw_sample(
                        extra_index,
                        low=np.asarray(low, dtype=np.float32),
                        high=np.asarray(high, dtype=np.float32),
                    )
                )
            image, boxes, labels = _four_image_mosaic(samples)
            mosaic_applied = True

        if np.random.random() < spectral_probability:
            image = _spectral_augment(
                image,
                gain=spectral[0],
                slope=spectral[1],
                noise=spectral[2],
                dropout=spectral[3],
            )
        if not mosaic_applied and np.random.random() < crop_probability:
            image, boxes, labels = _object_preserving_crop(
                image, boxes, labels, min_scale=0.60
            )

        horizontal = bool(np.random.random() < 0.5)
        vertical = bool(np.random.random() < vertical_probability)
        height, width = image.shape[:2]
        if horizontal:
            image = np.ascontiguousarray(image[:, ::-1])
        if vertical:
            image = np.ascontiguousarray(image[::-1])
        boxes = _flip_boxes(boxes, width, height, horizontal, vertical)

        target_width = int(np.random.choice(self.widths[stage]))
        target_height = target_width // 2
        canvas, sx, sy, resized_width, resized_height = _resize_to_canvas(
            image, target_width, target_height
        )
        if len(boxes):
            boxes = boxes * np.asarray((sx, sy, sx, sy), dtype=np.float32)
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, resized_width)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, resized_height)
            valid = (boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2)
            boxes = boxes[valid]
            labels = labels[valid]
        if len(boxes) == 0:
            return None

        results["img"] = canvas
        results["gt_bboxes"] = np.ascontiguousarray(boxes, dtype=np.float32)
        results["gt_labels"] = np.ascontiguousarray(labels, dtype=np.int64)
        results["img_shape"] = (resized_height, resized_width, 16)
        results["pad_shape"] = canvas.shape
        results["scale_factor"] = np.asarray((sx, sy, sx, sy), dtype=np.float32)
        results["flip"] = horizontal or vertical
        results["flip_direction"] = (
            "diagonal"
            if horizontal and vertical
            else "horizontal" if horizontal else "vertical" if vertical else None
        )
        results["hod26_stage"] = stage
        results["hod26_mosaic"] = mosaic_applied
        results["hod26_mixup"] = False
        return results


@PIPELINES.register_module()
class HOD26ResizePad:
    """Deterministic keep-ratio resize/pad used by validation and inference."""

    def __init__(self, width: int = 1280, height: int = 640) -> None:
        if width % 32 or height % 32:
            raise ValueError("Validation canvas must be divisible by 32")
        self.width = int(width)
        self.height = int(height)

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        image = results["img"]
        canvas, sx, sy, resized_width, resized_height = _resize_to_canvas(
            image, self.width, self.height
        )
        results["img"] = canvas
        results["img_shape"] = (resized_height, resized_width, 16)
        results["pad_shape"] = canvas.shape
        results["scale_factor"] = np.asarray((sx, sy, sx, sy), dtype=np.float32)
        results.setdefault("flip", False)
        results.setdefault("flip_direction", None)
        return results


@DATASETS.register_module()
class HOD26CocoDataset(CocoDataset):
    """18-class COCO view with bounded weak-class image repetition."""

    CLASSES = HOD26_CLASSES

    def __init__(
        self,
        *args,
        repeat_class_factors: Mapping[int, int] | None = None,
        max_images: int | None = None,
        **kwargs,
    ) -> None:
        self.repeat_class_factors = {
            int(class_id): int(factor)
            for class_id, factor in (repeat_class_factors or {}).items()
        }
        if any(
            class_id < 0 or class_id >= len(self.CLASSES)
            for class_id in self.repeat_class_factors
        ):
            raise ValueError("repeat_class_factors contains an invalid class id")
        if any(factor < 1 for factor in self.repeat_class_factors.values()):
            raise ValueError("repeat_class_factors values must be >= 1")
        super().__init__(*args, **kwargs)
        if max_images is not None:
            max_images = int(max_images)
            if max_images < 1:
                raise ValueError("max_images must be positive")
            self.data_infos = self.data_infos[:max_images]
            self.img_ids = self.img_ids[:max_images]
            if not self.test_mode:
                self._set_group_flag()
        self.base_image_count = len(self.data_infos)
        self.repeat_histogram: dict[int, int] = {1: self.base_image_count}
        if self.repeat_class_factors and not self.test_mode:
            original_infos = list(self.data_infos)
            original_ids = list(self.img_ids)
            repeated_infos: list[dict[str, Any]] = []
            repeated_ids: list[int] = []
            histogram: Counter[int] = Counter()
            for info, image_id in zip(original_infos, original_ids):
                ann_ids = self.coco.get_ann_ids(img_ids=[image_id])
                categories = {
                    int(ann["category_id"])
                    for ann in self.coco.load_anns(ann_ids)
                    if not ann.get("ignore", False)
                }
                factor = max(
                    (
                        self.repeat_class_factors.get(category_id, 1)
                        for category_id in categories
                    ),
                    default=1,
                )
                histogram[factor] += 1
                repeated_infos.extend([info] * factor)
                repeated_ids.extend([image_id] * factor)
            self.data_infos = repeated_infos
            self.img_ids = repeated_ids
            self.repeat_histogram = dict(histogram)
            self._set_group_flag()

    def prepare_train_img(self, idx: int):
        """Attach bounded dataset context used only by the Mosaic transform."""

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(
            img_info=img_info,
            ann_info=ann_info,
            hod26_dataset=self,
            hod26_sample_idx=int(idx),
        )
        if self.proposals is not None:
            results["proposals"] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def sample_mosaic_indices(self, current_index: int, count: int) -> list[int]:
        """Draw repeat-weighted sources while avoiding duplicate scenes."""

        if count < 0:
            raise ValueError("Mosaic source count must be non-negative")
        if not 0 <= current_index < len(self):
            raise IndexError(current_index)
        selected: list[int] = []
        used_image_ids = {int(self.img_ids[current_index])}
        max_attempts = max(64, count * 32)
        for _ in range(max_attempts):
            if len(selected) >= count:
                break
            candidate = int(np.random.randint(len(self)))
            image_id = int(self.img_ids[candidate])
            if image_id in used_image_ids:
                continue
            used_image_ids.add(image_id)
            selected.append(candidate)
        if len(selected) < count:
            # This only matters for tiny engineering smoke datasets. Formal
            # Fold0 has thousands of unique scenes and never reaches fallback.
            candidates = [
                index
                for index, image_id in enumerate(self.img_ids)
                if int(image_id) not in used_image_ids
            ]
            for candidate in candidates:
                selected.append(candidate)
                used_image_ids.add(int(self.img_ids[candidate]))
                if len(selected) >= count:
                    break
        if len(selected) != count:
            raise RuntimeError(
                f"Need {count + 1} distinct scenes for Mosaic, "
                f"dataset provides only {len(used_image_ids)}"
            )
        return selected

    def load_raw_sample(
        self,
        index: int,
        *,
        low: np.ndarray,
        high: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load one additional training scene without recursively augmenting it."""

        if not 0 <= index < len(self):
            raise IndexError(index)
        info = self.data_infos[index]
        image_id = self.img_ids[index]
        filename = info["filename"]
        if self.img_prefix:
            filename = osp.join(self.img_prefix, filename)
        image = _load_normalized_cube(filename, low, high)
        ann_ids = self.coco.get_ann_ids(img_ids=[image_id])
        annotations = self.coco.load_anns(ann_ids)
        parsed = self._parse_ann_info(info, annotations)
        return image, parsed["bboxes"], parsed["labels"]

    def evaluate(self, results, *args, **kwargs):
        metrics = super().evaluate(results, *args, **kwargs)
        metric_arg = kwargs.get("metric", args[0] if args else "bbox")
        if "bbox" not in ([metric_arg] if isinstance(metric_arg, str) else metric_arg):
            return metrics

        import tempfile

        from mmdet.datasets.api_wrappers import COCOeval

        with tempfile.TemporaryDirectory() as tmp:
            prefix = str(Path(tmp) / "hod26")
            result_files, _ = self.format_results(results, prefix)
            predictions = mmcv.load(result_files["bbox"])
            coco_det = self.coco.loadRes(predictions)
            evaluator = COCOeval(self.coco, coco_det, "bbox")
            evaluator.params.catIds = self.cat_ids
            evaluator.params.imgIds = self.img_ids
            evaluator.params.maxDets = [1, 10, 100]
            evaluator.evaluate()
            evaluator.accumulate()
            precision = evaluator.eval["precision"]
            per_class = {}
            per_class_75 = {}
            iou75 = int(np.argmin(np.abs(evaluator.params.iouThrs - 0.75)))
            for index, name in enumerate(self.CLASSES):
                values = precision[:, :, index, 0, -1]
                values = values[values > -1]
                values75 = precision[iou75, :, index, 0, -1]
                values75 = values75[values75 > -1]
                per_class[name] = float(values.mean()) if values.size else float("nan")
                per_class_75[name] = (
                    float(values75.mean()) if values75.size else float("nan")
                )
            weak = [per_class[self.CLASSES[index]] for index in WEAK4_IDS]
            weak75 = [per_class_75[self.CLASSES[index]] for index in WEAK4_IDS]
            finite_weak = np.asarray(weak, dtype=np.float64)
            finite_weak = finite_weak[np.isfinite(finite_weak)]
            finite_weak75 = np.asarray(weak75, dtype=np.float64)
            finite_weak75 = finite_weak75[np.isfinite(finite_weak75)]
            metrics["bbox_weak4_mAP"] = (
                float(finite_weak.mean()) if finite_weak.size else float("nan")
            )
            metrics["bbox_weak4_mAP_75"] = (
                float(finite_weak75.mean()) if finite_weak75.size else float("nan")
            )
            for name in self.CLASSES:
                metrics[f"bbox_AP_{name}"] = per_class[name]
        return metrics
