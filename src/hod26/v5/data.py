"""V5 staged augmentation without fictitious wavelength ordering."""

from __future__ import annotations

from typing import Any

import numpy as np
from mmdet.datasets.builder import PIPELINES

from hod26.v4.data import (
    HOD26StagedAugment,
    _flip_boxes,
    _four_image_mosaic,
    _object_preserving_crop,
    _resize_to_canvas,
)


def spectral_augment_order_agnostic(
    image: np.ndarray,
    *,
    gain: float,
    band_jitter: float,
    noise: float,
    dropout: float,
) -> np.ndarray:
    """Perturb sensor bands without treating mosaic IDs as wavelengths."""

    bands = image.shape[2]
    global_gain = np.float32(1.0 + np.random.uniform(-gain, gain))
    independent = np.random.normal(0.0, band_jitter, bands).astype(np.float32)
    independent -= independent.mean()
    augmented = image * (global_gain + independent).reshape(1, 1, bands)
    if noise > 0:
        augmented += np.random.normal(0.0, noise, image.shape).astype(np.float32)
    if dropout > 0 and np.random.random() < dropout:
        band = int(np.random.randint(bands))
        replacement = (augmented.sum(axis=2) - augmented[..., band]) / (bands - 1)
        augmented[..., band] = replacement
    return np.ascontiguousarray(np.clip(augmented, 0.0, 1.0))


@PIPELINES.register_module()
class HOD26V5StagedAugment(HOD26StagedAugment):
    """Long-horizon V5 stages with bounded native-HSI Mosaic."""

    def __call__(self, results: dict[str, Any]) -> dict[str, Any] | None:
        stage = self.stage
        image = results["img"]
        boxes = results.get("gt_bboxes", np.zeros((0, 4), dtype=np.float32))
        labels = results.get("gt_labels", np.zeros(0, dtype=np.int64))
        mosaic_applied = False
        if stage == "stabilize":
            spectral_probability, spectral = 0.30, (0.04, 0.015, 0.003, 0.01)
            crop_probability, vertical_probability = 0.0, 0.04
        elif stage == "strong":
            spectral_probability, spectral = 0.80, (0.12, 0.045, 0.009, 0.06)
            crop_probability, vertical_probability = 0.34, 0.10
        elif stage == "refine":
            spectral_probability, spectral = 0.55, (0.07, 0.025, 0.004, 0.025)
            crop_probability, vertical_probability = 0.12, 0.06
        else:
            spectral_probability, spectral = 0.15, (0.02, 0.008, 0.002, 0.0)
            crop_probability, vertical_probability = 0.0, 0.02

        if (
            stage == "strong"
            and self.mosaic_probability > 0
            and np.random.random() < self.mosaic_probability
        ):
            dataset = results.get("hod26_dataset")
            sample_index = results.get("hod26_sample_idx")
            low, high = results.get("hod26_band_low"), results.get("hod26_band_high")
            if dataset is None or sample_index is None or low is None or high is None:
                raise RuntimeError("V5 Mosaic requires dataset context and band limits")
            samples = [(image, boxes, labels)]
            for index in dataset.sample_mosaic_indices(int(sample_index), count=3):
                samples.append(
                    dataset.load_raw_sample(
                        index,
                        low=np.asarray(low, dtype=np.float32),
                        high=np.asarray(high, dtype=np.float32),
                    )
                )
            image, boxes, labels = _four_image_mosaic(samples)
            mosaic_applied = True

        if np.random.random() < spectral_probability:
            image = spectral_augment_order_agnostic(
                image,
                gain=spectral[0],
                band_jitter=spectral[1],
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
            valid = (boxes[:, 2] - boxes[:, 0] >= 2) & (
                boxes[:, 3] - boxes[:, 1] >= 2
            )
            boxes, labels = boxes[valid], labels[valid]
        if len(boxes) == 0:
            return None
        results.update(
            img=canvas,
            gt_bboxes=np.ascontiguousarray(boxes, dtype=np.float32),
            gt_labels=np.ascontiguousarray(labels, dtype=np.int64),
            img_shape=(resized_height, resized_width, 16),
            pad_shape=canvas.shape,
            scale_factor=np.asarray((sx, sy, sx, sy), dtype=np.float32),
            flip=horizontal or vertical,
            flip_direction=(
                "diagonal"
                if horizontal and vertical
                else "horizontal"
                if horizontal
                else "vertical"
                if vertical
                else None
            ),
            hod26_stage=stage,
            hod26_mosaic=mosaic_applied,
            hod26_mixup=False,
        )
        return results
