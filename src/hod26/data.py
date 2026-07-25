from __future__ import annotations

import math
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def decode_x2cube(mosaic: np.ndarray, cell_size: int = 4) -> np.ndarray:
    """Decode the snapshot mosaic exactly like the organizer's X2Cube."""
    if mosaic.ndim != 2:
        raise ValueError(f"Expected a 2-D mosaic, got shape {mosaic.shape}")
    height, width = mosaic.shape
    if height % cell_size or width % cell_size:
        raise ValueError(
            f"Mosaic dimensions must be divisible by {cell_size}: {width}x{height}"
        )
    cube = (
        mosaic.reshape(height // cell_size, cell_size, width // cell_size, cell_size)
        .transpose(0, 2, 1, 3)
        .reshape(height // cell_size, width // cell_size, cell_size * cell_size)
    )
    return np.ascontiguousarray(cube)


def load_cube(path: str | Path) -> np.ndarray:
    mosaic = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mosaic is None:
        raise FileNotFoundError(f"Unable to decode PNG: {path}")
    if mosaic.dtype != np.uint16:
        raise ValueError(f"Expected uint16 PNG at {path}, got {mosaic.dtype}")
    return decode_x2cube(mosaic)


def load_classes(path: str | Path) -> list[str]:
    classes = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(classes) != 18 or len(set(classes)) != len(classes):
        raise ValueError(f"Expected 18 unique classes, got {len(classes)}")
    return classes


def _raw_png_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return width, height


def parse_voc_annotation(
    path: str | Path,
    class_to_id: dict[str, int],
    actual_width: int,
    actual_height: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Parse and sanitize an official XML without changing the raw file.

    Degenerate boxes are removed and genuine out-of-range boxes are clipped to
    the decoded cube. XML dimensions are retained in the audit metadata.
    """
    root = ET.parse(path).getroot()
    xml_width = int(root.findtext("./size/width", "0"))
    xml_height = int(root.findtext("./size/height", "0"))
    xml_depth = int(root.findtext("./size/depth", "0"))
    objects: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    clipped = 0

    for index, node in enumerate(root.findall("./object")):
        name = node.findtext("name", "").strip()
        if name not in class_to_id:
            raise ValueError(f"Unknown class {name!r} in {path}")
        box_node = node.find("bndbox")
        if box_node is None:
            dropped.append({"index": index, "reason": "missing_bndbox", "class": name})
            continue
        original = np.asarray(
            [
                float(box_node.findtext("xmin", "nan")),
                float(box_node.findtext("ymin", "nan")),
                float(box_node.findtext("xmax", "nan")),
                float(box_node.findtext("ymax", "nan")),
            ],
            dtype=np.float32,
        )
        if not np.isfinite(original).all():
            dropped.append({"index": index, "reason": "non_finite", "class": name})
            continue
        clean = original.copy()
        clean[[0, 2]] = clean[[0, 2]].clip(0, actual_width)
        clean[[1, 3]] = clean[[1, 3]].clip(0, actual_height)
        if not np.array_equal(clean, original):
            clipped += 1
        if clean[2] <= clean[0] or clean[3] <= clean[1]:
            dropped.append(
                {
                    "index": index,
                    "reason": "degenerate_after_clip",
                    "class": name,
                    "box": original.tolist(),
                }
            )
            continue
        objects.append(
            {
                "class_id": class_to_id[name],
                "class_name": name,
                "bbox": [float(value) for value in clean],
                "difficult": int(node.findtext("difficult", "0")),
                "truncated": int(node.findtext("truncated", "0")),
            }
        )

    audit = {
        "xml_width": xml_width,
        "xml_height": xml_height,
        "xml_depth": xml_depth,
        "dimension_mismatch": (xml_width, xml_height, xml_depth)
        != (actual_width, actual_height, 16),
        "dropped": dropped,
        "clipped_boxes": clipped,
    }
    return objects, audit


def _perceptual_hash(cube: np.ndarray) -> int:
    """64-bit low-frequency hash used only to group near-duplicate scenes."""
    gray = cube.astype(np.float32).mean(axis=2)
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(gray)
    coefficients = dct[:8, :8].reshape(-1)
    threshold = float(np.median(coefficients[1:]))
    bits = coefficients > threshold
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def build_manifests(
    data_root: Path,
    train_images: Path,
    train_annotations: Path,
    test_images: Path,
    classes: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    class_to_id = {name: index for index, name in enumerate(classes)}
    train_samples: list[dict[str, Any]] = []
    test_samples: list[dict[str, Any]] = []
    issue_files: list[dict[str, Any]] = []

    train_paths = sorted(train_images.glob("*.png"), key=lambda p: int(p.stem))
    xml_stems = {path.stem for path in train_annotations.glob("*.xml")}
    if {path.stem for path in train_paths} != xml_stems:
        raise ValueError("Training PNG and XML stems do not match")

    for image_path in train_paths:
        raw_width, raw_height = _raw_png_size(image_path)
        if raw_width % 4 or raw_height % 4:
            raise ValueError(f"Raw shape is not divisible by four: {image_path}")
        width, height = raw_width // 4, raw_height // 4
        objects, audit = parse_voc_annotation(
            train_annotations / f"{image_path.stem}.xml",
            class_to_id,
            width,
            height,
        )
        cube = load_cube(image_path)
        if cube.shape != (height, width, 16):
            raise ValueError(f"X2Cube shape mismatch for {image_path}: {cube.shape}")
        record = {
            "image_id": image_path.stem,
            "image": str(image_path.relative_to(data_root)),
            "annotation": str(
                (train_annotations / f"{image_path.stem}.xml").relative_to(data_root)
            ),
            "raw_width": raw_width,
            "raw_height": raw_height,
            "width": width,
            "height": height,
            "phash": f"{_perceptual_hash(cube):016x}",
            "objects": objects,
            "class_counts": dict(Counter(obj["class_id"] for obj in objects)),
        }
        train_samples.append(record)
        if audit["dimension_mismatch"] or audit["dropped"] or audit["clipped_boxes"]:
            issue_files.append({"image_id": image_path.stem, **audit})

    for image_path in sorted(test_images.glob("*.png"), key=lambda p: int(p.stem)):
        raw_width, raw_height = _raw_png_size(image_path)
        if raw_width % 4 or raw_height % 4:
            raise ValueError(f"Raw shape is not divisible by four: {image_path}")
        width, height = raw_width // 4, raw_height // 4
        test_samples.append(
            {
                "image_id": image_path.stem,
                "image": str(image_path.relative_to(data_root)),
                "raw_width": raw_width,
                "raw_height": raw_height,
                "width": width,
                "height": height,
            }
        )

    train_manifest = {
        "version": 1,
        "classes": list(classes),
        "samples": train_samples,
        "audit": {
            "images": len(train_samples),
            "objects_after_cleaning": sum(len(x["objects"]) for x in train_samples),
            "issue_files": issue_files,
        },
    }
    test_manifest = {
        "version": 1,
        "classes": list(classes),
        "samples": test_samples,
        "audit": {"images": len(test_samples)},
    }
    return train_manifest, test_manifest


def compute_band_statistics(
    samples: Sequence[dict[str, Any]],
    data_root: Path,
    bins: int = 4096,
    quantile_low: float = 0.001,
    quantile_high: float = 0.999,
    sample_stride: int = 4,
) -> dict[str, Any]:
    """Compute robust global uint16 band limits with a streaming histogram."""
    histogram = np.zeros((16, bins), dtype=np.int64)
    bit_shift = 16 - int(round(math.log2(bins)))
    if 2**int(round(math.log2(bins))) != bins:
        raise ValueError("histogram_bins must be a power of two")

    for record in samples:
        cube = load_cube(data_root / record["image"])
        view = cube[::sample_stride, ::sample_stride]
        quantized = np.right_shift(view, bit_shift)
        for band in range(16):
            histogram[band] += np.bincount(
                quantized[..., band].reshape(-1), minlength=bins
            )

    lows: list[float] = []
    highs: list[float] = []
    means: list[float] = []
    stds: list[float] = []
    centers = (np.arange(bins, dtype=np.float64) + 0.5) * (2**bit_shift)
    for band in range(16):
        counts = histogram[band]
        cumulative = counts.cumsum()
        total = int(cumulative[-1])
        low_index = int(np.searchsorted(cumulative, total * quantile_low))
        high_index = int(np.searchsorted(cumulative, total * quantile_high))
        low = float(low_index * (2**bit_shift))
        high = float(min(65535, (high_index + 1) * (2**bit_shift) - 1))
        probability = counts.astype(np.float64) / max(total, 1)
        mean = float((probability * centers).sum())
        variance = float((probability * (centers - mean) ** 2).sum())
        lows.append(low)
        highs.append(max(high, low + 1))
        means.append(mean)
        stds.append(math.sqrt(max(variance, 0.0)))

    return {
        "version": 1,
        "quantile_low": quantile_low,
        "quantile_high": quantile_high,
        "sample_stride": sample_stride,
        "histogram_bins": bins,
        "low": lows,
        "high": highs,
        "raw_mean": means,
        "raw_std": stds,
        "samples": len(samples),
    }


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _near_duplicate_groups(
    samples: Sequence[dict[str, Any]], max_hamming: int = 2
) -> list[list[int]]:
    hashes = [int(sample["phash"], 16) for sample in samples]
    union = _UnionFind(len(samples))
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, value in enumerate(hashes):
        for chunk in range(4):
            buckets[(chunk, (value >> (chunk * 16)) & 0xFFFF)].append(index)
    seen: set[tuple[int, int]] = set()
    for indices in buckets.values():
        if len(indices) > 100:
            continue
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                pair = (min(left, right), max(left, right))
                if pair in seen:
                    continue
                seen.add(pair)
                if (hashes[left] ^ hashes[right]).bit_count() <= max_hamming:
                    union.union(left, right)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(samples)):
        grouped[union.find(index)].append(index)
    return list(grouped.values())


def make_balanced_folds(
    samples: Sequence[dict[str, Any]],
    num_folds: int,
    num_classes: int,
    seed: int,
) -> dict[str, Any]:
    """Greedy grouped multilabel split with class-instance balancing."""
    rng = random.Random(seed)
    groups = _near_duplicate_groups(samples)
    group_vectors: list[np.ndarray] = []
    for group in groups:
        vector = np.zeros(num_classes, dtype=np.float64)
        for index in group:
            for class_id, count in samples[index]["class_counts"].items():
                vector[int(class_id)] += int(count)
        group_vectors.append(vector)

    totals = np.sum(group_vectors, axis=0)
    target = totals / num_folds
    target_images = len(samples) / num_folds
    rarity = 1.0 / np.sqrt(np.maximum(totals, 1.0))
    order = list(range(len(groups)))
    rng.shuffle(order)
    order.sort(
        key=lambda idx: (
            -float((group_vectors[idx] * rarity).sum()),
            -len(groups[idx]),
        )
    )

    fold_counts = np.zeros((num_folds, num_classes), dtype=np.float64)
    fold_sizes = np.zeros(num_folds, dtype=np.float64)
    assignment = np.full(len(samples), -1, dtype=np.int64)
    class_denominator = np.maximum(target, 1.0)
    for group_index in order:
        vector = group_vectors[group_index]
        group_size = len(groups[group_index])
        scores = []
        for fold in range(num_folds):
            # Compare the change in the *global* objective. Looking only at a
            # candidate fold's absolute error favors an already-filled fold and
            # can leave later folds empty.
            current_class_error = np.mean(
                ((fold_counts[fold] - target) / class_denominator) ** 2
            )
            candidate_class_error = np.mean(
                ((fold_counts[fold] + vector - target) / class_denominator) ** 2
            )
            current_size_error = (
                (fold_sizes[fold] - target_images) / max(target_images, 1.0)
            ) ** 2
            candidate_size_error = (
                (fold_sizes[fold] + group_size - target_images)
                / max(target_images, 1.0)
            ) ** 2
            scores.append(
                candidate_class_error
                - current_class_error
                + 0.35 * (candidate_size_error - current_size_error)
            )
        chosen = min(range(num_folds), key=lambda fold: (scores[fold], fold_sizes[fold]))
        for sample_index in groups[group_index]:
            assignment[sample_index] = chosen
        fold_counts[chosen] += vector
        fold_sizes[chosen] += group_size

    return {
        "version": 2,
        "seed": seed,
        "num_folds": num_folds,
        "near_duplicate_hamming": 2,
        "fold_by_image_id": {
            sample["image_id"]: int(assignment[index])
            for index, sample in enumerate(samples)
        },
        "fold_image_counts": fold_sizes.astype(int).tolist(),
        "fold_class_counts": fold_counts.astype(int).tolist(),
    }


@dataclass
class LetterboxMeta:
    scale: float
    pad_x: float
    pad_y: float
    input_width: int
    input_height: int
    original_width: int
    original_height: int


def letterbox_cube(
    cube: np.ndarray,
    boxes: np.ndarray,
    output_height: int,
    output_width: int,
) -> tuple[np.ndarray, np.ndarray, LetterboxMeta]:
    height, width = cube.shape[:2]
    scale = min(output_width / width, output_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        cube, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    pad_x = (output_width - resized_width) // 2
    pad_y = (output_height - resized_height) // 2
    output = np.zeros((output_height, output_width, cube.shape[2]), dtype=np.float32)
    output[
        pad_y : pad_y + resized_height, pad_x : pad_x + resized_width
    ] = resized
    transformed = boxes.copy()
    if transformed.size:
        transformed[:, [0, 2]] = transformed[:, [0, 2]] * scale + pad_x
        transformed[:, [1, 3]] = transformed[:, [1, 3]] * scale + pad_y
    return output, transformed, LetterboxMeta(
        scale=scale,
        pad_x=float(pad_x),
        pad_y=float(pad_y),
        input_width=output_width,
        input_height=output_height,
        original_width=width,
        original_height=height,
    )


def restore_boxes(boxes: torch.Tensor, metadata: LetterboxMeta) -> torch.Tensor:
    restored = boxes.clone()
    restored[:, [0, 2]] = (restored[:, [0, 2]] - metadata.pad_x) / metadata.scale
    restored[:, [1, 3]] = (restored[:, [1, 3]] - metadata.pad_y) / metadata.scale
    restored[:, [0, 2]].clamp_(0, metadata.original_width)
    restored[:, [1, 3]].clamp_(0, metadata.original_height)
    return restored


class HSIDetectionDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[dict[str, Any]],
        data_root: str | Path,
        stats: dict[str, Any],
        output_height: int,
        output_width: int,
        augment: bool = False,
        augmentation: dict[str, float] | None = None,
    ):
        self.samples = list(samples)
        self.data_root = Path(data_root)
        self.low = np.asarray(stats["low"], dtype=np.float32).reshape(1, 1, 16)
        self.high = np.asarray(stats["high"], dtype=np.float32).reshape(1, 1, 16)
        self.output_height = output_height
        self.output_width = output_width
        self.augment = augment
        self.augmentation = augmentation or {}
        cv2.setNumThreads(0)

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize(self, cube: np.ndarray) -> np.ndarray:
        cube = cube.astype(np.float32)
        return np.clip((cube - self.low) / (self.high - self.low), 0.0, 1.0)

    def _spatial_augment(
        self, cube: np.ndarray, boxes: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = cube.shape[:2]
        if random.random() < float(self.augmentation.get("horizontal_flip", 0.0)):
            cube = np.ascontiguousarray(cube[:, ::-1])
            if boxes.size:
                old_x1 = boxes[:, 0].copy()
                boxes[:, 0] = width - boxes[:, 2]
                boxes[:, 2] = width - old_x1
        if random.random() < float(self.augmentation.get("vertical_flip", 0.0)):
            cube = np.ascontiguousarray(cube[::-1])
            if boxes.size:
                old_y1 = boxes[:, 1].copy()
                boxes[:, 1] = height - boxes[:, 3]
                boxes[:, 3] = height - old_y1

        scale_limit = float(self.augmentation.get("scale", 0.0))
        translate_limit = float(self.augmentation.get("translate", 0.0))
        if scale_limit or translate_limit:
            scale = random.uniform(1.0 - scale_limit, 1.0 + scale_limit)
            tx = random.uniform(-translate_limit, translate_limit) * width
            ty = random.uniform(-translate_limit, translate_limit) * height
            matrix = np.asarray(
                [
                    [scale, 0.0, (1.0 - scale) * width / 2.0 + tx],
                    [0.0, scale, (1.0 - scale) * height / 2.0 + ty],
                ],
                dtype=np.float32,
            )
            cube = cv2.warpAffine(
                cube,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            if cube.ndim == 2:
                cube = cube[..., None]
            if boxes.size:
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + matrix[0, 2]
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + matrix[1, 2]
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
                keep = (boxes[:, 2] - boxes[:, 0] >= 1.0) & (
                    boxes[:, 3] - boxes[:, 1] >= 1.0
                )
                boxes, labels = boxes[keep], labels[keep]
        return cube, boxes, labels

    def _spectral_augment(self, cube: np.ndarray) -> np.ndarray:
        bands = cube.shape[2]
        gain_limit = float(self.augmentation.get("spectral_gain", 0.0))
        slope_limit = float(self.augmentation.get("spectral_slope", 0.0))
        gain = random.uniform(1.0 - gain_limit, 1.0 + gain_limit)
        slope = random.uniform(-slope_limit, slope_limit)
        wavelength_position = np.linspace(-1.0, 1.0, bands, dtype=np.float32)
        cube = cube * (gain * (1.0 + slope * wavelength_position))[None, None]

        noise_std = float(self.augmentation.get("noise_std", 0.0))
        if noise_std and random.random() < 0.5:
            cube = cube + np.random.normal(0.0, noise_std, cube.shape).astype(np.float32)

        if random.random() < float(self.augmentation.get("band_dropout", 0.0)):
            band = random.randrange(bands)
            if band == 0:
                cube[..., band] = cube[..., 1]
            elif band == bands - 1:
                cube[..., band] = cube[..., bands - 2]
            else:
                cube[..., band] = 0.5 * (
                    cube[..., band - 1] + cube[..., band + 1]
                )
        return np.clip(cube, 0.0, 1.0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.samples[index]
        cube = self._normalize(load_cube(self.data_root / record["image"]))
        objects = record.get("objects", [])
        boxes = np.asarray([obj["bbox"] for obj in objects], dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(
            [obj["class_id"] for obj in objects], dtype=np.int64
        ).reshape(-1)
        original_boxes = boxes.copy()
        original_labels = labels.copy()

        if self.augment:
            cube, boxes, labels = self._spatial_augment(cube, boxes, labels)
            cube = self._spectral_augment(cube)

        cube, boxes, metadata = letterbox_cube(
            cube, boxes, self.output_height, self.output_width
        )
        if boxes.size:
            xywh = np.empty_like(boxes)
            xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2 / self.output_width
            xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2 / self.output_height
            xywh[:, 2] = (boxes[:, 2] - boxes[:, 0]) / self.output_width
            xywh[:, 3] = (boxes[:, 3] - boxes[:, 1]) / self.output_height
        else:
            xywh = np.empty((0, 4), dtype=np.float32)

        return {
            "img": torch.from_numpy(cube.transpose(2, 0, 1)).float(),
            "cls": torch.from_numpy(labels[:, None]).float(),
            "bboxes": torch.from_numpy(xywh).float(),
            "metadata": metadata,
            "image_id": record["image_id"],
            "original_boxes": torch.from_numpy(original_boxes).float(),
            "original_labels": torch.from_numpy(original_labels).long(),
        }


def collate_detection_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([item["img"] for item in items])
    classes: list[torch.Tensor] = []
    boxes: list[torch.Tensor] = []
    batch_indices: list[torch.Tensor] = []
    for batch_index, item in enumerate(items):
        count = len(item["cls"])
        classes.append(item["cls"])
        boxes.append(item["bboxes"])
        batch_indices.append(torch.full((count,), batch_index, dtype=torch.float32))
    return {
        "img": images,
        "cls": torch.cat(classes, dim=0) if classes else torch.empty((0, 1)),
        "bboxes": torch.cat(boxes, dim=0) if boxes else torch.empty((0, 4)),
        "batch_idx": (
            torch.cat(batch_indices, dim=0)
            if batch_indices
            else torch.empty((0,), dtype=torch.float32)
        ),
        "metadata": [item["metadata"] for item in items],
        "image_id": [item["image_id"] for item in items],
        "original_boxes": [item["original_boxes"] for item in items],
        "original_labels": [item["original_labels"] for item in items],
    }


def select_fold_samples(
    manifest: dict[str, Any],
    folds: dict[str, Any],
    fold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mapping = folds["fold_by_image_id"]
    train, validation = [], []
    for sample in manifest["samples"]:
        (validation if int(mapping[sample["image_id"]]) == fold else train).append(sample)
    return train, validation


def count_classes(samples: Iterable[dict[str, Any]], num_classes: int) -> list[int]:
    counts = np.zeros(num_classes, dtype=np.int64)
    for sample in samples:
        for class_id, count in sample.get("class_counts", {}).items():
            counts[int(class_id)] += int(count)
    return counts.tolist()
