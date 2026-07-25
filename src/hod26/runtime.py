from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import resolve_path
from .data import (
    HSIDetectionDataset,
    collate_detection_batch,
    load_classes,
    select_fold_samples,
)


def load_data_assets(config: dict[str, Any]) -> dict[str, Any]:
    project_root = Path.cwd()
    data_cfg = config["data"]
    data_root = resolve_path(data_cfg["root"], project_root)
    classes = load_classes(data_root / data_cfg["classes"])
    manifest = json.loads(
        resolve_path(data_cfg["manifest"], project_root).read_text(encoding="utf-8")
    )
    test_manifest = json.loads(
        resolve_path(data_cfg["test_manifest"], project_root).read_text(
            encoding="utf-8"
        )
    )
    stats = json.loads(
        resolve_path(data_cfg["stats"], project_root).read_text(encoding="utf-8")
    )
    folds = json.loads(
        resolve_path(data_cfg["folds"], project_root).read_text(encoding="utf-8")
    )
    return {
        "data_root": data_root,
        "classes": classes,
        "manifest": manifest,
        "test_manifest": test_manifest,
        "stats": stats,
        "folds": folds,
    }


def build_train_loaders(
    config: dict[str, Any],
    assets: dict[str, Any],
    rank: int = -1,
    world_size: int = 1,
    full_data: bool = False,
):
    data_cfg = config["data"]
    fold = int(data_cfg["fold"])
    if full_data:
        train_samples = assets["manifest"]["samples"]
        validation_samples = []
    else:
        train_samples, validation_samples = select_fold_samples(
            assets["manifest"], assets["folds"], fold
        )

    train_dataset = HSIDetectionDataset(
        train_samples,
        assets["data_root"],
        assets["stats"],
        int(data_cfg["image_height"]),
        int(data_cfg["image_width"]),
        augment=True,
        augmentation=config["augmentation"],
    )
    validation_dataset = HSIDetectionDataset(
        validation_samples,
        assets["data_root"],
        assets["stats"],
        int(data_cfg["image_height"]),
        int(data_cfg["image_width"]),
        augment=False,
    )
    sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(config["project"]["seed"]),
            drop_last=True,
        )
        if world_size > 1
        else None
    )
    workers = int(data_cfg["workers"])
    batch_size = int(config["train"]["batch_size_per_gpu"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=True,
        collate_fn=collate_detection_batch,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=max(1, workers // 2),
        pin_memory=True,
        persistent_workers=workers > 0 and len(validation_dataset) > 0,
        collate_fn=collate_detection_batch,
    )
    return train_loader, validation_loader, sampler


def build_inference_loader(
    config: dict[str, Any],
    assets: dict[str, Any],
    samples: list[dict[str, Any]],
) -> DataLoader:
    data_cfg = config["data"]
    dataset = HSIDetectionDataset(
        samples,
        assets["data_root"],
        assets["stats"],
        int(data_cfg["image_height"]),
        int(data_cfg["image_width"]),
        augment=False,
    )
    return DataLoader(
        dataset,
        batch_size=int(config["inference"]["batch_size_per_gpu"]),
        shuffle=False,
        num_workers=int(data_cfg["workers"]),
        pin_memory=True,
        persistent_workers=int(data_cfg["workers"]) > 0,
        collate_fn=collate_detection_batch,
    )

