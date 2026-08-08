from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


UPSTREAM = Path("storage/upstream/DEIM")
TRAIN_ANNOTATIONS = Path("storage/v2/dataset/fold0/train.json")
STATS = Path("storage/cache/band_stats.json")
WEAK_CLASS_FACTORS = {5: 2, 8: 2, 14: 2, 16: 2}


def _recording_transform(
    name: str,
    calls: list[str],
) -> torch.nn.Module:
    class Recorder(torch.nn.Module):
        def forward(self, sample):
            calls.append(name)
            return sample

    Recorder.__name__ = name
    return Recorder()


@pytest.mark.skipif(
    not UPSTREAM.is_dir(),
    reason="official pinned DEIM checkout is bootstrapped on the training host",
)
def test_v3_fresh_update_schedule_applies_warmup_ratio() -> None:
    sys.path.insert(0, str(UPSTREAM.resolve()))
    from hod26.v2.extensions import HSIDetSolver

    parameter = torch.nn.Parameter(torch.ones(()))
    solver = HSIDetSolver.__new__(HSIDetSolver)
    solver.optimizer = torch.optim.SGD([parameter], lr=2e-4)
    solver.optimizer_updates = 0
    solver._loaded_v2_training_state = False
    solver._resume_lr_ratio = None
    solver._active_lr_schedule_points = []
    solver._current_lr_ratio = 1.0
    solver.cfg = SimpleNamespace(
        resume=None,
        yaml_cfg={
            "update_lr_schedule": {
                "points": [[0, 0.1], [800, 1.0], [8000, 1.0]]
            }
        },
    )

    scheduler = solver._build_update_scheduler(updates_per_epoch=127)

    assert scheduler is not None
    assert solver._current_lr_ratio == pytest.approx(0.1)
    assert solver.optimizer.param_groups[0]["lr"] == pytest.approx(2e-5)


@pytest.mark.skipif(
    not (UPSTREAM.is_dir() and TRAIN_ANNOTATIONS.is_file() and STATS.is_file()),
    reason="official Fold0 data view is available only on the training host",
)
def test_v3_weak_sampling_is_uniform_and_capped_at_two() -> None:
    sys.path.insert(0, str(UPSTREAM.resolve()))
    from hod26.v2.extensions import HSICocoDetection

    dataset = HSICocoDetection(
        img_folder="data/raw/data_train/data_train/VIS",
        ann_file=str(TRAIN_ANNOTATIONS),
        stats_file=str(STATS),
        transforms=None,
        repeat_class_factors=WEAK_CLASS_FACTORS,
    )
    occurrences = Counter(dataset.ids)

    assert dataset.base_image_count == 2400
    assert len(dataset) == 2812
    assert set(occurrences.values()) <= {1, 2}
    assert max(occurrences.values()) == 2

    for image_id in set(dataset.ids):
        annotations = dataset.coco.loadAnns(
            dataset.coco.getAnnIds(imgIds=image_id)
        )
        categories = {
            int(annotation["category_id"]) for annotation in annotations
        }
        expected = 2 if categories & set(WEAK_CLASS_FACTORS) else 1
        assert occurrences[image_id] == expected


@pytest.mark.skipif(
    not (UPSTREAM.is_dir() and TRAIN_ANNOTATIONS.is_file() and STATS.is_file()),
    reason="official Fold0 data view is available only on the training host",
)
def test_v3_sampling_rejects_unbounded_or_unknown_factors() -> None:
    sys.path.insert(0, str(UPSTREAM.resolve()))
    from hod26.v2.extensions import HSICocoDetection

    common = {
        "img_folder": "data/raw/data_train/data_train/VIS",
        "ann_file": str(TRAIN_ANNOTATIONS),
        "stats_file": str(STATS),
        "transforms": None,
    }
    with pytest.raises(ValueError, match="must be >= 1"):
        HSICocoDetection(**common, repeat_class_factors={16: 0})
    with pytest.raises(ValueError, match="unknown category"):
        HSICocoDetection(**common, repeat_class_factors={99: 2})


@pytest.mark.skipif(
    not UPSTREAM.is_dir(),
    reason="official pinned DEIM checkout is bootstrapped on the training host",
)
def test_v3_staged_augmentation_excludes_mosaic_from_clean_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.path.insert(0, str(UPSTREAM.resolve()))
    from hod26.v2 import extensions
    from hod26.v2.extensions import HSICompose

    calls: list[str] = []
    compose = HSICompose.__new__(HSICompose)
    torch.nn.Module.__init__(compose)
    compose.transforms = torch.nn.ModuleList(
        [
            _recording_transform("HSIMosaic", calls),
            _recording_transform("HSISpectralAugment", calls),
            _recording_transform("RandomIoUCrop", calls),
            _recording_transform("RandomHorizontalFlip", calls),
        ]
    )
    compose.policy = {
        "epoch": [4, 150, 220],
        "ops": [
            "HSIMosaic",
            "HSISpectralAugment",
            "RandomIoUCrop",
            "RandomHorizontalFlip",
        ],
    }
    compose.mosaic_prob = 0.45
    monkeypatch.setattr(extensions.random, "random", lambda: 0.1)
    sample = (torch.zeros(1), {}, SimpleNamespace(epoch=50))

    compose.stop_epoch_forward(sample)
    assert calls == [
        "HSIMosaic",
        "HSISpectralAugment",
        "RandomHorizontalFlip",
    ]

    calls.clear()
    sample[-1].epoch = 180
    compose.stop_epoch_forward(sample)
    assert calls == [
        "HSISpectralAugment",
        "RandomIoUCrop",
        "RandomHorizontalFlip",
    ]

    calls.clear()
    sample[-1].epoch = 220
    compose.stop_epoch_forward(sample)
    assert calls == []
