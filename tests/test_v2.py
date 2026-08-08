from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from hod26.v2.infer import _keep_best_class_per_query


UPSTREAM = Path("storage/upstream/DEIM")
V2_DATA = Path("storage/v2/dataset/fold0/train.json")


@pytest.mark.skipif(
    not UPSTREAM.is_dir(),
    reason="official pinned DEIM checkout is bootstrapped on the training host",
)
def test_v2_semantic_head_transfer_preserves_unmapped_classes() -> None:
    sys.path.insert(0, str(UPSTREAM.resolve()))
    from hod26.v2.extensions import (
        COCO_CLASSES,
        HOD26_CLASSES,
        _semantic_head_tensor,
    )

    source = torch.arange(80, dtype=torch.float32).view(80, 1)
    current = torch.full((18, 1), -1.0)
    mapped = _semantic_head_tensor(
        current,
        source,
        "decoder.enc_score_head.weight",
    )
    assert mapped is not None
    assert mapped[HOD26_CLASSES.index("apple")].item() == COCO_CLASSES.index(
        "apple"
    )
    assert mapped[HOD26_CLASSES.index("people")].item() == COCO_CLASSES.index(
        "person"
    )
    assert mapped[HOD26_CLASSES.index("egg")].item() == -1.0


@pytest.mark.skipif(
    not UPSTREAM.is_dir(),
    reason="official pinned DEIM checkout is bootstrapped on the training host",
)
def test_v2_update_scheduler_is_resume_safe() -> None:
    sys.path.insert(0, str(UPSTREAM.resolve()))
    from hod26.v2.extensions import PiecewiseCosineLRScheduler

    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=2e-4)
    scheduler = PiecewiseCosineLRScheduler(
        optimizer,
        [(100, 0.5), (200, 0.5), (400, 0.1), (500, 0.01)],
    )

    assert scheduler.ratio_at(0) == pytest.approx(0.5)
    assert scheduler.ratio_at(200) == pytest.approx(0.5)
    assert scheduler.ratio_at(300) == pytest.approx(0.3)
    assert scheduler.ratio_at(450) == pytest.approx(0.055)
    assert scheduler.ratio_at(600) == pytest.approx(0.01)
    scheduler.step(200, optimizer)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


@pytest.mark.skipif(
    not UPSTREAM.is_dir(),
    reason="official pinned DEIM checkout is bootstrapped on the training host",
)
def test_v2_convergence_requires_every_metric_and_low_lr_tail() -> None:
    sys.path.insert(0, str(UPSTREAM.resolve()))
    from hod26.v2.extensions import MultiMetricConvergenceTracker

    metrics = {
        "map_50_95": 0.70,
        "map_75": 0.82,
        "map_small": 0.67,
        "weak4_map_50_95": 0.42,
    }
    tracker = MultiMetricConvergenceTracker(
        thresholds={name: 0.01 for name in metrics},
        patience_evals=2,
        min_optimizer_updates=100,
        max_lr_ratio=0.02,
        initial_metrics=metrics,
    )

    converged, meaningful = tracker.update(
        metrics,
        optimizer_updates=100,
        lr_ratio=0.02,
    )
    assert not converged
    assert not any(meaningful.values())

    tracker = MultiMetricConvergenceTracker(
        thresholds={name: 0.01 for name in metrics},
        patience_evals=2,
        min_optimizer_updates=100,
        max_lr_ratio=0.02,
        state=tracker.state_dict(),
    )
    improved = dict(metrics)
    improved["map_small"] += 0.02
    converged, meaningful = tracker.update(
        improved,
        optimizer_updates=101,
        lr_ratio=0.02,
    )
    assert not converged
    assert meaningful["map_small"]

    converged, _ = tracker.update(
        improved,
        optimizer_updates=102,
        lr_ratio=0.02,
    )
    assert not converged
    converged, _ = tracker.update(
        improved,
        optimizer_updates=103,
        lr_ratio=0.02,
    )
    assert converged


@pytest.mark.skipif(
    not V2_DATA.is_file(),
    reason="V2 COCO views are generated from the official local dataset",
)
def test_v2_coco_view_is_complete_and_0_based() -> None:
    payload = json.loads(V2_DATA.read_text(encoding="utf-8"))
    assert len(payload["images"]) == 2400
    assert [item["id"] for item in payload["categories"]] == list(range(18))
    assert all(
        annotation["bbox"][2] > 0 and annotation["bbox"][3] > 0
        for annotation in payload["annotations"]
    )


def test_single_label_inference_keeps_only_query_argmax() -> None:
    logits = torch.tensor(
        [[[0.1, 0.9, 0.2], [2.0, -1.0, 1.0]]],
        dtype=torch.float32,
    )
    boxes = torch.rand(1, 2, 4)

    filtered = _keep_best_class_per_query(
        {"pred_logits": logits, "pred_boxes": boxes}
    )

    assert filtered["pred_boxes"] is boxes
    assert filtered["pred_logits"][0, 0, 1] == logits[0, 0, 1]
    assert filtered["pred_logits"][0, 1, 0] == logits[0, 1, 0]
    assert torch.isneginf(filtered["pred_logits"]).sum().item() == 4
