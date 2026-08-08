from __future__ import annotations

import json

import pytest

from tools.diagnose_v6_evidence import (
    checkpoint_delta,
    dataset_stats,
    parameter_group,
)


def test_dataset_stats_reports_exact_coco_size_counts(tmp_path) -> None:
    annotation = tmp_path / "tiny.json"
    annotation.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "width": 100, "height": 100}],
                "categories": [{"id": 0, "name": "car"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10]},
                    {"id": 2, "image_id": 1, "category_id": 0, "bbox": [0, 0, 40, 40]},
                ],
            }
        ),
        encoding="utf-8",
    )

    report = dataset_stats(annotation)

    assert report["size_distribution"]["small"] == {
        "instances": 1,
        "fraction": 0.5,
    }
    assert report["size_distribution"]["medium"] == {
        "instances": 1,
        "fraction": 0.5,
    }
    assert report["size_distribution"]["large"] == {
        "instances": 0,
        "fraction": 0.0,
    }


def test_checkpoint_delta_exposes_strong_only_gain() -> None:
    parent = {
        "bbox_mAP": 0.7,
        "bbox_mAP_75": 0.8,
        "per_class": {"car": 0.4},
        "derived": {
            "weak4_mean": 0.5,
            "strong14_mean": 0.75,
            "strong14_minus_weak4": 0.25,
        },
    }
    candidate = {
        "bbox_mAP": 0.71,
        "bbox_mAP_75": 0.79,
        "per_class": {"car": 0.38},
        "derived": {
            "weak4_mean": 0.49,
            "strong14_mean": 0.77,
            "strong14_minus_weak4": 0.28,
        },
    }

    delta = checkpoint_delta(parent, candidate)

    assert delta["metrics"]["bbox_mAP"] == pytest.approx(0.01)
    assert delta["metrics"]["bbox_mAP_75"] == pytest.approx(-0.01)
    assert delta["derived"]["weak4_mean"] == pytest.approx(-0.01)
    assert delta["derived"]["strong14_mean"] == pytest.approx(0.02)
    assert delta["per_class"]["car"] == pytest.approx(-0.02)


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("ema_backbone_spectral_stem.0.0.weight", "spectral_encoder"),
        ("neck.spectral_fusions.2.gate", "spectral_fusion"),
        ("ema_query_head_fdr_branches_0_layers_0_weight", "fdr"),
        ("backbone.blocks.0.attn.qkv.weight", "visual_backbone"),
    ),
)
def test_parameter_group_understands_raw_and_ema_names(name, expected) -> None:
    assert parameter_group(name) == expected
