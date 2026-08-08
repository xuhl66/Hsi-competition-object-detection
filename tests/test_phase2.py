from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from hod26.config import atomic_json_dump
from hod26.submission import write_submission
from hod26.v2.phase2 import (
    build_ranking_views,
    combine_phase2_submissions,
)


CLASSES = [f"class_{index}" for index in range(18)]


def _sample(image_id: str, width: int = 5, height: int = 4) -> dict:
    return {
        "image_id": image_id,
        "image": f"{image_id}.png",
        "raw_width": width * 4,
        "raw_height": height * 4,
        "width": width,
        "height": height,
    }


def _manifest(path: Path, samples: list[dict]) -> None:
    atomic_json_dump(
        {"version": 1, "classes": CLASSES, "samples": samples},
        path,
    )


def _detection(image_id: str, confidence: float) -> dict:
    return {
        "image_id": image_id,
        "class_id": 3,
        "confidence": confidence,
        "x1": 0.5,
        "y1": 0.5,
        "x2": 4.5,
        "y2": 3.5,
    }


def _audit(csv_path: Path, checkpoint_sha256: str = "abc123") -> None:
    atomic_json_dump(
        {
            "valid": True,
            "checkpoint_sha256": checkpoint_sha256,
            "weights": "ema",
            "architecture": "HSI-DEIM-D-FINE-X",
        },
        csv_path.with_suffix(csv_path.suffix + ".audit.json"),
    )


def test_build_ranking_views_audits_uint16_and_writes_views(
    tmp_path: Path,
) -> None:
    image_folder = tmp_path / "ranking"
    image_folder.mkdir()
    assert cv2.imwrite(
        str(image_folder / "20.png"),
        np.zeros((16, 20), dtype=np.uint16),
    )
    assert cv2.imwrite(
        str(image_folder / "21.png"),
        np.ones((16, 20), dtype=np.uint16),
    )
    classes_path = tmp_path / "class.txt"
    classes_path.write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    test_manifest = tmp_path / "test_manifest.json"
    ranking_manifest = tmp_path / "ranking_manifest.json"
    ranking_coco = tmp_path / "ranking.json"
    combined_manifest = tmp_path / "phase2_manifest.json"
    _manifest(test_manifest, [_sample("10"), _sample("11")])

    report = build_ranking_views(
        image_folder,
        expected_images=2,
        classes_file=classes_path,
        test_manifest_path=test_manifest,
        ranking_manifest_path=ranking_manifest,
        ranking_coco_path=ranking_coco,
        combined_manifest_path=combined_manifest,
    )

    assert report["valid"]
    assert report["ranking_images"] == 2
    ranking = json.loads(ranking_manifest.read_text(encoding="utf-8"))
    assert [sample["image_id"] for sample in ranking["samples"]] == [
        "20",
        "21",
    ]
    assert all(Path(sample["image"]).is_absolute() for sample in ranking["samples"])
    coco = json.loads(ranking_coco.read_text(encoding="utf-8"))
    assert len(coco["images"]) == 2
    assert coco["annotations"] == []
    combined = json.loads(combined_manifest.read_text(encoding="utf-8"))
    assert len(combined["samples"]) == 4
    assert not combined["audit"]["labels_used"]


def test_combine_phase2_requires_and_records_one_checkpoint(
    tmp_path: Path,
) -> None:
    test_manifest = tmp_path / "test_manifest.json"
    ranking_manifest = tmp_path / "ranking_manifest.json"
    _manifest(test_manifest, [_sample("10")])
    _manifest(ranking_manifest, [_sample("20")])
    test_csv = write_submission(
        [_detection("10", 0.7)],
        tmp_path / "test.csv",
        include_id=False,
    )
    ranking_csv = write_submission(
        [_detection("20", 0.8)],
        tmp_path / "ranking.csv",
        include_id=True,
    )
    _audit(test_csv)
    _audit(ranking_csv)
    output = tmp_path / "phase2.csv"

    report = combine_phase2_submissions(
        test_csv,
        ranking_csv,
        output,
        test_manifest_path=test_manifest,
        ranking_manifest_path=ranking_manifest,
    )

    assert report["valid"]
    assert report["checkpoint_sha256"] == "abc123"
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["id"] == "0"
    assert rows[0]["image_id"] == "10"
    assert rows[1]["id"] == "1"
    assert rows[1]["image_id"] == "20"


def test_combine_phase2_rejects_different_checkpoints(
    tmp_path: Path,
) -> None:
    test_manifest = tmp_path / "test_manifest.json"
    ranking_manifest = tmp_path / "ranking_manifest.json"
    _manifest(test_manifest, [_sample("10")])
    _manifest(ranking_manifest, [_sample("20")])
    test_csv = write_submission(
        [_detection("10", 0.7)],
        tmp_path / "test.csv",
    )
    ranking_csv = write_submission(
        [_detection("20", 0.8)],
        tmp_path / "ranking.csv",
    )
    _audit(test_csv, checkpoint_sha256="model-a")
    _audit(ranking_csv, checkpoint_sha256="model-b")

    with pytest.raises(ValueError, match="same single-model"):
        combine_phase2_submissions(
            test_csv,
            ranking_csv,
            tmp_path / "phase2.csv",
            test_manifest_path=test_manifest,
            ranking_manifest_path=ranking_manifest,
        )
