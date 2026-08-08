from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from hod26.config import atomic_json_dump, resolve_path
from hod26.submission import (
    BASE_COLUMNS,
    validate_submission,
    write_submission,
)
from hod26.v2.prepare import _write_coco


def _classes(path: Path) -> list[str]:
    classes = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(classes) != 18 or len(set(classes)) != len(classes):
        raise ValueError(
            f"Expected 18 unique official classes in {path}, got "
            f"{len(classes)}"
        )
    return classes


def build_ranking_views(
    image_folder: str | Path,
    *,
    expected_images: int = 1000,
    classes_file: str | Path = "data/raw/class.txt",
    test_manifest_path: str | Path = "storage/cache/test_manifest.json",
    ranking_manifest_path: str | Path = (
        "storage/cache/ranking_manifest.json"
    ),
    ranking_coco_path: str | Path = "storage/v2/dataset/ranking.json",
    combined_manifest_path: str | Path = (
        "storage/cache/phase2_manifest.json"
    ),
) -> dict[str, Any]:
    """Audit released ranking images and build label-free inference views."""
    if expected_images <= 0:
        raise ValueError("expected_images must be positive")
    folder = resolve_path(image_folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    classes = _classes(resolve_path(classes_file))
    test_manifest = json.loads(
        resolve_path(test_manifest_path).read_text(encoding="utf-8")
    )
    if test_manifest["classes"] != classes:
        raise ValueError("Test manifest class order differs from class.txt")
    test_ids = {
        str(sample["image_id"]) for sample in test_manifest["samples"]
    }

    image_paths = list(folder.glob("*.png"))
    nonnumeric = [path.name for path in image_paths if not path.stem.isdigit()]
    if nonnumeric:
        raise ValueError(
            "Ranking filenames must have numeric stems: "
            f"{sorted(nonnumeric)[:10]}"
        )

    samples: list[dict[str, Any]] = []
    ranking_ids: set[str] = set()
    for image_path in sorted(image_paths, key=lambda path: int(path.stem)):
        try:
            image_id = str(int(image_path.stem))
        except ValueError as exc:
            raise ValueError(
                f"Ranking filename must have a numeric stem: {image_path.name}"
            ) from exc
        if image_id in ranking_ids:
            raise ValueError(f"Duplicate ranking image_id {image_id}")
        raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"Cannot read ranking image {image_path}")
        if raw.dtype != np.uint16 or raw.ndim != 2:
            raise ValueError(
                f"Expected uint16 single-channel PNG at {image_path}, "
                f"got dtype={raw.dtype}, shape={raw.shape}"
            )
        raw_height, raw_width = (int(value) for value in raw.shape)
        if raw_height % 4 or raw_width % 4:
            raise ValueError(
                f"Raw dimensions must be divisible by four: "
                f"{image_path.name}={raw_width}x{raw_height}"
            )
        ranking_ids.add(image_id)
        samples.append(
            {
                "image_id": image_id,
                "image": str(image_path.resolve()),
                "raw_width": raw_width,
                "raw_height": raw_height,
                "width": raw_width // 4,
                "height": raw_height // 4,
            }
        )

    if len(samples) != expected_images:
        raise ValueError(
            f"Expected {expected_images} ranking PNGs, got {len(samples)}"
        )
    overlap = test_ids & ranking_ids
    if overlap:
        raise ValueError(
            "Test and ranking image_id values overlap; image_id alone would "
            f"be ambiguous. First overlaps: {sorted(overlap)[:10]}"
        )

    ranking_manifest = {
        "version": 1,
        "classes": classes,
        "samples": samples,
        "audit": {
            "images": len(samples),
            "expected_images": expected_images,
            "source": str(folder),
            "labels_used": False,
        },
    }
    combined_manifest = {
        "version": 1,
        "classes": classes,
        "samples": test_manifest["samples"] + samples,
        "audit": {
            "test_images": len(test_manifest["samples"]),
            "ranking_images": len(samples),
            "total_images": len(test_manifest["samples"]) + len(samples),
            "labels_used": False,
        },
    }
    ranking_manifest_destination = resolve_path(ranking_manifest_path)
    ranking_coco_destination = resolve_path(ranking_coco_path)
    combined_manifest_destination = resolve_path(combined_manifest_path)
    atomic_json_dump(ranking_manifest, ranking_manifest_destination)
    _write_coco(
        ranking_coco_destination,
        samples,
        classes,
        include_annotations=False,
    )
    atomic_json_dump(combined_manifest, combined_manifest_destination)
    return {
        "valid": True,
        "ranking_images": len(samples),
        "combined_images": len(combined_manifest["samples"]),
        "ranking_manifest": str(ranking_manifest_destination),
        "ranking_coco": str(ranking_coco_destination),
        "combined_manifest": str(combined_manifest_destination),
    }


def _read_detection_rows(path: Path) -> tuple[list[dict[str, Any]], bool]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        include_id = fields == ["id", *BASE_COLUMNS]
        if not include_id and fields != BASE_COLUMNS:
            raise ValueError(
                f"Unexpected columns in {path}: {fields}"
            )
        rows = [
            {
                "image_id": str(row["image_id"]).strip(),
                "class_id": int(row["class_id"]),
                "confidence": float(row["confidence"]),
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
            }
            for row in reader
        ]
    return rows, include_id


def _read_inference_audit(csv_path: Path) -> dict[str, Any]:
    audit_path = csv_path.with_suffix(csv_path.suffix + ".audit.json")
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"Missing inference audit sidecar for {csv_path}: {audit_path}"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("valid"):
        raise ValueError(f"Inference audit is not valid: {audit_path}")
    required = ("checkpoint_sha256", "weights", "architecture")
    missing = [key for key in required if not audit.get(key)]
    if missing:
        raise ValueError(
            f"Inference audit lacks {missing}: {audit_path}"
        )
    return audit


def combine_phase2_submissions(
    test_csv: str | Path,
    ranking_csv: str | Path,
    output_csv: str | Path,
    *,
    test_manifest_path: str | Path = "storage/cache/test_manifest.json",
    ranking_manifest_path: str | Path = (
        "storage/cache/ranking_manifest.json"
    ),
) -> dict[str, Any]:
    """Combine two passes of one checkpoint into the official Phase 2 CSV."""
    test_manifest = json.loads(
        resolve_path(test_manifest_path).read_text(encoding="utf-8")
    )
    ranking_manifest = json.loads(
        resolve_path(ranking_manifest_path).read_text(encoding="utf-8")
    )
    if test_manifest["classes"] != ranking_manifest["classes"]:
        raise ValueError("Test and ranking class orders differ")
    test_csv_path = resolve_path(test_csv)
    ranking_csv_path = resolve_path(ranking_csv)
    test_audit = _read_inference_audit(test_csv_path)
    ranking_audit = _read_inference_audit(ranking_csv_path)
    model_identity = (
        test_audit["checkpoint_sha256"],
        test_audit["weights"],
        test_audit["architecture"],
    )
    ranking_model_identity = (
        ranking_audit["checkpoint_sha256"],
        ranking_audit["weights"],
        ranking_audit["architecture"],
    )
    if model_identity != ranking_model_identity:
        raise ValueError(
            "Phase 2 splits were not inferred by the same single-model "
            "checkpoint/weight source"
        )
    test_rows, test_has_id = _read_detection_rows(test_csv_path)
    ranking_rows, ranking_has_id = _read_detection_rows(ranking_csv_path)
    test_ids = {
        str(sample["image_id"]) for sample in test_manifest["samples"]
    }
    ranking_ids = {
        str(sample["image_id"]) for sample in ranking_manifest["samples"]
    }
    if test_ids & ranking_ids:
        raise ValueError("Test and ranking image_id sets overlap")
    wrong_test = {
        row["image_id"] for row in test_rows if row["image_id"] not in test_ids
    }
    wrong_ranking = {
        row["image_id"]
        for row in ranking_rows
        if row["image_id"] not in ranking_ids
    }
    if wrong_test or wrong_ranking:
        raise ValueError(
            "Split CSV contains unknown image ids: "
            f"test={sorted(wrong_test)[:10]}, "
            f"ranking={sorted(wrong_ranking)[:10]}"
        )

    rows = test_rows + ranking_rows
    rows.sort(
        key=lambda row: (
            int(row["image_id"]),
            -float(row["confidence"]),
            int(row["class_id"]),
        )
    )
    destination = write_submission(
        rows,
        resolve_path(output_csv),
        include_id=True,
    )
    combined_samples = (
        test_manifest["samples"] + ranking_manifest["samples"]
    )
    report = validate_submission(
        destination,
        combined_samples,
        num_classes=len(test_manifest["classes"]),
        include_id=True,
    )
    report.update(
        {
            "phase": 2,
            "test_rows": len(test_rows),
            "ranking_rows": len(ranking_rows),
            "input_test_had_id": test_has_id,
            "input_ranking_had_id": ranking_has_id,
            "single_checkpoint_required": True,
            "checkpoint_sha256": model_identity[0],
            "weights": model_identity[1],
            "architecture": model_identity[2],
        }
    )
    atomic_json_dump(
        report,
        destination.with_suffix(destination.suffix + ".audit.json"),
    )
    if not report["valid"]:
        raise ValueError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and validate HOD26 Phase 2 single-model output"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-ranking")
    prepare.add_argument("--image-folder", required=True)
    prepare.add_argument("--expected-images", type=int, default=1000)
    prepare.add_argument("--classes-file", default="data/raw/class.txt")
    prepare.add_argument(
        "--test-manifest",
        default="storage/cache/test_manifest.json",
    )
    prepare.add_argument(
        "--ranking-manifest",
        default="storage/cache/ranking_manifest.json",
    )
    prepare.add_argument(
        "--ranking-coco",
        default="storage/v2/dataset/ranking.json",
    )
    prepare.add_argument(
        "--combined-manifest",
        default="storage/cache/phase2_manifest.json",
    )

    combine = subparsers.add_parser("combine")
    combine.add_argument("--test-csv", required=True)
    combine.add_argument("--ranking-csv", required=True)
    combine.add_argument("--output", required=True)
    combine.add_argument(
        "--test-manifest",
        default="storage/cache/test_manifest.json",
    )
    combine.add_argument(
        "--ranking-manifest",
        default="storage/cache/ranking_manifest.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-ranking":
        report = build_ranking_views(
            args.image_folder,
            expected_images=args.expected_images,
            classes_file=args.classes_file,
            test_manifest_path=args.test_manifest,
            ranking_manifest_path=args.ranking_manifest,
            ranking_coco_path=args.ranking_coco,
            combined_manifest_path=args.combined_manifest,
        )
    else:
        report = combine_phase2_submissions(
            args.test_csv,
            args.ranking_csv,
            args.output,
            test_manifest_path=args.test_manifest,
            ranking_manifest_path=args.ranking_manifest,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
