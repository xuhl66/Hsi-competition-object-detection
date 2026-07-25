from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import load_config, resolve_path


BASE_COLUMNS = [
    "image_id",
    "class_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
]


def submission_columns(include_id: bool = True) -> list[str]:
    return (["id"] if include_id else []) + BASE_COLUMNS


def write_submission(
    detections: Iterable[dict[str, Any]],
    output_path: str | Path,
    include_id: bool = True,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    columns = submission_columns(include_id)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for index, row in enumerate(detections):
            clean = {
                "image_id": str(row["image_id"]),
                "class_id": int(row["class_id"]),
                "confidence": f"{float(row['confidence']):.8f}",
                "x1": f"{float(row['x1']):.4f}",
                "y1": f"{float(row['y1']):.4f}",
                "x2": f"{float(row['x2']):.4f}",
                "y2": f"{float(row['y2']):.4f}",
            }
            if include_id:
                clean["id"] = index
            writer.writerow(clean)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output_path)
    return output_path


def validate_submission(
    csv_path: str | Path,
    expected_samples: Sequence[dict[str, Any]],
    num_classes: int,
    include_id: bool = True,
) -> dict[str, Any]:
    csv_path = Path(csv_path)
    expected = {str(sample["image_id"]): sample for sample in expected_samples}
    columns = submission_columns(include_id)
    errors: list[str] = []
    error_count = 0
    seen_images: set[str] = set()
    class_counts = [0] * num_classes
    rows = 0

    def add_error(message: str) -> None:
        nonlocal error_count
        error_count += 1
        if len(errors) < 100:
            errors.append(message)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            add_error(
                f"header mismatch: expected {columns}, got {reader.fieldnames}"
            )
        for row_number, row in enumerate(reader, start=2):
            rows += 1
            try:
                image_id = str(row["image_id"]).strip()
                class_id = int(row["class_id"])
                confidence = float(row["confidence"])
                coordinates = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
                if include_id and int(row["id"]) != row_number - 2:
                    add_error(
                        f"row {row_number}: id must be contiguous from zero"
                    )
                    continue
            except (KeyError, TypeError, ValueError) as exc:
                add_error(f"row {row_number}: invalid value ({exc})")
                continue

            if image_id not in expected:
                add_error(f"row {row_number}: unknown image_id {image_id!r}")
                continue
            if not 0 <= class_id < num_classes:
                add_error(f"row {row_number}: class_id {class_id} out of range")
            elif 0 <= class_id < num_classes:
                class_counts[class_id] += 1
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                add_error(f"row {row_number}: confidence must be in [0, 1]")
            if not all(math.isfinite(value) for value in coordinates):
                add_error(f"row {row_number}: non-finite coordinate")
                continue
            x1, y1, x2, y2 = coordinates
            sample = expected[image_id]
            width, height = float(sample["width"]), float(sample["height"])
            epsilon = 1e-3
            if x1 < -epsilon or y1 < -epsilon or x2 > width + epsilon or y2 > height + epsilon:
                add_error(
                    f"row {row_number}: box outside {int(width)}x{int(height)} cube"
                )
            if x2 <= x1 or y2 <= y1:
                add_error(f"row {row_number}: degenerate box")
            seen_images.add(image_id)

    if rows == 0:
        add_error("submission contains no detections")
    report = {
        "valid": error_count == 0,
        "path": str(csv_path.resolve()),
        "rows": rows,
        "expected_test_images": len(expected),
        "images_with_detections": len(seen_images),
        "images_without_detections": len(expected) - len(seen_images),
        "class_detection_counts": class_counts,
        "columns": columns,
        "error_count": error_count,
        "errors": errors,
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strictly validate a HOD26 CSV")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--csv", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    manifest_path = resolve_path(config["data"]["test_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = validate_submission(
        args.csv,
        manifest["samples"],
        num_classes=int(config["model"]["num_classes"]),
        include_id=bool(config["inference"]["include_id"]),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
