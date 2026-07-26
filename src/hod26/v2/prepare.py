from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from hod26.config import atomic_json_dump, resolve_path


OFFICIAL_DEIM_COMMIT = "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"
OFFICIAL_OBJECT365_COCO_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1RMNrHh3bYN0FfT5ZlWhXtQxkG23xb2xj&export=download&confirm=t"
)
OFFICIAL_OBJECT365_COCO_SHA256 = (
    "f240a1128280eb6ce0fff911e8811a8dff669e18be25f5160db7b4263452277d"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_pretrained(path: str | Path) -> dict[str, Any]:
    checkpoint = resolve_path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing official V2 checkpoint: {checkpoint}. "
            "Run tools/bootstrap_v2.sh first."
        )
    actual = _sha256(checkpoint)
    if actual != OFFICIAL_OBJECT365_COCO_SHA256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch: expected "
            f"{OFFICIAL_OBJECT365_COCO_SHA256}, got {actual}"
        )
    provenance = {
        "file": str(checkpoint),
        "sha256": actual,
        "url": OFFICIAL_OBJECT365_COCO_URL,
        "upstream_commit": OFFICIAL_DEIM_COMMIT,
        "architecture": "DEIM-D-FINE-X / HGNetv2-B5",
        "source_pretraining": "Objects365 followed by 24-epoch MS COCO fine-tuning",
        "reported_coco_ap_50_95": 0.595,
        "declared_external_pretraining": True,
    }
    atomic_json_dump(
        provenance,
        checkpoint.with_suffix(checkpoint.suffix + ".provenance.json"),
    )
    return provenance


def _coco_categories(classes: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {"id": index, "name": name, "supercategory": "object"}
        for index, name in enumerate(classes)
    ]


def _coco_images(
    samples: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "id": int(sample["image_id"]),
            "file_name": Path(sample["image"]).name,
            "width": int(sample["width"]),
            "height": int(sample["height"]),
        }
        for sample in samples
    ]


def _coco_annotations(
    samples: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for sample in samples:
        for obj in sample.get("objects", []):
            x1, y1, x2, y2 = (float(value) for value in obj["bbox"])
            width = x2 - x1
            height = y2 - y1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": int(sample["image_id"]),
                    "category_id": int(obj["class_id"]),
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return annotations


def _write_coco(
    path: Path,
    samples: Sequence[dict[str, Any]],
    classes: Sequence[str],
    include_annotations: bool = True,
) -> None:
    payload = {
        "info": {
            "description": "HOD26 16-band X2Cube detection",
            "version": "2",
        },
        "licenses": [],
        "images": _coco_images(samples),
        "annotations": (
            _coco_annotations(samples) if include_annotations else []
        ),
        "categories": _coco_categories(classes),
    }
    atomic_json_dump(payload, path)


def prepare_v2(
    manifest_path: str | Path = "storage/cache/train_manifest.json",
    test_manifest_path: str | Path = "storage/cache/test_manifest.json",
    folds_path: str | Path = "storage/cache/folds.json",
    output_root: str | Path = "storage/v2/dataset",
    fold: int = 0,
) -> dict[str, Any]:
    manifest = json.loads(resolve_path(manifest_path).read_text(encoding="utf-8"))
    test_manifest = json.loads(
        resolve_path(test_manifest_path).read_text(encoding="utf-8")
    )
    folds = json.loads(resolve_path(folds_path).read_text(encoding="utf-8"))
    fold_by_id = folds["fold_by_image_id"]
    classes = manifest["classes"]
    samples = manifest["samples"]
    train = [
        sample
        for sample in samples
        if int(fold_by_id[str(sample["image_id"])]) != fold
    ]
    validation = [
        sample
        for sample in samples
        if int(fold_by_id[str(sample["image_id"])]) == fold
    ]
    output = resolve_path(output_root)
    fold_root = output / f"fold{fold}"
    fold_root.mkdir(parents=True, exist_ok=True)
    _write_coco(fold_root / "train.json", train, classes)
    _write_coco(fold_root / "val.json", validation, classes)
    _write_coco(output / "full.json", samples, classes)
    _write_coco(
        output / "test.json",
        test_manifest["samples"],
        classes,
        include_annotations=False,
    )
    report = {
        "fold": fold,
        "train_images": len(train),
        "validation_images": len(validation),
        "full_images": len(samples),
        "test_images": len(test_manifest["samples"]),
        "classes": classes,
        "fold_root": str(fold_root),
    }
    atomic_json_dump(report, fold_root / "alignment.json")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare immutable COCO views over the official 16-band data"
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        default="storage/pretrained/v2/deim_dfine_x_object365.pth",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = prepare_v2(fold=args.fold)
    report["pretraining"] = verify_pretrained(args.checkpoint)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
