from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import atomic_json_dump, load_config, resolve_path, seed_everything
from .data import (
    build_manifests,
    compute_band_statistics,
    count_classes,
    load_classes,
    make_balanced_folds,
)


def prepare(config: dict, force: bool = False) -> dict:
    project_root = Path.cwd()
    data_cfg = config["data"]
    seed = int(config["project"]["seed"])
    seed_everything(seed)

    data_root = resolve_path(data_cfg["root"], project_root)
    train_images = data_root / data_cfg["train_images"]
    train_annotations = data_root / data_cfg["train_annotations"]
    test_images = data_root / data_cfg["test_images"]
    class_path = data_root / data_cfg["classes"]
    manifest_path = resolve_path(data_cfg["manifest"], project_root)
    test_manifest_path = resolve_path(data_cfg["test_manifest"], project_root)
    stats_path = resolve_path(data_cfg["stats"], project_root)
    folds_path = resolve_path(data_cfg["folds"], project_root)

    classes = load_classes(class_path)
    started = time.time()
    if force or not (manifest_path.exists() and test_manifest_path.exists()):
        print("Building sanitized manifests and scene hashes...")
        train_manifest, test_manifest = build_manifests(
            data_root,
            train_images,
            train_annotations,
            test_images,
            classes,
        )
        atomic_json_dump(train_manifest, manifest_path)
        atomic_json_dump(test_manifest, test_manifest_path)
    else:
        train_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        test_manifest = json.loads(test_manifest_path.read_text(encoding="utf-8"))

    if force or not stats_path.exists():
        print("Computing streaming 16-band robust statistics...")
        stats = compute_band_statistics(
            train_manifest["samples"],
            data_root,
            bins=int(data_cfg["histogram_bins"]),
            quantile_low=float(data_cfg["quantile_low"]),
            quantile_high=float(data_cfg["quantile_high"]),
        )
        atomic_json_dump(stats, stats_path)
    else:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))

    existing_folds = None
    if folds_path.exists():
        existing_folds = json.loads(folds_path.read_text(encoding="utf-8"))
    if (
        force
        or existing_folds is None
        or int(existing_folds.get("version", 0)) < 2
    ):
        print("Building grouped multilabel-balanced folds...")
        folds = make_balanced_folds(
            train_manifest["samples"],
            num_folds=int(data_cfg["num_folds"]),
            num_classes=len(classes),
            seed=seed,
        )
        atomic_json_dump(folds, folds_path)
    else:
        folds = existing_folds

    audit = train_manifest["audit"]
    report = {
        "train_images": len(train_manifest["samples"]),
        "test_images": len(test_manifest["samples"]),
        "objects_after_cleaning": audit["objects_after_cleaning"],
        "class_counts_after_cleaning": count_classes(
            train_manifest["samples"], len(classes)
        ),
        "official_issue_files": len(audit["issue_files"]),
        "fold_image_counts": folds["fold_image_counts"],
        "fold_class_counts": folds["fold_class_counts"],
        "stats_low": stats["low"],
        "stats_high": stats["high"],
        "elapsed_seconds": round(time.time() - started, 3),
        "manifest": str(manifest_path),
        "test_manifest": str(test_manifest_path),
        "stats": str(stats_path),
        "folds": str(folds_path),
    }
    report_path = manifest_path.parent / "data_alignment_report.json"
    atomic_json_dump(report, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit and align the HOD26 dataset")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare(load_config(args.config), force=args.force)


if __name__ == "__main__":
    main()
