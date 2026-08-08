"""Create independent Fold0/full-data robust statistics for V6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hod26.config import atomic_json_dump
from hod26.data import compute_band_statistics


def prepare_statistics(
    *,
    manifest_path: str | Path,
    folds_path: str | Path,
    data_root: str | Path,
    fold_output: str | Path,
    full_output: str | Path,
    fold: int = 0,
) -> dict:
    manifest_path, folds_path = Path(manifest_path), Path(folds_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    fold_by_id = folds["fold_by_image_id"]
    samples = manifest["samples"]
    train_samples = [
        sample for sample in samples
        if int(fold_by_id[str(sample["image_id"])]) != int(fold)
    ]
    fold_stats = compute_band_statistics(train_samples, Path(data_root))
    fold_stats.update(
        scope=f"fold{fold}_train_only",
        excluded_validation_fold=int(fold),
        manifest=str(manifest_path.resolve()),
        folds=str(folds_path.resolve()),
    )
    full_stats = compute_band_statistics(samples, Path(data_root))
    full_stats.update(
        scope="all_3000_official_train_images",
        excluded_validation_fold=None,
        manifest=str(manifest_path.resolve()),
        folds=str(folds_path.resolve()),
    )
    atomic_json_dump(fold_stats, fold_output)
    atomic_json_dump(full_stats, full_output)
    return {"fold0": fold_stats, "full": full_stats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="storage/cache/train_manifest.json")
    parser.add_argument("--folds", default="storage/cache/folds.json")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--fold-output", default="storage/v6/dataset/fold0_band_stats.json")
    parser.add_argument("--full-output", default="storage/v6/dataset/full_band_stats.json")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    report = prepare_statistics(
        manifest_path=args.manifest,
        folds_path=args.folds,
        data_root=args.data_root,
        fold_output=args.fold_output,
        full_output=args.full_output,
        fold=args.fold,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
