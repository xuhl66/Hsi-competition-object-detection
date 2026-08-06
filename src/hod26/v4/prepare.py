"""Prepare leakage-free Fold0 statistics for the V4 data path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hod26.config import atomic_json_dump
from hod26.data import compute_band_statistics


def prepare_fold_statistics(
    *,
    manifest_path: str | Path,
    folds_path: str | Path,
    data_root: str | Path,
    output_path: str | Path,
    fold: int = 0,
) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    folds = json.loads(Path(folds_path).read_text(encoding="utf-8"))
    fold_by_id = folds["fold_by_image_id"]
    samples = [
        sample
        for sample in manifest["samples"]
        if int(fold_by_id[str(sample["image_id"])]) != int(fold)
    ]
    statistics = compute_band_statistics(samples, Path(data_root))
    statistics.update(
        {
            "scope": f"fold{fold}_train_only",
            "excluded_validation_fold": int(fold),
            "manifest": str(Path(manifest_path).resolve()),
            "folds": str(Path(folds_path).resolve()),
        }
    )
    atomic_json_dump(statistics, output_path)
    return statistics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute train-only robust band limits for HOD26 V4"
    )
    parser.add_argument("--manifest", default="storage/cache/train_manifest.json")
    parser.add_argument("--folds", default="storage/cache/folds.json")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output", default="storage/v4/dataset/fold0_band_stats.json")
    parser.add_argument("--fold", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = prepare_fold_statistics(
        manifest_path=args.manifest,
        folds_path=args.folds,
        data_root=args.data_root,
        output_path=args.output,
        fold=args.fold,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
