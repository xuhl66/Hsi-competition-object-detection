"""Fail-closed gate for public-initialized V6 all-3,000 retraining."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch
from mmcv import Config

from .full import (
    FULL_CANDIDATE_EPOCHS,
    FULL_CLASS_COUNTS,
    FULL_IMAGES,
    FULL_LR_POINTS,
    FULL_SOFT_HORIZON_UPDATES,
    FULL_STAGE_UPDATES,
    FULL_UPDATES_PER_EPOCH,
)
from .load_gate import (
    REPO_ROOT,
    V6LoadGateError,
    _build_schema,
    _validate_primary_state,
    sha256_file,
    validate as validate_v6,
    verify_runtime_source_lock,
)


FULL_CONFIG = "configs/v6/co_spec_dino_vitl_full_scratch.py"
FOLD0_CONFIG = "configs/v6/co_spec_dino_vitl_fold0.py"
FULL_SOURCE_LOCK = "configs/v6/runtime_source_lock_full_scratch.json"
FOLD0_SOURCE_LOCK = "configs/v6/runtime_source_lock.json"
FULL_ANNOTATIONS = "storage/v2/dataset/full.json"
FULL_STATS = "storage/v6/dataset/full_band_stats.json"
PUBLIC_INIT = "storage/pretrained/v6/co_spec_dino_vitl_public_init.pth"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return (REPO_ROOT / value).resolve() if not value.is_absolute() else value.resolve()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V6LoadGateError(message)


def _annotation_contract(path: str | Path) -> dict[str, Any]:
    annotation_path = _resolve(path)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    counts = Counter(int(row["category_id"]) for row in payload["annotations"])
    ordered = tuple(counts[index] for index in range(18))
    _require(len(payload["images"]) == FULL_IMAGES, "V6 Full is not exactly 3,000 images")
    _require(ordered == FULL_CLASS_COUNTS, "V6 Full annotation class counts changed")
    return {
        "annotations": str(annotation_path),
        "annotations_sha256": sha256_file(annotation_path),
        "images": len(payload["images"]),
        "annotations_count": len(payload["annotations"]),
        "class_counts": list(ordered),
    }


def _stats_contract(path: str | Path) -> dict[str, Any]:
    stats_path = _resolve(path)
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    _require(
        payload.get("scope") == "all_3000_official_train_images",
        "V6 Full statistics scope changed",
    )
    _require(int(payload.get("samples", -1)) == FULL_IMAGES, "V6 Full stats sample count changed")
    _require(len(payload.get("low", ())) == 16, "V6 Full low statistics are malformed")
    _require(len(payload.get("high", ())) == 16, "V6 Full high statistics are malformed")
    return {
        "stats": str(stats_path),
        "stats_sha256": sha256_file(stats_path),
        "stats_scope": payload["scope"],
        "stats_samples": int(payload["samples"]),
    }


def _config_contract(path: str | Path) -> dict[str, Any]:
    config_path = _resolve(path)
    config = Config.fromfile(str(config_path))
    fold = Config.fromfile(str(_resolve(FOLD0_CONFIG)))
    pipeline = config.data.train.pipeline
    hooks = config.custom_hooks
    _require(config.model.type == "HOD26V6CoSpecDINO", "V6 Full model type changed")
    _require(config.model.neck.fusion_open_updates == 7500, "V6 Full fusion timing changed")
    _require(config.data.train.ann_file == FULL_ANNOTATIONS, "V6 Full annotation path changed")
    _require(
        pipeline[0].stats_file == FULL_STATS,
        "V6 Full normalization must use all-3,000 statistics",
    )
    _require(tuple(pipeline[2].stage_updates) == FULL_STAGE_UPDATES, "V6 Full stages changed")
    _require(pipeline[2].mosaic_probability == 0.25, "V6 Full Mosaic policy changed")
    _require(config.runner.max_epochs == 80, "V6 Full default soft horizon changed")
    _require(
        tuple(tuple(point) for point in config.lr_config.points) == FULL_LR_POINTS,
        "V6 Full LR schedule changed",
    )
    _require(config.checkpoint_config.interval == 2, "V6 Full checkpoint interval changed")
    _require(config.checkpoint_config.max_keep_ckpts == -1, "V6 Full snapshots may be deleted")
    _require(config.evaluation.interval == 999999, "V6 Full validation must stay disabled")
    _require(config.optimizer == fold.optimizer, "V6 Full optimizer differs from Fold0")
    _require(hooks[0].type == "HOD26V6ProgressHook", "V6 Full progress hook changed")
    _require(hooks[1].type == "HOD26V6EMAHook", "V6 Full EMA owner changed")
    _require(hooks[1].total_iter == 2500, "V6 Full EMA exposure scaling changed")
    _require(hooks[3].type == "HOD26V6FullDataContractHook", "V6 Full contract hook missing")
    _require(hooks[4].source_lock == FULL_SOURCE_LOCK, "V6 Full source lock changed")
    _require(config.load_from == PUBLIC_INIT, "V6 Full no longer starts from public init")
    _require(config.resume_from is None and config.auto_resume is False, "Unsafe V6 Full auto-resume")
    return {
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "updates_per_epoch": FULL_UPDATES_PER_EPOCH,
        "soft_horizon_epochs": 80,
        "soft_horizon_attempted_updates": FULL_SOFT_HORIZON_UPDATES,
        "stage_exposure_updates": list(FULL_STAGE_UPDATES),
        "lr_successful_update_points": [list(point) for point in FULL_LR_POINTS],
        "candidate_epochs": list(FULL_CANDIDATE_EPOCHS),
        "candidate_attempted_updates": [
            epoch * FULL_UPDATES_PER_EPOCH for epoch in FULL_CANDIDATE_EPOCHS
        ],
    }


def _strict_public_target_schema(
    checkpoint_path: str | Path, config_path: str | Path
) -> dict[str, Any]:
    _, schema, _, _ = _build_schema(config_path)
    payload = torch.load(_resolve(checkpoint_path), map_location="cpu")
    try:
        _require("optimizer" not in payload, "V6 public init unexpectedly has optimizer state")
        report = _validate_primary_state(payload["state_dict"], schema, with_ema=False)
    finally:
        del payload, schema
        gc.collect()
    return report


def _full_resume_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    payload = torch.load(_resolve(checkpoint_path), map_location="cpu")
    try:
        meta = payload.get("meta")
        _require(isinstance(meta, Mapping), "V6 Full checkpoint metadata is missing")
        expected = {
            "hod26_v6_full_data_stage": True,
            "hod26_v6_full_training_mode": "public_init_all3000_complete_retrain",
            "hod26_v6_data_scope": "all_3000_official_train_images",
            "hod26_v6_full_images": FULL_IMAGES,
            "hod26_v6_full_updates_per_epoch": FULL_UPDATES_PER_EPOCH,
            "hod26_v6_full_soft_horizon_updates": FULL_SOFT_HORIZON_UPDATES,
            "hod26_v6_full_stage_updates": list(FULL_STAGE_UPDATES),
            "hod26_v6_full_candidate_epochs": list(FULL_CANDIDATE_EPOCHS),
            "hod26_v6_full_candidate_updates": [
                epoch * FULL_UPDATES_PER_EPOCH for epoch in FULL_CANDIDATE_EPOCHS
            ],
            "hod26_v6_full_class_counts": list(FULL_CLASS_COUNTS),
            "hod26_v6_full_class_weight_scope": "all_3000_annotations",
            "hod26_v6_full_augmentation_coordinate": "completed_dataset_passes",
            "hod26_v6_full_validation_policy": "none_all_labeled_images_in_training",
        }
        for key, value in expected.items():
            _require(meta.get(key) == value, f"V6 Full checkpoint contract changed: {key}")
        return {key: meta[key] for key in expected}
    finally:
        del payload
        gc.collect()


def validate(
    *,
    mode: str,
    checkpoint_path: str | Path = PUBLIC_INIT,
    config_path: str | Path = FULL_CONFIG,
    source_lock: str | Path = FULL_SOURCE_LOCK,
    target_max_epochs: int | None = None,
) -> dict[str, Any]:
    runtime = verify_runtime_source_lock(source_lock)
    contract = {
        **_config_contract(config_path),
        **_annotation_contract(FULL_ANNOTATIONS),
        **_stats_contract(FULL_STATS),
    }
    if mode == "public_init":
        public = validate_v6(
            mode="public_init",
            checkpoint_path=checkpoint_path,
            config_path=FOLD0_CONFIG,
            source_lock=FOLD0_SOURCE_LOCK,
        )
        target_schema = _strict_public_target_schema(checkpoint_path, config_path)
        report = {
            **public,
            **runtime,
            **contract,
            **target_schema,
            "mode": mode,
            "initialization": "public_checkpoint_only_no_v6_parent",
            "optimizer": "fresh",
            "ema": "fresh_post_successful_step",
            "fp16_scaler": "fresh",
            "class_weight_retarget": "fold0_buffer_to_all3000_before_update_zero",
        }
    elif mode == "resume":
        report = validate_v6(
            mode="resume",
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            source_lock=source_lock,
        )
        report.update(contract)
        report.update(_full_resume_metadata(checkpoint_path))
        _require(
            report["attempted_updates"]
            == report["epoch"] * FULL_UPDATES_PER_EPOCH,
            "V6 Full checkpoint is not at an exact 1,500-update epoch boundary",
        )
        if target_max_epochs is not None:
            _require(
                int(target_max_epochs) > report["epoch"],
                "V6 Full target max epoch must exceed the resumed epoch",
            )
    else:
        raise V6LoadGateError(f"Unknown V6 Full gate mode: {mode}")
    report.update(
        gate="passed",
        full_source_lock=str(_resolve(source_lock)),
        full_source_lock_sha256=runtime["source_lock_sha256"],
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("public_init", "resume"))
    parser.add_argument("--checkpoint", default=PUBLIC_INIT)
    parser.add_argument("--config", default=FULL_CONFIG)
    parser.add_argument("--source-lock", default=FULL_SOURCE_LOCK)
    parser.add_argument("--target-max-epochs", type=int)
    args = parser.parse_args()
    report = validate(
        mode=args.mode,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        source_lock=args.source_lock,
        target_max_epochs=args.target_max_epochs,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
