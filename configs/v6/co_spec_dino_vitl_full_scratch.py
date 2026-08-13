"""Public-initialized, all-3,000 complete V6 retraining configuration."""

import hod26.v6  # noqa: F401

_base_ = ["./co_spec_dino_vitl_fold0.py"]

custom_imports = dict(imports=["hod26.v6.full"], allow_failed_imports=False)

# Preserve the Fold0 schedule in completed full-dataset exposures.  With a
# global batch of two, all 3,000 scenes produce 1,500 attempted updates/epoch.
full_train_pipeline = [
    dict(
        type="LoadHOD26V6HSIFromFile",
        stats_file="storage/v6/dataset/full_band_stats.json",
    ),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="HOD26V6StagedAugment",
        stage_updates=(7500, 70500, 100500),
        mosaic_probability=0.25,
        widths=dict(
            stabilize=(896, 1024, 1152),
            capacity=(896, 960, 1024, 1088, 1152, 1216, 1280),
            clean=(1152, 1280),
            polish=(1280,),
        ),
    ),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

model = dict(neck=dict(fusion_open_updates=7500))
data = dict(
    train=dict(
        ann_file="storage/v2/dataset/full.json",
        pipeline=full_train_pipeline,
    ),
)

# 80 epochs / 120,000 attempted updates is the complete soft horizon.  It is
# intentionally long enough to retain every predeclared Fold0-matched window.
runner = dict(max_epochs=80)
lr_config = dict(
    _delete_=True,
    policy="HOD26V6SuccessfulCosine",
    points=[
        (0, 0.02),
        (1875, 1.00),
        (7500, 1.00),
        (70500, 0.30),
        (100500, 0.06),
        (115000, 0.015),
        (120000, 0.004),
    ],
)

# Keep all 40 two-epoch snapshots.  The predeclared e34/e40/e46/e52/e60
# candidates must survive an e80 run and any later seamless extension.
checkpoint_config = dict(
    interval=2, max_keep_ckpts=-1, save_optimizer=True, create_symlink=True,
)
evaluation = dict(interval=999999)
custom_hooks = [
    dict(type="HOD26V6ProgressHook", priority="VERY_HIGH"),
    dict(
        type="HOD26V6EMAHook",
        momentum=0.0001,
        total_iter=2500,
        priority="HIGHEST",
    ),
    dict(type="HOD26V6PostOptimizerEMAHook", priority="BELOW_NORMAL"),
    dict(
        type="HOD26V6FullDataContractHook",
        expected_images=3000,
        updates_per_epoch=1500,
        class_counts=(
            447, 493, 1375, 621, 369, 565, 355, 451, 461,
            475, 658, 258, 184, 310, 1095, 977, 272, 700,
        ),
        stage_updates=(7500, 70500, 100500),
        soft_horizon_updates=120000,
        candidate_epochs=(34, 40, 46, 52, 60),
        priority="HIGH",
    ),
    dict(
        type="HOD26V6CheckpointContractHook",
        source_lock="configs/v6/runtime_source_lock_full_scratch.json",
        initialization_provenance=(
            "storage/pretrained/v6/"
            "co_spec_dino_vitl_public_init.pth.provenance.json"
        ),
        priority="NORMAL",
    ),
]

load_from = "storage/pretrained/v6/co_spec_dino_vitl_public_init.pth"
resume_from = None
auto_resume = False
work_dir = "storage/v6/runs/co_spec_dino_vitl_full_scratch"
