"""Two-stage engineering smoke for the all-3,000 V6 retraining path."""

_base_ = ["./co_spec_dino_vitl_full_scratch.py"]

model = dict(neck=dict(fusion_open_updates=8))
data = dict(
    train=dict(
        max_images=8,
        pipeline=[
            dict(
                type="LoadHOD26V6HSIFromFile",
                stats_file="storage/v6/dataset/full_band_stats.json",
            ),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                type="HOD26V6StagedAugment",
                stage_updates=(0, 2, 8),
                mosaic_probability=1.0,
                widths=dict(
                    stabilize=(1280,),
                    capacity=(1280,),
                    clean=(1280,),
                    polish=(1280,),
                ),
            ),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
        ],
    ),
)
runner = dict(max_epochs=2)
lr_config = dict(
    _delete_=True,
    policy="HOD26V6SuccessfulCosine",
    points=[(0, 0.10), (1, 1.00), (8, 0.10)],
)
checkpoint_config = dict(
    interval=1, max_keep_ckpts=2, save_optimizer=True, create_symlink=True,
)
evaluation = dict(interval=999999)
custom_hooks = [
    dict(type="HOD26V6ProgressHook", priority="VERY_HIGH"),
    dict(
        type="HOD26V6EMAHook",
        momentum=0.0001,
        total_iter=8,
        priority="HIGHEST",
    ),
    dict(type="HOD26V6PostOptimizerEMAHook", priority="BELOW_NORMAL"),
    dict(
        type="HOD26V6FullDataContractHook",
        expected_images=8,
        updates_per_epoch=4,
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
log_config = dict(interval=1, hooks=[dict(type="TextLoggerHook")])
log_level = "WARNING"
work_dir = "storage/v6/smoke/co_spec_dino_vitl_full_scratch"
