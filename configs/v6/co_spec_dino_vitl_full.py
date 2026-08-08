"""State-continuous all-3,000 clean finalization for a promoted V6 Fold0 model."""

import os

_base_ = ["./co_spec_dino_vitl_fold0.py"]

start_update = int(os.environ.get("V6_FULL_START_UPDATE", "48000"))
end_update = int(os.environ.get("V6_FULL_END_UPDATE", "58500"))
start_ratio = float(os.environ.get("V6_FULL_START_RATIO", "0.06"))
target_max_epoch = int(os.environ.get("V6_FULL_TARGET_MAX_EPOCH", "47"))
if start_update <= 0 or end_update <= start_update or not 0 < start_ratio <= 0.06:
    raise ValueError("Invalid V6 full-data continuation coordinates")

full_train_pipeline = [
    dict(
        type="LoadHOD26V6HSIFromFile",
        stats_file="storage/v6/dataset/full_band_stats.json",
    ),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="HOD26V6StagedAugment",
        stage_updates=(0, 0, 0),
        mosaic_probability=0.0,
        widths=dict(
            stabilize=(1280,), capacity=(1280,), clean=(1280,), polish=(1280,),
        ),
    ),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

data = dict(
    train=dict(
        ann_file="storage/v2/dataset/full.json",
        pipeline=full_train_pipeline,
    ),
)
runner = dict(max_epochs=target_max_epoch)
lr_config = dict(
    _delete_=True,
    policy="HOD26V6SuccessfulCosine",
    points=[
        (0, start_ratio),
        (start_update, start_ratio),
        (end_update, 0.004),
    ],
)
checkpoint_config = dict(
    interval=1, max_keep_ckpts=12, save_optimizer=True, create_symlink=True,
)
evaluation = dict(interval=999999)
work_dir = "storage/v6/runs/co_spec_dino_vitl_full"
