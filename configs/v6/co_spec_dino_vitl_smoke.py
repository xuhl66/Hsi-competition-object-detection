"""Two-epoch, eight-image, full-resolution V6 engineering smoke."""

_base_ = ["./co_spec_dino_vitl_fold0.py"]

data = dict(
    train=dict(
        max_images=8,
        pipeline=[
            dict(
                type="LoadHOD26V6HSIFromFile",
                stats_file="storage/v6/dataset/fold0_band_stats.json",
            ),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                type="HOD26V6StagedAugment",
                stage_updates=(0, 2, 8),
                mosaic_probability=1.0,
                widths=dict(
                    stabilize=(1280,), capacity=(1280,),
                    clean=(1280,), polish=(1280,),
                ),
            ),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
        ],
    ),
    val=dict(max_images=2),
    train_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=1,
        persistent_workers=False, pin_memory=True,
    ),
    val_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=1,
        persistent_workers=False, pin_memory=True,
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
evaluation = dict(interval=1, metric="bbox", classwise=True, save_best="bbox_mAP", rule="greater")
log_config = dict(interval=1, hooks=[dict(type="TextLoggerHook")])
log_level = "WARNING"
work_dir = "storage/v6/smoke/co_spec_dino_vitl"
