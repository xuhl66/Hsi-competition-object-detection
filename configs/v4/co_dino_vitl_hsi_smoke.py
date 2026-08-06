"""Engineering-only V4 smoke; never use its metrics as model evidence."""

_base_ = ["./co_dino_vitl_hsi_fold0.py"]

data = dict(
    train=dict(
        max_images=4,
        repeat_class_factors=None,
        pipeline=[
            dict(
                type="LoadHOD26HSIFromFile",
                stats_file="storage/v4/dataset/fold0_band_stats.json",
            ),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                type="HOD26StagedAugment",
                # Epoch 0 deliberately exercises Mosaic; the resumed epoch
                # exercises the no-Mosaic refinement transition.
                stage_epochs=(0, 1, 2),
                mosaic_probability=1.0,
                mixup_probability=0.0,
                widths=dict(
                    stabilize=(1280,),
                    strong=(1280,),
                    refine=(1280,),
                    localize=(1280,),
                ),
            ),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
        ],
    ),
    val=dict(max_images=2),
    train_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=1,
        persistent_workers=False,
        pin_memory=True,
    ),
    val_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=1,
        persistent_workers=False,
        pin_memory=True,
    ),
)

runner = dict(max_epochs=2)
lr_config = dict(
    _delete_=True,
    policy="HOD26UpdateCosine",
    points=[(0, 0.1), (1, 1.0), (4, 0.1)],
)
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=2,
    save_optimizer=True,
    create_symlink=True,
)
evaluation = dict(
    interval=1,
    metric="bbox",
    classwise=True,
    save_best="bbox_mAP",
    rule="greater",
)
log_config = dict(interval=1, hooks=[dict(type="TextLoggerHook")])
log_level = "WARNING"
work_dir = "storage/v4/smoke/co_dino_vitl_hsi_object_aware"
