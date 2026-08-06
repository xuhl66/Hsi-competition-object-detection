"""Post-selection full-data V4 schedule with the same absolute update budget."""

_base_ = ["./co_dino_vitl_hsi_fold0.py"]

data = dict(
    train=dict(
        ann_file="storage/v2/dataset/full.json",
        repeat_class_factors={5: 2, 8: 2, 14: 2, 16: 2},
        pipeline=[
            dict(
                type="LoadHOD26HSIFromFile",
                stats_file="storage/cache/band_stats.json",
            ),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                type="HOD26StagedAugment",
                stage_epochs=(3, 32, 45),
                mosaic_probability=0.25,
                mixup_probability=0.0,
                widths=dict(
                    stabilize=(896, 1024),
                    strong=(896, 960, 1024, 1088, 1152, 1216, 1280),
                    refine=(1024, 1152, 1280),
                    localize=(1152, 1280),
                ),
            ),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
        ],
    )
)

# Full data has 3,518 exposures and therefore 1,759 updates/epoch on two GPUs.
# 58 epochs provide 102,022 continuous optimizer updates.
runner = dict(max_epochs=58)
work_dir = "storage/v4/runs/co_dino_vitl_hsi_full"
