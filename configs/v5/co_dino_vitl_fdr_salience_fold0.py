"""HOD26 V5: converged V4 parent plus localization-first mature modules."""

import hod26.v5  # noqa: F401

_base_ = ["../v4/co_dino_vitl_hsi_fold0.py"]

model = dict(
    type="HOD26V5CoDETR",
    salience_loss_weight=0.6,
    backbone=dict(type="HOD26V5HSIViT"),
    neck=dict(
        type="HOD26V5SalienceSFP",
        transition_updates=4000,
        salience_size_boundaries=(64, 128, 256),
    ),
    query_head=dict(
        type="HOD26V5CoDINOHead",
        reg_max=32,
        # Matches the loaded official D-FINE-X Objects365->COCO FDR weights.
        reg_scale=8.0,
        transition_updates=4000,
        fgl_weight=0.15,
        go_lsd_weight=1.5,
        go_lsd_temperature=5.0,
        transformer=dict(
            decoder=dict(type="HOD26V5DinoTransformerDecoder"),
        ),
    ),
)

train_pipeline = [
    dict(
        type="LoadHOD26HSIFromFile",
        stats_file="storage/v4/dataset/fold0_band_stats.json",
    ),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="HOD26V5StagedAugment",
        # 1,406 updates/epoch: 4k≈e3, 36k≈e26, 90k≈e64.
        stage_epochs=(3, 26, 64),
        mosaic_probability=0.18,
        mixup_probability=0.0,
        widths=dict(
            stabilize=(896, 1024),
            strong=(896, 960, 1024, 1088, 1152, 1216, 1280),
            refine=(1152, 1280),
            localize=(1152, 1280),
        ),
    ),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
data = dict(train=dict(pipeline=train_pipeline))

# 2,812 exposures/epoch, global batch 2 => 1,406 optimizer updates/epoch.
# 84 epochs is a 118,104-update soft horizon, never an automatic stop.
runner = dict(max_epochs=84)
lr_config = dict(
    _delete_=True,
    policy="HOD26UpdateCosine",
    points=[
        (0, 0.02),
        (1500, 1.00),
        (12000, 1.00),
        (42000, 0.55),
        (75000, 0.18),
        (100000, 0.04),
        (118104, 0.006),
    ],
)
optimizer = dict(
    constructor="HOD26V5LayerDecayOptimizerConstructor",
    paramwise_cfg=dict(
        num_layers=24,
        layer_decay_rate=0.8,
        hsi_lr_mult=8.0,
        class_lr_mult=4.0,
    ),
)
checkpoint_config = dict(interval=2, max_keep_ckpts=16, save_optimizer=True)
evaluation = dict(interval=2, save_best="bbox_mAP", rule="greater")
custom_hooks = [
    dict(type="HOD26DataStageHook", priority="VERY_HIGH"),
    dict(
        type="HOD26V5InheritedEMAHook",
        momentum=0.0001,
        total_iter=2000,
        inherited_updates=42180,
        priority="HIGHEST",
    ),
    dict(type="HOD26V5ProgressHook", priority="VERY_HIGH"),
]

load_from = "storage/pretrained/v5/v4_e30_dfine_to_v5_init.pth"
resume_from = None
auto_resume = False
work_dir = "storage/v5/runs/co_dino_vitl_fdr_salience_fold0"
