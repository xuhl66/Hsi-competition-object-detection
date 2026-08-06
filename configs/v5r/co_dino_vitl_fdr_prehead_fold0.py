"""HOD26 V5R: clean, supervised-anchor repair of the V5 localization path."""

import hod26.v5r  # noqa: F401

_base_ = ["../v5/co_dino_vitl_fdr_salience_fold0.py"]

transition_updates = 12000

model = dict(
    type="HOD26V5RCoDETR",
    transition_updates=transition_updates,
    neck=dict(transition_updates=transition_updates),
    query_head=dict(
        type="HOD26V5RCoDINOHead",
        transition_updates=transition_updates,
        pre_loss_weight=1.0,
        transformer=dict(
            decoder=dict(type="HOD26V5RDinoTransformerDecoder"),
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
        # The FDR transition reaches full strength at about e8.5.  Strong
        # Mosaic begins only after e10, so structural migration and the
        # strongest data-distribution shift cannot hit on the same update.
        stage_epochs=(10, 36, 72),
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

# 96 epochs x 1,406 attempted updates = 134,976.  This remains a soft
# horizon: extend the same checkpoint if the multi-metric boundary rises.
runner = dict(max_epochs=96)
lr_config = dict(
    _delete_=True,
    policy="HOD26UpdateCosine",
    points=[
        (0, 0.02),
        (1500, 1.00),
        (12000, 1.00),
        (50000, 0.55),
        (90000, 0.18),
        (115000, 0.04),
        (134976, 0.006),
    ],
)
optimizer = dict(
    constructor="HOD26V5RLayerDecayOptimizerConstructor",
    paramwise_cfg=dict(
        num_layers=24,
        layer_decay_rate=0.8,
        hsi_lr_mult=8.0,
        localization_lr_mult=1.0,
        class_lr_mult=4.0,
    ),
)
checkpoint_config = dict(interval=2, max_keep_ckpts=18, save_optimizer=True)
evaluation = dict(interval=2, save_best="bbox_mAP", rule="greater")
custom_hooks = [
    dict(type="HOD26DataStageHook", priority="VERY_HIGH"),
    dict(
        type="HOD26V5RInheritedEMAHook",
        momentum=0.0001,
        total_iter=2000,
        inherited_updates=42180,
        parent_successful_steps=42168,
        priority="HIGHEST",
    ),
    dict(type="HOD26V5ProgressHook", priority="VERY_HIGH"),
    dict(type="HOD26V5RPostOptimizerEMAHook", priority="BELOW_NORMAL"),
]

load_from = "storage/pretrained/v5r/v4_e30_dfine_to_v5r_init.pth"
resume_from = None
auto_resume = False
work_dir = "storage/v5r/runs/co_dino_vitl_fdr_prehead_fold0"
