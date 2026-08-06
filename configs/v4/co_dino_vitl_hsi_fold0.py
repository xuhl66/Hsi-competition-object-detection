"""HOD26 V4: final-capacity Co-DINO ViT-L 16-band Fold0 long run."""

import hod26.v4  # noqa: F401 - register V4 modules before Config construction


_base_ = [
    "../../storage/upstream/Co-DETR/projects/configs/co_dino_vit/"
    "co_dino_5scale_vit_large_coco.py"
]

num_classes = 18
spectral_channels = (64, 96, 128, 192)

# Preserve the complete official Co-DINO/ViT-L graph and replace only the
# input/fusion paths and target-specific heads. Query, RPN/ROI and ATSS heads
# all train jointly; inference remains the single Co-DINO query detector.
model = dict(
    type="HOD26ObjectAwareCoDETR",
    # OSSDet's published optimum: alpha=0.6 for the activation objective and
    # gamma=0.1 for foreground/background balance.
    object_activation_loss_weight=0.6,
    object_activation_background_weight=0.1,
    backbone=dict(
        type="HOD26HSIViT",
        input_bands=16,
        spectral_channels=spectral_channels,
    ),
    neck=dict(
        type="HOD26SpectralSFP",
        spectral_channels=spectral_channels,
        use_p2=True,
    ),
    query_head=dict(
        num_classes=num_classes,
        # V2/V3 diagnostics identify high-IoU localization as the dominant
        # remaining error. Assignment and regression use the same geometry.
        loss_bbox=dict(type="L1Loss", loss_weight=7.5),
        loss_iou=dict(type="GIoULoss", loss_weight=3.0),
    ),
    roi_head=[
        dict(
            type="CoStandardRoIHead",
            bbox_roi_extractor=dict(
                type="SingleRoIExtractor",
                roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56,
            ),
            bbox_head=dict(
                type="ConvFCBBoxHead",
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1],
                ),
                reg_class_agnostic=True,
                reg_decoded_bbox=True,
                norm_cfg=dict(type="GN", num_groups=32),
                loss_cls=dict(
                    type="CrossEntropyLoss",
                    use_sigmoid=False,
                    loss_weight=12.0,
                ),
                loss_bbox=dict(type="GIoULoss", loss_weight=120.0),
            ),
        )
    ],
    bbox_head=[
        dict(
            type="CoATSSHead",
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type="AnchorGenerator",
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128],
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=12.0,
            ),
            loss_bbox=dict(type="GIoULoss", loss_weight=24.0),
            loss_centerness=dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                loss_weight=12.0,
            ),
        )
    ],
    train_cfg=[
        dict(
            assigner=dict(
                type="HungarianAssigner",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBoxL1Cost", weight=7.5, box_format="xywh"),
                iou_cost=dict(type="IoUCost", iou_mode="giou", weight=3.0),
            )
        ),
        dict(
            rpn=dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False,
                ),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            ),
            rpn_proposal=dict(
                nms_pre=4000,
                max_per_img=1000,
                nms=dict(type="nms", iou_threshold=0.7),
                min_bbox_size=0,
            ),
            rcnn=dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True,
                ),
                pos_weight=-1,
                debug=False,
            ),
        ),
        dict(
            assigner=dict(type="ATSSAssigner", topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
    ],
    test_cfg=[
        dict(
            max_per_img=300,
            nms=dict(
                type="soft_nms",
                iou_threshold=0.75,
                min_score=0.0001,
            ),
        ),
        dict(
            rpn=dict(
                nms_pre=8000,
                max_per_img=2000,
                nms=dict(type="nms", iou_threshold=0.9),
                min_bbox_size=0,
            ),
            rcnn=dict(
                score_thr=0.0,
                mask_thr_binary=0.5,
                nms=dict(type="soft_nms", iou_threshold=0.5),
                max_per_img=300,
            ),
        ),
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.0,
            nms=dict(type="soft_nms", iou_threshold=0.6),
            max_per_img=300,
        ),
    ],
)

train_pipeline = [
    dict(
        type="LoadHOD26HSIFromFile",
        stats_file="storage/v4/dataset/fold0_band_stats.json",
    ),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="HOD26StagedAugment",
        stage_epochs=(4, 40, 56),
        # Native four-scene Mosaic is confined to the strong stage. MixUp is
        # explicitly disabled because full-scene spectral mixtures have no
        # physically faithful hard-box target.
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
]

test_pipeline = [
    dict(
        type="LoadHOD26HSIFromFile",
        stats_file="storage/v4/dataset/fold0_band_stats.json",
    ),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1280, 640),
        flip=False,
        transforms=[
            dict(type="HOD26ResizePad", width=1280, height=640),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    _delete_=True,
    train=dict(
        type="HOD26CocoDataset",
        ann_file="storage/v2/dataset/fold0/train.json",
        img_prefix="data/raw/data_train/data_train/VIS",
        pipeline=train_pipeline,
        classes=None,
        filter_empty_gt=False,
        repeat_class_factors={5: 2, 8: 2, 14: 2, 16: 2},
    ),
    val=dict(
        type="HOD26CocoDataset",
        ann_file="storage/v2/dataset/fold0/val.json",
        img_prefix="data/raw/data_train/data_train/VIS",
        pipeline=test_pipeline,
        classes=None,
        test_mode=True,
    ),
    test=dict(
        type="HOD26CocoDataset",
        ann_file="storage/v2/dataset/test.json",
        img_prefix="data/raw/data_test/data_test/VIS",
        pipeline=test_pipeline,
        classes=None,
        test_mode=True,
    ),
    train_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=4,
        persistent_workers=False,
        pin_memory=True,
    ),
    val_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=4,
        persistent_workers=False,
        pin_memory=True,
    ),
)

# Fold0 repeat sampling is exactly 2,812 exposures. With two GPUs and one
# image/GPU, each epoch is 1,406 optimizer updates. The 72-epoch soft horizon
# is therefore 101,232 updates; reaching it is not permission to stop while
# the multi-metric validation trend is still rising.
runner = dict(type="EpochBasedRunner", max_epochs=72)
lr_config = dict(
    _delete_=True,
    policy="HOD26UpdateCosine",
    points=[
        (0, 0.02),
        (1500, 1.00),
        (12000, 1.00),
        (52000, 0.25),
        (78000, 0.04),
        (95000, 0.01),
        (101232, 0.003),
    ],
)
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=1.25e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    constructor="HOD26LayerDecayOptimizerConstructor",
    paramwise_cfg=dict(
        num_layers=24,
        layer_decay_rate=0.8,
        hsi_lr_mult=8.0,
        class_lr_mult=4.0,
    ),
)
optimizer_config = dict(
    grad_clip=dict(max_norm=0.1, norm_type=2),
)
fp16 = dict(
    loss_scale=dict(
        init_scale=1.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
    )
)

checkpoint_config = dict(
    interval=2,
    max_keep_ckpts=12,
    save_optimizer=True,
    create_symlink=True,
)
evaluation = dict(
    interval=2,
    metric="bbox",
    classwise=True,
    # Kaggle names the standard COCO metric but does not publish maxDets.
    # Keep Fold0 directly comparable with the existing V1/V2 COCO evaluator.
    proposal_nums=(1, 10, 100),
    save_best="bbox_mAP",
    rule="greater",
)
custom_hooks = [
    dict(type="HOD26DataStageHook", priority="VERY_HIGH"),
    dict(
        type="ExpMomentumEMAHook",
        momentum=0.0001,
        total_iter=2000,
        # On resume this hook must register EMA buffers and restore runner
        # metadata before LR/FP16 hooks consume the update/scaler state.
        priority="HIGHEST",
    ),
]

load_from = "storage/pretrained/v4/co_dino_vitl_hod26_init.pth"
resume_from = None
auto_resume = False
find_unused_parameters = False
work_dir = "storage/v4/runs/co_dino_vitl_hsi_fold0"

log_config = dict(
    interval=25,
    hooks=[dict(type="TextLoggerHook")],
)
log_level = "INFO"
dist_params = dict(backend="nccl")
workflow = [("train", 1)]
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = "fork"
