"""HOD26 V6 CoSpec-DINO ViT-L Fold0 convergence-controlled training."""

import hod26.v6  # noqa: F401

_base_ = [
    "../../storage/upstream/Co-DETR/projects/configs/co_dino_vit/"
    "co_dino_5scale_vit_large_coco.py"
]

num_classes = 18
spectral_channels = (96, 128, 192)

model = dict(
    type="HOD26V6CoSpecDINO",
    backbone=dict(
        type="HOD26V6WavelengthViT",
        input_bands=16,
        spectral_channels=spectral_channels,
        wavelength_basis_size=8,
        use_spectral_checkpoint=True,
    ),
    neck=dict(
        type="HOD26V6CrossSpectralSFP",
        spectral_channels=spectral_channels,
        fusion_open_updates=6000,
        fusion_hidden=64,
        use_fusion_checkpoint=True,
        use_p2=True,
    ),
    query_head=dict(
        type="HOD26V6AlignCoDINOHead",
        num_classes=num_classes,
        align_alpha=0.25,
        align_gamma=2.0,
        align_rank_tau=1.5,
        class_weight_beta=0.999,
        class_weight_cap=1.5,
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
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=12.0),
                loss_bbox=dict(type="GIoULoss", loss_weight=120.0),
            ),
        )
    ],
    bbox_head=[
        dict(
            type="HOD26V6SpectralProposalHead",
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            class_weight_beta=0.999,
            class_weight_cap=1.5,
            level_loss_weights=(1.50, 1.25, 1.00, 0.75, 0.50, 0.25),
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
                type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25,
                loss_weight=12.0,
            ),
            loss_bbox=dict(type="GIoULoss", loss_weight=24.0),
            loss_centerness=dict(
                type="CrossEntropyLoss", use_sigmoid=True, loss_weight=12.0,
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
                    type="MaxIoUAssigner", pos_iou_thr=0.7, neg_iou_thr=0.3,
                    min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler", num=256, pos_fraction=0.5,
                    neg_pos_ub=-1, add_gt_as_proposals=False,
                ),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            ),
            rpn_proposal=dict(
                nms_pre=4000, max_per_img=1000,
                nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0,
            ),
            rcnn=dict(
                assigner=dict(
                    type="MaxIoUAssigner", pos_iou_thr=0.5, neg_iou_thr=0.5,
                    min_pos_iou=0.5, match_low_quality=False, ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler", num=512, pos_fraction=0.25,
                    neg_pos_ub=-1, add_gt_as_proposals=True,
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
            nms=dict(type="soft_nms", iou_threshold=0.75, min_score=0.0001),
        ),
        dict(
            rpn=dict(
                nms_pre=8000, max_per_img=2000,
                nms=dict(type="nms", iou_threshold=0.9), min_bbox_size=0,
            ),
            rcnn=dict(
                score_thr=0.0, mask_thr_binary=0.5,
                nms=dict(type="soft_nms", iou_threshold=0.5), max_per_img=300,
            ),
        ),
        dict(
            nms_pre=1000, min_bbox_size=0, score_thr=0.0,
            nms=dict(type="soft_nms", iou_threshold=0.6), max_per_img=300,
        ),
    ],
)

train_pipeline = [
    dict(
        type="LoadHOD26V6HSIFromFile",
        stats_file="storage/v6/dataset/fold0_band_stats.json",
    ),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="HOD26V6StagedAugment",
        stage_updates=(6000, 56400, 80400),
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

test_pipeline = [
    dict(
        type="LoadHOD26V6HSIFromFile",
        stats_file="storage/v6/dataset/fold0_band_stats.json",
    ),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1280, 640),
        flip=False,
        transforms=[
            dict(type="HOD26V6ResizePad", width=1280, height=640),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    _delete_=True,
    train=dict(
        type="HOD26V6CocoDataset",
        ann_file="storage/v2/dataset/fold0/train.json",
        img_prefix="data/raw/data_train/data_train/VIS",
        pipeline=train_pipeline,
        classes=None,
        filter_empty_gt=False,
    ),
    val=dict(
        type="HOD26V6CocoDataset",
        ann_file="storage/v2/dataset/fold0/val.json",
        img_prefix="data/raw/data_train/data_train/VIS",
        pipeline=test_pipeline,
        classes=None,
        test_mode=True,
    ),
    test=dict(
        type="HOD26V6CocoDataset",
        ann_file="storage/v2/dataset/test.json",
        img_prefix="data/raw/data_test/data_test/VIS",
        pipeline=test_pipeline,
        classes=None,
        test_mode=True,
    ),
    train_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=4,
        persistent_workers=False, pin_memory=True,
    ),
    val_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=4,
        persistent_workers=False, pin_memory=True,
    ),
)

# 2,400 images / global batch 2 = 1,200 attempted updates per epoch.
# 80 epochs is the 96,000-update soft horizon, never an automatic stop claim.
runner = dict(type="EpochBasedRunner", max_epochs=80)
lr_config = dict(
    _delete_=True,
    policy="HOD26V6SuccessfulCosine",
    points=[
        (0, 0.02),
        (1500, 1.00),
        (6000, 1.00),
        (56400, 0.30),
        (80400, 0.06),
        (92000, 0.015),
        (96000, 0.004),
    ],
)
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=1.25e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    constructor="HOD26V6OptimizerConstructor",
    paramwise_cfg=dict(
        num_layers=24,
        layer_decay_rate=0.90,
        visual_lr_mult=1.0,
        new_lr_mult=8.0,
        class_lr_mult=4.0,
    ),
)
optimizer_config = dict(
    type="HOD26V6StableFp16OptimizerHook",
    grad_clip=dict(max_norm=0.1, norm_type=2),
    loss_scale=dict(
        init_scale=1.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
    ),
    distributed=True,
)
# The custom optimizer hook owns autocast/GradScaler registration. Keeping the
# legacy top-level fp16 switch disabled prevents mmdet.apis.train from replacing
# it with MMCV's FP32-overflowing global-norm implementation.
fp16 = None

checkpoint_config = dict(
    interval=2, max_keep_ckpts=20, save_optimizer=True, create_symlink=True,
)
evaluation = dict(
    interval=2,
    metric="bbox",
    classwise=True,
    proposal_nums=(1, 10, 100),
    save_best="bbox_mAP",
    rule="greater",
)
custom_hooks = [
    dict(type="HOD26V6ProgressHook", priority="VERY_HIGH"),
    dict(
        type="HOD26V6EMAHook",
        momentum=0.0001,
        total_iter=2000,
        priority="HIGHEST",
    ),
    dict(type="HOD26V6PostOptimizerEMAHook", priority="BELOW_NORMAL"),
    dict(
        type="HOD26V6CheckpointContractHook",
        source_lock="configs/v6/runtime_source_lock.json",
        initialization_provenance=(
            "storage/pretrained/v6/co_spec_dino_vitl_public_init.pth.provenance.json"
        ),
        priority="NORMAL",
    ),
]

load_from = "storage/pretrained/v6/co_spec_dino_vitl_public_init.pth"
resume_from = None
auto_resume = False
find_unused_parameters = False
work_dir = "storage/v6/runs/co_spec_dino_vitl_fold0"

log_config = dict(interval=25, hooks=[dict(type="TextLoggerHook")])
log_level = "INFO"
dist_params = dict(backend="nccl")
workflow = [("train", 1)]
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = "fork"
