# Config for HRH-SCO Extension on Scale-Subset (Baseline equivalent)
# Based on yolc_subset.py which takes ~2-3 hours on CPU
# Uses 10% of data, smaller images (640x400), 16 epochs

dataset_type = 'VisDroneDataset'
data_root = 'data/VisDrone2019/'
classes = ('pedestrian', "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor")
num_classes = 10
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Training pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 384), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(640, 384), keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/Visdrone2019/subset/train_25pct.json',
        img_prefix='data/Visdrone2019/unzipped_train/VisDrone2019-DET-train/images',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/Visdrone2019/subset/val_25pct.json',
        img_prefix='data/Visdrone2019/unzipped_val/VisDrone2019-DET-val/images',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/Visdrone2019/subset/val_25pct.json',
        img_prefix='data/Visdrone2019/unzipped_val/VisDrone2019-DET-val/images',
        pipeline=test_pipeline))

evaluation = dict(interval=2, metric='bbox')
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[8, 14])

runner = dict(type='EpochBasedRunner', max_epochs=16)

checkpoint_config = dict(interval=2)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

model = dict(
    type='YOLC',
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384))),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w48'),
        # Force standard convolutions instead of Deformable Convolutions
        conv_cfg=dict(type='Conv'),
        norm_cfg=dict(type='BN', requires_grad=True)
    ),
    neck=dict(
        type='HRFPN',
        in_channels=[48, 96, 192, 384],
        out_channels=384,
        num_outs=1),
    bbox_head=dict(
        type='YOLCHead',
        num_classes=10,
        in_channel=384,
        feat_channel=96,
        loss_center_local=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_xywh=dict(type='GWDLoss', loss_weight=2.0),
        # HRH-SCO Configuration
        use_hrh_sco=True,
        heatmap_upsample_num=1 # Stride 4 (Input) -> Stride 2 (Heatmap)
    ),
    train_cfg=None,
    test_cfg=dict(topk=1000, local_maximum_kernel=3, max_per_img=300, nms_cfg=dict(iou_threshold=0.60)))

work_dir = './work_dirs/yolc_hrh_sco_25pct'
gpu_ids = [0]
