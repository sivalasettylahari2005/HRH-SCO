# Config for HRH-SCO Extension on 100% full dataset
# Fine-tuning from the 25% subset results for better performance

dataset_type = 'VisDroneDataset'
data_root = 'data/Visdrone2019/'
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
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/Visdrone2019/train_coco.json',
        img_prefix='data/Visdrone2019/unzipped_train/VisDrone2019-DET-train/images',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/Visdrone2019/val_coco.json',
        img_prefix='data/Visdrone2019/unzipped_val/VisDrone2019-DET-val/images',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/Visdrone2019/val_coco.json',
        img_prefix='data/Visdrone2019/unzipped_val/VisDrone2019-DET-val/images',
        pipeline=test_pipeline))

evaluation = dict(interval=4, metric='bbox') # Validation only every 4th epoch (Saves 2 hours per epoch) # Evaluate every epoch
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001) # Lower LR for fine-tuning
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 12])

runner = dict(type='EpochBasedRunner', max_epochs=16)

checkpoint_config = dict(interval=200, by_epoch=False) # Save every 200 images for extra safety # Save every 500 images to prevent loss during shutdowns
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = './work_dirs/yolc_hrh_sco_25pct/epoch_16.pth' # Load previous 25% weights
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
        use_hrh_sco=True,
        heatmap_upsample_num=1
    ),
    train_cfg=None,
    test_cfg=dict(topk=1000, local_maximum_kernel=3, max_per_img=300, nms_cfg=dict(iou_threshold=0.60)))

work_dir = './work_dirs/yolc_hrh_sco_100pct'
gpu_ids = [0]
