_base_ = '/opt/ml/baseline/mmdetection/configs/_base_/default_runtime.py'

class_list = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# model settings
model = dict(
    type='YOLOV3',
    backbone=dict(
        type='MobileNetV2',
        out_indices=(2, 4, 6),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2')),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[320, 96, 32],
        out_channels=[96, 96, 96]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=10,
        in_channels=[96, 96, 96],
        out_channels=[96, 96, 96],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100))

# dataset settings
dataset_type = 'CocoDataset'
data_root ='/opt/ml/dataset'
img_norm_cfg = dict(
    mean=[109.629, 103.211, 96.742], std=[51.046, 50.947, 55.714], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(
        type='Resize',
        img_scale=[(320, 320), (416, 416)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(416, 416),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',  # use RepeatDataset to speed up training
        times=10,
        dataset=dict(
            classes=class_list,
            type=dataset_type,
            ann_file='/opt/ml/level2_objectdetection_cv-level2-cv-13/stratified_fold_dataset/train_fold_1_of_5.json',
            img_prefix=data_root,
            pipeline=train_pipeline)),
    val=dict(
        classes=class_list,
        type=dataset_type,
        ann_file='/opt/ml/level2_objectdetection_cv-level2-cv-13/stratified_fold_dataset/validation_fold_1_of_5.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        classes=class_list,
        type=dataset_type,
        ann_file=data_root + '/test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.00005,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=4000,
#     warmup_ratio=0.0001,
#     step=[24, 28])
lr_config=None
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=50)
evaluation = dict(interval=1, metric=['bbox'])
find_unused_parameters = True

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (24 samples per GPU)
# auto_scale_lr = dict(base_batch_size=192)

model_name=model['backbone']['type']
neck_name=""
if type(model['neck'])==list:
    neck_name="_".join([neck['type'] for neck in model['neck']])
else:
    neck_name=model['neck']['type']   

log_config=dict(
    interval=50,
    hooks = [
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', interval=1000,
            init_kwargs= dict(
                project= 'Object Detection',
                entity = 'boostcamp-cv-13',
                name = 'atss_swin_dyhead',
                # config= dict(
                #     'optimizer_type':optimizer['type'],
                #     'optimizer_lr':optimizer['lr'],
                #     'neck_type':neck_name,
                #     'lr_scheduler_type':lr_config['policy'] if lr_config != None else None,
                #     'resize': data['train']['dataset']['pipeline'][2]['img_scale']
                # )
            ),
            log_artifact=True
        )
    ]
)
work_dir = f'/opt/ml/level2_objectdetection_cv-level2-cv-13/outputs/{model_name}_{neck_name}'

seed=2022
gpu_ids=[0]

checkpoint_config = dict(max_keep_ckpts=1, interval=1)
device='cuda'
evaluation = dict(interval=1, metric='bbox',save_best='bbox_mAP_50')