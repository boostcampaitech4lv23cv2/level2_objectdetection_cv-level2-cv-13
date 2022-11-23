_base_ = [
    '/opt/ml/baseline/mmdetection/configs/_base_/models/retinanet_r50_fpn.py',
    '/opt/ml/baseline/mmdetection/configs/_base_/datasets/coco_detection.py', '/opt/ml/baseline/mmdetection/configs/_base_/default_runtime.py'
]

cudnn_benchmark = True
norm_cfg = dict(type='BN', requires_grad=True)
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth'  # noqa
model = dict(
    backbone=dict(
        _delete_=True,
        type='EfficientNet',
        arch='b3',
        drop_path_rate=0.2,
        out_indices=(3, 4, 5),
        frozen_stages=0,
        norm_cfg=dict(
            type='BN', requires_grad=True, eps=1e-3, momentum=0.01),
        norm_eval=False,
        init_cfg=dict(
            type='Pretrained', prefix='backbone', checkpoint=checkpoint)),
    neck=dict(
        type='FPN',
        in_channels=[48, 136, 384],
        start_level=0,
        out_channels=256,
        relu_before_extra_convs=True,
        no_norm_on_lateral=True,
        norm_cfg=norm_cfg),
    bbox_head=dict(type='RetinaSepBNHead', num_ins=5, norm_cfg=norm_cfg,num_classes=10),
    # training and testing settings
    train_cfg=dict(assigner=dict(neg_iou_thr=0.5)))

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_size = (512,512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=img_size,
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_size),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=img_size),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=img_size),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
dataset_type = 'CocoDataset'
data_root ='/opt/ml/dataset'
class_list = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
            classes=class_list,
            type=dataset_type,
            ann_file='/opt/ml/level2_objectdetection_cv-level2-cv-13/stratified_fold_dataset/train_fold_1_of_5.json',
            img_prefix=data_root,
            pipeline=train_pipeline),
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
evaluation = dict(interval=1, metric='bbox',save_best='bbox_mAP_50')
# optimizer
optimizer_config = dict(grad_clip=None)
optimizer = dict(
    type='AdamW',
    lr=0.00005,
    betas=(0.9, 0.999),
    weight_decay=0.05
    )
# learning policy
lr_config = None
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)

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
                name = 'eff_retinanet',
                config= {
                    'optimizer_type':optimizer['type'],
                    'optimizer_lr':optimizer['lr'],
                    'neck_type':neck_name,
                    'lr_scheduler_type':lr_config['policy'] if lr_config != None else None,
                    'resize': img_size
                }
            ),
            log_artifact=True
        )
    ]
)
work_dir = f'/opt/ml/level2_objectdetection_cv-level2-cv-13/outputs/{model_name}_{neck_name}'
seed=2022
gpu_ids=[0]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (4 samples per GPU)
auto_scale_lr = dict(base_batch_size=32)
checkpoint_config = dict(max_keep_ckpts=3, interval=1,)
device='cuda'