_base_ = [
    '/opt/ml/baseline/mmdetection/configs/_base_/models/retinanet_r50_fpn.py',
    '/opt/ml/baseline/mmdetection/configs/_base_/datasets/coco_detection.py',
    '/opt/ml/baseline/mmdetection/configs/_base_/default_runtime.py'
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
class_list = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

model = dict(
    type = 'RetinaNet',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(type= 'FPN',
              in_channels=[192, 384, 768],
              start_level=0,
              num_outs=5),
    bbox_head=dict(num_classes=10)
    )

# dataset settings
dataset_type = 'CocoDataset'
data_root ='/opt/ml/dataset'
img_norm_cfg = dict(
    mean=[109.629, 103.211, 96.742], std=[51.046, 50.947, 55.714], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(512,512),
        multiscale_mode='range',
        keep_ratio=True,
        backend='pillow'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=128),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512,512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, backend='pillow'),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=128),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Use RepeatDataset to speed up training
data = dict(
    samples_per_gpu=4, 
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
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
evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer_config = dict(grad_clip=None)

##수정
optimizer = dict(
    type='AdamW',
    lr=0.00005,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    )

# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
lr_config=None
runner = dict(type='EpochBasedRunner', max_epochs=50)

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
                name = 'retinanet_swin-t-p4-w7_fpn_1x_coco',
                 config= {
                     'optimizer_type':optimizer['type'],
                     'optimizer_lr':optimizer['lr'],
                     'neck_type':neck_name,
                     'lr_scheduler_type':lr_config['policy'] if lr_config != None else None,
                     'resize': (512,512)
            
                }
            ),
            log_artifact=True
        )
    ]
)
work_dir = f'/opt/ml/level2_objectdetection_cv-level2-cv-13/outputs/{model_name}_{neck_name}'

seed=2022
gpu_ids=[0]

checkpoint_config = dict(max_keep_ckpts=3, interval=1)
device='cuda'