_base_ = '/opt/ml/baseline/mmdetection/configs/_base_/default_runtime.py'

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
class_list = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

model = dict(
    type='ATSS',
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
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
    neck=[
        dict(
            type='FPN',
            in_channels=[384, 768, 1536],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=5),
        dict(
            type='DyHead',
            in_channels=256,
            out_channels=256,
            num_blocks=6,
            # disable zero_init_offset to follow official implementation
            zero_init_offset=False)
    ],
    bbox_head=dict(
        type='ATSSHead',
        num_classes=10,
        in_channels=256,
        pred_kernel_size=1,  # follow DyHead official implementation
        stacked_convs=0,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
            center_offset=0.5),  # follow DyHead official implementation
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.4,
            alpha=0.2,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
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
        type='Resize',
        img_scale=[(512, 512), (1024, 1024)],
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
        img_scale=(1024,1024),
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
    samples_per_gpu=2, 
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

##수정
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

# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])


# lr_config = dict(
#    policy='fixed', #내가 목표로 한 lr에 도달하면 어떻게 할건지 cos등을 설정할 수 있따
#    warmup='linear',
#    warmup_iters=1000,
#    warmup_ratio=0.001
# )

lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=1099,
    warmup_ratio=0.001,
    periods=[5495, 5495, 6594, 6594, 8792, 8792, 13188, 13188, 26376, 26376],
    restart_weights=[1, 0.85, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4],
    by_epoch=False,
    min_lr=5e-6
    )

runner = dict(type='EpochBasedRunner', max_epochs=12)

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
                name = 'atss_swin_dyhead_s',
                config= {
                    'optimizer_type':optimizer['type'],
                    'optimizer_lr':optimizer['lr'],
                    'neck_type':neck_name,
                    'lr_scheduler_type':lr_config['policy'] if lr_config != None else None,
                    'batch_size':data['samples_per_gpu'],
                    'epoch_size':runner['max_epochs']
                }
            ),
            log_artifact=True
            
        )
    ]
)
opt_name=optimizer['type']
lr=optimizer['lr']
epoch=runner['max_epochs']

version=2
work_dir = f'/opt/ml/level2_objectdetection_cv-level2-cv-13/outputs/{model_name}_agc_warmup_{neck_name}_{opt_name}_{lr}_{epoch}_{version}'

seed=2022
gpu_ids=[0]

checkpoint_config = dict(max_keep_ckpts=1, interval=1)
device='cuda'
