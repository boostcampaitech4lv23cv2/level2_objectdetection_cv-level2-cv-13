_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                reg_decoded_bbox=True,
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                reg_decoded_bbox=True,
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                reg_decoded_bbox=True,
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ]))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0005,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(warmup_iters=500, step=[8, 11])
runner = dict(max_epochs=24)
#(256, 256),  (512, 512), (768, 768), (1024, 1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile',to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(256, 256), (384, 384),  (512, 512), (640, 640), 
                           (768, 768), (896, 896), (1024, 1024)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(256, 256),  (512, 512), (768, 768), (1024, 1024)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(384, 600),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(256, 256), (384, 384),  (512, 512), (640, 640), 
                                     (768, 768), (896, 896), (1024, 1024)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True),
                      dict(
                            type='PhotoMetricDistortion',
                            brightness_delta=32,
                            contrast_range=(0.5, 1.5),
                            saturation_range=(0.5, 1.5),
                            hue_delta=18),
                    dict(
                            type='MinIoURandomCrop',
                            min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                            min_crop_size=0.3),
                    dict(
                            type='CutOut',
                            n_holes=(5, 10),
                            cutout_shape=[(4, 4), (4, 8), (8, 4), (8, 8),
                                          (16, 32), (32, 16), (32, 32),
                                          (32, 48), (48, 32), (48, 48)]
                            )
                  ]]),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
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
#Wandb Config
log_config=dict(
    interval=50,
    hooks = [
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', interval=1000,
            init_kwargs= dict(
                project= 'Object Detection',
                entity = 'boostcamp-cv-13',
                name = 'cascade_rcnn_r50_fpn_2x_coco-Augcustom',
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
#best metric
evaluation = dict(interval=1, metric='bbox',save_best='bbox_mAP_50')
