_base_ = '/opt/ml/level2_objectdetection_cv-level2-cv-13/configs/swin_dyhead_baseline.py'

img_norm_cfg = dict(
    mean=[109.629, 103.211, 96.742], std=[51.046, 50.947, 55.714], to_rgb=True)

albu_train_transforms = [
    dict(
        type='Sharpen',
        p=0.5)
]

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
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    
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
