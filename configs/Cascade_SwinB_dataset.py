# dataset settings

dataset_type = 'CocoDataset'
data_root ='/opt/ml/dataset'

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")


albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(type='Flip',p=0.5),
            dict(type='RandomRotate90',p=0.5)
        ],
        p = 0.1
    ),
]

img_norm_cfg = dict(
    mean=[109.629, 103.211, 96.742], std=[51.046, 50.947, 55.714], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize', 
        img_scale= [(x,x) for x in range(512, 1024+1, 32)],
        multiscale_mode='value',
        keep_ratio=True,
        ),
    dict(type='RandomFlip', flip_ratio=0.5),
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
        'gt_bboxes': 'bboxes'
    },
    update_pad_shape=False,
    skip_img_without_anno=True
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024,1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
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
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='/opt/ml/level2_objectdetection_cv-level2-cv-13/stratified_fold_dataset/train_fold_5_of_5.json',
        img_prefix=data_root,
        pipeline=train_pipeline,
        classes=classes,
        ),
    val=dict(
        type=dataset_type,
        ann_file='/opt/ml/level2_objectdetection_cv-level2-cv-13/stratified_fold_dataset/validation_fold_5_of_5.json',
        img_prefix=data_root,
        pipeline=val_pipeline,
        classes=classes,
        ),
    test=dict(
        type=dataset_type,
        ann_file='/opt/ml/level2_objectdetection_cv-level2-cv-13/stratified_fold_dataset/validation_fold_1_of_5.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes,
        ))
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50')