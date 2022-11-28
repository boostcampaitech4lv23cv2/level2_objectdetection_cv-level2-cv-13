from mmdet.datasets.coco import CocoDataset

_base_ = ['/opt/ml/level2_objectdetection_cv-level2-cv-13/configs/swin_dyhead_baseline.py']


pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

class_list = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
model = dict(
    backbone=dict(
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=[
        dict(
            type='FPN',
            in_channels=[192, 384, 768],
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
)
# img_norm_cfg = dict(
#     mean=[109.629, 103.211, 96.742], std=[51.046, 50.947, 55.714], to_rgb=True)

# CocoDataset.CLASSES=class_list

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='Mosaic',
#         img_scale=(1024,1024),
#         prob=0.5  
#     ),
#     dict(
#         type='Resize',
#         img_scale=[(512, 512), (1024, 1024)],
#         multiscale_mode='range',
#         keep_ratio=True,
#         backend='pillow'),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=128),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]

dataset_type = 'CocoDataset'
data_root ='/opt/ml/dataset'


data = dict(
    samples_per_gpu=8,
    train=dict(
        ann_file='/opt/ml/level2_objectdetection_cv-level2-cv-13/stratified_fold_dataset/train_fold_1_of_5_cleaned_10_1000.json',
    )
)
work_dir = '/opt/ml/level2_objectdetection_cv-level2-cv-13/outputs/swin_t_dataset_cleaning_test_10_img_no_mosaic'