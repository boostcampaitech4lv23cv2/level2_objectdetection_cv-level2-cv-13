_base_ = [
          '/opt/ml/level2_objectdetection_cv-level2-cv-13/configs/swin_dyhead_baseline_modi.py',
    ]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa

model = dict(
    type='ATSS',
    backbone=dict(
        embed_dims=96,
        num_heads=[3, 6, 12, 24],
        depths=[2, 2, 18, 2],
        out_indices=(1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
        window_size=7,
    ),
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
    ]
    )


# Use RepeatDataset to speed up training
data = dict(
    samples_per_gpu=5,
)