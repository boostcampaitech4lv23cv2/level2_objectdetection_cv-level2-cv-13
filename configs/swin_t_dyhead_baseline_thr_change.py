_base_ = ['/opt/ml/level2_objectdetection_cv-level2-cv-13/configs/swin_dyhead_baseline.py']

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
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
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)
    
)
data = dict(
    samples_per_gpu=8
)


lr_config = dict(
    _delete_=True,
    policy='fixed',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001
)

runner = dict(type='EpochBasedRunner', max_epochs=24)
work_dir = f'/opt/ml/level2_objectdetection_cv-level2-cv-13/outputs/swin_t_score_thr_0.001_with_warmup'