_base_ = [
    './Cascade_SwinB_models_swin_l.py',
    './Cascade_SwinB_dataset_swin_l.py',
    './Cascade_SwinB_runtime_swin_l.py'
]

# 총 epochs 사이즈
checkpoint_config = dict(max_keep_ckpts=1, interval=1)


# samples_per_gpu -> batch size라 생각하면 됨
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4)

checkpoint_config = dict(interval=-1)
optimizer_config = dict(grad_clip=None)
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
            }
        )
    )

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

#Wandb Config
log_config=dict(
    interval=50,
    hooks = [
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', interval=1000,
            init_kwargs= dict(
                project= 'Object Detection',
                entity = 'boostcamp-cv-13',
                name = 'casscade_rcnn_swin_l_kfold1',
                config= {
                    'optimizer_type':'AdamW',
                    'optimizer_lr':optimizer['lr'],
                    'neck_type':'FPN',
                    'lr_scheduler_type':lr_config['policy'] if lr_config != None else None,
                    'batch_size':data['samples_per_gpu'],
                    'epoch_size':runner['max_epochs']
                }
            ),
            log_artifact=True
            
        )
    ]
)

#best metric
evaluation = dict(interval=1, metric='bbox',save_best='bbox_mAP_50')
seed=42
gpu_ids=[0]
model_name = 'Cascade RCNN'
neck_name = 'FPN'
lr=optimizer['lr']
epoch=runner['max_epochs']
version=1
work_dir = f'/opt/ml/level2_objectdetection_cv-level2-cv-13/outputs/{model_name}_{neck_name}_AdamW_{lr}_{epoch}_{version}'
device='cuda'
log_level = 'INFO'
resume_from = None
load_from = None