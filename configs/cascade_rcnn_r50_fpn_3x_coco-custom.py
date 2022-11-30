_base_ = [
    '/opt/ml/level2_objectdetection_cv-level2-cv-13/configs/dataset.py',
    '/opt/ml/level2_objectdetection_cv-level2-cv-13/configs/models.py',
    '/opt/ml/level2_objectdetection_cv-level2-cv-13/configs/runtime.py', '/opt/ml/level2_objectdetection_cv-level2-cv-13/configs/schedule.py'
]

# 총 epochs 사이즈
runner = dict(max_epochs=3)
checkpoint_config = dict(max_keep_ckpts=3, interval=1)


# samples_per_gpu -> batch size라 생각하면 됨
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2)

checkpoint_config = dict(interval=-1)
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
            }
        )
    )

# lr_config = dict(
#     policy='CosineRestart',
#     warmup='linear',
#     warmup_iters=1099,
#     warmup_ratio=0.001,
#     periods=[5495, 5495, 6594, 8792, 8792],
#     restart_weights=[1, 0.85, 0.75, 0.7, 0.6],
#     by_epoch=False,
#     min_lr=5e-6
#     )

lr_config = None

#Wandb Config
log_config=dict(
    interval=50,
    hooks = [
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', interval=1000,
            init_kwargs= dict(
                project= 'Object Detection',
                entity = 'boostcamp-cv-13',
                name = 'cascade_rcnn_swinB384_2X_Pseudo_0.975_e3_f4_with_train',
                config= {
                    'optimizer_type':optimizer['type'],
                    'optimizer_lr':optimizer['lr'],
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
opt_name=optimizer['type']
lr=optimizer['lr']
epoch=runner['max_epochs']
version=1 
work_dir = f'/opt/ml/level2_objectdetection_cv-level2-cv-13/outputs/{model_name}_{neck_name}_{opt_name}_{lr}_{epoch}_{version}'
device='cuda'