checkpoint_config = dict(interval=1)
log_config=dict(
    interval=50,
    hooks = [
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', interval=1000,
            init_kwargs= dict(
                project= 'Object Detection',
                entity = 'boostcamp-cv-13',
                name = 'cascade_rcnn_r50_fpn_3x_Albumentation',
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
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'