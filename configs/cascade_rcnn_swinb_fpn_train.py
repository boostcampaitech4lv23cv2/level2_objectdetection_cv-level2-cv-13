# 모듈 import

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, set_random_seed
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device
from mmcv.runner import load_checkpoint
import torch
torch.cuda.empty_cache()

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
cfg = Config.fromfile('/opt/ml/level2_objectdetection_cv-level2-cv-13/configs/cascade_rcnn_r50_fpn_3x_coco-custom.py')
root='/opt/ml/dataset/'

# dataset config 수정
cfg.data.train.classes = classes
cfg.data.train.img_prefix = root
cfg.data.train.ann_file = '/opt/ml/level2_objectdetection_cv-level2-cv-13/pseudo_labeling/swinb_f4_0.975.json' # train json 정보
#cfg.data.train.ann_file = root + '/fold_dataset/train_fold_1_of_5.json' # train json 정보
#cfg.data.train.pipeline[2]['img_scale'] = (1024,1024) # Resize

cfg.data.val.classes = classes
cfg.data.val.img_prefix = root
cfg.data.val.ann_file = '/opt/ml/level2_objectdetection_cv-level2-cv-13/fold_dataset/validation_fold_4_of_5.json' # train json 정보
#cfg.data.val.pipeline[1]['img_scale'] = (1024,1024) # Resize


cfg.data.test.classes = classes
cfg.data.test.img_prefix = root
cfg.data.test.ann_file = root + 'test.json' # test json 정보
#cfg.data.test.pipeline[1]['img_scale'] = (1024,1024) # Resize

cfg.data.samples_per_gpu = 2
cfg. workers_per_gpu = 4

cfg.seed = 42
cfg.gpu_ids = [0]
cfg.work_dir = './work_dirs/cascade_rcnn_swinB384_2X_Pseudo_0.975_e3_f4_with_train_lr0_000025'


cfg.evaluation.save_best='auto'
#cfg.model.roi_head.bbox_head.num_classes = 10
cfg.fp16 = dict(loss_scale=dict(init_scale=512.))


#cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
cfg.device = get_device()
 
meta = dict()
meta['config'] = cfg.pretty_text

# build_dataset
datasets = [build_dataset(cfg.data.train)]

# 모델 build 및 pretrained network 불러오기
set_random_seed(42,deterministic=True)
model = build_detector(cfg.model)
model.init_weights()
checkpoint = load_checkpoint(model, '/opt/ml/Cascade_swinb_384_best_bbox_mAP_epoch_18.pth', map_location='cuda')

# 모델 학습
#distributed -> 분산 학습을 위한 파라미터 True로 할경우 key_error 'LOCAL RANK' 발생(local rank가 없어서 발생하는 문제로 보임)
train_detector(model, datasets, cfg, distributed=False, validate=True, meta = meta)
torch.cuda.empty_cache()