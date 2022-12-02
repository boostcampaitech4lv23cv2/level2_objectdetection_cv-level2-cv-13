# 모듈 import
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, set_random_seed
import argparse
import os
from mmcv.runner import load_checkpoint

# set a argument parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--configs',
        type=str,
        help='The config file which train model',
        default='swin_dyhead_baseline_lr_config_cosinerestart_mmdetwandb_12+12.py'
        )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print('/opt/ml/level2_objectdetection_cv-level2-cv-13/configs/' + args.configs)
    cfg = Config.fromfile('/opt/ml/level2_objectdetection_cv-level2-cv-13/configs/' + args.configs)
    # build_dataset
    datasets = [build_dataset(cfg.data.train)]

    # 모델 build 및 pretrained network 불러오기
    model = build_detector(cfg.model)
    #model.init_weights()
    checkpoint_path = os.path.join('/opt/ml/level2_objectdetection_cv-level2-cv-13/outputs/SwinTransformer_FPN_DyHead_AdamW_5e-05_12_2_cosinerestart_12-12', 'best_bbox_mAP_50_epoch_12.pth')
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load
    set_random_seed(2022, deterministic= True)
    train_detector(model, datasets, cfg, distributed=False, validate=True)