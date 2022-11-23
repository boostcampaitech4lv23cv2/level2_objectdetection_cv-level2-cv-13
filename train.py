# 모듈 import
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import argparse

# set a argument parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--configs',
        type=str,
        help='The config file which train model',
        default='atss_swin_dyhead_50e.py'
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
    model.init_weights()
    train_detector(model, datasets, cfg, distributed=False, validate=True)