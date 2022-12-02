# 모듈 import
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test, set_random_seed
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
import json
import argparse

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

def make_submission():
    # submission 양식에 맞게 output 후처리
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds()
    class_num = 10
    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[j]:
                prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                    o[2]) + ' ' + str(o[3]) + ' '
            
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(cfg.work_dir, f'base_submission_{args.configs}.csv'), index=None)

if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile('/opt/ml/level2_objectdetection_cv-level2-cv-13/configs/' + args.configs)
    cfg.data.test.test_mode = True
    cfg.gpu_ids = [0]
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None
    # build_dataset
    dataset = build_dataset(cfg.data.test)
    
    print("make dataset")
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    
    # checkpoint path
    checkpoint_path = os.path.join(cfg.work_dir, 'best_bbox_mAP_50_epoch_12.pth')
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])
    set_random_seed(2022, deterministic= True)
    output = single_gpu_test(model, data_loader, show_score_thr=0.5) # output 계산

    make_submission()
