# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmcv
import numpy as np

prog_description = '''K-Fold coco split.

To split coco data for semi-supervised object detection:
    python tools/misc/split_coco.py
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root',
        type=str,
        help='The data root of coco dataset.',
        default='/opt/ml/dataset/train.json')
    parser.add_argument(
        '--out-dir',
        type=str,
        help='The output directory of coco semi-supervised annotations.',
        default='/opt/ml/level2_objectdetection_cv-level2-cv-13/fold_dataset')
    parser.add_argument(
        '--percent',
        type=float,
        nargs='+',
        help='The percentage of labeled data in the training set.',
        default=0.8)
    parser.add_argument(
        '--fold',
        type=int,
        help='K-fold cross validation for semi-supervised object detection.',
        default=5)
    args = parser.parse_args()
    return args


def split_coco(data_root, out_dir, percent, fold):
    """Split COCO data for Semi-supervised object detection.

    Args:
        data_root (str): The data root of coco dataset.
        out_dir (str): The output directory of coco semi-supervised
            annotations.
        percent (float): The percentage of labeled data in the training set.
        fold (int): The fold of dataset and set as random seed for data split.
    """

    def save_anns(name, images, annotations):
        sub_anns = dict()
        sub_anns['info'] = anns['info'] #데이터셋 정보
        sub_anns['licenses'] = anns['licenses'] #CC BY 4.0
        sub_anns['images'] = images #이미지 너비,높이, 파일명, 라이센스, url 2개, 캡처시기,id
        sub_anns['categories'] = anns['categories'] # class label 10개
        sub_anns['annotations'] = annotations #image_id, category_id, area, bbox, iscrowd, id

        mmcv.mkdir_or_exist(out_dir)
        mmcv.dump(sub_anns, f'{out_dir}/{name}.json')

    # set random seed with the fold
    np.random.seed(fold)
    anns = mmcv.load(data_root)

    image_list = anns['images']
    labeled_total = int(percent * len(image_list))
    labeled_inds = set(
        np.random.choice(range(len(image_list)), size=labeled_total,replace=False))
    labeled_ids, labeled_images, validation_images = [], [], []

    for i in range(len(image_list)):
        if i in labeled_inds:
            labeled_images.append(image_list[i])
            labeled_ids.append(image_list[i]['id'])
        else:
            validation_images.append(image_list[i])

    # get all annotations of labeled images
    labeled_ids = set(labeled_ids)
    labeled_annotations, validation_annotation = [], []

    for ann in anns['annotations']:
        if ann['image_id'] in labeled_ids: #id 별 split 과정
            labeled_annotations.append(ann)
        else:
            validation_annotation.append(ann)

    # save train, validation file based on percentage
    labeled_name = f'train_fold_{fold}_of_{args.fold}'
    validation_name = f'validation_fold_{fold}_of_{args.fold}'

    save_anns(labeled_name, labeled_images, labeled_annotations)
    save_anns(validation_name, validation_images, validation_annotation)


if __name__=="__main__":
    args = parse_args()
    for fold in range(1,args.fold+1):
        split_coco(args.data_root, args.out_dir,args.percent,fold)
        
        