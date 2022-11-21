# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmcv
import numpy as np
import random
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

    #나중에 진짜 TRAIN 할때 SEED 해제
    random.seed(42)
    anns = mmcv.load(data_root)

    image_list = anns['images']
    images_shuffled=list(range(len(image_list)))
    random.shuffle(images_shuffled) #이거 return값 없다
    validation_length=int((1-percent)*len(image_list))
    for k in range(fold):
        labeled_inds=[]
        if k<fold-1:
            validation_inds=set(images_shuffled[k*validation_length:(k+1)*validation_length])
        else:
            validation_inds=set(images_shuffled[k*validation_length:])
        labeled_ids, labeled_images, validation_images = [], [], []
        for i in range(len(image_list)):
            if i not in validation_inds:
                labeled_inds.append(i)
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
        labeled_name = f'train_fold_{k+1}_of_{args.fold}'
        validation_name = f'validation_fold_{k+1}_of_{args.fold}'

        save_anns(labeled_name, labeled_images, labeled_annotations)
        save_anns(validation_name, validation_images, validation_annotation)


if __name__=="__main__":
    args = parse_args()
    split_coco(args.data_root, args.out_dir,args.percent,args.fold)
        
        