# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import mmcv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root',
        type=str,
        help='The data root of coco dataset.',
        default='/opt/ml/level2_objectdetection_cv-level2-cv-13/stratified_fold_dataset')
    parser.add_argument(
        '--file_name',
        type=str,
        help='The name of the json file to change',
        default='train_fold_1_of_5.json')
    parser.add_argument(
        '--remove_img_over',
        type=int,
        help='Threshold to clean from the target',
        default=20)
    parser.add_argument(
        '--remove_bbox_over',
        type=int,
        help='Threshold to clean from the target',
        default=1000)
    args = parser.parse_args()
    return args


def clean(data_root, file_name,remove_img_over,remove_bbox_over):
        

    anns = mmcv.load(data_root+"/"+file_name)
    
    
    image_list = anns['images']
    ids, images = [], []
    
    num_box=[0]*(10000)
    for ann in anns['annotations']:
        num_box[ann['image_id']]+=1
    
    image_removed=0
    bbox_removed=0
    
    for i in range(len(image_list)):
        if num_box[image_list[i]['id']]<=remove_img_over:
            images.append(image_list[i])
            ids.append(image_list[i]['id'])
        else:
            image_removed+=1

    # get all annotations of labeled images
    ids = set(ids)
    annotations=[]

    for ann in anns['annotations']:
        if ann['image_id'] in ids and ann['area']>remove_bbox_over: #id 별 split 과정
            annotations.append(ann)
        else:
            bbox_removed+=1

    # save train, validation file based on percentage
    sub_anns = dict()
    sub_anns['info'] = anns['info'] #데이터셋 정보
    sub_anns['licenses'] = anns['licenses'] #CC BY 4.0
    sub_anns['images'] = images #이미지 너비,높이, 파일명, 라이센스, url 2개, 캡처시기,id
    sub_anns['categories'] = anns['categories'] # class label 10개
    sub_anns['annotations'] = annotations #image_id, category_id, area, bbox, iscrowd, id

    mmcv.mkdir_or_exist(data_root)
    mmcv.dump(sub_anns, f'{data_root}/{file_name[:-5]}_cleaned_{remove_img_over}_{remove_bbox_over}.json')
    print(f"removed {image_removed} images and {bbox_removed} bbox")



if __name__=="__main__":
    args = parse_args()
    clean(args.data_root,args.file_name,args.remove_img_over,args.remove_bbox_over)