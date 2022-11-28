import argparse
from ensemble_boxes import *
import pandas as pd
import os
import numpy as np
#csv import
# k-fold 5개 csv 특정 폴더 안에 넣고 parser로 folder 위치 변경하게 설정
# label, score, xmin, ymin, xmax, ymax 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv-root',
        type=str,
        help='The csv root of prediction results',
        default='/opt/ml/level2_objectdetection_cv-level2-cv-13/Ensemble/submission_files')
    parser.add_argument(
        '--out-dir',
        type=str,
        help='The output directory of coco semi-supervised annotations.',
        default='/opt/ml/level2_objectdetection_cv-level2-cv-13/ensemble_result')
    parser.add_argument(
        '--iou-thr',
        type= float,
        help='The threshold of IoU to ensemble.',
        default= 0.65)
    parser.add_argument(
        '--skip-box-thr',
        type= float,
        help='The threshold of confidence to skip.',
        default= 0.001)
    parser.add_argument(
        '--sigma',
        type= float,
        help='Only for soft_nms',
        default= 0.1)
    parser.add_argument(
        '--mode',
        type= int,
        help='1. NMS, 2. Soft NMS, 3. Non Maximum Weighted, 4. Weighted Boxes Fusion',
        default= 4)
    args = parser.parse_args()
    return args 
def ensemble(csv_root,out_dir):
    file_list = os.listdir(csv_root)
    #Hyperparameters
    iou_thr = args.iou_thr
    skip_box_thr = args.skip_box_thr
    sigma = args.sigma 
    weights = [1]*len(file_list)  #weight manual 설정 가능

    # CSV 불러오기
    df_save=pd.read_csv('/opt/ml/sample_submission/submission_ensemble.csv')
    df_list=[pd.read_csv(os.path.join(csv_root, file)) for file in file_list]
    # image 별로 ensemble 돌리기
    for idx in range(len(df_save)):
        boxes_list = []
        scores_list = []
        labels_list = []
        for df in df_list:
            if type(df['PredictionString'][idx])==str:
                v=df['PredictionString'][idx].split()
            else:
                continue
            labels=[]
            boxes=[]
            scores=[]
            for i in range(0,len(v),6):
                l,s,x_min,y_min,x_max,y_max=v[i:i+6]
                labels.append(int(l))
                scores.append(float(s))
                boxes.append([float(x_min)/1024,float(y_min)/1024,float(x_max)/1024,float(y_max)/1024])
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)
        if args.mode == 1:
            boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
        elif args.mode == 2:
            boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
        elif args.mode == 3:
            boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        elif args.mode == 4:
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        pred=[]
        for box,score,label in zip(boxes,scores,labels):
            pred.append(int(label))
            pred.append(score)
            pred+=[coor* 1024 for coor in box]   
        df_save['PredictionString'][idx]=" ".join(str(x) for x in pred)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df_save.to_csv(os.path.join(out_dir,'ensemble.csv'),index=False)

if __name__=="__main__":
    args=parse_args()
    ensemble(args.csv_root,args.out_dir)
    