from map_boxes import mean_average_precision_for_boxes
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from pycocotools.coco import COCO
import argparse

LABEL_NAME = ["General trash", "Paper", "Paper pack", "Metal", 
              "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

def calculate_map(GT_JSON, PRED_CSV):
    # load ground truth
    with open(GT_JSON, 'r') as outfile:
        test_anno = (json.load(outfile))

    # load prediction
    pred_df = pd.read_csv(PRED_CSV)

    new_pred = []

    file_names = pred_df['image_id'].values.tolist()
    bboxes = pred_df['PredictionString'].values.tolist()

    for file_name, bbox in tqdm(zip(file_names, bboxes)):
        boxes = np.array(str(bbox).split(' '))
        
        if len(boxes) % 6 == 1:
            boxes = boxes[:-1].reshape(-1, 6)
        elif len(boxes) % 6 == 0:
            boxes = boxes.reshape(-1, 6)
        else:
            raise Exception('error', 'invalid box count')
        for box in boxes:
            new_pred.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])

    gt = []

    coco = COCO(GT_JSON)

    for image_id in coco.getImgIds():
            
        image_info = coco.loadImgs(image_id)[0]
        annotation_id = coco.getAnnIds(imgIds=image_info['id'])
        annotation_info_list = coco.loadAnns(annotation_id)
            
        file_name = image_info['file_name']
            
        for annotation in annotation_info_list:
            gt.append([file_name, annotation['category_id'],
                    float(annotation['bbox'][0]),
                    float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
                    float(annotation['bbox'][1]),
                    (float(annotation['bbox'][1]) + float(annotation['bbox'][3]))])

    mean_ap, average_precisions = mean_average_precision_for_boxes(gt, new_pred, iou_threshold=0.5)

    print(mean_ap)

# set a argument parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--json',
        type=str,
        help='The config file which train model',
        default='/opt/ml/dataset/train.json'
        )
    parser.add_argument(
        '--csv',
        type=str,
        default='/opt/ml/outputs/CasCadeRCNN_SwinB/submission_best_bbox_mAP_epoch_26.csv'
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    calculate_map(args.json, args.csv)