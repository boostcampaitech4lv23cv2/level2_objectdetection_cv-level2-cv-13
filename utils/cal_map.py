from map_boxes import mean_average_precision_for_boxes
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from pycocotools.coco import COCO
import argparse

LABEL_NAME = ["General trash", "Paper", "Paper pack", "Metal", 
              "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

def calculate_map(GT_JSON, PRED_CSV, IOU_threshold):
    # load ground truth
    with open(GT_JSON, 'r') as outfile:
        test_anno = (json.load(outfile))

    # load prediction
    pred_df = pd.read_csv(PRED_CSV)

    new_pred = []

    file_names = pred_df['image_id'].values.tolist()
    bboxes = pred_df['PredictionString'].values.tolist()

    bbox_0 = []
    bbox_1 = []
    bbox_2 = []
    bbox_3 = []
    bbox_4 = []
    bbox_5 = []
    bbox_6 = []
    bbox_7 = []
    bbox_8 = []
    bbox_9 = []

    for file_name, bbox in tqdm(zip(file_names, bboxes)):
        boxes = np.array(str(bbox).split(' '))
        
        if len(boxes) % 6 == 1:
            # for i in range(10):
            boxes = boxes[:-1].reshape(-1, 6)
            # print(boxes.shape)
        elif len(boxes) % 6 == 0:
            boxes = boxes.reshape(-1, 6)
        else:
            raise Exception('error', 'invalid box count')
        for box in boxes:
            new_pred.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])
            if box[0] == '0':
                bbox_0.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])
            elif box[0] == '1':
                bbox_1.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])
            elif box[0] == '2':
                bbox_2.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])
            elif box[0] == '3':
                bbox_3.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])
            elif box[0] == '4':
                bbox_4.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])
            elif box[0] == '5':
                bbox_5.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])
            elif box[0] == '6':
                bbox_6.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])
            elif box[0] == '7':
                bbox_7.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])
            elif box[0] == '8':
                bbox_8.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])
            else:
                bbox_9.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])
            

    print(len(bbox_0))
    print(len(new_pred))
    gt = []
    gt_0 = []
    gt_1 = []
    gt_2 = []
    gt_3 = []
    gt_4 = []
    gt_5 = []
    gt_6 = []
    gt_7 = []
    gt_8 = []
    gt_9 = []

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
            if annotation['category_id'] == 0:
                gt_0.append([file_name, annotation['category_id'],
                    float(annotation['bbox'][0]),
                    float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
                    float(annotation['bbox'][1]),
                    (float(annotation['bbox'][1]) + float(annotation['bbox'][3]))])
            elif annotation['category_id'] == 1:
                gt_1.append([file_name, annotation['category_id'],
                    float(annotation['bbox'][0]),
                    float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
                    float(annotation['bbox'][1]),
                    (float(annotation['bbox'][1]) + float(annotation['bbox'][3]))])
            elif annotation['category_id'] == 2:
                gt_2.append([file_name, annotation['category_id'],
                    float(annotation['bbox'][0]),
                    float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
                    float(annotation['bbox'][1]),
                    (float(annotation['bbox'][1]) + float(annotation['bbox'][3]))])
            elif annotation['category_id'] == 3:
                gt_3.append([file_name, annotation['category_id'],
                    float(annotation['bbox'][0]),
                    float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
                    float(annotation['bbox'][1]),
                    (float(annotation['bbox'][1]) + float(annotation['bbox'][3]))])
            elif annotation['category_id'] == 4:
                gt_4.append([file_name, annotation['category_id'],
                    float(annotation['bbox'][0]),
                    float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
                    float(annotation['bbox'][1]),
                    (float(annotation['bbox'][1]) + float(annotation['bbox'][3]))])
            elif annotation['category_id'] == 5:
                gt_5.append([file_name, annotation['category_id'],
                    float(annotation['bbox'][0]),
                    float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
                    float(annotation['bbox'][1]),
                    (float(annotation['bbox'][1]) + float(annotation['bbox'][3]))])
            elif annotation['category_id'] == 6:
                gt_6.append([file_name, annotation['category_id'],
                    float(annotation['bbox'][0]),
                    float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
                    float(annotation['bbox'][1]),
                    (float(annotation['bbox'][1]) + float(annotation['bbox'][3]))])
            elif annotation['category_id'] == 7:
                gt_7.append([file_name, annotation['category_id'],
                    float(annotation['bbox'][0]),
                    float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
                    float(annotation['bbox'][1]),
                    (float(annotation['bbox'][1]) + float(annotation['bbox'][3]))])
            elif annotation['category_id'] == 8:
                gt_8.append([file_name, annotation['category_id'],
                    float(annotation['bbox'][0]),
                    float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
                    float(annotation['bbox'][1]),
                    (float(annotation['bbox'][1]) + float(annotation['bbox'][3]))])
            else:
                gt_9.append([file_name, annotation['category_id'],
                    float(annotation['bbox'][0]),
                    float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
                    float(annotation['bbox'][1]),
                    (float(annotation['bbox'][1]) + float(annotation['bbox'][3]))])

    mean_ap, average_precisions = mean_average_precision_for_boxes(gt, new_pred, iou_threshold=IOU_threshold)
    mean_ap_0, average_precisions_0 = mean_average_precision_for_boxes(gt_0, bbox_0, iou_threshold=IOU_threshold)
    mean_ap_1, average_precisions_1 = mean_average_precision_for_boxes(gt_1, bbox_1, iou_threshold=IOU_threshold)
    mean_ap_2, average_precisions_2 = mean_average_precision_for_boxes(gt_2, bbox_2, iou_threshold=IOU_threshold)
    mean_ap_3, average_precisions_3 = mean_average_precision_for_boxes(gt_3, bbox_3, iou_threshold=IOU_threshold)
    mean_ap_4, average_precisions_4 = mean_average_precision_for_boxes(gt_4, bbox_4, iou_threshold=IOU_threshold)
    mean_ap_5, average_precisions_5 = mean_average_precision_for_boxes(gt_5, bbox_5, iou_threshold=IOU_threshold)
    mean_ap_6, average_precisions_6 = mean_average_precision_for_boxes(gt_6, bbox_6, iou_threshold=IOU_threshold)
    mean_ap_7, average_precisions_7 = mean_average_precision_for_boxes(gt_7, bbox_7, iou_threshold=IOU_threshold)
    mean_ap_8, average_precisions_8 = mean_average_precision_for_boxes(gt_8, bbox_8, iou_threshold=IOU_threshold)
    mean_ap_9, average_precisions_9 = mean_average_precision_for_boxes(gt_9, bbox_9, iou_threshold=IOU_threshold)
    
    print("mean_ap")
    print(mean_ap)
    print((mean_ap_0+mean_ap_1+mean_ap_2+mean_ap_3+mean_ap_4+mean_ap_5+mean_ap_6+mean_ap_7+mean_ap_8+mean_ap_9)/10)


# set a argument parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--json',
        type=str,
        help='The config file which train model',
        default='/opt/ml/level2_objectdetection_cv-level2-cv-13/fold_dataset/test_val.json'
        )
    parser.add_argument(
        '--csv',
        type=str,
        default='/opt/ml/level2_objectdetection_cv-level2-cv-13/outputs/SwinTransformer_FPN_DyHead/test_val.csv'
    )
    parser.add_argument(
        '--iou_threshold',
        type=float,
        default=0.5
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    calculate_map(args.json, args.csv, args.iou_threshold)