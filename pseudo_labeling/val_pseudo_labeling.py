import pandas as pd
import json
import argparse
from tqdm import tqdm
from tqdm.auto import tqdm

# json 경로, 저장 경로, csv 경로
def parse_args():                                                           # set a argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--empty-json-dir',
        type=str,
        help='The json file directory of train.json',
        default='./pseudo.json')
    parser.add_argument(
        '--val-json-dir',
        type=str,
        default='/opt/ml/level2_objectdetection_cv-level2-cv-13/stratified_fold_dataset/train_fold_1_of_5.json'
    )
    parser.add_argument(
        '--csv-dir',
        type=str,
        help='The csv file directory of model_output.csv',
        default='/opt/ml/level2_objectdetection_cv-level2-cv-13/outputs/SwinTransformer_FPN_DyHead/output_1.csv')
    parser.add_argument(
        '--out-dir',
        type=str,
        help='The directory of output json file.',
        default='/opt/ml/dataset/train_pseudo.json')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.3
    )
    args = parser.parse_args()
    return args

# pseudo labeling
def pseudo_labeling(empty_json_dir, val_json_dir, csv_dir, out_dir, threshold):
    df = pd.read_csv(csv_dir)                                               # model output csv file load
    
    with open(empty_json_dir, encoding='utf-8', errors='ignore') as json_data:    # json file open
        data = json.load(json_data, strict=False)
        old_id_num = len(data['images'])                                    # length of existing train image
        old_ann_id = int(data['annotations'][-1]['id']+1 if len(data['annotations']) != 0 else 0)                             # length of existing train annotated object
        for i in tqdm(range(len(df))):                                           # append new json data for each image

            data["images"].append({                                         # append a new images row
                    "width": 1024,
                    "height": 1024,
                    "file_name": df['image_id'][i],
                    "license": 0,
                    "flickr_url": None,
                    "coco_url": None,
                    "date_captured": None,
                    "id": int(df['image_id'][i].split('/')[1].split('.')[0])
                })

            ann = str(df['PredictionString'][i]).split()
            ann_num = len(ann)//6

            for j in range(ann_num):                                        # append new json data for each object annotations
                if float(ann[j*6 + 1]) >= threshold:
                    category_id = ann[j*6 + 0]
                    bbox_x = float(ann[j*6 + 2])
                    bbox_y = float(ann[j*6 + 3])
                    bbox_w = float(ann[j*6 + 4]) - float(ann[j*6 + 2])
                    bbox_h = float(ann[j*6 + 5]) - float(ann[j*6 + 3])
                    old_ann_id += 1
                    data['annotations'].append({                                # append a new annotations row
                        "image_id": int(df['image_id'][i].split('/')[1].split('.')[0]),
                        "category_id": int(category_id),
                        "area": bbox_w * bbox_h,
                        "bbox": [
                            bbox_x,
                            bbox_y,
                            bbox_w,
                            bbox_h
                        ],
                        "iscrowd": 0,
                        "id": old_ann_id
                        })

    with open(out_dir, 'w') as outfile:                                     # save the file
        json.dump(data, outfile, indent=4)


if __name__ =="__main__":
    args = parse_args()
    pseudo_labeling(args.empty_json_dir, args.val_json_dir, args.csv_dir, args.out_dir, args.threshold)