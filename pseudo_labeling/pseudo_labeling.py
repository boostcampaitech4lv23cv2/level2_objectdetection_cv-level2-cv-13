import pandas as pd
import json
import argparse
from tqdm import tqdm
from tqdm.auto import tqdm

# json 경로, 저장 경로, csv 경로
def parse_args():                                                           # set a argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--json-dir',
        type=str,
        help='The json file directory of train.json',
        default='/opt/ml/dataset/train.json')
    parser.add_argument(
        '--csv-dir',
        type=str,
        help='The csv file directory of model_output.csv',
        default='/opt/ml/sample_submission/train_sample.csv')
    parser.add_argument(
        '--out-dir',
        type=str,
        help='The directory of output json file.',
        default='/opt/ml/dataset/train_pseudo.json')
    args = parser.parse_args()
    return args

# pseudo labeling
def pseudo_labeling(json_dir, csv_dir, out_dir):
    df = pd.read_csv(csv_dir)                                               # model output csv file load
    with open(json_dir, encoding='utf-8', errors='ignore') as json_data:    # json file open
        data = json.load(json_data, strict=False)
        old_id_num = len(data['images'])                                    # length of existing train image
        old_ann_id = len(data['annotations'])                               # length of existing train annotated object
        for i in tqdm(range(len(df))):                                            # append new json data for each image

            data["images"].append({                                         # append a new images row
                    "width": 1024,
                    "height": 1024,
                    "file_name": df['image_id'][i],
                    "license": 0,
                    "flickr_url": None,
                    "coco_url": None,
                    "date_captured": None,
                    "id": old_id_num + i
                })

            ann = str(df['PredictionString'][i]).split()
            ann_num = len(ann)//6

            for j in range(ann_num):                                        # append new json data for each object annotations

                category_id = ann[j*6 + 0]
                bbox_x = float(ann[j*6 + 2])
                bbox_y = float(ann[j*6 + 3])
                bbox_w = float(ann[j*6 + 4]) - float(ann[j*6 + 2])
                bbox_h = float(ann[j*6 + 5]) - float(ann[j*6 + 3])

                data['annotations'].append({                                # append a new annotations row
                    "image_id": old_id_num + i,
                    "category_id": category_id,
                    "area": bbox_w * bbox_h,
                    "bbox": [
                        bbox_x,
                        bbox_y,
                        bbox_w,
                        bbox_h
                    ],
                    "iscrowd": 0,
                    "id": old_ann_id + j
                    })

            old_ann_id += ann_num                                           # renew old_ann_id

    with open(out_dir, 'w') as outfile:                                     # save the file
        json.dump(data, outfile, indent=4)


if __name__ =="__main__":
    args = parse_args()
    pseudo_labeling(args.json_dir, args.csv_dir, args.out_dir)