import argparse
import os
import mmcv
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


def check_kfold():
    val_ids=[]
    train_anns=mmcv.load(args.data_root)
    for k in range(args.fold):
        val_path=os.path.join(args.out_dir,f'validation_fold_{k+1}_of_{args.fold}.json')
        val_anns = mmcv.load(val_path)
        image_list=val_anns['images']
        for img in image_list:
            val_ids.append(img['id'])
    val_ids=set(val_ids)
    assert len(val_ids)==len(train_anns['images']),"Not all image is included in one of validation folds"

if __name__=="__main__":
    args = parse_args()
    check_kfold()