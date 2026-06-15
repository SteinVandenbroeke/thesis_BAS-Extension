import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from PIL import Image
from tqdm import tqdm


def run(args):
    if args.dataset == 'voc':
        dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)

        preds = []
        labels = []
        n_images = 0
        for i, id in enumerate(dataset.ids):
            n_images += 1
            cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
            cams = cam_dict['high_res']
            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            cls_labels = np.argmax(cams, axis=0)
            cls_labels = keys[cls_labels]
            preds.append(cls_labels.copy())
            labels.append(dataset.get_example_by_keys(i, (1,))[0])

        confusion = calc_semantic_segmentation_confusion(preds, labels)

        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj
        iou = gtjresj / denominator

        print("threshold:", args.cam_eval_thres, 'miou:', np.nanmean(iou), "i_imgs", n_images)
        print('among_predfg_bg', float((resj[1:].sum() - confusion[1:, 1:].sum()) / (resj[1:].sum())))
        miou = np.nanmean(iou)

    elif args.dataset == 'coco':
        # 1. Get the list of images directly from the dataset split folder
        split_dir = os.path.join(args.coco_root, args.chainer_eval_set)

        # Add a fallback just in case 'chainer_eval_set' is 'val' but the folder is 'val2017'
        if not os.path.isdir(split_dir):
            if os.path.isdir(os.path.join(args.coco_root, f"{args.chainer_eval_set}2017")):
                split_dir = os.path.join(args.coco_root, f"{args.chainer_eval_set}2017")
            elif os.path.isdir(os.path.join(args.coco_root, f"{args.chainer_eval_set}2014")):
                split_dir = os.path.join(args.coco_root, f"{args.chainer_eval_set}2014")
            else:
                raise FileNotFoundError(f"Could not find the dataset split folder at {split_dir}")

        # Extract IDs by listing the .jpg files in the directory
        ids = [os.path.splitext(f)[0] for f in os.listdir(split_dir) if f.endswith('.jpg')]

        if not ids:
            raise ValueError(f"No .jpg images found in {split_dir}. Check your coco_root and split.")

        preds = []
        labels = []
        n_images = 0

        for id in tqdm(ids):
            # --- Load Prediction (CAM) ---
            cam_path = os.path.join(args.cam_out_dir, id + '.npy')
            if not os.path.exists(cam_path):
                # Skip silently or print a warning if a CAM wasn't generated for an image
                print("Skipping ", id)
                continue

            n_images += 1
            cam_dict = np.load(cam_path, allow_pickle=True).item()
            cams = cam_dict['high_res']
            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            cls_labels = np.argmax(cams, axis=0)
            cls_labels = keys[cls_labels]
            preds.append(cls_labels.copy())

            # --- Load Ground Truth Mask ---
            # (Requires your pre-processed 8-bit PNG masks for COCO)
            mask_path = os.path.join(args.coco_root, 'SegmentationClass', id + '.png')
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Missing GT mask at {mask_path}. PNG masks are required for evaluation.")

            label = np.array(Image.open(mask_path), dtype=np.int32)

            # ChainerCV expects -1 for ignore labels (usually 255 in WSSS datasets)
            label[label == 255] = -1
            labels.append(label)

        # 2. Compute Confusion Matrix & mIoU
        confusion = calc_semantic_segmentation_confusion(preds, labels)

        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj
        iou = gtjresj / denominator

        print("threshold:", args.cam_eval_thres, 'miou:', np.nanmean(iou), "i_imgs", n_images)

        # Guard against division by zero if dataset is very small or corrupted
        if resj[1:].sum() > 0:
            print('among_predfg_bg', float((resj[1:].sum() - confusion[1:, 1:].sum()) / (resj[1:].sum())))
        else:
            print('among_predfg_bg: undefined (resj[1:].sum() is 0)')

        miou = np.nanmean(iou)

    else:
        print(f'{args.dataset} evaluation not implemented in eval_cam.py')
        miou = 0.0

    return miou