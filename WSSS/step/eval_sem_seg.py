import random

import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
import imageio
from tqdm import tqdm


# Fast histogram calculation to keep the confusion matrix fixed at 81x81
def fast_hist(pred, label, n_class):
    k = (label >= 0) & (label < n_class) # Automatically ignores label 255 (crowds/voids)
    return np.bincount(n_class * label[k].astype(int) + pred[k],
                       minlength=n_class ** 2).reshape(n_class, n_class)

def run(args):
    if args.dataset == 'voc':
        dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
        # labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

        preds = []
        labels = []
        n_img = 0
        for i, id in enumerate(dataset.ids):
            cls_labels = imageio.imread(os.path.join(args.sem_seg_out_dir, id + '.png')).astype(np.uint8)
            cls_labels[cls_labels == 255] = 0
            preds.append(cls_labels.copy())
            labels.append(dataset.get_example_by_keys(i, (1,))[0])
            n_img += 1

        confusion = calc_semantic_segmentation_confusion(preds, labels)[:21, :21]

        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj
        fp = 1. - gtj / denominator
        fn = 1. - resj / denominator
        iou = gtjresj / denominator
        print("total images", n_img)
        print(fp[0], fn[0])
        print(np.mean(fp[1:]), np.mean(fn[1:]))

        print({'iou': iou, 'miou': np.nanmean(iou)})
    elif args.dataset == 'coco':
        from pycocotools.coco import COCO

        ann_file = os.path.join(args.coco_root, 'annotations/instances_val2017.json')
        print("Loading COCO annotations...")
        coco_gt = COCO(ann_file)

        # Map COCO's sparse 90 IDs to continuous 1-80 IDs
        valid_cat_ids = sorted(coco_gt.getCatIds())
        cat_id_to_continuous = {cat_id: i + 1 for i, cat_id in enumerate(valid_cat_ids)}

        n_classes = 81  # 1 Background + 80 Objects
        confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
        n_img = 0

        img_ids = coco_gt.getImgIds()
        print(f"Calculating Semantic mIoU for {len(img_ids)} COCO images...")

        for img_id in tqdm(img_ids):
            img_info = coco_gt.loadImgs(img_id)[0]

            pred_filename = img_info['file_name'].replace('.jpg', '.png')
            pred_path = os.path.join(args.sem_seg_out_dir, pred_filename)

            if not os.path.exists(pred_path):
                continue

            pred_mask = imageio.imread(pred_path).astype(np.uint8)

            # --- THE FIX: Resize prediction to match Ground Truth if needed ---
            target_shape = (img_info['height'], img_info['width'])
            if pred_mask.shape != target_shape:
                from PIL import Image
                # PIL resize expects (width, height)
                target_size_pil = (img_info['width'], img_info['height'])
                pred_img = Image.fromarray(pred_mask)
                # Use NEAREST to prevent class IDs from blending into decimal values
                pred_mask = np.array(pred_img.resize(target_size_pil, resample=Image.NEAREST))
            # ------------------------------------------------------------------

            pred_mask[pred_mask == 255] = 0  # Handle WSSS ignore labels

            # 2. Build Ground Truth Mask on-the-fly
            gt_mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
            ann_ids = coco_gt.getAnnIds(imgIds=img_id)
            anns = coco_gt.loadAnns(ann_ids)

            for ann in anns:
                mask = coco_gt.annToMask(ann)
                if ann['iscrowd'] == 1:
                    # Ignore crowd regions by setting them to 255
                    gt_mask[mask == 1] = 255
                else:
                    continuous_id = cat_id_to_continuous[ann['category_id']]
                    gt_mask[mask == 1] = continuous_id

            # 3. Compute fixed-size confusion matrix
            confusion += fast_hist(pred_mask.flatten(), gt_mask.flatten(), n_classes)
            n_img += 1

        if n_img == 0:
            print("No prediction PNGs found! Check your sem_seg_out_dir and filenames.")
            return

        # 4. Final mIoU Calculation
        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)

        denominator = gtj + resj - gtjresj
        denominator = np.maximum(denominator, 1)  # Prevent division by zero

        fp = 1. - gtj / denominator
        fn = 1. - resj / denominator
        iou = gtjresj / denominator

        print("Total images evaluated:", n_img)
        print("Background FP/FN:", fp[0], fn[0])
        print("Mean Class FP/FN:", np.mean(fp[1:]), np.mean(fn[1:]))
        print({'iou': iou, 'miou': np.nanmean(iou)})

    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))