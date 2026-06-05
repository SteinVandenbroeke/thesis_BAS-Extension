
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
import imageio

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
        from pycocotools.cocoeval import COCOeval
        import json

        ann_file = os.path.join(args.coco_root, 'annotations/instances_val2017.json')
        coco_gt = COCO(ann_file)

        res_file = os.path.join(args.sem_seg_out_dir, 'instances_predictions.json')
        
        # Ensure the file exists and is not empty
        if not os.path.exists(res_file) or os.path.getsize(res_file) == 0:
            print(f"Result file not found or is empty: {res_file}")
            return

        try:
            coco_dt = coco_gt.loadRes(res_file)
        except Exception as e:
            print(f"Error loading result file: {e}")
            return

        coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
