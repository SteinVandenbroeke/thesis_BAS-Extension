import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
import numpy as np
import importlib
import os
import json

# New imports for connected components and COCO RLE encoding
from scipy.ndimage import label
import pycocotools.mask as maskUtils

from misc import torchutils, indexing, imutils

cudnn.enabled = True


def _work(process_id, model, dataset, args):
    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin,
                             shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    worker_results = []  # List to accumulate the JSON dictionaries for this GPU

    with torch.no_grad(), cuda.device(process_id):
        model.cuda()

        for iter, pack in enumerate(data_loader):
            if args.dataset == 'coco':
                img_name = pack['name'][0]
            else:
                import voc12.dataloader
                img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
            orig_img_size = np.asarray(pack['size'])

            edge, dp = model(pack['img'][0].cuda(non_blocking=True))

            cam_dict = np.load(args.cam_out_dir + '/' + img_name + '.npy', allow_pickle=True).item()
            cams = cam_dict['cam']
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            cams = np.power(cams, 1.5)
            cam_downsized_values = cams.cuda()

            rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times,
                                            radius=5)
            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0],
                    :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up)
            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=args.sem_seg_bg_thres)

            # Keep the probability map in CPU memory to calculate scores later
            prob_map_cpu = rw_up_bg.cpu().numpy()

            rw_pred_idx = torch.argmax(rw_up_bg, dim=0).cpu().numpy()
            rw_pred = keys[rw_pred_idx]

            if args.dataset == 'coco':
                # ---------------------------------------------------------
                # Process distinct instances instead of saving a .png mask
                # ---------------------------------------------------------
                unique_classes = np.unique(rw_pred)
                for cls in unique_classes:
                    if cls == 0:  # Assuming 0 is the background class index
                        continue

                    # 1. Isolate the mask for the current class
                    class_mask = (rw_pred == cls).astype(np.uint8)

                    # 2. Split into distinct, non-touching components
                    labeled_mask, num_features = label(class_mask)

                    # 3. Get the correct index to pull confidence scores for this class
                    cls_idx = np.where(keys == cls)[0][0]
                    cls_prob_map = prob_map_cpu[cls_idx]

                    for i in range(1, num_features + 1):
                        # Create a binary mask for just this specific isolated instance
                        instance_mask = (labeled_mask == i).astype(np.uint8)

                        # Calculate a dummy "score" using the mean random walk probability for this instance
                        score = float(np.mean(cls_prob_map[instance_mask == 1]))

                        # Convert to COCO RLE format (requires Fortran contiguous array)
                        rle = maskUtils.encode(np.asfortranarray(instance_mask))
                        rle_counts = rle['counts'].decode('utf-8')  # Convert bytes to string for JSON

                        worker_results.append({
                            "image_id": int(img_name),
                            "score": score,
                            "category_id": int(cls),
                            "segmentation": {
                                "size": [int(orig_img_size[0]), int(orig_img_size[1])],
                                "counts": rle_counts
                            }
                        })
            else:
                # Save as .png mask for VOC
                imageio.imwrite(os.path.join(args.sem_seg_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))


            if process_id == n_gpus - 1 and iter % max(1, (len(databin) // 20)) == 0:
                print("%d " % ((5 * iter + 1) // max(1, (len(databin) // 20))), end='')

    if args.dataset == 'coco':
        # Save the chunk of data handled by this process to a temporary file
        # This avoids multiprocessing locks and shared memory headaches
        tmp_json_path = os.path.join(args.sem_seg_out_dir, f'tmp_results_{process_id}.json')
        with open(tmp_json_path, 'w') as f:
            json.dump(worker_results, f)


def run(args):
    if args.dataset == 'coco':
        import coco.dataloader as dataloader
        root = args.coco_root
    else:
        import voc12.dataloader as dataloader
        root = args.voc12_root

    model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')()
    model.load_state_dict(torch.load(args.irn_weights_name), strict=False)
    model.eval()

    n_gpus = torch.cuda.device_count()

    if args.dataset == 'coco':
        dataset = dataloader.COCOClassificationDatasetMSF(args.infer_list,
                                                                 coco_root=root,
                                                                 scales=(1.0,))
    else:
        dataset = dataloader.VOC12ClassificationDatasetMSF(args.infer_list,
                                                                 voc12_root=root,
                                                                 scales=(1.0,))
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print("[", end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print("]")

    torch.cuda.empty_cache()

    if args.dataset == 'coco':
        # Merge the temporary JSON files from all workers into one final output
        all_results = []
        for i in range(n_gpus):
            tmp_json_path = os.path.join(args.sem_seg_out_dir, f'tmp_results_{i}.json')
            if os.path.exists(tmp_json_path):
                with open(tmp_json_path, 'r') as f:
                    all_results.extend(json.load(f))
                os.remove(tmp_json_path)  # Clean up temp files

        final_json_path = os.path.join(args.sem_seg_out_dir, 'instances_predictions.json')
        with open(final_json_path, 'w') as f:
            json.dump(all_results, f)

        print(f"\nSaved JSON instance segmentations to {final_json_path}")
