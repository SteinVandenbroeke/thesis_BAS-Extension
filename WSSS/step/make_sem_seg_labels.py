import imageio
import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
import numpy as np
import importlib
import os

from misc import torchutils, indexing, imutils
from tqdm import tqdm

cudnn.enabled = True


def _work(process_id, model, dataset, args):
    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin,
                             shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):
        model.cuda()

        for iter, pack in enumerate(tqdm(data_loader)):
            if args.dataset == 'coco':
                img_name = pack['name'][0]
            else:
                import voc12.dataloader
                img_name = voc12.dataloader.decode_int_filename(pack['name'][0])

            orig_img_size = [pack['size'][0].item(), pack['size'][1].item()]

            # --- BULLETPROOF IMAGE BATCHING ---
            img = pack['img'][0].cuda(non_blocking=True)

            if img.dim() == 3:
                img = img.unsqueeze(0)

            if img.shape[0] == 1:
                img = torch.cat([img, img.flip(-1)], dim=0)
            # ----------------------------------

            edge, dp = model(img)

            cam_dict = np.load(args.cam_out_dir + '/' + str(img_name) + '.npy', allow_pickle=True).item()
            cams = cam_dict['cam']
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            if isinstance(cams, np.ndarray):
                cams = torch.from_numpy(cams)
            cams = torch.pow(cams, 1.5)

            if cams.shape[-2:] != edge.shape[-2:]:
                cams = F.interpolate(cams.unsqueeze(0), size=edge.shape[-2:], mode='bilinear', align_corners=False)[0]

            cam_downsized_values = cams.cuda()

            rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times,
                                            radius=5)
            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0],
                    :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up)
            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=args.sem_seg_bg_thres)

            rw_pred_idx = torch.argmax(rw_up_bg, dim=0).cpu().numpy()
            rw_pred = keys[rw_pred_idx]

            # --- THE MASSIVE SPEEDUP ---
            # Save as a flat .png mask for BOTH datasets.
            # For COCO, we ensure the image name is padded to standard 12-digit format just in case
            if args.dataset == 'coco':
                save_name = str(img_name).zfill(12) + '.png'
            else:
                save_name = img_name + '.png'

            imageio.imwrite(os.path.join(args.sem_seg_out_dir, save_name), rw_pred.astype(np.uint8))

            if process_id == n_gpus - 1 and iter % max(1, (len(databin) // 20)) == 0:
                print("%d " % ((5 * iter + 1) // max(1, (len(databin) // 20))), end='')


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
    print(f"\nSaved all PNG semantic masks to {args.sem_seg_out_dir}")