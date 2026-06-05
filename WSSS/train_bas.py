import os
import argparse
from pickletools import optimize
import torch
import torch.nn as nn
from tqdm import tqdm

from net import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tool.accuracy import *
from tool.func import *
from tool.util import *
import torch.nn.functional as F
import pprint
import os
import math
import random
import shutil
from tool import pyutils, imutils, torchutils, visualization, optimizer
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def cross_entropy_copy(logits, y):    
    s = torch.exp(logits)
    logits = s / (torch.sum(s * (1-y), dim=1, keepdim=True) + s)
    c = -(y * torch.log(logits+1e-8)).sum(dim=-1)
    return torch.mean(c)

def cross_entropy(logits, y, device, num_classes=80):
    batch = logits.size(0)
    c = torch.tensor(0.).to(device)
    loss_func = nn.CrossEntropyLoss().to(device)
    ##  [gt_label, neg]
    for i in range(batch):
        class_num = int(y[i].sum(-1)) 
        if class_num == 0:
            continue
        p_batch_label = torch.zeros(class_num).to(device)
        p_batch_label = p_batch_label.long() 
        batch_neg_logits = logits[i][y[i]<0.5]
        batch_pos_logits = logits[i][y[i]>0.5]
        batch_pos_logits = batch_pos_logits.view(class_num, 1)
        batch_neg_logits = batch_neg_logits.view(1, num_classes - class_num).expand((class_num, num_classes - class_num))
        batch_logits = torch.cat([batch_pos_logits, batch_neg_logits], 1)
        c += loss_func(batch_logits, p_batch_label) * class_num/ batch
    return c

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        ##  save
        self.parser.add_argument('--save_path', type=str, default='sess')
        self.parser.add_argument('--log_file', type=str, default='log.txt')
        self.parser.add_argument('--log_code_dir', type=str, default='save_code')

        # Dataset
        self.parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'voc'])
        self.parser.add_argument("--train_list", default="coco/train.txt", type=str)
        self.parser.add_argument("--val_list", default="coco/val.txt", type=str)
        self.parser.add_argument("--coco_root", default='/home2/wpy/WSSS/COCO/', type=str)
        self.parser.add_argument("--voc12_root", default='/home2/wpy/WSSS/VOC2012/', type=str)

        ##  dataloader
        self.parser.add_argument('--crop_size', default=448, type=int)
        self.parser.add_argument('--num_workers', type=int, default=2)
        ##  train
        self.parser.add_argument('--num_classes', type=int, default=80)
        self.parser.add_argument('--batch_size', type=int, default=16)
        self.parser.add_argument('--epochs', type=int, default=10)
        self.parser.add_argument('--phase', type=str, default='train') ## train / test
        self.parser.add_argument('--lr', type=float, default=0.005)
        self.parser.add_argument('--wt_dec', type=float, default=1e-4)
        self.parser.add_argument('--power', type=float, default=0.9)
        self.parser.add_argument('--momentum', type=float, default=0.9)
        self.parser.add_argument("--local_rank", type=int,default=-1)
        self.parser.add_argument("--seed", type=int,default=0)
        ##  model
        self.parser.add_argument('--arch', type=str, default='resnet50_bas')   ##  choose  [ vgg, resnet, inception, mobilenet ]   
        
    def parse(self):
        opt = self.parser.parse_args()
        opt.arch = opt.arch
        if opt.dataset == 'voc':
            opt.train_list = "voc12/train_aug.txt"
            opt.val_list = "voc12/val.txt"
            opt.num_classes = 20
        return opt
  
args = opts().parse()
lr = args.lr

if args.dataset == 'coco':
    import coco.dataloader
else:
    import voc12.dataloader

args.batch_size = args.batch_size // max(1, int(torch.cuda.device_count()))
print(f"Batch size per GPU: {args.batch_size}")

## save_log_txt
makedirs(args.save_path + '/' + args.arch + '/' + args.log_code_dir)
sys.stdout = Logger(args.save_path  + '/' + args.arch +  '/' + args.log_code_dir + '/' + args.log_file)
sys.stdout.log.flush()

##  save_code
save_file = ['train_bas.py', 'run_sample.py']
for file_name in save_file:
    shutil.copyfile(file_name, args.save_path + '/' + args.arch + '/' + args.log_code_dir + '/' + file_name)
save_dir = ['net', 'step', 'misc']
for file_name in save_dir:
    copy_dir(file_name, args.save_path + '/' + args.arch + '/' + args.log_code_dir + '/' + file_name)


# Check if we are running in a distributed environment (e.g. via torchrun)
is_distributed = "LOCAL_RANK" in os.environ

if is_distributed:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda", local_rank)
    set_seed(args.seed + dist.get_rank())
    print(f"Running in distributed mode. Rank: {dist.get_rank()}")
else:
    # Single GPU / non-distributed setup
    local_rank = args.local_rank if args.local_rank != -1 else 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    set_seed(args.seed)
    print("Running in single-GPU / non-distributed mode.")


if args.dataset == 'coco':
    train_dataset = coco.dataloader.COCOClassificationDataset(args.train_list, coco_root=args.coco_root,
                                                                    resize_scale=512, hor_flip=True,
                                                                    crop_size=448, crop_method="random")
else:
    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_scale=512, hor_flip=True,
                                                                crop_size=448, crop_method="random")

if is_distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, 
                                    num_workers=args.num_workers, pin_memory=True, drop_last=True)
else:
    # Use standard dataloader without DistributedSampler
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=True)


max_step = len(train_dataset) // args.batch_size * args.epochs

model = eval(args.arch).model(num_classes=args.num_classes)

if is_distributed:
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda().to(local_rank)
    model = DDP(model, find_unused_parameters=False, device_ids=[local_rank], output_device=local_rank)
else:
    model = model.to(device)

model.train()

optimizer = optimizer.get_optimizer(model, args, max_step)

for epoch in range(args.epochs):
    if is_distributed:
        train_data_loader.sampler.set_epoch(epoch)
        
    loss_epoch_1 = AverageMeter()
    loss_epoch_2 = AverageMeter()
    loss_epoch_3 = AverageMeter()
    loss_epoch_4 = AverageMeter()
    for step, pack in enumerate(tqdm(train_data_loader)):
        if args.dataset == 'coco':
            imgs, label = pack['img'].to(device), pack['label'].to(device)
        else:
            imgs, label = pack[0].to(device), pack[1].to(device)

        optimizer.zero_grad()
        output1, loss2, loss3, loss4 = model(imgs, label)  
        
        loss1 = cross_entropy(output1, label, device, num_classes=args.num_classes)
        loss = loss1 + loss2 * 0.2 + loss3 + loss4 * 1.2 
        loss.backward()
        optimizer.step() 
        loss_epoch_1.updata(loss1.data, 1)
        loss_epoch_2.updata(loss2.data, 1)
        loss_epoch_3.updata(loss3.data, 1)
        loss_epoch_4.updata(loss4.data, 1)
        
    # Only log and save from the main process (rank 0 in DDP, or always in single GPU)
    if not is_distributed or dist.get_rank() == 0:
        print('  Epoch:[{}/{}]\t step:[{}/{}]\tcls_loss_1:{:.3f}\tcls_loss_2:{:.3f}\tbas_loss:{:.3f}\tarea_loss:{:.3f}  '.format(
                epoch+1, args.epochs, step+1, len(train_data_loader), loss_epoch_1.avg, loss_epoch_2.avg, loss_epoch_3.avg, loss_epoch_4.avg  
        ))
        sys.stdout.log.flush()    
        torch.save(model.state_dict(), os.path.join(args.save_path + '/' + args.arch  +'/' + 'epoch_'+ str(epoch+1) +'.pth.tar'),_use_new_zipfile_serialization=False)