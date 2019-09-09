from __future__ import absolute_import
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

import models
from util.losses import CrossEntropyLoss, DeepSupervision, CrossEntropyLabelSmooth, TripletLossAlignedReID
from util import data_manager
from util import transforms as T
from util.dataset_loader import ImageDataset
from util.utils import Logger
from util.utils import AverageMeter, Logger, save_checkpoint
from util.eval_metrics import evaluate
from util.optimizers import init_optim
from util.samplers import RandomIdentitySampler
#from IPython import embed


parser = argparse.ArgumentParser(description='Train AlignedReID with cross entropy loss and triplet hard loss')
# Datasets
parser.add_argument('--root', type=str, default='data', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='market1501',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0, help="split index")

# Optimization options
parser.add_argument('--labelsmooth', action='store_true', help="label smooth")
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=400, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")

parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=150, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
# triplet hard loss
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
# Miscs
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use_cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')


args = parser.parse_args()

def main():
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False
    pin_memory = True if use_gpu else False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_img_dataset(
        root=args.root, name=args.dataset, split_id=args.split_id,
    )

    # data augmentation
    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )


    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'softmax','metric'}, aligned =True, use_gpu=use_gpu)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    if args.labelsmooth:
        criterion_class = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    else:
        criterion_class = CrossEntropyLoss(use_gpu=use_gpu)
    criterion_metric = TripletLossAlignedReID(margin=args.margin)
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)

    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    if use_gpu:
        model = nn.DataParallel(model).cuda()



    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")
    cnt_n=0
    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, criterion_class, criterion_metric, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)
        cnt_n=cnt_n+1
        
         

        if args.stepsize > 0: scheduler.step()
        if (cnt_n % 40) == 0 :       #### Saving models after each 40 epochs
            print("==> Saving")
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            is_best = 0
            save_checkpoint({
                'state_dict': state_dict,             ### rank1 and is_best are kept same as original code. Don't want to mess up the saving
                'rank1': '###',
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))


    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

def train(epoch, model, criterion_class, criterion_metric, optimizer, trainloader, use_gpu):
    model.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    xent_losses = AverageMeter()
    global_losses = AverageMeter()
    local_losses = AverageMeter()

    end = time.time()
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        # measure data loading time
        data_time.update(time.time() - end)
        outputs, features, local_features = model(imgs)
        #print(features.size())
        #print(local_features.size())
        if args.htri_only:
            if isinstance(features, tuple):
                global_loss, local_loss = DeepSupervision(criterion_metric, features, pids, local_features)
            else:
                global_loss, local_loss = criterion_metric(features, pids, local_features)
        else:
            if isinstance(outputs, tuple):
                xent_loss = DeepSupervision(criterion_class, outputs, pids)
            else:
                xent_loss = criterion_class(outputs, pids)

            if isinstance(features, tuple):
                global_loss, local_loss = DeepSupervision(criterion_metric, features, pids, local_features)
            else:
                global_loss, local_loss = criterion_metric(features, pids, local_features)
        loss = xent_loss + global_loss + local_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.item(), pids.size(0))
        xent_losses.update(xent_loss.item(), pids.size(0))
        global_losses.update(global_loss.item(), pids.size(0))
        local_losses.update(local_loss.item(), pids.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'CLoss {xent_loss.val:.4f} ({xent_loss.avg:.4f})\t'
                  'GLoss {global_loss.val:.4f} ({global_loss.avg:.4f})\t'
                  'LLoss {local_loss.val:.4f} ({local_loss.avg:.4f})\t'.format(
                   epoch+1, batch_idx+1, len(trainloader), batch_time=batch_time,data_time=data_time,
                   loss=losses,xent_loss=xent_losses, global_loss=global_losses, local_loss = local_losses))



if __name__ == '__main__':
    main()
