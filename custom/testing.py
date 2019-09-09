#################################################################################
######################## CSCE 625 : AI PROJECT : TEAM 17 ########################
## Get predictions and evaluates the performance
#################################################################################
#################################################################################

from util.utils import AverageMeter, Logger, save_checkpoint
from custom.evaluation import evaluate
import time
import models
import numpy as np
from util.losses import CrossEntropyLoss, DeepSupervision, CrossEntropyLabelSmooth, TripletLossAlignedReID
from os import getcwd
import torch
import torch.nn as nn
from util.utils import AverageMeter, Logger, save_checkpoint

def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids, lqf = [], [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features, local_features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            local_features = local_features.data.cpu()
            qf.append(features)
            lqf.append(local_features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        lqf = torch.cat(lqf,0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids, lgf = [], [], [], []
        end = time.time()
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features, local_features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            local_features = local_features.data.cpu()
            gf.append(features)
            lgf.append(local_features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        lgf = torch.cat(lgf,0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)


        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, 4))
    # feature normlization
    qf = 1. * qf / (torch.norm(qf, 2, dim = -1, keepdim=True).expand_as(qf) + 1e-12)
    gf = 1. * gf / (torch.norm(gf, 2, dim = -1, keepdim=True).expand_as(gf) + 1e-12)
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()
    
    ## Calculating global distance
    print("Using global and local branches")
    from util.distance import low_memory_local_dist
    lqf = lqf.permute(0,2,1)
    lgf = lgf.permute(0,2,1)
    local_distmat = low_memory_local_dist(lqf.numpy(),lgf.numpy(),aligned=True)
    distmat = local_distmat+distmat
    
    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    
    return distmat
