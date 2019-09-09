#################################################################################
######################## CSCE 625 : AI PROJECT : TEAM 17 ########################
## EXAMPLE: Testing on a custom dataset with a trained model
## Custom dataset are defined in "~\examples\custom_valset.py"
## Currently supports:
##     ValSetCSCE625() : validation dataset given in the CSCE625 Fall18
#################################################################################
#################################################################################

import torch
from torch.utils.data import DataLoader
from util import transforms as T
from util.dataset_loader import ImageDataset
import custom.validationset as custom
import os.path as osp
import time
import models
import numpy as np
from util.losses import CrossEntropyLoss, DeepSupervision, CrossEntropyLabelSmooth, TripletLossAlignedReID
from os import getcwd
import torch
import torch.nn as nn
from util.utils import AverageMeter, Logger, save_checkpoint
from custom.testing import test


use_gpu = torch.cuda.is_available()
pin_memory = True if use_gpu else False
    
transform_test = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

## CUSTOM DATASET
dataset = custom.ValSetCSCE625()
    
queryloader = DataLoader(
    ImageDataset(dataset.query, transform=transform_test),
    batch_size=4, shuffle=False, num_workers=1,
    pin_memory=pin_memory, drop_last=False,
)

galleryloader = DataLoader(
    ImageDataset(dataset.gallery, transform=transform_test),
    batch_size=4, shuffle=False, num_workers=1,
    pin_memory=pin_memory, drop_last=False,
)

## MODEL OPTIONS
mdl_arch        = 'resnet50'                ## Network architecture
mdl_weight      = '\\checkpoint_ep300.pth'  ## Path to the weight file
mdl_num_classes = 751                       ## For MarketNet1501
labelsmooth = False

## Load the model
print("Initializing model: {}".format(mdl_arch))
model = models.init_model(name=mdl_arch,
                          num_classes=mdl_num_classes,
                          loss={'softmax','metric'}, 
                          aligned =True, 
                          use_gpu=use_gpu)

if labelsmooth:
    criterion_class = CrossEntropyLabelSmooth(num_classes=mdl_num_classes, use_gpu=use_gpu)
else:
    criterion_class = CrossEntropyLoss(use_gpu=use_gpu)

## Load the weights
print("Loading checkpoint from '{}'".format(mdl_weight))
checkpoint = torch.load(mdl_weight)
model.load_state_dict(checkpoint['state_dict'])

if use_gpu:
    model = nn.DataParallel(model).cuda()

print("Evaluate only")
distmat = test(model, queryloader, galleryloader, use_gpu)
