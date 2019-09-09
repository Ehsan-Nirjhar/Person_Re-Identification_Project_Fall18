#################################################################################
######################## CSCE 625 : AI PROJECT : TEAM 17 ########################
## EXAMPLE: Loading a custom dataset
## Custom dataset are defined in "~\examples\custom_valset.py"
## Currently supports:
##     ValSetCSCE625() : validation dataset given in the CSCE625 Fall18
#################################################################################
#################################################################################

import torch
from torch.utils.data import DataLoader
from util import transforms as T
from util.dataset_loader import ImageDataset
import examples.custom_valset as custom
import os.path as osp

use_gpu = torch.cuda.is_available()
pin_memory = True if use_gpu else False
    
transform_test = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

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
