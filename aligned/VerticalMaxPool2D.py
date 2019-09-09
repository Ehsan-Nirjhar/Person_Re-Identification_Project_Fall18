import torch
import torch.nn as nn

class VerticalMaxPool2d(nn.Module):
    def __init__(self):
        super(VerticalMaxPool2d, self).__init__()


    def forward(self, x):
        inp_size = x.size()

        return torch.transpose(nn.functional.max_pool2d(input=x,kernel_size= (inp_size[2], 1)),  2,3)