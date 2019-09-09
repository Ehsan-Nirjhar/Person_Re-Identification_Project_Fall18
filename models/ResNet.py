from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import sys
from aligned.HorizontalMaxPool2D import HorizontalMaxPool2d
from aligned.VerticalMaxPool2D import VerticalMaxPool2d

__all__ = ['ResNet50', 'ResNet101']

class ResNet50(nn.Module):
    def __init__(self, num_classes, loss={'softmax'}, aligned=False, **kwargs):
        super(ResNet50, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        self.vert_pool = VerticalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        ### lf is for horizontal stripes, while vf is for vertical.
        x = self.base(x)
        if not self.training:
            lf = self.horizon_pool(x)
            vf = self.vert_pool(x)
        if self.aligned and self.training:
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
#            print(lf.size())
            lf = self.conv1(lf)
#            print(lf.size())
            vf = self.bn(x)
            vf = self.relu(vf)
            vf = self.vert_pool(vf)
#            print(vf.size())
            vf = self.conv1(vf)
#            print(vf.size())
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
#            print(lf.size())
            vf = vf.view(vf.size()[0:3])
#            print(vf.size())
            lf = lf / torch.pow(lf,2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
#            print(lf.size())
            vf = vf / torch.pow(vf,2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
#            print(vf.size())
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        lf=torch.cat((lf,vf), dim=2)
#        print(lf.size())

        #f = 1. * f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)
        if not self.training:
            return f,lf
        y = self.classifier(f)
        if self.loss == {'softmax'}:
            return y
        elif self.loss == {'metric'}:
            if self.aligned: return  f, lf
            return f
        elif self.loss == {'softmax', 'metric'}:
            if self.aligned: return y, f, lf
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet101(nn.Module):
    def __init__(self, num_classes, loss={'softmax'}, aligned=False, **kwargs):
        super(ResNet101, self).__init__()
        self.loss = loss
        resnet101 = torchvision.models.resnet101(pretrained=False)
        self.base = nn.Sequential(*list(resnet101.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.base(x)
        if not self.training:
            lf = self.horizon_pool(x)
        if self.aligned:
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
            print(lf.size())
            lf = self.conv1(lf)
            print(lf.size())
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            print(lf.size())
            lf = lf / torch.pow(lf, 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
            print(lf.size())
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        sys.exit()
        # f = 1. * f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)
        if not self.training:
            return f, lf
        y = self.classifier(f)
        if self.loss == {'softmax'}:
            return y
        elif self.loss == {'metric'}:
            if self.aligned: return f, lf
            return f
        elif self.loss == {'softmax', 'metric'}:
            if self.aligned: return y, f, lf
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
