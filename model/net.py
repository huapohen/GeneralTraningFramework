import os
import cv2
import sys
import ipdb
import copy
import torch
import imageio
import logging
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from easydict import EasyDict
from ipdb import set_trace as ip
from .efficientformer_v2 import efficientformerv2_l

warnings.filterwarnings("ignore")


class Net(nn.Module):
    """
    https://pytorch.org/vision/stable/models.html
    pytorch version:
        1.8.0 use pretrained=True
        other use weights='DEFAULT'
    """
    def __init__(self, params):
        super().__init__()
        is_pretrained = True if params.forward_mode == 'train' else False
        # Replace the last layer with a custom classifier
        if params.net_type in ['basic', 'mobilenet']:
            # model = models.mobilenet_v2(pretrained=is_pretrained) # GFLOPS: 0.30
            # model.classifier[1] = nn.Linear(model.last_channel, params.num_classes)
            model = models.mobilenet_v3_large(pretrained=is_pretrained)
            # model = models.mobilenet_v3_large(weights='DEFAULT') # GFLOPS: 0.22
            model.classifier[3] = nn.Linear(1280, params.num_classes)
            param_names = ['classifier']
        if params.net_type == 'shufflenet':
            model = models.shufflenet_v2_x2_0(weights='DEFAULT') # GFLOPS: 0.58
            model.fc = nn.Linear(2048, params.num_classes)
            param_names = ['fc']
        elif params.net_type == 'resnet':
            model = models.resnext101_32x8d(pretrained=is_pretrained) # GFLOPS: 16.41
            # model = models.resnext101_64x4d(weights='DEFAULT') # GFLOPS: 15.46
            model.fc = nn.Linear(2048, params.num_classes)
            param_names = ['fc']
        elif params.net_type == 'densenet':
            # model = models.densenet201(weights='DEFAULT') # GFLOPS: 4.29
            model = models.densenet201(pretrained=True)
            model.classifier = nn.Linear(1920, params.num_classes)
            param_names = ['classifier']
        elif params.net_type == 'vit':
            model = models.vit_b_16(weights='DEFAULT') # GFLOPS: 17.56
            model.heads[3] = nn.Linear(512, params.num_classes)
            param_names = ['heads.3']
        elif params.net_type == 'efficientnet':
            model = models.efficientnet_v2_l(weights='DEFAULT') # GFLOPS: 56.08
            # model = models.efficientnet_v2_s(weights='DEFAULT') # GFLOPS: 8.37
            model.classifier[1] = nn.Linear(1280, params.num_classes)
            param_names = ['classifier.1']
        elif params.net_type == 'swin':
            model = models.swin_b(weights='DEFAULT') # GFLOPS: 15.43
            model.classifier[2] = nn.Linear(1024, params.num_classes)
            param_names = ['classifier.2']
        elif params.net_type == 'convnext':
            model = models.convnext_base(weights='DEFAULT') # GFLOPS: 15.36
            model.classifier[2] = nn.Linear(1024, params.num_classes)
            param_names = ['classifier.2']
        elif params.net_type == 'maxvit':
            model = models.maxvit_t(weights='DEFAULT') # GFLOPS: 5.56
            model.classifier[5] = nn.Linear(512, params.num_classes)
            param_names = ['classifier.5']
        elif params.net_type == 'efficientformer':
            model = efficientformerv2_l(pretrained=True) # MACs: 2.56B
            ckp = torch.load(f'/home/data/lwb/ckp/eformer_l_450.pth')
            model.load_state_dict(ckp['model'], strict=False)
            model.head = nn.Linear(384, params.num_classes)
            param_names = ['head']
        # named_parameters for freezing parameters
        params.param_names = param_names
        self.model = model
        
    def forward(self, x):
        x = self.model(x)

        return x


def fetch_net(params):
    net = Net(params)

    return net


if __name__ == '__main__':
    params = EasyDict()
    params.forward_mode = 'train'
    params.num_classes = 5

    with torch.no_grad():
        model = models.mobilenet_v3_large(params)
        x = torch.randn(2, 3, 224, 224)
        y = model(x)
    _, indexs = torch.max(y, 1)
    print(indexs.detach().numpy())
    ip()
    pass
