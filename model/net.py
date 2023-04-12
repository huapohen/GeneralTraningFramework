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

warnings.filterwarnings("ignore")


class Net(nn.Module):
    def __init__(self, params):
        super().__init__()
        is_pretrained = True if params.forward_mode == 'train' else False
        if params.net_type in ['basic', 'mobilenet_v2']:
            model = models.mobilenet_v2(pretrained=is_pretrained)
            model.classifier[1] = nn.Linear(model.last_channel, params.num_classes)
        elif params.net_type == 'convnext':
            # model = models.convnext_small(weights='DEFAULT')
            model = models.resnext101_32x8d(pretrained=is_pretrained)
            model.fc = nn.Linear(2048, params.num_classes)
        # Replace the last layer with a custom classifier
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
        model = MobileNet_v2(params)
        x = torch.randn(2, 3, 224, 224)
        y = model(x)
    _, indexs = torch.max(y, 1)
    print(indexs.detach().numpy())
    ip()
    pass
