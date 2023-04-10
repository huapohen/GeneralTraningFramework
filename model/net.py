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


class MobileNet_v2(nn.Module):
    def __init__(self, params):
        super().__init__()
        is_pretrained = True if params.forward_mode == 'train' else False
        model = models.mobilenet_v2(pretrained=is_pretrained)
        # Replace the last layer with a custom classifier
        model.classifier[1] = nn.Linear(model.last_channel, params.num_classes)
        self.model = model

    def forward(self, x):
        x = self.model(x)

        return x


def fetch_net(params):

    if params.net_type == "basic":
        net = MobileNet_v2(params)
    else:
        raise NotImplementedError("Unkown model: {}".format(params.net_type))
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
