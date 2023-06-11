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
    """
    https://pytorch.org/vision/stable/models.html
    pytorch version:
        1.8.0 use pretrained=True
        other use weights='DEFAULT'
    """

    def __init__(self, params):
        super().__init__()
        is_pretrained = True if params.forward_mode == "train" else False
        # Replace the last layer with a custom classifier
        if params.net_type in ["basic", "mobilenet"]:
            model = models.mobilenet_v2(pretrained=is_pretrained)  # GFLOPS: 0.30
            # model.classifier[1] = nn.Linear(model.last_channel, params.num_classes)
            # model.classifier[1] = MLP(model.last_channel, 1280, params.num_classes, 3)
            # param_names = ['classifier.1']
            model.classifier = MLP(model.last_channel, 1280, params.num_classes, 3)
            param_names = ["classifier"]  # Accuracy=83.45%
            # model = models.mobilenet_v3_large(pretrained=is_pretrained)
            # model = models.mobilenet_v3_large(weights='DEFAULT') # GFLOPS: 0.22
            # model.classifier[3] = nn.Linear(1280, params.num_classes)
            # model.classifier[3] = MLP(1280, 1280, params.num_classes, 3)
            # param_names = ['classifier']
            # param_names = ['classifier.3']
            # model.classifier = MLP(960, 1280, params.num_classes, 3)
            # param_names = ['classifier']
        elif params.net_type == "shufflenet":
            model = models.shufflenet_v2_x2_0(weights="DEFAULT")  # GFLOPS: 0.58
            model.fc = nn.Linear(2048, params.num_classes)
            param_names = ["fc"]
        elif params.net_type == "resnet":
            model = models.resnext101_32x8d(pretrained=is_pretrained)  # GFLOPS: 16.41
            # model = models.resnext101_64x4d(weights='DEFAULT') # GFLOPS: 15.46
            # model.fc = nn.Linear(2048, params.num_classes)
            model.fc = MLP(2048, 1280, params.num_classes, 3)
            param_names = ["fc"]
        elif params.net_type == "densenet":
            # model = models.densenet201(weights='DEFAULT') # GFLOPS: 4.29
            model = models.densenet201(pretrained=True)
            model.classifier = nn.Linear(1920, params.num_classes)
            # model.classifier = MLP(1920, 1280, params.num_classes, 3)
            param_names = ["classifier"]
        elif params.net_type == "vit":
            model = models.vit_b_16(weights="DEFAULT")  # GFLOPS: 17.56
            model.heads[3] = nn.Linear(512, params.num_classes)
            param_names = ["heads.3"]
        elif params.net_type == "efficientnet":
            model = models.efficientnet_v2_l(weights="DEFAULT")  # GFLOPS: 56.08
            # model = models.efficientnet_v2_s(weights='DEFAULT') # GFLOPS: 8.37
            model.classifier[1] = nn.Linear(1280, params.num_classes)
            param_names = ["classifier.1"]
        elif params.net_type == "swin":
            model = models.swin_b(weights="DEFAULT")  # GFLOPS: 15.43
            model.classifier[2] = nn.Linear(1024, params.num_classes)
            param_names = ["classifier.2"]
        elif params.net_type == "convnext":
            model = models.convnext_base(weights="DEFAULT")  # GFLOPS: 15.36
            model.classifier[2] = nn.Linear(1024, params.num_classes)
            param_names = ["classifier.2"]
        elif params.net_type == "maxvit":
            model = models.maxvit_t(weights="DEFAULT")  # GFLOPS: 5.56
            model.classifier[5] = nn.Linear(512, params.num_classes)
            param_names = ["classifier.5"]
        elif params.net_type == "efficientformer":
            from .efficientformer_v2 import efficientformerv2_l

            model = efficientformerv2_l(pretrained=True)  # MACs: 2.56B
            ckp = torch.load(f"/home/data/lwb/ckp/eformer_l_450.pth")
            model.load_state_dict(ckp["model"], strict=False)
            # model.head = nn.Linear(384, params.num_classes)
            model.head = MLP(384, 1280, params.num_classes, 3)  # Accuracy=96.78%
            param_names = ["head"]
        else:
            raise
        # named_parameters for freezing parameters
        params.param_names = param_names
        self.model = model

    def forward(self, x):
        x = self.model(x)

        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


def fetch_net(params):
    net = Net(params)

    return net


def main():
    params = EasyDict()
    params.forward_mode = "train"
    params.num_classes = 5

    with torch.no_grad():
        model = models.mobilenet_v3_large(params)
        x = torch.randn(2, 3, 224, 224)
        y = model(x)
    _, indexs = torch.max(y, 1)
    print(indexs.detach().numpy())
    # ip()
    print("pass")

    with torch.no_grad():
        model = models.mobilenet_v2(pretrained=True)
        model.classifier = MLP(model.last_channel, 1280, 7, 3)
        x = torch.randn(2, 3, 224, 224)
        y = model(x)
    _, indexs = torch.max(y, 1)
    print(indexs.detach().numpy())
    # ip()
    print("pass")

    pass


if __name__ == "__main__":
    main()
