import os
import sys
import cv2
import copy
import ipdb
import torch
import shutil
import imageio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import net
from torchvision.utils import save_image
from ipdb import set_trace as ip



def compute_losses(output, input, params, k=0):
    losses = {}
    losses['total'] = F.cross_entropy(output, input['label'])
    return losses


def compute_eval_results(data_batch, output_batch, params):
    eval_results = {}

    # Get the predicted class
    _, predicted_class = torch.max(output_batch, 1)
    # print("Predicted class:", predicted_class.item())
    
    eval_results["pred_class"] = predicted_class
    return eval_results
