import os
import cv2
import ipdb
import json
import torch
import random
import numpy as np
from PIL import Image
from easydict import EasyDict
from ipdb import set_trace as ip
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop, CenterCrop
from torchvision.transforms import InterpolationMode


class RainData(Dataset):
    def __init__(self, params, mode='train'):
        set_data = params.train_data if mode == 'train' else params.test_data
        self.mode = mode
        self.data_all = []
        self.sample_number = {}
        self.params = params
        
        for i in range(len(set_data)):
            set_name = set_data[i][0]
            set_ratio = set_data[i][1]
            mode_dir = os.path.join(params.data_dir, set_name)
            txt_path = os.path.join(mode_dir, f'{mode}_file_names.txt')
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            file_names = [[mode_dir, name_label.rsplit('\n')[0]] for name_label in lines]
            if mode == 'train':
                random.shuffle(file_names)
            num = int(len(file_names) * set_ratio)
            self.data_all += file_names[:num]
            self.sample_number[set_name] = {
                'ratio': set_ratio,
                "samples": num,
            }
            
        self.sample_number["total_samples"] = len(self.data_all)
        if mode == 'train':
            random.shuffle(self.data_all)
        # ipdb.set_trace()
        self.classes = {
            'no': 0,
            'light': 1,
            'light_moderate': 2,
            'moderate': 3,
            'moderate_heavy': 4,
            'heavy': 5,
        }
        
    def __len__(self):
        return len(self.data_all)
    
    def __getitem__(self, idx):
        '''
        debug dataloader: set num_workers=0
        '''
        data_info = self.data_all[idx]
        img_dir = data_info[0]
        img_name, cls_name, cls_ind = data_info[1].split(' ')

        img_path = os.path.join(img_dir, 'image', img_name)
        img = Image.open(img_path)
        img = self.data_aug(img)

        data_dict = {}
        data_dict['image'] = img
        data_dict['label'] = torch.tensor(int(cls_ind))
        data_dict['class_name'] = cls_name
        data_dict['path'] = img_path
        
        # ipdb.set_trace()
        return data_dict
    
    def data_aug(self, img):
        params = self.params
        def transform_func():
            # w, h = 1280, 720
            h, w = 720, 1280
            # Resize(size=(h, w))
            # Crop(size=(h, w))
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            ch, cw = 576, 1088
            rh, rw = params.resize_size
            ch, cw = params.crop_size
            crop = RandomCrop if self.mode == 'train' else CenterCrop
            transform = Compose([
                Resize((rh, rw), interpolation=InterpolationMode.BICUBIC),
                # Resize(512, interpolation=InterpolationMode.BILINEAR),
                # Resize((512, 512), interpolation=InterpolationMode.BILINEAR),
                # crop(448),
                crop((ch, cw)),
                ToTensor(),
                Normalize(mean, std),
            ])
            return transform
            
        img = transform_func()(img)
        return img


def fetch_dataloader(params):

    dataloaders = {}
    
    if params.dataset_type in ['basic', 'train']:
        train_ds = RainData(params, 'train')
        
        dl = DataLoader(
            train_ds,
            batch_size=params.train_batch_size,
            shuffle=True,
            num_workers=params.train_num_workers,
            pin_memory=params.cuda,
            drop_last=True,
            # prefetch_factor=3,
        )
        dl.sample_number = train_ds.sample_number
        dataloaders["train"] = dl

    if params.eval_type in ['val', 'test']:
        test_ds = RainData(params, 'test')
        dl = DataLoader(
            test_ds,
            batch_size=params.eval_batch_size,
            shuffle=False,
            num_workers=params.eval_num_workers,
            pin_memory=params.cuda,
            # prefetch_factor=3,
        )
        dl.sample_number = test_ds.sample_number
        dataloaders[params.eval_type] = dl

    return dataloaders


if __name__ == '__main__':
    
    param_path = '/home/data/lwb/code/rain/experiments/params.json'
    with open(param_path) as f:
        params = json.load(f)
    params = EasyDict(params)
    # params.cuda = True
    params.cuda = False
    params.train_batch_size = 2
    params.eval_batch_size = 2
    p = print
    
    for mode in ['train', 'test']:
        p(mode)
        params.mode = mode
        rain_data = RainData(params, mode)
        rain_dataloader = fetch_dataloader(params)[mode]
        
        for loader in [rain_data, rain_dataloader]:
            p(f"total_samples: {len(loader)}")
            for i, data_dict in enumerate(loader):
                p(data_dict['image'].shape)
                p(data_dict['label'])
                p(data_dict['class_name'])
                p(data_dict['path'])
                break
            p()
    