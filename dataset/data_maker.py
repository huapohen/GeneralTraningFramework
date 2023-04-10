import os
import sys
import json
import glob
import random
from tqdm import tqdm
from ipdb import set_trace as ip


classes = [
    'no_rain',
    'light_rain',
    'moderate_rain',
    'heavy_rain',
    'unkown',
]

classes_ind = {
            'no': 0,
            'light': 1,
            'light_moderate': 2,
            'moderate': 3,
            'moderate_heavy': 4,
            'heavy': 5,
        }

set_version = 'v1'
bp = f'/home/data/lwb/data/rain/{set_version}/'

lab_info_list = []
for file_name in os.listdir(bp + 'label'):
    with open(os.path.join(bp + 'label', file_name)) as f:
        data = json.load(f)
    num = len(data)
    for i in tqdm(range(num)):
        img_name = data[i]['image']
        prefix = '_'.join(img_name.split('_')[:2])
        img_name = (os.sep).join([prefix, img_name])
        label = data[i]['annotations']
        if len(label) == 1:
            lab = label[0].split('_')[0]
            if lab == 'unkown':
                continue
            ind = classes_ind[lab]
        else:
            label = [k.split('_') for k in label]
            if 'light' in label:
                lab = 'light_moderate'
                ind = 2
            else:
                lab = 'moderate_heavy'
                ind = 4
        line = ' '.join([img_name, lab, str(ind)])
        lab_info_list.append(line)
        
random.seed(925)
random.shuffle(lab_info_list)

for mode in ['train', 'test']:
    sv_path = f'{bp}/{mode}_file_names.txt'
    if os.path.exists(sv_path):
        os.remove(sv_path)
    ratio = 0.9 if mode == 'train' else 0.1
    num = int(len(lab_info_list) * ratio)
    line_list = lab_info_list[:num]
    if mode == 'test':
        line_list = sorted(line_list)
    with open(sv_path, 'a+') as f:
        for line in line_list:
            f.write(line + '\n')
    
        