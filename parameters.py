import os
import sys
import ipdb
from easydict import EasyDict
from yacs.config import CfgNode as CN


def train_config(cfg):
    cfg.exp_id = 3
    cfg.exp_description = f' exp_{cfg.exp_id}: '
    cfg.exp_description += ' mobilenet_v2 + resize_512_and_random_crop_448 '
    # cfg.net_type = 'convnext'
    cfg.net_type = 'mobilenet_v2'
    cfg.forward_mode = 'train'
    cfg.gpu_used = '3'
    cfg.learning_rate = 1e-3
    cfg.gamma = 0.8
    cfg.num_epochs = 40
    cfg.train_data_ratio = 1.0
    cfg.train_batch_size = 8
    cfg.train_num_workers = 8
    cfg.eval_freq = 10
    cfg.eval_batch_size = 1
    cfg.save_iteration = 1
    cfg.eval_num_workers = 0
    cfg = continue_train(cfg)
    # cfg.gpu_used = '0_1_2_3_4_5_6_7' # use 8 GPUs
    return cfg


def test_config(cfg, args=None):

    cfg.exp_id = 3
    cfg.gpu_used = '3'
    cfg.save_iteration = 1
    cfg.eval_batch_size = 1
    # cfg.eval_batch_size = 16
    cfg.eval_num_workers = 0
    # cfg.is_vis_and_exit = True
    # cfg.is_debug_dataloader = True

    cfg.dataset_type = "test"
    cfg.restore_file = "model_latest.pth"

    if 'exp_id' in vars(args):
        cfg.exp_id = args.exp_id

    return cfg


def continue_train(cfg):
    if 'is_continue_train' in vars(cfg) and cfg.is_continue_train:
        # cfg.restore_file = 'test_model_best.pth'
        cfg.restore_file = "model_latest.pth"
        cfg.only_weights = True
        # cfg.only_weights = False
    return cfg


def common_config(cfg):
    if "linux" in sys.platform:
        cfg.data_dir = "/home/data/lwb/data/rain"
    else:  # windows
        cfg.data_dir = ""
    if not os.path.exists(cfg.data_dir):
        raise ValueError
    cfg.exp_root_dir = '/home/data/lwb/experiments'
    cfg.exp_current_dir = 'experiments'
    cfg.exp_name = 'rain'
    cfg.extra_config_json_dir = os.path.join(cfg.exp_current_dir, 'config')
    exp_dir = os.path.join(cfg.exp_root_dir, cfg.exp_name)
    cfg.model_dir = os.path.join(exp_dir, f"exp_{cfg.exp_id}")
    cfg.tb_path = os.path.join(exp_dir, 'tf_log', f'exp_{cfg.exp_id}')
    if 'restore_file' in cfg and cfg.restore_file is not None:
        cfg.restore_file = os.path.join(cfg.model_dir, cfg.restore_file)
    if (
        'is_exp_rm_protect' in vars(cfg)
        and cfg.is_exp_rm_protect
        and os.path.exists(cfg.model_dir)
        and not cfg.is_continue_train
    ):
        print("Existing experiment, exit.")
        sys.exit()
    if 'is_debug_dataloader' in dictToObj(cfg) and cfg.is_debug_dataloader:
        cfg.train_num_workers = 0
        cfg.eval_num_workers = 0

    return cfg


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dictToObj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dictToObj(v)
    return d


def get_config(args=None, mode='train'):
    """Get a yacs CfgNode object with debug params values."""
    cfg = CN()

    assert mode in ['train', 'test', 'val', 'evaluate']

    if mode == 'train':
        cfg = train_config(cfg)
    else:
        cfg = test_config(cfg, args)

    cfg = common_config(cfg)

    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = cfg.clone()
    config.freeze()

    return config


'''
cfg = get_config(None, 'train'))
dic = json.loads(json.dumps(cfg))
'''
