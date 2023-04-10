import os
import sys
import json
import ipdb
import torch
import shutil
import random
import datetime
import argparse
import numpy as np
import torch.optim as optim
from tqdm import tqdm

import model.net as net
import dataset.data_loader as data_loader
from common import utils
from common.manager import Manager
from evaluate import evaluate
from loss.losses import compute_losses
from parameters import get_config, dictToObj


parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, default=None, help="params file")
parser.add_argument(
    "--exp_current_dir", type=str, default=None, help="parent dir of experiments"
)
parser.add_argument(
    "--exp_root_dir", type=str, default=None, help="parent dir of experiments"
)
parser.add_argument(
    "--model_dir", type=str, default=None, help="Directory containing params.json"
)
parser.add_argument(
    "--restore_file",
    default=None,
    help="Optional, name of the file in --model_dir "
    + "containing weights to reload before training",
)
parser.add_argument(
    "-ow",
    "--only_weights",
    action="store_true",
    help="Only use weights to load or load all train status.",
)
parser.add_argument(
    "-gu",
    "--gpu_used",
    type=str,
    default=None,
    help="select the gpu for train or evaluation.",
)
parser.add_argument(
    "-exp", "--exp_name", type=str, default=None, help="experiment name."
)
parser.add_argument("-eid", "--exp_id", type=int, default=None, help="experiment id.")
parser.add_argument(
    "-tb",
    "--tb_path",
    type=str,
    default=None,
    help="the path to save the tensorboardx log.",
)


def train(model, manager):

    # loss status initial
    manager.reset_loss_status()

    # set model to training mode
    torch.cuda.empty_cache()
    model.train()
    params.forward_mode = 'train'

    # Use tqdm for progress bar
    with tqdm(total=len(manager.dataloaders['train'])) as t:
        for i, data_batch in enumerate(manager.dataloaders['train']):

            # move to GPU if available
            data_batch = utils.tensor_gpu(data_batch)

            # compute model output and loss
            output_batch = model(data_batch['image'])
            loss = compute_losses(output_batch, data_batch, manager.params)

            # update loss status and print current loss and average loss
            manager.update_loss_status(loss=loss, split="train")

            # clear previous gradients, compute gradients of all variables loss
            manager.optimizer.zero_grad()
            loss['total'].backward()

            # performs updates using calculated gradients
            manager.optimizer.step()

            # update step: step += 1
            manager.update_step()

            # infor print
            print_str = manager.print_train_info()

            t.set_description(desc=print_str)
            t.update()

    manager.logger.info(print_str)
    manager.scheduler.step()

    # update epoch: epoch += 1
    manager.update_epoch()


def train_and_evaluate(model, manager):

    for epoch in range(manager.params.num_epochs):
        manager.params.current_epoch = epoch + 1

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, manager)

        if (epoch + 1) % manager.params.eval_freq == 0:
            evaluate(model, manager)
            
        # Save latest model, or best model weights accroding to the params.major_metric
        manager.check_best_save_last_checkpoints(latest_freq_val=999, latest_freq=1)


def set_params_lr(model):
    param_names = ['classifier']
    
    def match_name_keywords(n, name_keywords):
        for b in name_keywords:
            if b in n:
                return True
        return False
    
    for n, p in model.named_parameters():
        if not match_name_keywords(n, param_names) and p.requires_grad:
            p.requires_grad = False
    
    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not match_name_keywords(n, param_names) and p.requires_grad
            ],
            "lr": params.lr_backbone,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if match_name_keywords(n, param_names) and p.requires_grad
            ],
            "lr": params.learning_rate,
        },
    ]
    
    return param_dicts
        
        

if __name__ == '__main__':

    args = parser.parse_args()

    if args.params_path is not None and args.restore_file is None:
        # run train by DIY in designated diy_params.json
        '''python train.py --params_path diy_param_json_path'''
        params = utils.Params(args.params_path)
    elif args.gpu_used is not None:
        # run train.py through search_hyperparams.py
        print(args.__dict__)
        # ipdb.set_trace()
        exp_json_path = os.path.join(args.model_dir, "params.json")
        params = utils.Params(exp_json_path)
        params.update(args.__dict__)
        file_name = f"{params.exp_name}_exp_{params.exp_id}.json"
        params.exp_json_path1 = exp_json_path
        params.exp_json_path2 = os.path.join(params.extra_config_json_dir, file_name)
    else:
        # run by python train.py
        cfg = get_config(args, mode='train')
        dic_params = json.loads(json.dumps(cfg))
        obj_params = dictToObj(dic_params)
        default_json_path = os.path.join(cfg.exp_current_dir, "params.json")
        params = utils.Params(default_json_path)
        params.update(obj_params)
        file_name = f"{params.exp_name}_exp_{params.exp_id}.json"
        params.extra_config_json_dir = os.path.join(cfg.exp_current_dir, 'config')
        params.exp_json_path1 = os.path.join(params.model_dir, "params.json")
        params.exp_json_path2 = os.path.join(params.extra_config_json_dir, file_name)
        # resume
        if 'restore_file' not in obj_params:
            try:
                shutil.rmtree(params.model_dir)
                shutil.rmtree(params.tb_path)
                os.remove(params.exp_json_path2)
            except:
                pass
        else:
            model_json_path = os.path.join(obj_params.model_dir, "params.json")
            params_model = utils.Params(model_json_path)
            params.update(params_model.dict)

    os.makedirs(params.model_dir, exist_ok=True)
    os.makedirs(params.tb_path, exist_ok=True)
    os.makedirs(params.extra_config_json_dir, exist_ok=True)

    # Assign dataset
    if params.eval_freq < params.num_epochs:
        params.dataset_type = 'basic'

    # Save params
    if 'restore_file' not in vars(params) or params.restore_file is None:
        params.save(params.exp_json_path1)
        params.save(params.exp_json_path2)

    # use GPU if available
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if "_" in params.gpu_used:
        params.gpu_used = ",".join(params.gpu_used.split("_"))
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_used
    # cuda
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)
    if params.cuda:
        torch.cuda.manual_seed(params.seed)

    # Set the logger
    logger = utils.set_logger(os.path.join(params.model_dir, 'train.log'))

    # Set the tensorboard writer
    log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create the input data pipeline
    logger.info("Creating the dataset...")
    
    # DataLoder
    dataloaders = data_loader.fetch_dataloader(params)
    
    # Dataset information
    for set_mode, set_dl in dataloaders.items():
        sample_info = set_dl.sample_number
        ds_stats = ""
        for k, v in sample_info.items():
            if k != 'total_samples':
                ds_stats += f" {k}  r={v['ratio']} n={v['samples']}\t"
        logger.info(f"{set_mode} dataset: {ds_stats}")
        logger.info(f"total samples: {sample_info['total_samples']}")

    # model
    params.forward_mode == 'train'
    model = net.fetch_net(params)

    # gpu
    if params.cuda:
        model = model.cuda()
        gpu_num = len(params.gpu_used.split(","))
        device_ids = range(gpu_num)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    # assign learning_rate
    param_dicts = set_params_lr(model)
    
    # optimizer
    optimizer = optim.Adam(param_dicts, lr=params.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.gamma)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)
    
    # initial status for checkpoint manager
    manager = Manager(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        params=params,
        dataloaders=dataloaders,
        writer=None,
        logger=logger,
        exp_name=params.exp_name,
        tb_path=params.tb_path,
    )

    # ipdb.set_trace()
    # Continue training
    if 'restore_file' in vars(params) and params.restore_file is not None:
        manager.load_checkpoints()

    # Train the model
    logger.info("Starting training for {} epoch(s)".format(params.num_epochs))

    train_and_evaluate(model, manager)
