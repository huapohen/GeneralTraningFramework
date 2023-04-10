"""Evaluates the model"""

import os
import sys
import cv2
import json
import ipdb
import torch
import shutil
import imageio
import logging
import argparse
import numpy as np
import model.net as net
import dataset.data_loader as data_loader
from tqdm import tqdm
from common import utils
from easydict import EasyDict
from ipdb import set_trace as ip
from common.manager import Manager
from parameters import get_config, dictToObj
from loss.losses import compute_losses, compute_eval_results


parser = argparse.ArgumentParser()
parser.add_argument(
    '--params_path',
    default=None,
    help="Directory containing params.json",
)
parser.add_argument(
    '--model_dir',
    default=None,
    help="Directory containing params.json",
)
parser.add_argument(
    '--restore_file',
    default=None,
    help="name of the file in --model_dir containing weights to load",
)


def evaluate(model, manager):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        manager: a class instance that contains objects related to train and evaluate.
    """
    # print("eval begin!")

    # loss status and eval status initial
    manager.reset_loss_status()
    manager.reset_metric_status(manager.params.eval_type)
    model.eval()
    params = manager.params
    params.forward_mode = 'eval'

    kpr_list = []
    loss_total_list = []

    with torch.no_grad():
        # compute metrics over the dataset
        iter_max = len(manager.dataloaders[params.eval_type])
        with tqdm(total=iter_max) as t:
            for k, data_batch in enumerate(manager.dataloaders[params.eval_type]):
                labs = data_batch["label"]
                paths = data_batch["path"]
                data_batch = utils.tensor_gpu(data_batch)
                output_batch = model(data_batch['image'])
                losses = compute_losses(output_batch, data_batch, manager.params, k)
                loss_total_list.append(losses['total'].item())
                eval_results = compute_eval_results(data_batch, output_batch, params)
                bs = len(paths)
                if k % params.save_iteration == 0:
                    for j in range(bs):
                        ind = f'{k*bs+j+1}_b{k}_{j}'
                        pred = eval_results['pred_class'][j]
                        lab = labs[j]
                        is_right = 1 if pred == lab else 0
                        loss = losses['total'].item()
                        prt_str = f"index:{ind}, is_right:{is_right}, " + \
                            f"lab:{lab}, pred:{pred}, path:{paths[j]}, loss:{loss:.4f}"

                kpr_list.append(prt_str)
                # t.set_description(prt_str)
                t.update()
                # if k > 10:
                # break
                
        loss_total_avg = round(np.mean(np.array(loss_total_list)), 4)
        prt_loss = f'total_loss: {loss_total_avg}'
        print(prt_loss)
        kpr_list = [prt_loss + '\n'] + kpr_list
        kpr_dir = os.path.join(params.model_dir, 'loss')
        os.makedirs(kpr_dir, exist_ok=True)
        if 'current_epoch' not in vars(params):
            params.current_epoch = -1
        kpr_name = f'epoch={params.current_epoch:02d}.txt'
        kpr_path = os.path.join(kpr_dir, kpr_name)
        if os.path.exists(kpr_path):
            os.remove(kpr_path)
        kpr = open(kpr_path, 'a+')
        kpr.write(('\n').join(kpr_list))
        kpr.close()

        Metric = {
            "total_loss": loss_total_avg,
        }

        manager.update_metric_status(
            metrics=Metric, split=manager.params.eval_type, batch_size=1
        )

        # update data to logger
        manager.logger.info(
            "Loss/Test: epoch_{}, {} ".format(
                manager.epoch_val,
                prt_loss,
            )
        )

        # For each epoch, print the metric
        manager.print_metrics(
            manager.params.eval_type, title=manager.params.eval_type, color="green"
        )

        # manager.epoch_val += 1

        model.train()


def eval_save_result(save_file, save_name, manager, k, j, i, m):

    type_name = 'gif' if type(save_file) == list else 'jpg'
    save_dir_gif = os.path.join(manager.params.model_dir, type_name)
    if not os.path.exists(save_dir_gif):
        os.makedirs(save_dir_gif)

    # save_dir_gif_epoch = os.path.join(save_dir_gif, str(manager.epoch_val))
    save_dir_gif_epoch = os.path.join(save_dir_gif)
    if k == 0 and j == 0 and i == 0 and m == 0:
        if os.path.exists(save_dir_gif_epoch):
            shutil.rmtree(save_dir_gif_epoch)
        os.makedirs(save_dir_gif_epoch, exist_ok=True)

    save_path = os.path.join(save_dir_gif_epoch, save_name + '.' + type_name)
    if type(save_file) == list:  # save gif
        utils.create_gif(save_file, save_path)
    elif type(save_file) == str:  # save string information
        f = open(save_path, 'w')
        f.write(save_file)
        f.close()
    else:  # save single image
        cv2.imwrite(save_path, save_file)
    if manager.params.is_vis_and_exit:
        sys.exit()


def run_all_exps(exp_id):
    """
    Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()

    if args.params_path is not None and args.restore_file is None:
        # run test by DIY in designated diy_params.json
        '''python evaluate.py --params_path diy_param_json_path'''
        params = utils.Params(args.params_path)
    else:
        # run by python evaluate.py
        if exp_id is not None:
            args.exp_id = exp_id
        cfg = get_config(args, mode='test')
        dic_params = json.loads(json.dumps(cfg))
        obj_params = dictToObj(dic_params)
        params_default_path = os.path.join(obj_params.exp_current_dir, 'params.json')
        model_json_path = os.path.join(obj_params.model_dir, "params.json")
        assert os.path.isfile(
            model_json_path
        ), "No json configuration file found at {}".format(model_json_path)
        params = utils.Params(params_default_path)
        params_model = utils.Params(model_json_path)
        params.update(params_model.dict)
        params.update(obj_params)
        # ipdb.set_trace()

    # Only load model weights
    params.only_weights = True

    # use GPU if available
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if "_" in params.gpu_used:
        params.gpu_used = ",".join(params.gpu_used.split("_"))
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_used
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(params.seed)
    if params.cuda:
        torch.cuda.manual_seed(params.seed)

    # Get the logger
    logger = utils.set_logger(os.path.join(params.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # Fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

    # model
    params.forward_mode = 'test'
    model = net.fetch_net(params)
    if params.cuda:
        model = model.cuda()
        gpu_num = len(params.gpu_used.split(","))
        device_ids = range(gpu_num)
        # device_ids = range(torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        

    # Initial status for checkpoint manager
    manager = Manager(
        model=model,
        optimizer=None,
        scheduler=None,
        params=params,
        dataloaders=dataloaders,
        writer=None,
        logger=logger,
    )

    # Reload weights from the saved file
    manager.load_checkpoints()

    # Test the model
    logger.info("Starting test")

    # Evaluate
    evaluate(model, manager)


if __name__ == "__main__":

    # for i in range(5, 10):
    #     run_all_exps(i)
    run_all_exps(exp_id=None)
    # run_all_exps(exp_id=1)
