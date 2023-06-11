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
    "--params_path",
    default=None,
    help="Directory containing params.json",
)
parser.add_argument(
    "--model_dir",
    default=None,
    help="Directory containing params.json",
)
parser.add_argument(
    "--restore_file",
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
    params = manager.params
    model.eval()
    params.forward_mode = "eval"

    eval_info_list = []
    loss_total_list = []
    total_is_right = 0
    samples_count = 0

    classes_ind = {
        0: "无  ",
        1: "微  ",
        2: "小  ",
        3: "小中",
        4: "中  ",
        5: "中大",
        6: "大  ",
    }

    with torch.no_grad():
        iter_max = len(manager.dataloaders[params.eval_type])
        with tqdm(total=iter_max) as t:
            for k, data_batch in enumerate(manager.dataloaders[params.eval_type]):
                labs = data_batch["label"]
                paths = data_batch["path"]
                data_batch = utils.tensor_gpu(data_batch)
                output_batch = model(data_batch["image"])
                losses = compute_losses(
                    output_batch, data_batch, params, k, reduction="none"
                )
                eval_results = compute_eval_results(data_batch, output_batch, params)
                #
                avg_loss = losses["total"].mean().item()
                loss_total_list.append(avg_loss)
                lab_pd = eval_results["pred_class"].detach().cpu()
                lab_gt = labs.detach().cpu()
                right_count = (lab_pd == lab_gt).sum()
                total_is_right += right_count
                bs = len(data_batch["image"])
                samples_count += bs
                if k % params.save_iteration == 0:
                    for j in range(bs):
                        ind = f"{k*bs+j+1}_b{k}_{j}"
                        pred = eval_results["pred_class"][j]
                        lab = labs[j]
                        is_right = 1 if pred == lab else 0
                        loss = losses["total"][j].item()
                        prt_str = (
                            f"index:{ind:<12} is_right: {is_right}  "
                            + f"gt:{lab}, pred:{pred}, lab:{classes_ind[lab.item()]:<18} "
                            + f"loss:{loss:.4f}, "
                            + f"path:{paths[j].split(params.data_dir)[1]}"
                        )
                        # total_is_right += is_right
                        eval_info_list.append(prt_str)
                # t.set_description(prt_str)
                cur_str = (
                    f"loss:{avg_loss:.4f}"
                    + f"({np.mean(np.array(loss_total_list)):.4f}), "
                    + f"acc:{right_count / bs * 100:.2f}%"
                    + f"({total_is_right / samples_count * 100:.2f}%)"
                )
                t.set_description(cur_str)
                t.update()
                # if k > 6:
                # break

        loss_total_avg = round(np.mean(np.array(loss_total_list)), 4)
        # total_samples = manager.dataloaders[params.eval_type].sample_number['total_samples']
        # total_samples = iter_max * params.eval_batch_size
        total_samples = max(samples_count, 1)
        accuracy = round(np.float64(total_is_right / total_samples), 4)
        prt_loss = f"total_loss: {loss_total_avg}, "
        prt_loss += f"samples: {total_samples}, "
        prt_loss += f"is_right: {total_is_right}, "
        prt_loss += f"Accuracy: {accuracy * 100:.2f}% "
        # print(prt_loss)
        eval_info_list = [prt_loss + "\n"] + eval_info_list
        eval_info_dir = os.path.join(params.model_dir, "eval_info")
        os.makedirs(eval_info_dir, exist_ok=True)
        if "current_epoch" not in vars(params):
            params.current_epoch = -1
        res_name = f"epoch={params.current_epoch:02d}.txt"
        res_path = os.path.join(eval_info_dir, res_name)
        if os.path.exists(res_path):
            os.remove(res_path)
        res = open(res_path, "a+")
        res.write(("\n").join(eval_info_list))
        res.close()

        Metric = {
            "total_loss": loss_total_avg,
            "Accuracy": accuracy,
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
    type_name = "gif" if type(save_file) == list else "jpg"
    save_dir_gif = os.path.join(manager.params.model_dir, type_name)
    if not os.path.exists(save_dir_gif):
        os.makedirs(save_dir_gif)

    # save_dir_gif_epoch = os.path.join(save_dir_gif, str(manager.epoch_val))
    save_dir_gif_epoch = os.path.join(save_dir_gif)
    if k == 0 and j == 0 and i == 0 and m == 0:
        if os.path.exists(save_dir_gif_epoch):
            shutil.rmtree(save_dir_gif_epoch)
        os.makedirs(save_dir_gif_epoch, exist_ok=True)

    save_path = os.path.join(save_dir_gif_epoch, save_name + "." + type_name)
    if type(save_file) == list:  # save gif
        utils.create_gif(save_file, save_path)
    elif type(save_file) == str:  # save string information
        f = open(save_path, "w")
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
        """python evaluate.py --params_path diy_param_json_path"""
        params = utils.Params(args.params_path)
    else:
        # run by python evaluate.py
        if exp_id is not None:
            args.exp_id = exp_id
        cfg = get_config(args, mode="test")
        dic_params = json.loads(json.dumps(cfg))
        obj_params = dictToObj(dic_params)
        params_default_path = os.path.join(obj_params.exp_current_dir, "params.json")
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
    logger = utils.set_logger(os.path.join(params.model_dir, "evaluate.log"))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # Fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

    # model
    params.forward_mode = "eval"
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
