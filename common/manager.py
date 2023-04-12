import json
import logging
import os
import pickle
import shutil
import time
from collections import defaultdict
import ipdb
import numpy as np
import torch
from termcolor import colored
from tensorboardX import SummaryWriter

from common import utils


class Manager:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        params,
        dataloaders,
        writer,
        logger,
        tb_path=None,
        exp_name=None,
    ):
        # params status
        self.params = params

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders = dataloaders
        self.writer = writer
        self.logger = logger

        self.epoch = 1
        self.epoch_val = 0
        self.step = 0
        if self.params.metric_mode == 'ascend':
            self.best_val_score = 0
            self.best_test_score = 0
            self.cur_val_score = -9999
            self.cur_test_score = -9999
        else: # descend
            self.best_val_score = 9999
            self.best_test_score = 9999
            self.cur_val_score = 19999
            self.cur_test_score = 19999

        # tf_board handlle
        self.writer = SummaryWriter(tb_path) if exp_name is not None else None

        # train status
        self.train_status = defaultdict(utils.AverageMeter)

        # val status
        self.val_status = defaultdict(utils.AverageMeter)

        # test status
        self.test_status = defaultdict(utils.AverageMeter)

        # model status
        self.loss_status = defaultdict(utils.AverageMeter)

    def update_step(self):
        self.step += 1

    def update_epoch(self):
        self.epoch += 1
        self.epoch_val = 0

    def update_epoch_val(self):
        self.epoch_val += 1

    def update_loss_status(self, loss, split, batch_size=None):
        # loss: dict: ["loss_name": value, ...]

        if split == "train":
            for k, v in loss.items():
                self.loss_status[k].update(
                    val=v.item(),
                    num=batch_size
                    if batch_size is not None
                    else self.params.train_batch_size,
                )
        elif split == "val":
            for k, v in loss.items():
                self.loss_status[k].update(
                    val=v.item(),
                    num=batch_size
                    if batch_size is not None
                    else self.params.eval_batch_size,
                )
        elif split == "test":
            for k, v in loss.items():
                self.loss_status[k].update(
                    val=v.item(),
                    num=batch_size
                    if batch_size is not None
                    else self.params.eval_batch_size,
                )
        else:
            raise ValueError("Wrong eval type: {}".format(split))

    def update_metric_status(self, metrics, split, batch_size=None):
        # metric: dict: ["metric_name": value, ...]

        if split == "val":
            for k, v in metrics.items():
                self.val_status[k].update(
                    val=v.item(),
                    num=batch_size
                    if batch_size is not None
                    else self.params.eval_batch_size,
                )
                self.cur_val_score = self.val_status[self.params.major_metric].avg
        elif split == "test":
            for k, v in metrics.items():
                self.test_status[k].update(
                    val=v.item(),
                    num=batch_size
                    if batch_size is not None
                    else self.params.eval_batch_size,
                )
                self.cur_test_score = self.test_status[self.params.major_metric].avg
        else:
            raise ValueError("Wrong eval type: {}".format(split))

    def reset_loss_status(self):
        for k, v in self.loss_status.items():
            self.loss_status[k].reset()

    def reset_metric_status(self, split):
        if split == "val":
            for k, v in self.val_status.items():
                self.val_status[k].reset()
        elif split == "test":
            for k, v in self.test_status.items():
                self.test_status[k].reset()
        else:
            raise ValueError("Wrong eval type: {}".format(split))

    def print_train_info(self):
        exp_name = self.params.model_dir.split('/')[-1]
        exp_id = self.params.exp_id
        # get_last_lr() for pytorch version>=1.4.0, otherwise get_lr()
        print_str = "exp_{} Epoch: {:4d}, lr={:.7f} ".format(
            exp_id, self.epoch, self.scheduler.get_last_lr()[0]
        )
        for k, v in self.loss_status.items():
            print_str += "%s: %.4f(%.4f) " % (k, v.val, v.avg)

        return print_str

    def print_metrics(self, split, title="Eval", color="red"):
        if split == "val":
            metric_status = self.val_status
        elif split == "test":
            metric_status = self.test_status
        else:
            raise ValueError("Wrong eval type: {}".format(split))
        print_str = " | ".join(
            "{}: {:.4f}".format(k, v.avg) for k, v in metric_status.items()
        )
        exp_info = f'exp_{self.params.exp_id}, epoch={self.params.current_epoch}, '
        self.logger.info(
            colored("{} {} Results: {}".format(exp_info, title, print_str), color, attrs=["bold"])
        )

    def check_best_save_last_checkpoints(self, latest_freq_val=5, latest_freq=5):

        state = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
        }
        if "val" in self.dataloaders:
            state["best_val_score"] = self.best_val_score
        if "test" in self.dataloaders:
            state["best_test_score"] = self.best_test_score

        # save latest checkpoint
        if self.epoch % latest_freq == 0 or self.epoch % latest_freq_val == 0:
            latest_ckpt_name = os.path.join(self.params.model_dir, "model_latest.pth")
            if self.params.save_mode == "local":
                torch.save(state, latest_ckpt_name)
            else:
                raise NotImplementedError
            self.logger.info("Saved latest checkpoint to: {}".format(latest_ckpt_name))

        # check whether best model
        if self.params.current_epoch != self.params.eval_freq:
            return
        
        # save val latest metrics, and check if val is best checkpoints
        if "val" in self.dataloaders:
            val_latest_metrics_name = os.path.join(
                self.params.model_dir, "val_metrics_latest.json"
            )
            utils.save_dict_to_json(self.val_status, val_latest_metrics_name)
            if self.params.metric_mode == 'ascend':
                is_best = self.cur_val_score > self.best_val_score
            else:
                is_best = self.cur_val_score < self.best_val_score
            if is_best:
                # save metrics
                self.best_val_score = self.cur_val_score
                best_metrics_name = os.path.join(
                    self.params.model_dir, "val_metrics_best.json"
                )
                val_status_save = self.val_status.copy()
                val_status_save.update(
                    epoch=self.epoch, epoch_val=self.epoch_val, step=self.step
                )
                utils.save_dict_to_json(val_status_save, best_metrics_name)

                self.logger.info(
                    "Current is val best, score={:.4f}".format(self.best_val_score)
                )
                # save checkpoint
                best_ckpt_name = os.path.join(
                    self.params.model_dir, "val_model_best.pth"
                )
                if self.params.save_mode == "local":
                    torch.save(state, best_ckpt_name)

                self.logger.info(
                    "Saved val best checkpoint to: {}".format(best_ckpt_name)
                )

        # save test latest metrics, and check if test is best checkpoints
        # if self.dataloaders["test"] is not None:
        if "test" in self.dataloaders:
            test_latest_metrics_name = os.path.join(
                self.params.model_dir, "test_metrics_latest.json"
            )
            utils.save_dict_to_json(self.test_status, test_latest_metrics_name)
            if self.params.metric_mode == 'ascend':
                is_best = self.cur_test_score > self.best_val_score
            else:
                is_best = self.cur_test_score < self.best_val_score
            if is_best:
                # save metrics
                self.best_test_score = self.cur_test_score
                best_metrics_name = os.path.join(
                    self.params.model_dir, "test_metrics_best.json"
                )
                test_status_save = self.test_status.copy()
                test_status_save.update(
                    epoch=self.epoch, epoch_val=self.epoch_val, step=self.step
                )
                utils.save_dict_to_json(test_status_save, best_metrics_name)
                self.logger.info(
                    "Current is test best, score={:.4f}".format(self.best_test_score)
                )
                # save checkpoint
                best_ckpt_name = os.path.join(
                    self.params.model_dir, "test_model_best.pth"
                )
                if self.params.save_mode == "local":
                    torch.save(state, best_ckpt_name)

                self.logger.info(
                    "Saved test best checkpoint to: {}".format(best_ckpt_name)
                )

    def load_checkpoints(self):
        if self.params.save_mode == "local":
            if self.params.cuda:
                state = torch.load(self.params.restore_file)
            else:
                state = torch.load(
                    self.params.restore_file, map_location=torch.device('cpu')
                )

        ckpt_component = []
        if "state_dict" in state and self.model is not None:
            try:
                self.model.load_state_dict(state["state_dict"])

            except:
                print("Using custom loading net")
                net_dict = self.model.state_dict()

                if "module" not in list(state["state_dict"].keys())[0]:
                    state_dict = {
                        "module." + k: v
                        for k, v in state["state_dict"].items()
                        if "module." + k in net_dict.keys()
                    }
                else:
                    state_dict = {
                        k.replace("module.", ""): v
                        for k, v in state["state_dict"].items()
                        if k.replace("module.", "") in net_dict.keys()
                    }

                net_dict.update(state_dict)
                self.model.load_state_dict(net_dict, strict=False)

            ckpt_component.append("net")

        if not self.params.only_weights:

            if "optimizer" in state and self.optimizer is not None:
                try:
                    self.optimizer.load_state_dict(state["optimizer"])

                except:
                    print("Using custom loading optimizer")
                    optimizer_dict = self.optimizer.state_dict()
                    state_dict = {
                        k: v
                        for k, v in state["optimizer"].items()
                        if k in optimizer_dict.keys()
                    }
                    optimizer_dict.update(state_dict)
                    self.optimizer.load_state_dict(optimizer_dict)
                ckpt_component.append("opt")

            if "scheduler" in state and self.train_status["scheduler"] is not None:
                try:
                    self.scheduler.load_state_dict(state["scheduler"])

                except:
                    print("Using custom loading scheduler")
                    scheduler_dict = self.scheduler.state_dict()
                    state_dict = {
                        k: v
                        for k, v in state["scheduler"].items()
                        if k in scheduler_dict.keys()
                    }
                    scheduler_dict.update(state_dict)
                    self.scheduler.load_state_dict(scheduler_dict)
                ckpt_component.append("sch")

            if "step" in state:
                self.train_status["step"] = state["step"] + 1
                ckpt_component.append("step")

            if "epoch" in state:
                self.train_status["epoch"] = state["epoch"] + 1
                ckpt_component.append("epoch")

            if "best_val_score" in state:
                self.best_val_score = state["best_val_score"]
                ckpt_component.append(
                    "best val score: {:.3g}".format(self.best_val_score)
                )

            if "best_test_score" in state:
                self.best_test_score = state["best_test_score"]
                ckpt_component.append(
                    "best test score: {:.3g}".format(self.best_test_score)
                )

        ckpt_component = ", ".join(i for i in ckpt_component)
        self.logger.info("Loaded models from: {}".format(self.params.restore_file))
        self.logger.info("Ckpt load: {}".format(ckpt_component))
