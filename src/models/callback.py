import os
import yaml
import pickle
import numpy as np
import torch as th
from tabulate import tabulate

import helpers


class Callback:
    def __init__(self, epoch_interval=1, batch_interval=1):
        """
        训练回调的基类

        :param epoch_interval: 调用 `at_epoch_end` 之间的 epoch 数量。
        :type epoch_interval: int
        :param batch_interval: 调用 `at_batch_end` 之间的 batch 数量。
        :type batch_interval: int
        """
        self.epoch_interval = epoch_interval
        self.batch_interval = batch_interval

    def epoch_end(self, epoch, **kwargs):
        """
        epoch 结束时调用
        
        :param epoch: 当前 epoch
        :param kwargs: 额外参数
        :return: 回调结果
        """
        if not (epoch % self.epoch_interval):
            return self.at_epoch_end(epoch, **kwargs)

    def batch_end(self, epoch, batch, **kwargs):
        """
        batch 结束时调用
        
        :param epoch: 当前 epoch
        :param batch: 当前 batch
        :param kwargs: 额外参数
        :return: 回调结果
        """
        if (not (epoch % self.epoch_interval)) and (not (batch % self.batch_interval)):
            return self.at_batch_end(epoch, batch, **kwargs)

    def at_epoch_end(self, epoch, logs=None, net=None, **kwargs):
        """
        在 epoch 结束时执行的回调
        
        :param epoch: 当前 epoch
        :param logs: 日志
        :param net: 网络模型
        :param kwargs: 额外参数
        """
        pass

    def at_batch_end(self, epoch, batch, outputs=None, losses=None, net=None, **kwargs):
        """
        在 batch 结束时执行的回调
        
        :param epoch: 当前 epoch
        :param batch: 当前 batch
        :param outputs: 网络输出
        :param losses: 损失值
        :param net: 网络模型
        :param kwargs: 额外参数
        """
        pass

    def at_eval(self, net=None, logs=None, **kwargs):
        """
        在评估时执行的回调
        
        :param net: 网络模型
        :param logs: 日志
        :param kwargs: 额外参数
        """
        pass


class Printer(Callback):
    def __init__(self, print_confusion_matrix=True, **kwargs):
        """
        将日志打印到终端。

        :param print_confusion_matrix: 当混淆矩阵可用时打印它？
        :type print_confusion_matrix: bool
        :param kwargs:
        :type kwargs:
        """
        super().__init__(**kwargs)
        self.ignore_keys = ["iter_losses/"]
        if not print_confusion_matrix:
            self.ignore_keys.append("metrics/cmat")
            
        np.set_printoptions(edgeitems=20, linewidth=200)

    def at_epoch_end(self, epoch, logs=None, net=None, **kwargs):
        """
        在 epoch 结束时打印日志
        
        :param epoch: 当前 epoch
        :param logs: 日志
        :param net: 网络模型
        :param kwargs: 额外参数
        """
        print_logs = logs.copy()
        for key in logs.keys():
            if any([key.startswith(ik) for ik in self.ignore_keys]):
                del print_logs[key]

        headers = ["键", "值"]
        values = list(print_logs.items())
        print(tabulate(values, headers=headers), "\n")


class ModelSaver(Callback):
    def __init__(self, cfg, experiment_name, identifier, run, best_loss_term, checkpoint_interval=1, **kwargs):
        """
        模型保存回调。在指定的检查点保存模型，或当损失函数中的 `best_loss_term`
        达到观察到的最低值时保存模型。

        :param cfg: 实验配置
        :type cfg: config.defaults.Experiment
        :param experiment_name: 实验名称
        :type experiment_name: str
        :param identifier: 8字符的唯一实验标识符
        :type identifier: str
        :param run: 当前训练运行
        :type run: int
        :param best_loss_term: 要监控的损失函数项。
        :type best_loss_term: str
        :param checkpoint_interval: 保存模型检查点之间的 epoch 数量。
        :type checkpoint_interval: int
        :param kwargs:
        :type kwargs:
        """
        super().__init__(**kwargs)

        self.best_loss_term = f"eval_losses/{best_loss_term}"
        self.min_loss = np.inf
        self.checkpoint_interval = checkpoint_interval
        self.save_dir = helpers.get_save_dir(experiment_name, identifier, run)
        os.makedirs(self.save_dir, exist_ok=True)
        self._save_cfg(cfg)

    def _save_cfg(self, cfg):
        """
        保存配置
        
        :param cfg: 配置对象
        """
        with open(self.save_dir / "config.yml", "w") as f:
            yaml.dump(cfg.dict(), f)
        with open(self.save_dir / "config.pkl", "wb") as f:
            pickle.dump(cfg, f)

    def _save_model(self, file_name, net):
        """
        保存模型
        
        :param file_name: 文件名
        :param net: 网络模型
        """
        model_path = self.save_dir / file_name
        th.save(net.state_dict(), model_path)
        print(f"模型成功保存: {model_path}")

    def at_epoch_end(self, epoch, outputs=None, logs=None, net=None, **kwargs):
        """
        在 epoch 结束时保存模型
        
        :param epoch: 当前 epoch
        :param outputs: 网络输出
        :param logs: 日志
        :param net: 网络模型
        :param kwargs: 额外参数
        """
        if not (epoch % self.checkpoint_interval):
            # 保存模型检查点
            self._save_model(f"checkpoint_{str(epoch).zfill(4)}.pt", net)

        avg_loss = logs.get(self.best_loss_term, np.inf)
        # 如果当前损失是遇到的最低损失，则保存到 model_best
        if avg_loss < self.min_loss:
            self.min_loss = avg_loss
            self._save_model("best.pt", net)


class StopTraining(Exception):
    """
    停止训练异常
    """
    pass


class EarlyStopping(Callback):
    def __init__(self, patience, best_loss_term, **kwargs):
        """
        早停回调。当损失函数中的 `best_loss_term` 项在 `patience` 个 epoch 中没有减少时，
        抛出 `StopTraining` 异常。

        :param patience: 等待损失减少的 epoch 数量
        :type patience: int
        :param best_loss_term: 要监控的损失函数项。
        :type best_loss_term: str
        :param kwargs:
        :type kwargs:
        """
        super().__init__(**kwargs)
        self.best_loss_term = f"eval_losses/{best_loss_term}"
        self.patience = patience
        self.min_loss = np.inf
        self.best_epoch = 0

    def at_epoch_end(self, epoch, outputs=None, logs=None, net=None, **kwargs):
        """
        在 epoch 结束时检查是否需要早停
        
        :param epoch: 当前 epoch
        :param outputs: 网络输出
        :param logs: 日志
        :param net: 网络模型
        :param kwargs: 额外参数
        """
        avg_loss = logs.get(self.best_loss_term, np.inf)

        if np.isnan(avg_loss):
            raise StopTraining(f"获得损失 = NaN。训练停止。")

        if avg_loss < self.min_loss:
            self.min_loss = avg_loss
            self.best_epoch = epoch

        if (epoch - self.best_epoch) >= self.patience:
            raise StopTraining(f"损失在 {self.patience} 个 epoch 中没有减少。最小损失为 {self.min_loss}。 "
                               f"训练停止。")

