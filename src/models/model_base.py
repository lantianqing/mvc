import torch.nn as nn


class ModelBase(nn.Module):
    def __init__(self):
        """
        模型基类
        """
        super().__init__()

        self.fusion = None  # 融合模块
        self.optimizer = None  # 优化器
        self.loss = None  # 损失函数

    def calc_losses(self, ignore_in_total=tuple()):
        """
        计算损失
        
        :param ignore_in_total: 在总损失中忽略的项
        :return: 损失值字典
        """
        return self.loss(self, ignore_in_total=ignore_in_total)

    def train_step(self, batch, epoch, it, n_batches):
        """
        训练步骤
        
        :param batch: 当前批次
        :param epoch: 当前轮数
        :param it: 当前迭代次数
        :param n_batches: 每轮的批次数
        :return: 损失值字典
        """
        self.optimizer.zero_grad()  # 清零梯度
        _ = self(batch)  # 前向传播
        losses = self.calc_losses()  # 计算损失
        losses["tot"].backward()  # 反向传播
        self.optimizer.step(epoch + it / n_batches)  # 更新参数
        return losses
