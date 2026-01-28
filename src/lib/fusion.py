import numpy as np
import torch as th
import torch.nn as nn


class _Fusion(nn.Module):
    def __init__(self, cfg, input_sizes):
        """
        融合模块基类

        :param cfg: 融合配置。参见 config.defaults.Fusion
        :param input_sizes: 输入形状
        """
        super().__init__()
        self.cfg = cfg
        self.input_sizes = input_sizes
        self.output_size = None

    def forward(self, inputs):
        """
        前向传播
        
        :param inputs: 多视图输入
        :raise NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError()

    @classmethod
    def get_weighted_sum_output_size(cls, input_sizes):
        """
        获取加权和的输出大小
        
        :param input_sizes: 输入大小列表
        :return: 输出大小
        """
        flat_sizes = [np.prod(s) for s in input_sizes]
        assert all(s == flat_sizes[0] for s in flat_sizes), f"融合方法 {cls.__name__} 要求所有骨干网络的展平输出" \
                                                            f"形状相同。" \
                                                            f"得到的大小: {input_sizes} -> {flat_sizes}."
        return [flat_sizes[0]]

    def get_weights(self, softmax=True):
        """
        获取融合权重
        
        :param softmax: 是否使用softmax归一化
        :return: 权重
        """
        out = []
        if hasattr(self, "weights"):
            out = self.weights
            if softmax:
                out = nn.functional.softmax(self.weights, dim=-1)
        return out

    def update_weights(self, inputs, a):
        """
        更新权重
        
        :param inputs: 输入
        :param a: 更新参数
        """
        pass


class Mean(_Fusion):
    def __init__(self, cfg, input_sizes):
        """
        均值融合

        :param cfg: 融合配置。参见 config.defaults.Fusion
        :param input_sizes: 输入形状
        """
        super().__init__(cfg, input_sizes)
        self.output_size = self.get_weighted_sum_output_size(input_sizes)

    def forward(self, inputs):
        """
        前向传播
        
        :param inputs: 多视图输入
        :return: 融合结果
        """
        return th.mean(th.stack(inputs, -1), dim=-1)


class WeightedMean(_Fusion):
    """
    加权均值融合

    :param cfg: 融合配置。参见 config.defaults.Fusion
    :param input_sizes: 输入形状
    """
    def __init__(self, cfg, input_sizes):
        super().__init__(cfg, input_sizes)
        self.weights = nn.Parameter(th.full((self.cfg.n_views,), 1 / self.cfg.n_views), requires_grad=True)
        self.output_size = self.get_weighted_sum_output_size(input_sizes)

    def forward(self, inputs):
        """
        前向传播
        
        :param inputs: 多视图输入
        :return: 融合结果
        """
        return _weighted_sum(inputs, self.weights, normalize_weights=True)


def _weighted_sum(tensors, weights, normalize_weights=True):
    """
    计算加权和
    
    :param tensors: 张量列表
    :param weights: 权重
    :param normalize_weights: 是否归一化权重
    :return: 加权和结果
    """
    if normalize_weights:
        weights = nn.functional.softmax(weights, dim=0)
    out = th.sum(weights[None, None, :] * th.stack(tensors, dim=-1), dim=-1)
    return out


MODULES = {
    "mean": Mean,
    "weighted_mean": WeightedMean,
}


def get_fusion_module(cfg, input_sizes):
    """
    获取融合模块
    
    :param cfg: 融合配置
    :param input_sizes: 输入大小列表
    :return: 融合模块实例
    """
    return MODULES[cfg.method](cfg, input_sizes)
