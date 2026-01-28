import math
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

import config


def npy(t, to_cpu=True):
    """
    将张量转换为numpy数组。

    :param t: 输入张量
    :type t: th.Tensor
    :param to_cpu: 是否调用`t`的.cpu()方法？
    :type to_cpu: bool
    :return: numpy数组
    :rtype: np.ndarray
    """
    if isinstance(t, (list, tuple)):
        # 输入是列表。将每个元素转换为numpy
        return [npy(ti, to_cpu) for ti in t]
    elif isinstance(t, dict):
        # 输入是字典。将每个值转换为numpy
        return {k: npy(v, to_cpu) for k, v in t.items()}
    elif isinstance(t, np.ndarray):
        # 如果是NumPy数组，直接返回
        return t
    else:
        try:
            # 尝试转换为NumPy
            if to_cpu and hasattr(t, 'cpu'):
                t = t.cpu()
            if hasattr(t, 'detach'):
                t = t.detach()
            return t.numpy()
        except (AttributeError, RuntimeError):
            # 如果转换失败，直接返回原对象
            return t


def ensure_iterable(elem, expected_length=1):
    """
    确保元素是可迭代的
    
    :param elem: 元素
    :param expected_length: 期望长度
    :return: 可迭代对象
    """
    if isinstance(elem, (list, tuple)):
        assert len(elem) == expected_length, f"Expected iterable {elem} with length {len(elem)} does not have " \
                                             f"expected length {expected_length}"
    else:
        elem = expected_length * [elem]
    return elem


def dict_means(dicts):
    """
    计算字典列表中键的平均值

    :param dicts: 输入字典列表
    :type dicts: List[dict]
    :return: 平均值
    :rtype: dict
    """
    return pd.DataFrame(dicts).mean(axis=0).to_dict()


def add_prefix(dct, prefix, sep="/"):
    """
    为`dct`中的所有键添加前缀

    :param dct: 输入字典
    :type dct: dict
    :param prefix: 前缀
    :type prefix: str
    :param sep: 前缀和键之间的分隔符
    :type sep: str
    :return: 所有键都添加了前缀的字典
    :rtype: dict
    """
    return {prefix + sep + key: value for key, value in dct.items()}


def ordered_cmat(labels, pred):
    """
    计算对应最佳聚类到类分配的混淆矩阵和准确率

    :param labels: 标签数组
    :type labels: np.array
    :param pred: 预测数组
    :type pred: np.array
    :return: 准确率和混淆矩阵
    :rtype: Tuple[float, np.array]
    """
    cmat = confusion_matrix(labels, pred)  # 计算混淆矩阵
    ri, ci = linear_sum_assignment(-cmat)  # 找到最佳分配
    ordered = cmat[np.ix_(ri, ci)]  # 重新排序混淆矩阵
    acc = np.sum(np.diag(ordered))/np.sum(ordered)  # 计算准确率
    return acc, ordered


def get_save_dir(experiment_name, identifier, run):
    """
    获取实验的保存目录

    :param experiment_name: 配置名称
    :type experiment_name: str
    :param identifier: 当前实验的8字符唯一标识符
    :type identifier: str
    :param run: 当前训练运行
    :type run: int
    :return: 保存目录路径
    :rtype: pathlib.Path
    """
    if not str(run).startswith("run-"):
        run = f"run-{run}"
    return config.MODELS_DIR / f"{experiment_name}-{identifier}" / run


def he_init_weights(module):
    """
    使用He（Kaiming）初始化策略初始化网络权重

    :param module: 网络模块
    :type module: nn.Module
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)


def num2tuple(num):
    """
    将数字转换为元组
    
    :param num: 数字或可迭代对象
    :return: 元组
    """
    return num if isinstance(num, (tuple, list)) else (num, num)


def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    计算卷积操作的输出形状

    :param h_w: 输入的高度和宽度
    :type h_w: Tuple[int, int]
    :param kernel_size: 核大小
    :type kernel_size: Union[int, Tuple[int, int]]
    :param stride: 卷积步长
    :type stride: Union[int, Tuple[int, int]]
    :param pad: 填充（像素）
    :type pad: Union[int, Tuple[int, int]]
    :param dilation: 膨胀率
    :type dilation: Union[int, Tuple[int, int]]
    :return: 输出的高度和宽度
    :rtype: Tuple[int, int]
    """
    h_w, kernel_size, stride, = num2tuple(h_w), num2tuple(kernel_size), num2tuple(stride)
    pad, dilation = num2tuple(pad), num2tuple(dilation)

    h = math.floor((h_w[0] + 2 * pad[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w = math.floor((h_w[1] + 2 * pad[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return h, w
