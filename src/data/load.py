import numpy as np
import torch as th

import config


def _load_npz(name):
    """加载指定名称的npz文件
    
    Args:
        name: 数据集名称
    
    Returns:
        npz文件的内容
    """
    return np.load(config.DATA_DIR / "processed" / f"{name}.npz")


def _fix_labels(l):
    """修复标签，确保标签从0开始连续编号
    
    Args:
        l: 原始标签数组
    
    Returns:
        修复后的标签数组
    """
    uniq = np.unique(l)[None, :]
    new = (l[:, None] == uniq).argmax(axis=1)
    return new


def load_dataset(name, n_samples=None, select_views=None, select_labels=None, label_counts=None, noise_sd=None,
                 noise_views=None, to_dataset=True, **kwargs):
    """加载数据集，支持多种数据处理选项
    
    Args:
        name: 数据集名称
        n_samples: 加载的样本数量，None表示加载所有样本
        select_views: 要加载的视图子集，None表示加载所有视图
        select_labels: 要加载的标签（类别）子集，None表示加载所有类别
        label_counts: 每个类别的样本数量，None表示加载所有样本
        noise_sd: 添加到视图的噪声标准差
        noise_views: 要添加噪声的视图子集
        to_dataset: 是否返回TensorDataset对象
        **kwargs: 其他参数
    
    Returns:
        处理后的数据集
    """
    # 加载npz文件
    npz = _load_npz(name)
    labels = npz["labels"]
    views = [npz[f"view_{i}"] for i in range(npz["n_views"])]

    # 选择指定的标签
    if select_labels is not None:
        mask = np.isin(labels, select_labels)
        labels = labels[mask]
        views = [v[mask] for v in views]
        labels = _fix_labels(labels)

    # 按类别数量选择样本
    if label_counts is not None:
        idx = []
        unique_labels = np.unique(labels)
        assert len(unique_labels) == len(label_counts)
        for l, n in zip(unique_labels, label_counts):
            _idx = np.random.choice(np.where(labels == l)[0], size=n, replace=False)
            idx.append(_idx)

        idx = np.concatenate(idx, axis=0)
        labels = labels[idx]
        views = [v[idx] for v in views]

    # 随机选择指定数量的样本
    if n_samples is not None:
        idx = np.random.choice(labels.shape[0], size=min(labels.shape[0], int(n_samples)), replace=False)
        labels = labels[idx]
        views = [v[idx] for v in views]

    # 选择指定的视图
    if select_views is not None:
        if not isinstance(select_views, (list, tuple)):
            select_views = [select_views]
        views = [views[i] for i in select_views]

    # 添加噪声
    if noise_sd is not None:
        assert noise_views is not None, "当'noise_sd'不为None时，必须指定'noise_views'"
        if not isinstance(noise_views, (list, tuple)):
            noise_views = [int(noise_views)]
        for v in noise_views:
            views[v] += np.random.normal(loc=0, scale=float(noise_sd), size=views[v].shape)

    # 转换为float32类型
    views = [v.astype(np.float32) for v in views]
    
    # 返回TensorDataset对象或(views, labels)元组
    if to_dataset:
        dataset = th.utils.data.TensorDataset(*[th.Tensor(v).to(config.DEVICE, non_blocking=True) for v in views],
                                              th.Tensor(labels).to(config.DEVICE, non_blocking=True))
    else:
        dataset = (views, labels)
    return dataset
