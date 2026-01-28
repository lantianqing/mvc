import os
import argparse
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.metrics import normalized_mutual_info_score

import helpers
from models.build_model import from_file

IGNORE_IN_TOTAL = ("contrast",)  # 在总损失中忽略的项


def calc_metrics(labels, pred):
    """
    计算指标。

    :param labels: 标签张量
    :type labels: th.Tensor
    :param pred: 预测张量
    :type pred: th.Tensor
    :return: 包含计算指标的字典
    :rtype: dict
    """
    acc, cmat = helpers.ordered_cmat(labels, pred)
    metrics = {
        "acc": acc,
        "cmat": cmat,
        "nmi": normalized_mutual_info_score(labels, pred, average_method="geometric"),
    }
    return metrics


def get_log_params(net):
    """
    获取我们要记录的网络参数。

    :param net: 模型
    :type net:
    :return: 参数字典
    :rtype: dict
    """
    params_dict = {}
    weights = []
    if getattr(net, "fusion", None) is not None:
        with th.no_grad():
            weights = net.fusion.get_weights(softmax=True)

    elif hasattr(net, "attention"):
        weights = net.weights

    for i, w in enumerate(helpers.npy(weights)):
        params_dict[f"fusion/weight_{i}"] = w

    if hasattr(net, "discriminators"):
        for i, discriminator in enumerate(net.discriminators):
            d0, dv = helpers.npy([discriminator.d0, discriminator.dv])
            params_dict[f"discriminator_{i}/d0/mean"] = d0.mean()
            params_dict[f"discriminator_{i}/d0/std"] = d0.std()
            params_dict[f"discriminator_{i}/dv/mean"] = dv.mean()
            params_dict[f"discriminator_{i}/dv/std"] = dv.std()

    return params_dict


def get_eval_data(dataset, n_eval_samples, batch_size):
    """
    创建用于评估的数据加载器

    :param dataset: 输入数据集。
    :type dataset: th.utils.data.Dataset
    :param n_eval_samples: 评估数据集中包含的样本数量。设置为 None 以使用所有可用样本。
    :type n_eval_samples: int
    :param batch_size: 用于训练的批量大小。
    :type batch_size: int
    :return: 评估数据集加载器
    :rtype: th.utils.data.DataLoader
    """
    if n_eval_samples is not None:
        *views, labels = dataset.tensors
        n = views[0].size(0)
        idx = np.random.choice(n, min(n, n_eval_samples), replace=False)
        views, labels = [v[idx] for v in views], labels[idx]
        dataset = th.utils.data.TensorDataset(*views, labels)

    eval_loader = th.utils.data.DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0,
                                           drop_last=False, pin_memory=False)
    return eval_loader


def batch_predict(net, eval_data, batch_size):
    """
    批量计算 `eval_data` 的预测。批量处理不影响预测，但会影响损失计算。

    :param net: 模型
    :type net:
    :param eval_data: 评估数据加载器
    :type eval_data: th.utils.data.DataLoader
    :param batch_size: 批量大小
    :type batch_size: int
    :return: 标签张量，预测张量，损失值字典列表，包含聚类大小均值和标准差的数组。
    :rtype:
    """
    predictions = []
    labels = []
    losses = []
    cluster_sizes = []

    net.eval()
    with th.no_grad():
        for i, (*batch, label) in enumerate(eval_data):
            pred = net(batch)
            labels.append(helpers.npy(label))
            predictions.append(helpers.npy(pred).argmax(axis=1))

            # 计算损失：跳过不完整批次，避免错误计算
            if label.size(0) == batch_size:
                batch_losses = net.calc_losses(ignore_in_total=IGNORE_IN_TOTAL)
                losses.append(helpers.npy(batch_losses))
            # 对于不完整批次，使用累积的 losses 列表来获取前一批次的损失
            elif len(losses) > 0:
                # 从上一次的损失字典复制值
                prev_batch_losses = losses[-1]
                batch_losses = {k: prev_batch_losses[k] for k in prev_batch_losses.keys()}
                losses.append(helpers.npy(batch_losses))
            else:
                # 首次迭代，跳过
                pass

            cluster_sizes.append(helpers.npy(pred.sum(dim=0)))

    labels = np.concatenate(labels, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    net.train()
    return labels, predictions, losses, np.array(cluster_sizes).sum(axis=0)


def get_logs(cfg, net, eval_data, iter_losses=None, epoch=None, include_params=True):
    """
    获取日志
    
    :param cfg: 配置
    :param net: 模型
    :param eval_data: 评估数据
    :param iter_losses: 迭代损失
    :param epoch: 轮数
    :param include_params: 是否包含参数
    :return: 日志字典
    """
    if iter_losses is not None:
        logs = helpers.add_prefix(helpers.dict_means(iter_losses), "iter_losses")
    else:
        logs = {}
    if (epoch is None) or ((epoch % cfg.eval_interval) == 0):
        labels, pred, eval_losses, cluster_sizes = batch_predict(net, eval_data, cfg.batch_size)
        eval_losses = helpers.dict_means(eval_losses)
        logs.update(helpers.add_prefix(eval_losses, "eval_losses"))
        logs.update(helpers.add_prefix(calc_metrics(labels, pred), "metrics"))
        logs.update(helpers.add_prefix({"mean": cluster_sizes.mean(), "sd": cluster_sizes.std()}, "cluster_size"))
    if include_params:
        logs.update(helpers.add_prefix(get_log_params(net), "params"))
    if epoch is not None:
        logs["epoch"] = epoch
    return logs


def eval_run(cfg, cfg_name, experiment_identifier, run, net, eval_data, callbacks=tuple(), load_best=True):
    """
    评估训练运行。

    :param cfg: 实验配置
    :type cfg: config.defaults.Experiment
    :param cfg_name: 配置名称
    :type cfg_name: str
    :param experiment_identifier: 当前实验的 8 字符唯一标识符
    :type experiment_identifier: str
    :param run: 要评估的运行
    :type run: int
    :param net: 模型
    :type net:
    :param eval_data: 评估数据加载器
    :type eval_data: th.utils.data.DataLoader
    :param callbacks: 评估后要调用的回调列表
    :type callbacks: List
    :param load_best: 评估前加载 "best.pt" 模型？
    :type load_best: bool
    :return: 评估日志
    :rtype: dict
    """
    if load_best:
        model_path = helpers.get_save_dir(cfg_name, experiment_identifier, run) / "best.pt"
        if os.path.isfile(model_path):
            net.load_state_dict(th.load(model_path, weights_only=True))
        else:
            print(f"无法加载最佳模型进行评估。模型文件未找到: {model_path}")
    logs = get_logs(cfg, net, eval_data, include_params=True)
    for cb in callbacks:
        cb.at_eval(net=net, logs=logs)
    return logs


def eval_experiment(cfg_name, tag, plot=False):
    """
    评估完整实验

    :param cfg_name: 配置名称
    :type cfg_name: str
    :param tag: 当前实验的 8 字符唯一标识符
    :type tag: str
    :param plot: 显示融合前后表示的散点图？
    :type plot: bool
    """
    def move_all_tensors_to_cpu(module):
        """递归将模块中的所有张量移动到CPU"""
        for name, child in module.named_children():
            move_all_tensors_to_cpu(child)
        for name, param in module._parameters.items():
            if param is not None:
                module._parameters[name] = param.cpu()
        for name, buf in module._buffers.items():
            if buf is not None:
                module._buffers[name] = buf.cpu()
        # 移动其他可能不是参数或缓冲区的张量属性
        for attr_name in dir(module):
            # 检查是否为张量属性（包括以下划线开头的属性）
            try:
                attr = getattr(module, attr_name)
                if isinstance(attr, th.Tensor):
                    setattr(module, attr_name, attr.cpu())
            except AttributeError:
                pass

    # 在评估时强制使用CPU设备
    import config
    config.DEVICE = th.device('cpu')

    max_n_runs = 100
    best_logs = None
    best_run = None
    best_net = None
    best_loss = np.inf

    for run in range(max_n_runs):
        try:
            net, views, labels, cfg = from_file(cfg_name, tag, run, ckpt="best", return_data=True, return_config=True)
            net = net.cpu()  # 将模型移动到CPU进行评估
            move_all_tensors_to_cpu(net)  # 确保所有张量都在CPU上
            # 特别处理loss模块中的eye张量
            if hasattr(net, 'loss') and hasattr(net.loss, 'eye'):
                net.loss.eye = net.loss.eye.cpu()
        except FileNotFoundError:
            break

        eval_dataset = th.utils.data.TensorDataset(*[th.tensor(v) for v in views], th.tensor(labels))
        eval_data = get_eval_data(eval_dataset, cfg.n_eval_samples, cfg.batch_size)
        run_logs = eval_run(cfg, cfg_name, tag, run, net, eval_data, load_best=False)
        del run_logs["metrics/cmat"]

        if run_logs[f"eval_losses/{cfg.best_loss_term}"] < best_loss:
            best_loss = run_logs[f"eval_losses/{cfg.best_loss_term}"]
            best_logs = run_logs
            best_run = run
            best_net = net

    print(f"\n最佳运行是 {best_run}。", end="\n\n")
    headers = ["名称", "值"]
    values = list(best_logs.items())
    print(tabulate(values, headers=headers), "\n")
    
    if plot:
        plot_representations(views, labels, best_net)
        plt.show()
    

def plot_representations(views, labels, net, project_method="pca"):
    """
    绘制表示
    
    :param views: 多视图数据
    :param labels: 标签
    :param net: 模型
    :param project_method: 投影方法
    """
    with th.no_grad():
        output = net([th.tensor(v) for v in views])
        pred = helpers.npy(output).argmax(axis=1)

        hidden = helpers.npy(net.backbone_outputs)
        fused = helpers.npy(net.fused)

    hidden = np.concatenate(hidden, axis=0)
    view_hue = sum([labels.shape[0] * [str(i + 1)] for i in range(2)], [])
    fused_hue = [str(l + 1) for l in pred]

    view_cmap = "tab10"
    class_cmap = "hls"
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    plot_projection(X=hidden, method=project_method, hue=view_hue, ax=ax[0], title="Before Fusion",
                    legend_title="View", hue_order=sorted(list(set(view_hue))), cmap=view_cmap)
    plot_projection(X=fused, method=project_method, hue=fused_hue, ax=ax[1], title="After Fusion",
                    legend_title="Prediction", hue_order=sorted(list(set(fused_hue))), cmap=class_cmap)


def plot_projection(X, method, hue, ax, title=None, cmap="tab10", legend_title=None, legend_loc=1, **kwargs):
    """
    绘制投影
    
    :param X: 输入数据
    :param method: 投影方法
    :param hue: 色调
    :param ax: 轴
    :param title: 标题
    :param cmap: 颜色映射
    :param legend_title: 图例标题
    :param legend_loc: 图例位置
    :param kwargs: 额外参数
    """
    X = project(X, method)
    pl = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=hue, ax=ax, legend="full", palette=cmap, **kwargs)
    leg = pl.get_legend()
    leg._loc = legend_loc
    if title is not None:
        ax.set_title(title)
    if legend_title is not None:
        leg.set_title(legend_title)


def project(X, method):
    """
    投影方法
    
    :param X: 输入数据
    :param method: 投影方法
    :return: 投影后的数据
    """
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(X)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2).fit_transform(X)
    elif method is None:
        return X
    else:
        raise RuntimeError()


def parse_args():
    """
    解析参数
    
    :return: 参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="cfg_name", required=True)
    parser.add_argument("-t", "--tag", dest="tag", required=True)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    eval_experiment(args.cfg_name, args.tag, args.plot)
