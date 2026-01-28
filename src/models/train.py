import wandb
import torch as th
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

import config
import helpers

# ==========================================
# 固定随机种子（用于可复现性）
# ==========================================
import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

set_seed(42)
from data.load import load_dataset
from models import callback
from models.build_model import build_model
from models import evaluate


# ==========================================
# 设置 matplotlib 中文字体
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 可视化函数
# ==========================================

def get_label_mapping(y_true, y_pred):
    """使用匈牙利算法找到预测簇与真实标签的最佳对应关系"""
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return row_ind, col_ind, w


def cluster_acc(y_true, y_pred):
    """计算聚类准确率"""
    _, _, w = get_label_mapping(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def plot_history(history, save_dir):
    """绘制 ACC 和 Loss 曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss 曲线
    if 'loss' in history and history['loss']:
        axes[0].plot(history['loss'], label='Total Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

    # ACC 曲线
    if 'acc' in history and history['acc']:
        axes[1].plot(history['acc'], label='Accuracy', color='green')
        axes[1].set_title('Clustering Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_dir, title='Confusion Matrix'):
    """绘制混淆矩阵"""
    row_ind, col_ind, w = get_label_mapping(y_true, y_pred)
    map_dict = {row: col for row, col in zip(row_ind, col_ind)}
    mapped_y_pred = np.array([map_dict[p] for p in y_pred])

    cm = confusion_matrix(y_true, mapped_y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()


def plot_tsne(net, views, labels, save_dir, n_samples=500):
    """绘制 t-SNE 散点图"""
    net.eval()
    device = next(net.parameters()).device

    # 先采样，避免加载全部数据到内存
    total_samples = len(labels)
    indices = np.random.choice(total_samples, min(n_samples, total_samples), replace=False)

    # 采样后的数据
    labels_sampled = labels[indices]
    views_sampled = [v[indices] for v in views]

    with th.no_grad():
        # 获取融合后的表示 - 确保张量在同一设备上
        views_tensor = [th.tensor(v, device=device) for v in views_sampled]
        output = net(views_tensor)
        z = helpers.npy(net.fused)
        pred = helpers.npy(output).argmax(axis=1)

    # t-SNE 降维
    z_2d = TSNE(n_components=2, random_state=42).fit_transform(z)

    # 绘图 - 真实标签
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(x=z_2d[:, 0], y=z_2d[:, 1], hue=labels_sampled,
                   palette='tab10', alpha=0.6, ax=axes[0], legend='full')
    axes[0].set_title('t-SNE: True Labels')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')

    # 绘图 - 预测标签
    sns.scatterplot(x=z_2d[:, 0], y=z_2d[:, 1], hue=pred,
                   palette='tab10', alpha=0.6, ax=axes[1], legend='full')
    axes[1].set_title('t-SNE: Predicted Clusters')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tsne.png'), dpi=150)
    plt.close()


def train(cfg, net, loader, eval_data, callbacks=tuple(), history=None):
    """
    训练模型一次运行。

    :param cfg: 实验配置
    :type cfg: config.defaults.Experiment
    :param net: 模型
    :type net:
    :param loader: 训练数据的数据加载器
    :type loader:  th.utils.data.DataLoader
    :param eval_data: 评估数据的数据加载器
    :type eval_data:  th.utils.data.DataLoader
    :param callbacks: 训练回调
    :type callbacks: List
    :param history: 训练历史记录字典
    :type history: dict
    :return: None
    :rtype: None
    """
    if history is None:
        history = {'loss': [], 'acc': []}

    n_batches = len(loader)  # 批次数
    for e in range(1, cfg.n_epochs + 1):  # 遍历每个epoch
        iter_losses = []  # 迭代损失
        # 使用 tqdm 显示进度条
        pbar = tqdm(loader, desc=f'Epoch {e}/{cfg.n_epochs}', ncols=100)
        for i, data in enumerate(pbar):  # 遍历每个批次
            *batch, _ = data  # 解包批次数据
            try:
                batch_losses = net.train_step(batch, epoch=(e-1), it=i, n_batches=n_batches)  # 训练步骤
            except Exception as e:
                print(f"Training stopped due to exception: {e}")
                return

            iter_losses.append(helpers.npy(batch_losses))  # 记录损失

            # 更新进度条显示损失
            if len(iter_losses) > 0:
                avg_loss = np.mean([l.get('tot', 0) for l in iter_losses])
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

        logs = evaluate.get_logs(cfg, net, eval_data=eval_data, iter_losses=iter_losses, epoch=e, include_params=True)  # 获取日志

        # 记录历史
        if 'eval_losses/tot' in logs:
            history['loss'].append(logs['eval_losses/tot'])
        if 'metrics/acc' in logs:
            history['acc'].append(logs['metrics/acc'])

        print(f"Epoch {e:3d} | Loss: {logs.get('eval_losses/tot', 0):.4f} | "
              f"ACC: {logs.get('metrics/acc', 0):.4f} | NMI: {logs.get('metrics/nmi', 0):.4f}")

        try:
            for cb in callbacks:
                cb.epoch_end(e, logs=logs, net=net)  # 回调epoch结束
        except callback.StopTraining as err:
            print(err)
            break

    return history


def main():
    """
    运行实验。
    """
    experiment_name, cfg = config.get_experiment_config()  # 获取实验配置
    dataset = load_dataset(**cfg.dataset_config.dict())  # 加载数据集
    loader = th.utils.data.DataLoader(dataset, batch_size=int(cfg.batch_size), shuffle=True, num_workers=0,
                                      drop_last=True, pin_memory=False)  # 创建数据加载器
    eval_data = evaluate.get_eval_data(dataset, cfg.n_eval_samples, cfg.batch_size)  # 获取评估数据
    # 使用配置名作为标识符，不使用随机ID
    experiment_identifier = 'latest'  # 固定标识符

    # 创建保存目录 - 直接使用配置名，不添加标识符
    save_dir = config.MODELS_DIR / experiment_name / 'run-0'
    os.makedirs(save_dir, exist_ok=True)

    run_logs = []  # 运行日志
    history = {'loss': [], 'acc': []}  # 训练历史

    for run in range(cfg.n_runs):  # 遍历每个运行
        print(f"\n{'='*50}")
        print(f"Run {run + 1}/{cfg.n_runs}")
        print(f"{'='*50}\n")

        net = build_model(cfg.model_configuration)  # 构建模型
        print(net)
        callbacks = (
            callback.Printer(print_confusion_matrix=(cfg.model_configuration.cm_config.n_clusters <= 100)),  # 打印回调
            callback.ModelSaver(cfg=cfg, experiment_name=experiment_name, identifier=experiment_identifier,
                                run=run, epoch_interval=1, best_loss_term=cfg.best_loss_term,
                                checkpoint_interval=cfg.checkpoint_interval),  # 模型保存回调
            callback.EarlyStopping(patience=cfg.patience, best_loss_term=cfg.best_loss_term, epoch_interval=1)  # 早停回调
        )
        train(cfg, net, loader, eval_data=eval_data, callbacks=callbacks, history=history)  # 训练模型
        run_logs.append(evaluate.eval_run(cfg=cfg, cfg_name=experiment_name,
                                          experiment_identifier=experiment_identifier, run=run, net=net,
                                          eval_data=eval_data, callbacks=callbacks, load_best=True))  # 评估运行

    # --- 生成可视化报告 ---
    print(f"\n{'='*50}")
    print("Generating Visualization Reports...")
    print(f"{'='*50}")

    # 获取视图数据用于可视化
    # dataset.tensors 是 (view_0, view_1, ..., labels) 的元组
    all_tensors = dataset.tensors
    views = [v.cpu().numpy() if v.is_cuda else v.numpy() for v in all_tensors[:-1]]
    labels = all_tensors[-1].cpu().numpy() if all_tensors[-1].is_cuda else all_tensors[-1].numpy()

    # 1. 训练曲线
    plot_history(history, save_dir)
    print(f"[+] Training curves saved to: {os.path.join(save_dir, 'training_curves.png')}")

    # 2. 混淆矩阵
    best_net = build_model(cfg.model_configuration)
    model_path = save_dir / 'best.pt'
    if model_path.exists():
        best_net.load_state_dict(th.load(model_path, weights_only=True))
        # 使用配置中的设备
        device = config.DEVICE
        best_net = best_net.to(device)
        labels_np, pred_np, _, _ = evaluate.batch_predict(best_net, eval_data, cfg.batch_size)
        plot_confusion_matrix(labels_np, pred_np, save_dir)
        print(f"[+] Confusion matrix saved to: {os.path.join(save_dir, 'confusion_matrix.png')}")

    # 3. t-SNE 散点图
    plot_tsne(best_net, views, labels, save_dir)
    print(f"[+] t-SNE plot saved to: {os.path.join(save_dir, 'tsne.png')}")

    print(f"\n[✓] All reports saved to: {save_dir}")
    print(f"\nBest Results:")
    if run_logs:
        best_log = min(run_logs, key=lambda x: x.get(f"eval_losses/{cfg.best_loss_term}", float('inf')))
        print(f"  - ACC: {best_log.get('metrics/acc', 0):.4f}")
        print(f"  - NMI: {best_log.get('metrics/nmi', 0):.4f}")
        print(f"  - Loss: {best_log.get(f'eval_losses/{cfg.best_loss_term}', 0):.4f}")


if __name__ == '__main__':
    main()  # 运行主函数
