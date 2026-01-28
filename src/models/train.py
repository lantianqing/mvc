import wandb
import torch as th

import config
import helpers
from data.load import load_dataset
from models import callback
from models.build_model import build_model
from models import evaluate


def train(cfg, net, loader, eval_data, callbacks=tuple()):
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
    :return: None
    :rtype: None
    """
    n_batches = len(loader)  # 批次数
    for e in range(1, cfg.n_epochs + 1):  # 遍历每个epoch
        iter_losses = []  # 迭代损失
        for i, data in enumerate(loader):  # 遍历每个批次
            *batch, _ = data  # 解包批次数据
            try:
                batch_losses = net.train_step(batch, epoch=(e-1), it=i, n_batches=n_batches)  # 训练步骤
            except Exception as e:
                print(f"Training stopped due to exception: {e}")
                return

            iter_losses.append(helpers.npy(batch_losses))  # 记录损失
        logs = evaluate.get_logs(cfg, net, eval_data=eval_data, iter_losses=iter_losses, epoch=e, include_params=True)  # 获取日志
        try:
            for cb in callbacks:
                cb.epoch_end(e, logs=logs, net=net)  # 回调epoch结束
        except callback.StopTraining as err:
            print(err)
            break


def main():
    """
    运行实验。
    """
    experiment_name, cfg = config.get_experiment_config()  # 获取实验配置
    dataset = load_dataset(**cfg.dataset_config.dict())  # 加载数据集
    loader = th.utils.data.DataLoader(dataset, batch_size=int(cfg.batch_size), shuffle=True, num_workers=0,
                                      drop_last=True, pin_memory=False)  # 创建数据加载器
    eval_data = evaluate.get_eval_data(dataset, cfg.n_eval_samples, cfg.batch_size)  # 获取评估数据
    experiment_identifier = wandb.util.generate_id()  # 生成实验标识符

    run_logs = []  # 运行日志
    for run in range(cfg.n_runs):  # 遍历每个运行
        net = build_model(cfg.model_configuration)  # 构建模型
        print(net)
        callbacks = (
            callback.Printer(print_confusion_matrix=(cfg.model_configuration.cm_config.n_clusters <= 100)),  # 打印回调
            callback.ModelSaver(cfg=cfg, experiment_name=experiment_name, identifier=experiment_identifier,
                                run=run, epoch_interval=1, best_loss_term=cfg.best_loss_term,
                                checkpoint_interval=cfg.checkpoint_interval),  # 模型保存回调
            callback.EarlyStopping(patience=cfg.patience, best_loss_term=cfg.best_loss_term, epoch_interval=1)  # 早停回调
        )
        train(cfg, net, loader, eval_data=eval_data, callbacks=callbacks)  # 训练模型
        run_logs.append(evaluate.eval_run(cfg=cfg, cfg_name=experiment_name,
                                          experiment_identifier=experiment_identifier, run=run, net=net,
                                          eval_data=eval_data, callbacks=callbacks, load_best=True))  # 评估运行


if __name__ == '__main__':
    main()  # 运行主函数
