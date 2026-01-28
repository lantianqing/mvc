"""
端到端对抗注意力网络的多模态聚类（EAMC）的自定义实现。
https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_End-to-End_Adversarial-Attention_Network_for_Multi-Modal_Clustering_CVPR_2020_paper.pdf
基于原始作者提供的代码。
"""

from typing import Tuple, Union, Optional

from config.config import Config
from pydantic import Field
from config.defaults import MLP, DDC, CNN, Dataset, Fusion


class Loss(Config):
    # sigma 超参数的乘法因子
    rel_sigma: float = 0.15
    # 对抗损失的权重
    gamma: float = 10
    # 聚类数量
    n_clusters: int = None
    # 损失项的可选权重。设置为 None 使所有权重等于 1
    weights: Tuple[Union[float, int], ...] = None
    # 损失中使用的项，用 '|' 分隔。例如，"ddc_1|ddc_2|ddc_3|" 用于 DDC 聚类损失
    funcs: str = "ddc_1|ddc_2_flipped|ddc_3|att|gen|disc"


class AttentionLayer(Config):
    # Softmax 温度参数
    tau: float = 10.0
    # 注意力网络的配置。最终层将自动添加
    mlp_config: MLP = MLP(
        layers=(100, 50),
        activation=None
    )
    # 输入视图数量
    n_views: int = 2


class Discriminator(Config):
    # 判别器的配置
    mlp_config: MLP = MLP(
        layers=(256, 256, 128),
        activation="leaky_relu:0.2"
    )


class Optimizer(Config):
    # 判别器学习率
    lr_disc: float = 1e-3
    # 编码器学习率
    lr_backbones: float = 1e-5
    # 注意力学习率
    lr_att: float = 1e-4
    # 聚类模块学习率
    lr_clustering_module: float = 1e-5
    # 判别器的 Beta 参数
    betas_disc: Tuple[float, float] = (0.5, 0.999)
    # 编码器的 Beta 参数
    betas_backbones: Tuple[float, float] = (0.95, 0.999)
    # 注意力网络的 Beta 参数
    betas_att: Tuple[float, float] = (0.95, 0.999)
    # 聚类模块的 Beta 参数
    betas_clustering_module: Tuple[float, float] = (0.95, 0.999)


class EAMC(Config):
    # 编码器配置
    backbone_configs: Tuple[Union[MLP, CNN], ...]
    # 注意力网络配置。设置为 None 以移除注意力网络
    attention_config: Optional[AttentionLayer] = AttentionLayer()
    # 可选的融合配置，用于替代注意力网络
    fusion_config: Fusion = None
    # 判别器配置
    discriminator_config: Optional[Discriminator] = Discriminator()
    # 聚类模块配置
    cm_config: DDC
    # 损失配置
    loss_config: Loss = Loss()
    # 优化器配置
    optimizer_config: Optimizer = Optimizer()
    # 梯度裁剪的最大范数
    clip_norm: float = 0.5
    # 连续训练编码器、注意力网络和聚类模块的批次数
    t: int = 1
    # 连续训练判别器的批次数
    t_disc: int = 1


class EAMCExperiment(Config):
    # 数据集配置
    dataset_config: Dataset
    # 模型配置
    model_configuration: EAMC = Field(alias="model_config")
    # 训练运行次数
    n_runs: int = 20
    # 每次运行的训练轮数
    n_epochs: int = 500
    # 批量大小
    batch_size: int = 100
    # 模型评估之间的轮数
    eval_interval: int = 5
    # 模型检查点之间的轮数
    checkpoint_interval: int = 50
    # 用于评估的样本数量。设置为 None 以使用数据集中的所有样本
    n_eval_samples: int = None
    # 早停的耐心值
    patience: int = 1e9
    # 用于模型选择的损失函数项。设置为 "tot" 以使用所有项的总和
    best_loss_term: str = "tot"
