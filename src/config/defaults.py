from typing import Tuple, List, Union, Optional
from typing_extensions import Literal

from config import Config
from pydantic import Field


class Dataset(Config):
    # 数据集名称。必须与 data/processed/ 中的文件名对应
    name: str
    # 加载的样本数量。设置为 None 以加载所有样本
    n_samples: int = None
    # 要加载的视图子集。设置为 None 以加载所有视图
    select_views: Tuple[int, ...] = None
    # 要加载的标签（类别）子集。设置为 None 以加载所有类别
    select_labels: Tuple[int, ...] = None
    # 每个类别要加载的样本数量。设置为 None 以加载所有样本
    label_counts: Tuple[int, ...] = None
    # 添加到视图 `noise_views` 的噪声标准差
    noise_sd: float = None
    # 要添加噪声的视图子集
    noise_views: Tuple[int, ...] = None


class Loss(Config):
    # 聚类数量
    n_clusters: int = None
    # 损失中使用的项，用 '|' 分隔。例如，"ddc_1|ddc_2|ddc_3|" 用于 DDC 聚类损失
    funcs: str
    # 损失项的可选权重。设置为 None 使所有权重等于 1
    weights: Tuple[Union[float, int], ...] = None
    # sigma 超参数的乘法因子
    rel_sigma: float = 0.15
    # Tau 超参数
    tau: float = 0.1
    # Delta 超参数
    delta: float = 0.1
    # 在对比损失中用作负样本数量的批量大小的比例。设置为 -1 以使用所有
    # 对（除了正样本）作为负样本对
    negative_samples_ratio: float = 0.25
    # 对比损失的相似性函数。支持 "cos"（默认）和 "gauss"
    contrastive_similarity: Literal["cos", "gauss"] = "cos"
    # 是否启用自适应对比权重
    adaptive_contrastive_weight: bool = True


class Optimizer(Config):
    # 基础学习率
    learning_rate: float = 0.001
    # 梯度裁剪的最大梯度范数
    clip_norm: float = 5.0
    # 学习率调度器的步长。None 禁用调度器
    scheduler_step_size: int = None
    # 学习率调度器的乘法因子
    scheduler_gamma: float = 0.1


class DDC(Config):
    # 聚类数量
    n_clusters: int = None
    # 第一个全连接层中的单元数
    n_hidden: int = 100
    # 在第一个全连接层后使用批归一化？
    use_bn: bool = True


class CNN(Config):
    # 输入图像的形状。格式：CHW
    input_size: Tuple[int, ...] = None
    # 网络层
    layers: Tuple = (
        ("conv", 5, 5, 32, "relu"),  # 卷积层：核大小 5x5，输出通道 32，激活函数 relu
        ("conv", 5, 5, 32, None),     # 卷积层：核大小 5x5，输出通道 32，无激活函数
        ("bn",),                      # 批归一化层
        ("relu",),                    # ReLU 激活层
        ("pool", 2, 2),               # 池化层：大小 2x2
        ("conv", 3, 3, 32, "relu"),  # 卷积层：核大小 3x3，输出通道 32，激活函数 relu
        ("conv", 3, 3, 32, None),     # 卷积层：核大小 3x3，输出通道 32，无激活函数
        ("bn",),                      # 批归一化层
        ("relu",),                    # ReLU 激活层
        ("pool", 2, 2),               # 池化层：大小 2x2
    )


class MLP(Config):
    # 输入的形状
    input_size: Tuple[int, ...] = None
    # 网络层中的单元数
    layers: Tuple[Union[int, str], ...] = (512, 512, 256)
    # 激活函数。可以是单个字符串，指定所有层的激活函数，或者是列表/元组
    # 字符串，指定每层的激活函数
    activation: Union[str, None, List[Union[None, str]], Tuple[Union[None, str], ...]] = "relu"
    # 是否包含偏置参数？单个布尔值适用于所有层，或布尔值的列表/元组适用于各个层
    use_bias: Union[bool, Tuple[bool, ...]] = True
    # 是否在层后包含批归一化？单个布尔值适用于所有层，或布尔值的列表/元组适用于各个层
    use_bn: Union[bool, Tuple[bool, ...]] = False


class Fusion(Config):
    # 融合方法。"mean" 固定权重 = 1/V。"weighted_mean"：使用学习的权重进行加权平均
    method: Literal["mean", "weighted_mean"]
    # 数据集中的视图数量
    n_views: int


class DDCModel(Config):
    # 编码器网络配置
    backbone_config: Union[MLP, CNN]
    # 聚类模块配置
    cm_config: Union[DDC]
    # 损失函数配置
    loss_config: Loss
    # 优化器配置
    optimizer_config: Optimizer = Optimizer()


class SiMVC(Config):
    # 编码器配置元组。每个模态一个
    backbone_configs: Tuple[Union[MLP, CNN], ...]
    # 融合模块配置
    fusion_config: Fusion
    # 聚类模块配置
    cm_config: Union[DDC]
    # 损失函数配置
    loss_config: Loss
    # 优化器配置
    optimizer_config: Optimizer = Optimizer()


class CoMVC(Config):
    # 编码器配置元组。每个模态一个
    backbone_configs: Tuple[Union[MLP, CNN], ...]
    # 投影头配置。设置为 None 以移除投影头
    projector_config: Optional[MLP]
    # 融合模块配置
    fusion_config: Fusion
    # 聚类模块配置
    cm_config: Union[DDC]
    # 损失函数配置
    loss_config: Loss
    # 优化器配置
    optimizer_config: Optimizer = Optimizer()


class Experiment(Config):
    # 数据集配置
    dataset_config: Dataset
    # 模型配置
    model_configuration: Union[CoMVC, SiMVC, DDC] = Field(alias="model_config")
    # 训练运行次数
    n_runs: int = 20
    # 训练轮数
    n_epochs: int = 100
    # 批量大小
    batch_size: int = 100
    # 模型评估之间的轮数
    eval_interval: int = 4
    # 模型检查点之间的轮数
    checkpoint_interval: int = 20
    # 早停的耐心值
    patience: int = 50000
    # 用于评估的样本数量。设置为 None 以使用数据集中的所有样本
    n_eval_samples: int = None
    # 用于模型选择的损失函数项。设置为 "tot" 以使用所有项的总和
    best_loss_term: str = "ddc_1"
