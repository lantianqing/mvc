import torch as th
import torch.nn as nn

import helpers
from lib.loss import Loss
from lib.optimizer import Optimizer
from lib.backbones import Backbones, MLP
from lib.fusion import get_fusion_module
from models.clustering_module import DDC
from models.model_base import ModelBase


class CoMVC(ModelBase):
    def __init__(self, cfg):
        """
        CoMVC 模型的实现。

        :param cfg: 模型配置。有关配置对象的文档，请参见 `config.defaults.CoMVC`。
        """
        super().__init__()

        self.cfg = cfg
        self.output = self.hidden = self.fused = self.backbone_outputs = self.projections = None

        # 定义骨干网络和融合模块
        self.backbones = Backbones(cfg.backbone_configs)
        self.fusion = get_fusion_module(cfg.fusion_config, self.backbones.output_sizes)

        bb_sizes = self.backbones.output_sizes
        assert all([bb_sizes[0] == s for s in bb_sizes]), f"CoMVC 要求所有骨干网络具有相同的 " \
                                                          f"输出大小。得到: {bb_sizes}"

        if cfg.projector_config is None:
            self.projector = nn.Identity()  # 恒等映射
        else:
            self.projector = MLP(cfg.projector_config, input_size=bb_sizes[0])  # 投影器

        # 定义聚类模块
        self.ddc = DDC(input_dim=self.fusion.output_size, cfg=cfg.cm_config)
        # 定义损失模块
        self.loss = Loss(cfg=cfg.loss_config)
        # 初始化权重
        self.apply(helpers.he_init_weights)
        # 实例化优化器
        self.optimizer = Optimizer(cfg.optimizer_config, self.parameters())

    def forward(self, views):
        """
        前向传播
        
        :param views: 多视图输入
        :return: 聚类分配
        """
        self.backbone_outputs = self.backbones(views)  # 骨干网络输出
        self.fused = self.fusion(self.backbone_outputs)  # 特征融合
        self.projections = self.projector(th.cat(self.backbone_outputs, dim=0))  # 投影
        self.output, self.hidden = self.ddc(self.fused)  # 聚类
        return self.output

