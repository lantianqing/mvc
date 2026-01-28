import numpy as np

import helpers
from lib.loss import Loss
from lib.optimizer import Optimizer
from lib.backbones import Backbones
from models.model_base import ModelBase
from models.clustering_module import DDC


class DDCModel(ModelBase):
    def __init__(self, cfg):
        """
        完整的 DDC 模型

        :param cfg: DDC 模型配置
        :type cfg: config.defaults.DDCModel
        """
        super().__init__()

        self.cfg = cfg
        self.backbone_output = self.output = self.hidden = None
        self.backbone = Backbones.create_backbone(cfg.backbone_config)  # 骨干网络
        self.ddc_input_size = np.prod(self.backbone.output_size)  # DDC 输入大小
        self.ddc = DDC([self.ddc_input_size], cfg.cm_config)  # DDC 聚类模块
        self.loss = Loss(cfg.loss_config)  # 损失模块

        # 初始化权重
        self.apply(helpers.he_init_weights)
        # 实例化优化器
        self.optimizer = Optimizer(cfg.optimizer_config, self.parameters())

    def forward(self, x):
        """
        前向传播
        
        :param x: 输入
        :return: 聚类分配
        """
        if isinstance(x, list):
            # 由于多视图兼容性，我们可能会得到一个单元素列表作为输入
            assert len(x) == 1
            x = x[0]

        self.backbone_output = self.backbone(x).view(-1, self.ddc_input_size)  # 骨干网络输出
        self.output, self.hidden = self.ddc(self.backbone_output)  # 聚类
        return self.output
