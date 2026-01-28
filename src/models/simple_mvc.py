import helpers
from lib.loss import Loss
from lib.fusion import get_fusion_module
from lib.optimizer import Optimizer
from lib.backbones import Backbones
from models.model_base import ModelBase
from models.clustering_module import DDC


class SiMVC(ModelBase):
    def __init__(self, cfg):
        """
        SiMVC 模型的实现。

        :param cfg: 模型配置。有关配置对象的文档，请参见 `config.defaults.SiMVC`。
        """
        super().__init__()

        self.cfg = cfg
        self.output = self.hidden = self.fused = self.backbone_outputs = None

        # 定义骨干网络和融合模块
        self.backbones = Backbones(cfg.backbone_configs)  # 骨干网络
        self.fusion = get_fusion_module(cfg.fusion_config, self.backbones.output_sizes)  # 融合模块
        # 定义聚类模块
        self.ddc = DDC(input_dim=self.fusion.output_size, cfg=cfg.cm_config)  # 聚类模块
        # 定义损失模块
        self.loss = Loss(cfg=cfg.loss_config)  # 损失模块
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
        self.output, self.hidden = self.ddc(self.fused)  # 聚类
        return self.output
