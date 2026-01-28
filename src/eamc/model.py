"""
端到端对抗注意力网络的多模态聚类（EAMC）的自定义实现。
https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_End-to-End_Adversarial-Attention_Network_for_Multi-Modal_Clustering_CVPR_2020_paper.pdf
基于原始作者提供的代码。
"""

import torch as th
import torch.nn as nn

import helpers
import config
from lib.fusion import get_fusion_module
from lib.backbones import Backbones, MLP
from models.clustering_module import DDC
from eamc.loss import Loss


class Discriminator(nn.Module):
    def __init__(self, cfg, input_size):
        """
        EAMC 判别器

        :param cfg: 判别器配置
        :type cfg: config.eamc.defaults.Discriminator
        :param input_size: 输入大小
        :type input_size: Union[List[int, ...], Tuple[int, ...], ...]
        """
        super().__init__()
        self.mlp = MLP(cfg.mlp_config, input_size=input_size)
        self.output_layer = nn.Sequential(
            nn.Linear(self.mlp.output_size[0], 1, bias=True),
            nn.Sigmoid()
        )
        self.d0 = self.dv = None

    def forward(self, x0, xv):
        """
        前向传播
        
        :param x0: 真实样本
        :param xv: 伪造样本
        :return: 判别器输出
        """
        self.d0 = self.output_layer(self.mlp(x0))
        self.dv = self.output_layer(self.mlp(xv))
        return [self.d0, self.dv]


class AttentionLayer(nn.Module):
    def __init__(self, cfg, input_size):
        """
        EAMC 注意力网络

        :param cfg: 注意力配置
        :type cfg: config.eamc.defaults.AttentionLayer
        :param input_size: 输入大小
        :type input_size: Union[List[int, ...], Tuple[int, ...], ...]
        """
        super().__init__()
        self.tau = cfg.tau
        self.mlp = MLP(cfg.mlp_config, input_size=[input_size[0] * cfg.n_views])
        self.output_layer = nn.Linear(self.mlp.output_size[0], cfg.n_views, bias=True)
        self.weights = None

    def forward(self, xs):
        """
        前向传播
        
        :param xs: 多视图输入
        :return: 注意力权重
        """
        h = th.cat(xs, dim=1)
        act = self.output_layer(self.mlp(h))
        e = nn.functional.softmax(th.sigmoid(act) / self.tau, dim=1)
        # e = nn.functional.softmax(act, dim=1)
        self.weights = th.mean(e, dim=0)
        return self.weights


class EAMC(nn.Module):
    def __init__(self, cfg):
        """
        EAMC 模型

        :param cfg: EAMC 配置
        :type cfg: config.eamc.defaults.EAMC
        """
        super().__init__()

        self.cfg = cfg
        self.backbones = Backbones(cfg.backbone_configs)

        backbone_output_sizes = self.backbones.output_sizes
        assert all([backbone_output_sizes[0] == s for s in backbone_output_sizes])
        assert len(backbone_output_sizes[0]) == 1
        hidden_size = backbone_output_sizes[0]

        if cfg.attention_config is not None:
            self.fusion = None
            self.attention = AttentionLayer(cfg.attention_config, input_size=hidden_size)
            self.weights = None
            assert getattr(self.cfg, "fusion_config", None) is None, "EAMC 的 attention_config 和 fusion_config 不能同时非空"

        elif getattr(cfg, "fusion_config", None) is not None:
            self.fusion = get_fusion_module(cfg.fusion_config, input_sizes=backbone_output_sizes)
            self.attention = None
            self.weights = None

        else:
            self.attention = None
            self.weights = th.full([len(cfg.backbone_configs)], 1/len(cfg.backbone_configs), device=config.DEVICE)

        if cfg.discriminator_config is not None:
            self.discriminators = nn.ModuleList(
                [Discriminator(cfg.discriminator_config, input_size=hidden_size)
                 for _ in range(len(cfg.backbone_configs) - 1)]
            )
        else:
            self.discriminators = None

        self.ddc = DDC(hidden_size, cfg.cm_config)
        self.loss = Loss(cfg.loss_config)

        # 初始化权重
        self.apply(helpers.he_init_weights)

        self.backbone_outputs = None
        self.discriminator_outputs = None
        self.fused = None
        self.hidden = None
        self.output = None

        self.clustering_optimizer, self.discriminator_optimizer = self.get_optimizers()

    def get_optimizers(self):
        """
        获取优化器
        
        :return: 聚类优化器和判别器优化器
        """
        opt = self.cfg.optimizer_config

        # 聚类优化器
        clustering_optimizer_spec = [
            {"params": self.backbones.parameters(), "lr": opt.lr_backbones, "betas": opt.betas_backbones},
            {"params": self.ddc.parameters(), "lr": opt.lr_clustering_module, "betas": opt.betas_clustering_module}
        ]
        if self.cfg.attention_config is not None:
            clustering_optimizer_spec.append(
                {"params": self.attention.parameters(), "lr": opt.lr_att, "betas": opt.betas_att}
            )
        if getattr(self.cfg, "fusion_config", None) is not None:
            clustering_optimizer_spec.append(
                {"params": self.fusion.parameters(), "lr": 1e-3}
            )
        clustering_optimizer = th.optim.Adam(clustering_optimizer_spec)

        # 判别器优化器
        if self.cfg.discriminator_config is None:
            discriminator_optimizer = None
        else:
            discriminator_optimizer = th.optim.Adam([
                {"params": self.discriminators.parameters(), "lr": opt.lr_disc, "betas": opt.betas_disc}
            ])

        return clustering_optimizer, discriminator_optimizer

    def forward(self, views):
        """
        前向传播
        
        :param views: 多视图输入
        :return: 模型输出
        """
        self.backbone_outputs = self.backbones(views)
        if self.discriminators is not None:
            self.discriminator_outputs = [
                self.discriminators[i](self.backbone_outputs[0], self.backbone_outputs[i+1])
                for i in range(len(self.backbone_outputs) - 1)
            ]

        if self.fusion is not None:
            self.fused = self.fusion(self.backbone_outputs)
        else:
            if self.attention is not None:
                self.weights = self.attention(self.backbone_outputs)

            self.fused = th.sum(self.weights[None, None, :] * th.stack(self.backbone_outputs, dim=-1), dim=-1)

        self.output, self.hidden = self.ddc(self.fused)
        return self.output

    def calc_losses(self, ignore_in_total=tuple()):
        """
        计算损失
        
        :param ignore_in_total: 在总损失中忽略的项
        :return: 损失字典
        """
        return self.loss(self, ignore_in_total=ignore_in_total)

    @staticmethod
    def _get_train_mode(i, cfg):
        """
        获取训练模式
        
        :param i: 迭代次数
        :param cfg: 配置
        :return: 训练模式
        """
        if cfg.discriminator_config is None:
            return "gen"
        return "gen" if (i % (cfg.t + cfg.t_disc) < cfg.t) else "disc"

    def train_step(self, batch, epoch, it, n_batches):
        """
        训练步骤
        
        :param batch: 批次数据
        :param epoch: 当前轮数
        :param it: 当前迭代次数
        :param n_batches: 每轮的批次数
        :return: 损失字典
        """
        train_mode = self._get_train_mode(it, self.cfg)
        if train_mode == "disc":
            # 训练判别器
            opt = self.discriminator_optimizer
            loss_key = "disc"
            ignore_in_total = ("ddc_1", "ddc_2_flipped", "ddc_3", "att", "gen")
        else:
            opt = self.clustering_optimizer
            loss_key = "tot"
            ignore_in_total = ("disc",)

        opt.zero_grad()
        _ = self(batch)
        losses = self.calc_losses(ignore_in_total=ignore_in_total)
        losses[loss_key].backward()

        # 梯度裁剪
        if train_mode == "gen" and self.cfg.clip_norm is not None:
            th.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.clip_norm)

        opt.step()
        return losses
