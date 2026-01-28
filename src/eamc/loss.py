"""
端到端对抗注意力网络的多模态聚类（EAMC）的自定义实现。
https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_End-to-End_Adversarial-Attention_Network_for_Multi-Modal_Clustering_CVPR_2020_paper.pdf
基于原始作者提供的代码。
"""

import torch as th
from torch.nn.functional import binary_cross_entropy

import config
from lib import loss, kernel


class AttLoss(loss.LossTerm):
    """
    注意力损失
    """
    required_tensors = ["backbone_kernels", "fusion_kernel"]

    def __call__(self, net, cfg, extra):
        # 计算加权核
        kc = th.sum(net.weights[None, None, :] * th.stack(extra["backbone_kernels"], dim=-1), dim=-1)
        # 计算与融合核的差异
        dif = (extra["fusion_kernel"] - kc)
        # 返回迹作为损失
        return th.trace(dif @ th.t(dif))


class GenLoss(loss.LossTerm):
    """
    生成器损失
    """
    def __call__(self, net, cfg, extra):
        tot = th.tensor(0., device=config.DEVICE)
        target = th.ones(net.output.size(0), device=config.DEVICE)
        # 计算每个判别器输出的损失
        for _, dv in net.discriminator_outputs:
            tot += binary_cross_entropy(dv.squeeze(), target)
        return cfg.gamma * tot


class DiscLoss(loss.LossTerm):
    """
    判别器损失
    """
    def __call__(self, net, cfg, extra):
        tot = th.tensor(0., device=config.DEVICE)
        real_target = th.ones(net.output.size(0), device=config.DEVICE)
        fake_target = th.zeros(net.output.size(0), device=config.DEVICE)
        # 计算真实和伪造样本的损失
        for d0, dv in net.discriminator_outputs:
            tot += binary_cross_entropy(dv.squeeze(), fake_target) + binary_cross_entropy(d0.squeeze(), real_target)
        return tot


def backbone_kernels(net, cfg):
    """计算每个 backbone 输出的核矩阵
    
    Args:
        net: 网络模型
        cfg: 配置
    
    Returns:
        核矩阵列表
    """
    return [kernel.vector_kernel(h, cfg.rel_sigma) for h in net.backbone_outputs]


def fusion_kernel(net, cfg):
    """计算融合输出的核矩阵
    
    Args:
        net: 网络模型
        cfg: 配置
    
    Returns:
        核矩阵
    """
    return kernel.vector_kernel(net.fused, cfg.rel_sigma)


class Loss(loss.Loss):
    # 覆盖 Loss 类的 TERM_CLASSES 和 EXTRA_FUNCS，以便包含 EAMC 损失
    TERM_CLASSES = {
        "ddc_1": loss.DDC1,
        "ddc_2_flipped": loss.DDC2Flipped,
        "ddc_2": loss.DDC2,
        "ddc_3": loss.DDC3,
        "att": AttLoss,
        "gen": GenLoss,
        "disc": DiscLoss
    }
    EXTRA_FUNCS = {
        "hidden_kernel": loss.hidden_kernel,
        "backbone_kernels": backbone_kernels,
        "fusion_kernel": fusion_kernel
    }
