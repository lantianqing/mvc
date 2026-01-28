import torch as th
import torch.nn as nn

import config
from lib import kernel

EPSILON = 1E-9  # 数值稳定性常量
DEBUG_MODE = False  # 调试模式


def triu(X):
    # 严格上三角部分的和
    return th.sum(th.triu(X, diagonal=1))


def _atleast_epsilon(X, eps=EPSILON):
    """
    确保所有元素 >= `eps`。

    :param X: 输入元素
    :type X: th.Tensor
    :param eps: epsilon
    :type eps: float
    :return: X的新版本，其中小于 `eps` 的元素已被替换为 `eps`。
    :rtype: th.Tensor
    """
    return th.where(X < eps, X.new_tensor(eps), X)


def d_cs(A, K, n_clusters):
    """
    柯西-施瓦茨散度。

    :param A: 聚类分配矩阵
    :type A:  th.Tensor
    :param K: 核矩阵
    :type K: th.Tensor
    :param n_clusters: 聚类数量
    :type n_clusters: int
    :return: CS散度
    :rtype: th.Tensor
    """
    nom = th.t(A) @ K @ A
    dnom_squared = th.unsqueeze(th.diagonal(nom), -1) @ th.unsqueeze(th.diagonal(nom), 0)

    nom = _atleast_epsilon(nom)
    dnom_squared = _atleast_epsilon(dnom_squared, eps=EPSILON**2)

    d = 2 / (n_clusters * (n_clusters - 1)) * triu(nom / th.sqrt(dnom_squared))
    return d


# ======================================================================================================================
# 损失项
# ======================================================================================================================

class LossTerm:
    # 损失计算所需的张量名称
    required_tensors = []

    def __init__(self, *args, **kwargs):
        """
        损失函数项的基类。

        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        """
        pass

    def __call__(self, net, cfg, extra):
        """
        计算损失项
        
        :param net: 网络模型
        :param cfg: 配置
        :param extra: 额外的张量
        :raise NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError()


class DDC1(LossTerm):
    """
    DDC 的 L_1 损失
    """
    required_tensors = ["hidden_kernel"]

    def __call__(self, net, cfg, extra):
        """
        计算 DDC1 损失
        
        :param net: 网络模型
        :param cfg: 配置
        :param extra: 额外的张量
        :return: 损失值
        """
        return d_cs(net.output, extra["hidden_kernel"], cfg.n_clusters)


class DDC2(LossTerm):
    """
    DDC 的 L_2 损失
    """
    def __call__(self, net, cfg, extra):
        """
        计算 DDC2 损失
        
        :param net: 网络模型
        :param cfg: 配置
        :param extra: 额外的张量
        :return: 损失值
        """
        n = net.output.size(0)
        return 2 / (n * (n - 1)) * triu(net.output @ th.t(net.output))


class DDC2Flipped(LossTerm):
    """
    DDC 的 L_2 损失的翻转版本。由 EAMC 使用
    """

    def __call__(self, net, cfg, extra):
        """
        计算翻转的 DDC2 损失
        
        :param net: 网络模型
        :param cfg: 配置
        :param extra: 额外的张量
        :return: 损失值
        """
        return 2 / (cfg.n_clusters * (cfg.n_clusters - 1)) * triu(th.t(net.output) @ net.output)


class DDC3(LossTerm):
    """
    DDC 的 L_3 损失
    """
    required_tensors = ["hidden_kernel"]

    def __init__(self, cfg):
        """
        初始化 DDC3 损失
        
        :param cfg: 配置
        """
        super().__init__()
        self.eye = th.eye(cfg.n_clusters, device=config.DEVICE)

    def __call__(self, net, cfg, extra):
        """
        计算 DDC3 损失
        
        :param net: 网络模型
        :param cfg: 配置
        :param extra: 额外的张量
        :return: 损失值
        """
        m = th.exp(-kernel.cdist(net.output, self.eye))
        return d_cs(m, extra["hidden_kernel"], cfg.n_clusters)


class Contrastive(LossTerm):
    large_num = 1e9  # 大数值，用于掩码

    def __init__(self, cfg):
        """
        对比损失函数

        :param cfg: 损失函数配置
        :type cfg: config.defaults.Loss
        """
        super().__init__()
        # 选择使用哪种实现
        if cfg.negative_samples_ratio == -1:
            self._loss_func = self._loss_without_negative_sampling
        else:
            self.eye = th.eye(cfg.n_clusters, device=config.DEVICE)
            self._loss_func = self._loss_with_negative_sampling

        # 设置相似性函数
        if cfg.contrastive_similarity == "cos":
            self.similarity_func = self._cosine_similarity
        elif cfg.contrastive_similarity == "gauss":
            self.similarity_func = kernel.vector_kernel
        else:
            raise RuntimeError(f"无效的对比相似性: {cfg.contrastive_similarity}")

    @staticmethod
    def _norm(mat):
        """
        归一化矩阵
        
        :param mat: 输入矩阵
        :return: 归一化后的矩阵
        """
        return th.nn.functional.normalize(mat, p=2, dim=1)

    @staticmethod
    def get_weight(net):
        """
        获取权重
        
        :param net: 网络模型
        :return: 权重
        """
        w = th.min(th.nn.functional.softmax(net.fusion.weights.detach(), dim=0))
        return w

    @classmethod
    def _normalized_projections(cls, net):
        """
        获取归一化的投影
        
        :param net: 网络模型
        :return: 样本数量和归一化的投影
        """
        n = net.projections.size(0) // 2
        h1, h2 = net.projections[:n], net.projections[n:]
        h2 = cls._norm(h2)
        h1 = cls._norm(h1)
        return n, h1, h2

    @classmethod
    def _cosine_similarity(cls, projections):
        """
        计算余弦相似性
        
        :param projections: 投影
        :return: 相似性矩阵
        """
        h = cls._norm(projections)
        return h @ h.t()

    def _draw_negative_samples(self, net, cfg, v, pos_indices):
        """
        构建负样本集。

        :param net: 模型
        :type net: Union[models.simple_mvc.SiMVC, models.contrastive_mvc.CoMVC]
        :param cfg: 损失配置
        :type cfg: config.defaults.Loss
        :param v: 视图数量
        :type v: int
        :param pos_indices: 连接的相似性矩阵中阳性样本的行索引
        :type pos_indices: th.Tensor
        :return: 负样本的索引
        :rtype: th.Tensor
        """
        cat = net.output.detach().argmax(dim=1)
        cat = th.cat(v * [cat], dim=0)

        weights = (1 - self.eye[cat])[:, cat[[pos_indices]]].T
        n_negative_samples = int(cfg.negative_samples_ratio * cat.size(0))
        negative_sample_indices = th.multinomial(weights, n_negative_samples, replacement=True)
        if DEBUG_MODE:
            self._check_negative_samples_valid(cat, pos_indices, negative_sample_indices)
        return negative_sample_indices

    @staticmethod
    def _check_negative_samples_valid(cat, pos_indices, neg_indices):
        """
        检查负样本是否有效
        
        :param cat: 类别
        :param pos_indices: 阳性样本索引
        :param neg_indices: 负样本索引
        """
        pos_cats = cat[pos_indices].view(-1, 1)
        neg_cats = cat[neg_indices]
        assert (pos_cats != neg_cats).detach().cpu().numpy().all()

    @staticmethod
    def _get_positive_samples(logits, v, n):
        """
        获取阳性样本

        :param logits: 输入相似性
        :type logits: th.Tensor
        :param v: 视图数量
        :type v: int
        :param n: 每个视图的样本数量（批量大小）
        :type n: int
        :return: 阳性对的相似性及其索引
        :rtype: Tuple[th.Tensor, th.Tensor]
        """
        diagonals = []
        inds = []
        for i in range(1, v):
            diagonal_offset = i * n
            diag_length = (v - i) * n
            _upper = th.diagonal(logits, offset=diagonal_offset)
            _lower = th.diagonal(logits, offset=-1 * diagonal_offset)
            _upper_inds = th.arange(0, diag_length)
            _lower_inds = th.arange(i * n, v * n)
            if DEBUG_MODE:
                assert _upper.size() == _lower.size() == _upper_inds.size() == _lower_inds.size() == (diag_length,)
            diagonals += [_upper, _lower]
            inds += [_upper_inds, _lower_inds]

        pos = th.cat(diagonals, dim=0)
        pos_inds = th.cat(inds, dim=0)
        return pos, pos_inds

    def _loss_with_negative_sampling(self, net, cfg, extra):
        """
        带负采样的对比损失实现。

        :param net: 模型
        :type net: Union[models.simple_mvc.SiMVC, models.contrastive_mvc.CoMVC]
        :param cfg: 损失配置
        :type cfg: config.defaults.Loss
        :param extra: 额外的张量
        :type extra:
        :return: 损失值
        :rtype: th.Tensor
        """
        n = net.output.size(0)
        v = len(net.backbone_outputs)
        logits = self.similarity_func(net.projections) / cfg.tau

        pos, pos_inds = self._get_positive_samples(logits, v, n)
        neg_inds = self._draw_negative_samples(net, cfg, v, pos_inds)
        neg = logits[pos_inds.view(-1, 1), neg_inds]

        inputs = th.cat((pos.view(-1, 1), neg), dim=1)
        labels = th.zeros(v * (v - 1) * n, device=config.DEVICE, dtype=th.long)
        loss = th.nn.functional.cross_entropy(inputs, labels)

        if cfg.adaptive_contrastive_weight:
            loss *= self.get_weight(net)

        return cfg.delta * loss

    def _loss_without_negative_sampling(self, net, cfg, extra):
        """
        不带负采样的对比损失实现。
        改编自: https://github.com/google-research/simclr/blob/master/objective.py

        :param net: 模型
        :type net: Union[models.simple_mvc.SiMVC, models.contrastive_mvc.CoMVC]
        :param cfg: 损失配置
        :type cfg: config.defaults.Loss
        :param extra: 额外的张量
        :type extra:
        :return: 损失值
        :rtype: th.Tensor
        """
        assert len(net.backbone_outputs) == 2, "不带负采样的对比损失仅支持 2 个视图。"
        n, h1, h2 = self._normalized_projections(net)

        labels = th.arange(0, n, device=config.DEVICE, dtype=th.long)
        masks = th.eye(n, device=config.DEVICE)

        logits_aa = ((h1 @ h1.t()) / cfg.tau) - masks * self.large_num
        logits_bb = ((h2 @ h2.t()) / cfg.tau) - masks * self.large_num

        logits_ab = (h1 @ h2.t()) / cfg.tau
        logits_ba = (h2 @ h1.t()) / cfg.tau

        loss_a = th.nn.functional.cross_entropy(th.cat((logits_ab, logits_aa), dim=1), labels)
        loss_b = th.nn.functional.cross_entropy(th.cat((logits_ba, logits_bb), dim=1), labels)

        loss = (loss_a + loss_b)

        if cfg.adaptive_contrastive_weight:
            loss *= self.get_weight(net)

        return cfg.delta * loss

    def __call__(self, net, cfg, extra):
        """
        计算对比损失
        
        :param net: 网络模型
        :param cfg: 配置
        :param extra: 额外的张量
        :return: 损失值
        """
        return self._loss_func(net, cfg, extra)


# ======================================================================================================================
# 额外函数
# ======================================================================================================================

def hidden_kernel(net, cfg):
    """
    计算隐藏层的核矩阵
    
    :param net: 网络模型
    :param cfg: 配置
    :return: 核矩阵
    """
    return kernel.vector_kernel(net.hidden, cfg.rel_sigma)


# ======================================================================================================================
# 损失类
# ======================================================================================================================

class Loss(nn.Module):
    # 损失中可能包含的项
    TERM_CLASSES = {
        "ddc_1": DDC1,
        "ddc_2": DDC2,
        "ddc_2_flipped": DDC2Flipped,
        "ddc_3": DDC3,
        "contrast": Contrastive,
    }
    # 计算项所需张量的函数
    EXTRA_FUNCS = {
        "hidden_kernel": hidden_kernel,
    }

    def __init__(self, cfg):
        """
        通用损失函数的实现

        :param cfg: 损失函数配置
        :type cfg: config.defaults.Loss
        """
        super().__init__()
        self.cfg = cfg

        self.names = cfg.funcs.split("|")
        self.weights = cfg.weights if cfg.weights is not None else len(self.names) * [1]

        self.terms = []
        for term_name in self.names:
            self.terms.append(self.TERM_CLASSES[term_name](cfg))

        self.required_extras_names = list(set(sum([t.required_tensors for t in self.terms], [])))

    def forward(self, net, ignore_in_total=tuple()):
        """
        前向传播
        
        :param net: 网络模型
        :param ignore_in_total: 在总损失中忽略的项
        :return: 损失值字典
        """
        extra = {name: self.EXTRA_FUNCS[name](net, self.cfg) for name in self.required_extras_names}
        loss_values = {}
        for name, term, weight in zip(self.names, self.terms, self.weights):
            value = term(net, self.cfg, extra)
            # 如果得到一个字典，将字典中的每个项添加到 "name/" 作用域下
            if isinstance(value, dict):
                for key, _value in value.items():
                    loss_values[f"{name}/{key}"] = weight * _value
            # 否则，直接将值添加到字典中
            else:
                loss_values[name] = weight * value

        loss_values["tot"] = sum([loss_values[k] for k in loss_values.keys() if k not in ignore_in_total])
        return loss_values

