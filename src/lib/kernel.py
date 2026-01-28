import torch as th
from torch.nn.functional import relu


EPSILON = 1E-9  # 数值稳定性常量


def kernel_from_distance_matrix(dist, rel_sigma, min_sigma=EPSILON):
    """
    从距离矩阵计算高斯核矩阵

    :param dist: 距离矩阵
    :type dist: th.Tensor
    :param rel_sigma: sigma超参数的乘法因子
    :type rel_sigma: float
    :param min_sigma: sigma的最小值，用于数值稳定性
    :type min_sigma: float
    :return: 核矩阵
    :rtype: th.Tensor
    """
    # 由于浮点误差，`dist`有时会包含负值，所以将这些值设为零
    dist = relu(dist)
    sigma2 = rel_sigma * th.median(dist)
    # 禁用sigma的梯度
    sigma2 = sigma2.detach()
    sigma2 = th.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
    k = th.exp(- dist / (2 * sigma2))
    return k


def vector_kernel(x, rel_sigma=0.15):
    """
    从矩阵的行计算核矩阵

    :param x: 输入矩阵
    :type x: th.Tensor
    :param rel_sigma: sigma超参数的乘法因子
    :type rel_sigma: float
    :return: 核矩阵
    :rtype: th.Tensor
    """
    return kernel_from_distance_matrix(cdist(x, x), rel_sigma)


def cdist(X, Y):
    """
    X的行和Y的行之间的成对距离

    :param X: 第一个输入矩阵
    :type X: th.Tensor
    :param Y: 第二个输入矩阵
    :type Y: th.Tensor
    :return: 包含X的行和Y的行之间成对距离的矩阵
    :rtype: th.Tensor
    """
    xyT = X @ th.t(Y)
    x2 = th.sum(X**2, dim=1, keepdim=True)
    y2 = th.sum(Y**2, dim=1, keepdim=True)
    d = x2 - 2 * xyT + th.t(y2)
    return d
