import torch as th


class Optimizer:
    def __init__(self, cfg, params):
        """
        优化器包装类

        :param cfg: 优化器配置
        :type cfg: config.defaults.Optimizer
        :param params: 与优化器关联的参数
        :type params: 
        """
        self.clip_norm = cfg.clip_norm  # 梯度裁剪范数
        self.params = params  # 模型参数
        self._opt = th.optim.Adam(params, lr=cfg.learning_rate)  # Adam优化器
        if cfg.scheduler_step_size is not None:
            assert cfg.scheduler_gamma is not None
            self._sch = th.optim.lr_scheduler.StepLR(self._opt, step_size=cfg.scheduler_step_size,
                                                     gamma=cfg.scheduler_gamma)  # 学习率调度器
        else:
            self._sch = None

    def zero_grad(self):
        """
        清零梯度
        
        :return: 优化器的zero_grad方法返回值
        """
        return self._opt.zero_grad()

    def step(self, epoch):
        """
        执行优化步骤
        
        :param epoch: 当前epoch
        :return: 优化器的step方法返回值
        """
        if self._sch is not None:
            # 只在整数epoch时更新调度器，且不在第一个epoch更新
            if epoch.is_integer() and epoch > 0:
                self._sch.step()

        if self.clip_norm is not None:
            th.nn.utils.clip_grad_norm_(self.params, self.clip_norm)  # 梯度裁剪

        out = self._opt.step()  # 执行优化步骤
        return out
