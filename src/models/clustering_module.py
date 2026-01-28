import torch.nn as nn


class DDC(nn.Module):
    def __init__(self, input_dim, cfg):
        """
        DDC 聚类模块

        :param input_dim: 输入的形状。
        :param cfg: DDC 配置。参见 `config.defaults.DDC`
        """
        super().__init__()

        hidden_layers = [nn.Linear(input_dim[0], cfg.n_hidden), nn.ReLU()]
        if cfg.use_bn:
            hidden_layers.append(nn.BatchNorm1d(num_features=cfg.n_hidden))
        self.hidden = nn.Sequential(*hidden_layers)  # 隐藏层
        self.output = nn.Sequential(nn.Linear(cfg.n_hidden, cfg.n_clusters), nn.Softmax(dim=1))  # 输出层

    def forward(self, x):
        """
        前向传播
        
        :param x: 输入张量
        :return: 聚类分配和隐藏表示
        """
        hidden = self.hidden(x)
        output = self.output(hidden)
        return output, hidden
