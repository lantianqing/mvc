import torch.nn as nn
import numpy as np

import helpers


class Backbone(nn.Module):
    def __init__(self):
        """
        骨干网络基类
        """
        super().__init__()
        self.layers = nn.ModuleList()

    def forward(self, x):
        """
        前向传播
        
        :param x: 输入数据
        :return: 输出特征
        """
        for layer in self.layers:
            x = layer(x)
        return x


class CNN(Backbone):
    def __init__(self, cfg, flatten_output=True, **_):
        """
        CNN 骨干网络

        :param cfg: CNN 配置
        :type cfg: config.defaults.CNN
        :param flatten_output: 是否展平骨干网络输出
        :type flatten_output: bool
        :param _: 其他参数
        :type _:
        """
        super().__init__()

        self.output_size = list(cfg.input_size)

        for layer_type, *layer_params in cfg.layers:
            if layer_type == "conv":
                self.layers.append(nn.Conv2d(in_channels=self.output_size[0], out_channels=layer_params[2],
                                             kernel_size=layer_params[:2]))
                # 更新输出大小
                self.output_size[0] = layer_params[2]
                self.output_size[1:] = helpers.conv2d_output_shape(self.output_size[1:], kernel_size=layer_params[:2])
                # 添加激活函数
                if layer_params[3] == "relu":
                    self.layers.append(nn.ReLU())

            elif layer_type == "pool":
                self.layers.append(nn.MaxPool2d(kernel_size=layer_params))
                # 更新输出大小
                self.output_size[1:] = helpers.conv2d_output_shape(self.output_size[1:], kernel_size=layer_params,
                                                                   stride=layer_params)

            elif layer_type == "relu":
                self.layers.append(nn.ReLU())

            elif layer_type == "lrelu":
                self.layers.append(nn.LeakyReLU(layer_params[0]))

            elif layer_type == "bn":
                if len(self.output_size) > 1:
                    self.layers.append(nn.BatchNorm2d(num_features=self.output_size[0]))
                else:
                    self.layers.append(nn.BatchNorm1d(num_features=self.output_size[0]))

            elif layer_type == "fc":
                self.layers.append(nn.Flatten())
                self.output_size = [np.prod(self.output_size)]
                self.layers.append(nn.Linear(self.output_size[0], layer_params[0], bias=True))
                self.output_size = [layer_params[0]]

            else:
                raise RuntimeError(f"未知层类型: {layer_type}")

        if flatten_output:
            self.layers.append(nn.Flatten())
            self.output_size = [np.prod(self.output_size)]


class MLP(Backbone):
    def __init__(self, cfg, input_size=None, **_):
        """
        MLP 骨干网络

        :param cfg: MLP 配置
        :type cfg: config.defaults.MLP
        :param input_size: 可选输入大小，覆盖 `cfg` 中设置的大小
        :type input_size: Optional[Union[List, Tuple]]
        :param _: 其他参数
        :type _:
        """
        super().__init__()
        self.output_size = self.create_linear_layers(cfg, self.layers, input_size=input_size)

    @staticmethod
    def get_activation_module(a):
        """
        获取激活函数模块
        
        :param a: 激活函数名称
        :return: 激活函数模块
        """
        if a == "relu":
            return nn.ReLU()
        elif a == "sigmoid":
            return nn.Sigmoid()
        elif a == "tanh":
            return nn.Tanh()
        elif a == "softmax":
            return nn.Softmax(dim=1)
        elif a.startswith("leaky_relu"):
            neg_slope = float(a.split(":")[1])
            return nn.LeakyReLU(neg_slope)
        else:
            raise RuntimeError(f"无效的 MLP 激活函数: {a}.")

    @classmethod
    def create_linear_layers(cls, cfg, layer_container, input_size=None):
        """
        创建线性层
        
        :param cfg: 配置
        :param layer_container: 层容器
        :param input_size: 输入大小
        :return: 输出大小
        """
        # `input_size` 优先级高于 `cfg.input_size`
        if input_size is not None:
            output_size = list(input_size)
        else:
            output_size = list(cfg.input_size)

        if len(output_size) > 1:
            layer_container.append(nn.Flatten())
            output_size = [np.prod(output_size)]

        n_layers = len(cfg.layers)
        activations = helpers.ensure_iterable(cfg.activation, expected_length=n_layers)
        use_bias = helpers.ensure_iterable(cfg.use_bias, expected_length=n_layers)
        use_bn = helpers.ensure_iterable(cfg.use_bn, expected_length=n_layers)

        for n_units, act, _use_bias, _use_bn in zip(cfg.layers, activations, use_bias, use_bn):
            # 如果 n_units = -1，则单元数应与前一个单元数相同，或与输入维度相同
            if n_units == -1:
                n_units = output_size[0]

            layer_container.append(nn.Linear(in_features=output_size[0], out_features=n_units, bias=_use_bias))
            if _use_bn:
                # 在激活前添加 BN
                layer_container.append(nn.BatchNorm1d(num_features=n_units))
            if act is not None:
                # 添加激活函数
                layer_container.append(cls.get_activation_module(act))
            output_size[0] = n_units

        return output_size


class Backbones(nn.Module):
    BACKBONE_CONSTRUCTORS = {
        "CNN": CNN,
        "MLP": MLP
    }

    def __init__(self, backbone_configs, flatten_output=True):
        """
        表示多个骨干网络的类。使用输入列表调用，其中 inputs[0] 进入第一个骨干网络，依此类推。

        :param backbone_configs: 骨干网络配置列表。每个元素对应一个骨干网络。
        :type backbone_configs: List[Union[config.defaults.MLP, config.defaults.CNN], ...]
        :param flatten_output: 是否展平骨干网络输出
        :type flatten_output: bool
        """
        super().__init__()

        self.backbones = nn.ModuleList()
        for cfg in backbone_configs:
            self.backbones.append(self.create_backbone(cfg, flatten_output=flatten_output))

    @property
    def output_sizes(self):
        """
        获取所有骨干网络的输出大小
        
        :return: 输出大小列表
        """
        return [bb.output_size for bb in self.backbones]

    @classmethod
    def create_backbone(cls, cfg, flatten_output=True):
        """
        创建骨干网络
        
        :param cfg: 骨干网络配置
        :param flatten_output: 是否展平输出
        :return: 骨干网络实例
        """
        if cfg.class_name not in cls.BACKBONE_CONSTRUCTORS:
            raise RuntimeError(f"无效的骨干网络: '{cfg.class_name}'")
        return cls.BACKBONE_CONSTRUCTORS[cfg.class_name](cfg, flatten_output=flatten_output)

    def forward(self, views):
        """
        前向传播
        
        :param views: 多视图输入
        :return: 多视图输出
        """
        assert len(views) == len(self.backbones), f"视图数量 ({len(views)}) != 骨干网络数量 ({len(self.backbones)})."
        outputs = [bb(v) for bb, v in zip(self.backbones, views)]
        return outputs


