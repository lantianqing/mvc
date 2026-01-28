import torch as th

import config
import helpers
from models.ddc import DDCModel
from models.simple_mvc import SiMVC
from models.contrastive_mvc import CoMVC
from eamc.model import EAMC
from data.load import load_dataset


MODEL_CONSTRUCTORS = {
    "DDCModel": DDCModel,
    "SiMVC": SiMVC,
    "CoMVC": CoMVC,
    "EAMC": EAMC
}


def build_model(model_cfg):
    """
    构建由 `model_cfg` 指定的模型。

    :param model_cfg: 要构建的模型的配置
    :type model_cfg: Union[config.defaults.DDCModel, config.defaults.SiMVC, config.defaults.CoMVC,
                           config.eamc.defaults.EAMC]
    :return: 模型
    :rtype: Union[DDCModel, SiMVC, CoMVC, EAMC]
    """
    if model_cfg.class_name not in MODEL_CONSTRUCTORS:
        raise ValueError(f"无效的模型类型: {model_cfg.class_name}")
    model = MODEL_CONSTRUCTORS[model_cfg.class_name](model_cfg).to(config.DEVICE, non_blocking=True)
    return model


def from_file(experiment_name=None, tag=None, run=None, ckpt="best", return_data=False, return_config=False, **kwargs):
    """
    从磁盘加载训练好的模型

    :param experiment_name: 实验名称（配置的名称）
    :type experiment_name: str
    :param tag: 8字符的实验标识符
    :type tag: str
    :param run: 要加载的训练运行
    :type run: int
    :param ckpt: 要加载的检查点。指定有效的检查点，或使用 "best" 加载最佳模型。
    :type ckpt: Union[int, str]
    :param return_data: 是否返回数据集？
    :type return_data: bool
    :param return_config: 是否返回实验配置？
    :type return_config: bool
    :param kwargs: 额外的参数
    :type kwargs:
    :return: 加载的模型，数据集（如果 return_data == True），配置（如果 return_config == True）
    :rtype:
    """
    try:
        cfg = config.get_config_from_file(name=experiment_name, tag=tag)
    except FileNotFoundError:
        print("警告: 无法获取序列化的配置。")
        cfg = config.get_config_by_name(experiment_name)

    model_dir = helpers.get_save_dir(experiment_name, identifier=tag, run=run)
    if ckpt == "best":
        model_file = "best.pt"
    else:
        model_file = f"checkpoint_{str(ckpt).zfill(4)}.pt"

    model_path = model_dir / model_file
    net = build_model(cfg.model_configuration)
    print(f"从 {model_path} 加载模型")
    net.load_state_dict(th.load(model_path, map_location='cpu', weights_only=True))
    net.eval()

    out = [net]

    if return_data:
        dataset_kwargs = cfg.dataset_config.dict()
        for key, value in kwargs.items():
            dataset_kwargs[key] = value
        views, labels = load_dataset(to_dataset=False, **dataset_kwargs)
        out = [net, views, labels]

    if return_config:
        out.append(cfg)

    if len(out) == 1:
        out = out[0]

    return out
