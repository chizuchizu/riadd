from adabelief_pytorch import AdaBelief
import torch_optimizer
from torch import optim

__OPTIMIZERS__ = {
    "AdaBelief": AdaBelief,
    "RAdam": torch_optimizer.RAdam
}


def get_optimizer(cfg, model):
    optimizer_name = cfg.optimizer.name

    if __OPTIMIZERS__.get(optimizer_name) is not None:
        return __OPTIMIZERS__[optimizer_name](model.parameters(), **cfg.optimizer.param)
    else:
        return optim.__getattribute__(optimizer_name)(model.parameters(), **cfg.optimizer.param)
