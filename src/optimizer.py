from adabelief_pytorch import AdaBelief
import torch_optimizer
from torch import optim
from src.sam import SAM

__OPTIMIZERS__ = {
    "AdaBelief": AdaBelief,
    "RAdam": torch_optimizer.RAdam,
    "SAM": SAM
}


def get_optimizer(cfg, model):
    optimizer_name = cfg.optimizer.name

    if optimizer_name == "SAM":
        base_optimizer_name = cfg.optimizer.base
        if __OPTIMIZERS__.get(base_optimizer_name) is not None:
            base_optimizer = __OPTIMIZERS__[base_optimizer_name]
        else:
            base_optimizer = optim.__getattribute__(base_optimizer_name)
            return SAM(model.parameters(), base_optimizer, **cfg.optimizer.param)

    if __OPTIMIZERS__.get(optimizer_name) is not None:
        return __OPTIMIZERS__[optimizer_name](model.parameters(), **cfg.optimizer.param)
    else:
        return optim.__getattribute__(optimizer_name)(model.parameters(), **cfg.optimizer.param)
