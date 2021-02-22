import torch
from torch import nn


class BCEFocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=None, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        loss = - (1 - pt) ** self.gamma * target * torch.log(pt) - \
               pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.alpha:
            loss = loss * self.alpha
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

__CRITERIONS__ = {
    "BCEFocalLoss": BCEFocalLoss
}


def get_criterion(cfg):
    if hasattr(nn, cfg.loss.name):
        return nn.__getattribute__(cfg.loss.name)(**cfg.loss.param)
    elif __CRITERIONS__.get(cfg.loss.name) is not None:
        return __CRITERIONS__[cfg.loss.name](**cfg.loss.param)
    else:
        raise NotImplementedError
