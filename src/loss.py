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


# Code taken from https://github.com/fhopfmueller/bi-tempered-loss-pytorch/blob/master/bi_tempered_loss_pytorch.py

def log_t(u, t):
    """Compute log_t for `u'."""
    if t == 1.0:
        return u.log()
    else:
        return (u.pow(1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u, t):
    """Compute exp_t for `u'."""
    if t == 1:
        return u.exp()
    else:
        return (1.0 + (1.0 - t) * u).relu().pow(1.0 / (1.0 - t))


def compute_normalization_fixed_point(activations, t, num_iters):
    """Returns the normalization value for each example (t > 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same shape as activation with the last dimension being 1.
    """
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations_step_0 = activations - mu

    normalized_activations = normalized_activations_step_0

    for _ in range(num_iters):
        logt_partition = torch.sum(
            exp_t(normalized_activations, t), -1, keepdim=True)
        normalized_activations = normalized_activations_step_0 * \
                                 logt_partition.pow(1.0 - t)

    logt_partition = torch.sum(
        exp_t(normalized_activations, t), -1, keepdim=True)
    normalization_constants = - log_t(1.0 / logt_partition, t) + mu

    return normalization_constants


def compute_normalization_binary_search(activations, t, num_iters):
    """Returns the normalization value for each example (t < 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (< 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations = activations - mu

    effective_dim = \
        torch.sum(
            (normalized_activations > -1.0 / (1.0 - t)).to(torch.int32),
            dim=-1, keepdim=True).to(activations.dtype)

    shape_partition = activations.shape[:-1] + (1,)
    lower = torch.zeros(shape_partition, dtype=activations.dtype, device=activations.device)
    upper = -log_t(1.0 / effective_dim, t) * torch.ones_like(lower)

    for _ in range(num_iters):
        logt_partition = (upper + lower) / 2.0
        sum_probs = torch.sum(
            exp_t(normalized_activations - logt_partition, t),
            dim=-1, keepdim=True)
        update = (sum_probs < 1.0).to(activations.dtype)
        lower = torch.reshape(
            lower * update + (1.0 - update) * logt_partition,
            shape_partition)
        upper = torch.reshape(
            upper * (1.0 - update) + update * logt_partition,
            shape_partition)

    logt_partition = (upper + lower) / 2.0
    return logt_partition + mu


class ComputeNormalization(torch.autograd.Function):
    """
    Class implementing custom backward pass for compute_normalization. See compute_normalization.
    """

    @staticmethod
    def forward(ctx, activations, t, num_iters):
        if t < 1.0:
            normalization_constants = compute_normalization_binary_search(activations, t, num_iters)
        else:
            normalization_constants = compute_normalization_fixed_point(activations, t, num_iters)

        ctx.save_for_backward(activations, normalization_constants)
        ctx.t = t
        return normalization_constants

    @staticmethod
    def backward(ctx, grad_output):
        activations, normalization_constants = ctx.saved_tensors
        t = ctx.t
        normalized_activations = activations - normalization_constants
        probabilities = exp_t(normalized_activations, t)
        escorts = probabilities.pow(t)
        escorts = escorts / escorts.sum(dim=-1, keepdim=True)
        grad_input = escorts * grad_output

        return grad_input, None, None


def compute_normalization(activations, t, num_iters=5):
    """Returns the normalization value for each example.
    Backward pass is implemented.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """
    return ComputeNormalization.apply(activations, t, num_iters)


def tempered_sigmoid(activations, t, num_iters=5):
    """Tempered sigmoid function.
    Args:
      activations: Activations for the positive class for binary classification.
      t: Temperature tensor > 0.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A probabilities tensor.
    """
    internal_activations = torch.stack([activations,
                                        torch.zeros_like(activations)],
                                       dim=-1)
    internal_probabilities = tempered_softmax(internal_activations, t, num_iters)
    return internal_probabilities[..., 0]


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature > 1.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A probabilities tensor.
    """
    if t == 1.0:
        return activations.softmax(dim=-1)

    normalization_constants = compute_normalization(activations, t, num_iters)
    return exp_t(activations - normalization_constants, t)


# def bi_tempered_binary_logistic_loss(activations,
#                                      labels,
#                                      t1,
#                                      t2,
#                                      label_smoothing=0.0,
#                                      num_iters=5,
#                                      reduction='mean'):
#     """Bi-Tempered binary logistic loss.
#     Args:
#       activations: A tensor containing activations for class 1.
#       labels: A tensor with shape as activations, containing probabilities for class 1
#       t1: Temperature 1 (< 1.0 for boundedness).
#       t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
#       label_smoothing: Label smoothing
#       num_iters: Number of iterations to run the method.
#     Returns:
#       A loss tensor.
#     """
#     internal_activations = torch.stack([activations,
#                                         torch.zeros_like(activations)],
#                                        dim=-1)
#     internal_labels = torch.stack([labels.to(activations.dtype),
#                                    1.0 - labels.to(activations.dtype)],
#                                   dim=-1)
#     return bi_tempered_logistic_loss(internal_activations,
#                                      internal_labels,
#                                      t1,
#                                      t2,
#                                      label_smoothing=label_smoothing,
#                                      num_iters=num_iters,
#                                      reduction=reduction)

class BiT(nn.Module):
    def __init__(self, t1, t2, label_smoothing=0, num_iters=5, reduction="mean"):
        super().__init__()

        self.t1 = t1
        self.t2 = t2
        self.label_smoothing = label_smoothing
        self.num_iters = num_iters
        self.reduction = reduction

    def forward(self, activations,
                labels):
        """Bi-Tempered Logistic Loss.
        Args:
          activations: A multi-dimensional tensor with last dimension `num_classes`.
          labels: A tensor with shape and dtype as activations (onehot),
            or a long tensor of one dimension less than activations (pytorch standard)
          t1: Temperature 1 (< 1.0 for boundedness).
          t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
          label_smoothing: Label smoothing parameter between [0, 1). Default 0.0.
          num_iters: Number of iterations to run the method. Default 5.
          reduction: ``'none'`` | ``'mean'`` | ``'sum'``. Default ``'mean'``.
            ``'none'``: No reduction is applied, return shape is shape of
            activations without the last dimension.
            ``'mean'``: Loss is averaged over minibatch. Return shape (1,)
            ``'sum'``: Loss is summed over minibatch. Return shape (1,)
        Returns:
          A loss tensor.
        """

        activations = torch.sigmoid(activations)

        if len(labels.shape) < len(activations.shape):  # not one-hot
            labels_onehot = torch.zeros_like(activations)
            labels_onehot.scatter_(1, labels[..., None], 1)
        else:
            labels_onehot = labels

        if self.label_smoothing > 0:
            num_classes = labels_onehot.shape[-1]
            labels_onehot = (1 - self.label_smoothing * num_classes / (num_classes - 1)) \
                            * labels_onehot + \
                            self.label_smoothing / (num_classes - 1)

        probabilities = tempered_softmax(activations, self.t2, self.num_iters)

        loss_values = labels_onehot * log_t(labels_onehot + 1e-10, self.t1) \
                      - labels_onehot * log_t(probabilities, self.t1) \
                      - labels_onehot.pow(2.0 - self.t1) / (2.0 - self.t1) \
                      + probabilities.pow(2.0 - self.t1) / (2.0 - self.t1)
        loss_values = loss_values.sum(dim=-1)  # sum over classes

        if self.reduction == 'none':
            return loss_values
        if self.reduction == 'sum':
            return loss_values.sum()
        if self.reduction == 'mean':
            return loss_values.mean()


class TaylorSoftmax(nn.Module):
    '''
    This is the autograd version
    '''

    def __init__(self, dim=1, n=2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        '''
        usage similar to nn.Softmax:
            >>> mod = TaylorSoftmax(dim=1, n=4)
            >>> inten = torch.randn(1, 32, 64, 64)
            >>> out = mod(inten)
        '''
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.n + 1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=5, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.
                            smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data, self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class TaylorCrossEntropyLoss(nn.Module):
    def __init__(self, target_size, n=2, ignore_index=-1, reduction='mean', smoothing=0.05):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lab_smooth = LabelSmoothingLoss(target_size, smoothing=smoothing)

    def forward(self, logits, labels):
        logits = torch.sigmoid(logits)
        log_probs = self.taylor_softmax(logits).log()
        # loss = F.nll_loss(log_probs, labels, reduction=self.reduction,
        #        ignore_index=self.ignore_index)
        loss = self.lab_smooth(log_probs, labels.long())
        return loss


__CRITERIONS__ = {
    "BCEFocalLoss": BCEFocalLoss,
    "BiT": BiT,
    "TaylorCrossEntropyLoss": TaylorCrossEntropyLoss
}


def get_criterion(cfg):
    if hasattr(nn, cfg.loss.name):
        return getattr(nn, cfg.loss.name)(**cfg.loss.param)
    elif __CRITERIONS__.get(cfg.loss.name) is not None:
        return __CRITERIONS__[cfg.loss.name](**cfg.loss.param)
    else:
        raise NotImplementedError
