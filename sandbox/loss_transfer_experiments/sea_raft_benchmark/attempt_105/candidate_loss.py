import math
import torch
import torch.nn.functional as F

MAX_FLOW = 400
gamma = 0.85
use_var = True
var_min = 0
var_max = 10
epsilon = 1e-8


def _to_bhwc(x):
    if x is None:
        return None
    if not torch.is_tensor(x):
        return None
    if x.dim() == 3:
        x = x.unsqueeze(-1)
    if x.dim() < 4:
        return x
    if x.shape[-1] <= 8:
        return x
    if x.shape[1] <= 8:
        return x.permute(0, 2, 3, 1).contiguous()
    return x


def _extract_sequence(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _broadcast_channels(x, channels):
    if x is None:
        return None
    if x.dim() == 3:
        x = x.unsqueeze(-1)
    if x.shape[-1] == channels:
        return x
    if x.shape[-1] == 1:
        return x.expand(*x.shape[:-1], channels)
    if channels == 1:
        return x.mean(dim=-1, keepdim=True)
    if x.shape[-1] > channels:
        return x[..., :channels]
    repeat = (channels + x.shape[-1] - 1) // x.shape[-1]
    x = x.repeat_interleave(repeat, dim=-1)
    return x[..., :channels]


def _resolve_extra(name, kwargs):
    if name in kwargs and kwargs[name] is not None:
        return kwargs[name]
    loss_inputs = kwargs.get("loss_inputs", None)
    if isinstance(loss_inputs, dict):
        return loss_inputs.get(name, None)
    return None


def _logsumexp_two(a, b):
    stacked = torch.stack([a, b], dim=-1)
    return torch.logsumexp(stacked, dim=-1)


def _masked_mean(x, valid_mask):
    valid_mask = valid_mask.to(dtype=x.dtype)
    denom = valid_mask.sum().clamp_min(1.0)
    return (x * valid_mask).sum() / denom


def _mol_nll_single(pred, target, mask, weight, log_b):
    pred = _to_bhwc(pred)
    target = _to_bhwc(target)
    weight = _to_bhwc(weight)
    log_b = _to_bhwc(log_b)

    if pred is None or target is None:
        ref = pred if pred is not None else target
        if ref is None:
            return torch.tensor(0.0)
        return ref.new_zeros(())

    channels = pred.shape[-1]
    target = _broadcast_channels(target, channels)

    if weight is None:
        weight = pred.new_zeros(pred.shape[:-1] + (2,))
    else:
        weight = _broadcast_channels(weight, 2)

    if log_b is None:
        log_b = pred.new_zeros(pred.shape[:-1] + (1,))
    log_b = _broadcast_channels(log_b, 1)

    if use_var:
        log_b = torch.clamp(log_b, min=var_min, max=var_max)
    else:
        log_b = torch.zeros_like(log_b)

    residual = (target - pred).abs()
    log_component_1 = weight[..., 0:1] - math.log(2.0) - residual
    log_component_2 = weight[..., 1:2] - math.log(2.0) - log_b - residual * torch.exp(-log_b)
    log_mix = _logsumexp_two(log_component_1, log_component_2)
    nll = -log_mix

    finite_mask = torch.isfinite(nll).all(dim=-1, keepdim=True)
    finite_mask = finite_mask & torch.isfinite(pred).all(dim=-1, keepdim=True)
    finite_mask = finite_mask & torch.isfinite(target).all(dim=-1, keepdim=True)
    finite_mask = finite_mask & torch.isfinite(weight).all(dim=-1, keepdim=True)
    finite_mask = finite_mask & torch.isfinite(log_b).all(dim=-1, keepdim=True)

    mag = torch.sqrt(torch.sum(target * target, dim=-1, keepdim=True) + epsilon)
    flow_valid = mag < MAX_FLOW

    if mask is None:
        mask = pred.new_ones(pred.shape[:-1] + (1,))
    else:
        mask = _to_bhwc(mask)
        if mask.dim() == 3:
            mask = mask.unsqueeze(-1)
        if mask.shape[-1] != 1:
            mask = mask[..., :1]
        mask = mask.to(dtype=pred.dtype)

    valid_mask = (mask >= 0.5) & flow_valid & finite_mask
    valid_mask = valid_mask.expand_as(nll)

    return _masked_mean(nll, valid_mask)


def sandbox_loss(pred, target, mask=None, **kwargs):
    pred_seq = _extract_sequence(pred)
    weight_seq = _extract_sequence(_resolve_extra("weight", kwargs))
    log_b_seq = _extract_sequence(_resolve_extra("log_b", kwargs))

    if len(pred_seq) == 0:
        if torch.is_tensor(target):
            return target.new_zeros(())
        return torch.tensor(0.0)

    n_predictions = len(pred_seq)
    total_loss = None

    for i, pred_i in enumerate(pred_seq):
        if len(weight_seq) == 0:
            weight_i = None
        else:
            weight_i = weight_seq[i] if i < len(weight_seq) else weight_seq[-1]

        if len(log_b_seq) == 0:
            log_b_i = None
        else:
            log_b_i = log_b_seq[i] if i < len(log_b_seq) else log_b_seq[-1]

        i_weight = gamma ** (n_predictions - i - 1)
        loss_i = _mol_nll_single(pred_i, target, mask, weight_i, log_b_i)
        weighted_loss_i = loss_i * i_weight

        if total_loss is None:
            total_loss = weighted_loss_i
        else:
            total_loss = total_loss + weighted_loss_i

    if total_loss is None:
        ref = pred_seq[0] if len(pred_seq) > 0 and torch.is_tensor(pred_seq[0]) else None
        if ref is not None:
            return ref.new_zeros(())
        if torch.is_tensor(target):
            return target.new_zeros(())
        return torch.tensor(0.0)

    return total_loss
