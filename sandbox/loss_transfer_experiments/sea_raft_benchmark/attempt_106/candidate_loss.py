import math
import torch
import torch.nn.functional as F

MAX_FLOW = 400
gamma = 0.85
use_var = True
var_min = 0
var_max = 10
epsilon = 1e-08


def _to_bhwc(x):
    if x is None:
        return None
    if not torch.is_tensor(x):
        return x
    if x.ndim != 4:
        return x
    if x.shape[-1] <= 8:
        return x
    if x.shape[1] <= 8:
        return x.permute(0, 2, 3, 1)
    return x


def _as_sequence(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _get_loss_input(kwargs, name):
    if name in kwargs and kwargs[name] is not None:
        return kwargs[name]
    loss_inputs = kwargs.get("loss_inputs", None)
    if isinstance(loss_inputs, dict):
        return loss_inputs.get(name, None)
    return None


def _broadcast_mask(mask, target, MAX_FLOW):
    if mask is None:
        mask = torch.ones(target.shape[0], target.shape[1], target.shape[2], 1, device=target.device, dtype=target.dtype)
    else:
        mask = _to_bhwc(mask)
        if mask.ndim == 3:
            mask = mask.unsqueeze(-1)
        if mask.ndim == 4 and mask.shape[-1] > 1:
            mask = mask[..., :1]
        mask = mask.to(device=target.device, dtype=target.dtype)

    mag = torch.sqrt(torch.sum(target * target, dim=-1, keepdim=True) + epsilon)
    valid = (mask >= 0.5).to(target.dtype) * (mag < MAX_FLOW).to(target.dtype)
    return valid


def _prepare_weight(weight, pred_item):
    weight = _to_bhwc(weight)
    if weight.ndim == 3:
        weight = weight.unsqueeze(-1)
    if weight.shape[-1] == 1:
        ordinary = weight
        ambiguous = torch.zeros_like(weight)
        weight = torch.cat([ordinary, ambiguous], dim=-1)
    elif weight.shape[-1] > 2:
        weight = weight[..., :2]
    return weight.to(device=pred_item.device, dtype=pred_item.dtype)


def _prepare_log_b(log_b, pred_item, use_var, var_min, var_max):
    log_b = _to_bhwc(log_b)
    if log_b.ndim == 3:
        log_b = log_b.unsqueeze(-1)
    if log_b.shape[-1] == 1:
        log_b = torch.cat([torch.zeros_like(log_b), log_b], dim=-1)
    elif log_b.shape[-1] > 2:
        log_b = log_b[..., :2]

    if not use_var:
        return torch.zeros_like(log_b, device=pred_item.device, dtype=pred_item.dtype)

    first = torch.clamp(log_b[..., :1], min=0.0, max=0.0)
    second = torch.clamp(log_b[..., 1:2], min=var_min, max=var_max)
    return torch.cat([first, second], dim=-1).to(device=pred_item.device, dtype=pred_item.dtype)


def _mol_nll(pred_item, target, valid, weight_item, log_b_item, epsilon):
    pred_item = _to_bhwc(pred_item).to(target.dtype)
    target = target.to(pred_item.dtype)
    valid = valid.to(pred_item.dtype)

    residual = (target - pred_item).abs().unsqueeze(-1)
    beta = log_b_item.unsqueeze(-2)
    log_component = weight_item.unsqueeze(-2) - math.log(2.0) - beta - residual * torch.exp(-beta)
    log_mix = torch.logsumexp(log_component, dim=-1)
    nll = -log_mix

    finite = torch.isfinite(nll).to(nll.dtype)
    valid_xy = valid.expand_as(nll) * finite
    denom = valid_xy.sum()
    if denom <= 0:
        return torch.zeros((), device=pred_item.device, dtype=pred_item.dtype)
    return (nll * valid_xy).sum() / (denom + epsilon)


def sandbox_loss(pred, target, mask=None, **kwargs):
    pred_seq = _as_sequence(pred)
    if len(pred_seq) == 0:
        pred_seq = [pred]

    target = _to_bhwc(target)
    target = target.to(dtype=pred_seq[-1].dtype, device=pred_seq[-1].device)
    valid = _broadcast_mask(mask, target, MAX_FLOW)

    weight = _get_loss_input(kwargs, "weight")
    log_b = _get_loss_input(kwargs, "log_b")

    weight_seq = _as_sequence(weight)
    log_b_seq = _as_sequence(log_b)

    if len(weight_seq) == 0:
        weight_seq = [torch.zeros_like(_to_bhwc(pred_seq[-1])[..., :2]) for _ in pred_seq]
    if len(log_b_seq) == 0:
        log_b_seq = [torch.zeros_like(_to_bhwc(pred_seq[-1])[..., :2]) for _ in pred_seq]

    if len(weight_seq) == 1 and len(pred_seq) > 1:
        weight_seq = weight_seq * len(pred_seq)
    if len(log_b_seq) == 1 and len(pred_seq) > 1:
        log_b_seq = log_b_seq * len(pred_seq)

    n_predictions = min(len(pred_seq), len(weight_seq), len(log_b_seq))
    flow_loss = torch.zeros((), device=target.device, dtype=target.dtype)

    for i in range(n_predictions):
        pred_item = _to_bhwc(pred_seq[i]).to(device=target.device, dtype=target.dtype)
        weight_item = _prepare_weight(weight_seq[i], pred_item)
        log_b_item = _prepare_log_b(log_b_seq[i], pred_item, use_var, var_min, var_max)
        i_weight = gamma ** (n_predictions - i - 1)
        loss_i = _mol_nll(pred_item, target, valid, weight_item, log_b_item, epsilon)
        flow_loss = flow_loss + i_weight * loss_i

    return flow_loss
