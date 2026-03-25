import math
import torch
import torch.nn.functional as F


MAX_FLOW = 400
gamma = 0.85
use_var = True
var_min = 0.0
var_max = 10.0
epsilon = 1e-8


def _get_loss_input(kwargs, name):
    value = kwargs.get(name)
    if value is not None:
        return value
    loss_inputs = kwargs.get("loss_inputs", {})
    if isinstance(loss_inputs, dict):
        return loss_inputs.get(name)
    return None


def _to_sequence(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _prepare_mask(mask, target):
    if mask is None:
        return torch.ones(target.shape[0], target.shape[1], target.shape[2], 1, device=target.device, dtype=target.dtype)
    if mask.dim() == 3:
        mask = mask.unsqueeze(-1)
    if mask.dim() != 4:
        raise ValueError("mask must be None or BHWC/BHW")
    return mask.to(device=target.device, dtype=target.dtype)


def _reduce_mask(mask):
    if mask.shape[-1] == 1:
        return mask
    return mask.amax(dim=-1, keepdim=True)


def _valid_flow_mask(target, mask):
    mag = torch.sqrt(torch.sum(target * target, dim=-1, keepdim=True) + epsilon)
    valid = (mag < MAX_FLOW).to(target.dtype)
    return _reduce_mask(mask) * valid


def _reshape_component(x, reference):
    if x is None:
        return None
    if not torch.is_tensor(x):
        raise TypeError("loss input must be a torch tensor")
    if x.dim() == 3:
        x = x.unsqueeze(-1)
    if x.dim() != 4:
        raise ValueError("loss input tensor must be BHWC or BHW")
    if x.shape[0] != reference.shape[0] or x.shape[1] != reference.shape[1] or x.shape[2] != reference.shape[2]:
        raise ValueError("loss input spatial shape must match pred/target")
    return x.to(device=reference.device, dtype=reference.dtype)


def _expand_weight(weight, pred):
    if weight.shape[-1] == 1:
        weight = torch.cat([weight, torch.zeros_like(weight)], dim=-1)
    if weight.shape[-1] != 2:
        raise ValueError("weight must have 1 or 2 channels in BHWC format")
    return weight


def _expand_log_b(log_b, pred):
    channels = pred.shape[-1]
    if log_b.shape[-1] == 1 and channels > 1:
        log_b = log_b.expand(-1, -1, -1, channels)
    if log_b.shape[-1] != channels:
        raise ValueError("log_b channel count must match pred channels or be 1")
    return log_b


def _mixture_laplace_nll(pred, target, weight, log_b, valid_mask):
    log_b = torch.clamp(log_b, min=var_min, max=var_max)
    abs_error = (target - pred).abs()
    ordinary_log_prob = weight[..., :1] - math.log(2.0) - abs_error
    ambiguous_log_prob = weight[..., 1:2] - math.log(2.0) - log_b - abs_error * torch.exp(-log_b)
    stacked_log_prob = torch.cat([ordinary_log_prob, ambiguous_log_prob], dim=-1)
    log_partition = torch.logsumexp(weight, dim=-1, keepdim=True)
    per_dim_nll = log_partition - torch.logsumexp(stacked_log_prob, dim=-1, keepdim=True)
    weighted = per_dim_nll * valid_mask
    denom = valid_mask.sum() * pred.shape[-1]
    denom = torch.clamp(denom, min=1.0)
    return weighted.sum() / denom


def sandbox_loss(pred, target, mask=None, **kwargs):
    pred_seq = _to_sequence(pred)
    target_seq = _to_sequence(target)
    if len(target_seq) == 1 and len(pred_seq) > 1:
        target_seq = target_seq * len(pred_seq)
    if len(pred_seq) != len(target_seq):
        raise ValueError("pred and target sequence lengths must match, or target must be a single tensor")

    weight = _get_loss_input(kwargs, "weight")
    log_b = _get_loss_input(kwargs, "log_b")
    if weight is None or log_b is None:
        raise ValueError("sandbox_loss requires weight and log_b in kwargs or loss_inputs")

    weight_seq = _to_sequence(weight)
    log_b_seq = _to_sequence(log_b)
    if len(weight_seq) == 1 and len(pred_seq) > 1:
        weight_seq = weight_seq * len(pred_seq)
    if len(log_b_seq) == 1 and len(pred_seq) > 1:
        log_b_seq = log_b_seq * len(pred_seq)
    if len(weight_seq) != len(pred_seq) or len(log_b_seq) != len(pred_seq):
        raise ValueError("weight/log_b sequence lengths must match pred")

    gamma_value = kwargs.get("gamma", gamma)
    total_loss = None
    num_predictions = len(pred_seq)

    for index, (pred_i, target_i, weight_i, log_b_i) in enumerate(zip(pred_seq, target_seq, weight_seq, log_b_seq)):
        if pred_i.dim() != 4 or target_i.dim() != 4:
            raise ValueError("pred and target must be BHWC tensors")
        pred_i = pred_i.to(dtype=target_i.dtype)
        mask_i = _prepare_mask(mask, target_i)
        valid_mask = _valid_flow_mask(target_i, mask_i)
        weight_i = _expand_weight(_reshape_component(weight_i, pred_i), pred_i)
        log_b_i = _expand_log_b(_reshape_component(log_b_i, pred_i), pred_i)
        loss_i = _mixture_laplace_nll(pred_i, target_i, weight_i, log_b_i, valid_mask)
        i_weight = gamma_value ** (num_predictions - index - 1)
        total_loss = loss_i * i_weight if total_loss is None else total_loss + loss_i * i_weight

    return total_loss
