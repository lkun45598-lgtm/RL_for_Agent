import math
import torch
import torch.nn.functional as F

MAX_FLOW = 400
gamma = 0.85
use_var = True
var_min = 0
var_max = 10
epsilon = 1e-8


def _as_bhwc(x):
    if x is None:
        return None
    if not torch.is_tensor(x):
        return x
    if x.ndim == 4:
        if x.shape[-1] in (1, 2):
            return x
        if x.shape[1] in (1, 2):
            return x.permute(0, 2, 3, 1).contiguous()
    if x.ndim == 5:
        if x.shape[-1] in (1, 2):
            return x
        if x.shape[2] in (1, 2):
            return x.permute(0, 1, 3, 4, 2).contiguous()
    return x


def _ensure_sequence(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    if torch.is_tensor(x) and x.ndim == 5:
        return [x[:, i] for i in range(x.shape[1])]
    return [x]


def _to_tensor_like(x, ref):
    if torch.is_tensor(x):
        return x.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)


def _get_from_kwargs(kwargs, name):
    value = kwargs.get(name, None)
    if value is not None:
        return value
    loss_inputs = kwargs.get("loss_inputs", None)
    if isinstance(loss_inputs, dict):
        return loss_inputs.get(name, None)
    return None


def _expand_mask(mask, target):
    if mask is None:
        return torch.ones(target.shape[:-1] + (1,), device=target.device, dtype=target.dtype)

    mask = _as_bhwc(mask)
    mask = _to_tensor_like(mask, target)

    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(-1)
    elif mask.ndim == 3:
        if mask.shape == target.shape[:-1]:
            mask = mask.unsqueeze(-1)
        elif mask.shape[0] == target.shape[0] and mask.shape[1] == target.shape[1] and mask.shape[2] == 1:
            mask = mask
        else:
            mask = mask.unsqueeze(0)
    elif mask.ndim == 4 and mask.shape[-1] != 1:
        mask = mask[..., :1]

    if mask.shape[0] == 1 and target.shape[0] != 1:
        mask = mask.expand(target.shape[0], mask.shape[1], mask.shape[2], mask.shape[3])
    if mask.shape[1] == 1 and target.shape[1] != 1:
        mask = mask.expand(mask.shape[0], target.shape[1], mask.shape[2], mask.shape[3])
    if mask.shape[2] == 1 and target.shape[2] != 1:
        mask = mask.expand(mask.shape[0], mask.shape[1], target.shape[2], mask.shape[3])

    return mask


def _broadcast_param(param, pred_item, name):
    param = _as_bhwc(param)
    param = _to_tensor_like(param, pred_item)

    out_channels = 2 if name == "weight" else 1

    if param.ndim == 0:
        return param.view(1, 1, 1, 1).expand(pred_item.shape[0], pred_item.shape[1], pred_item.shape[2], out_channels)

    if param.ndim == 1:
        if param.shape[0] == out_channels:
            return param.view(1, 1, 1, out_channels).expand(pred_item.shape[0], pred_item.shape[1], pred_item.shape[2], out_channels)
        if param.shape[0] == 1:
            return param.view(1, 1, 1, 1).expand(pred_item.shape[0], pred_item.shape[1], pred_item.shape[2], out_channels)
        if param.shape[0] == pred_item.shape[0]:
            return param.view(pred_item.shape[0], 1, 1, 1).expand(pred_item.shape[0], pred_item.shape[1], pred_item.shape[2], out_channels)

    if param.ndim == 2:
        if param.shape == pred_item.shape[1:3]:
            return param.unsqueeze(0).unsqueeze(-1).expand(pred_item.shape[0], pred_item.shape[1], pred_item.shape[2], out_channels)
        if param.shape[-1] == out_channels:
            return param.reshape(1, 1, 1, out_channels).expand(pred_item.shape[0], pred_item.shape[1], pred_item.shape[2], out_channels)

    if param.ndim == 3:
        if param.shape == pred_item.shape[:-1]:
            return param.unsqueeze(-1).expand(pred_item.shape[0], pred_item.shape[1], pred_item.shape[2], out_channels)
        if param.shape[0] == pred_item.shape[0] and param.shape[1] == pred_item.shape[1] and param.shape[2] in (1, out_channels):
            return param.unsqueeze(2).expand(pred_item.shape[0], pred_item.shape[1], pred_item.shape[2], param.shape[2])
        if param.shape[0] == pred_item.shape[0] and param.shape[2] in (1, out_channels):
            return param.unsqueeze(1).expand(pred_item.shape[0], pred_item.shape[1], pred_item.shape[2], param.shape[2])

    if param.ndim == 4:
        if param.shape[0] == 1 and pred_item.shape[0] != 1:
            param = param.expand(pred_item.shape[0], param.shape[1], param.shape[2], param.shape[3])
        if param.shape[1] == 1 and pred_item.shape[1] != 1:
            param = param.expand(param.shape[0], pred_item.shape[1], param.shape[2], param.shape[3])
        if param.shape[2] == 1 and pred_item.shape[2] != 1:
            param = param.expand(param.shape[0], param.shape[1], pred_item.shape[2], param.shape[3])
        if param.shape[-1] == 1 and out_channels != 1:
            param = param.expand(param.shape[0], param.shape[1], param.shape[2], out_channels)
        if param.shape[-1] >= out_channels:
            return param[..., :out_channels]

    raise ValueError(name + " must be broadcastable to BHWC")


def _prepare_param_sequence(param, pred_seq, name):
    if param is None:
        return None
    param_seq = _ensure_sequence(param)
    if len(param_seq) == 1 and len(pred_seq) > 1:
        param_seq = param_seq * len(pred_seq)
    if len(param_seq) != len(pred_seq):
        raise ValueError(name + " sequence length must match pred sequence length")
    return [_broadcast_param(item, pred_item, name) for item, pred_item in zip(param_seq, pred_seq)]


def _masked_l1_fallback(pred, target, valid_mask):
    abs_error = (target - pred).abs()
    if valid_mask.shape[-1] == 1:
        valid_mask = valid_mask.expand_as(abs_error)
    finite_mask = torch.isfinite(abs_error).to(dtype=abs_error.dtype)
    final_mask = valid_mask * finite_mask
    denom = final_mask.sum().clamp_min(epsilon)
    return (abs_error * final_mask).sum() / denom


def _mixture_laplace_nll(pred, target, weight, log_b, valid_mask):
    abs_error = (target - pred).abs()

    log_b = _broadcast_param(log_b, pred, "log_b")
    if use_var:
        log_b = torch.clamp(log_b, min=var_min, max=var_max)
    else:
        log_b = torch.zeros_like(log_b)
    beta2 = log_b.expand_as(abs_error)

    weight = _broadcast_param(weight, pred, "weight")
    if weight.shape[-1] == 1:
        weight = torch.cat([weight, torch.zeros_like(weight)], dim=-1)

    term2 = abs_error.unsqueeze(-1) * torch.exp(-torch.stack([torch.zeros_like(beta2), beta2], dim=-1))
    term1 = weight.unsqueeze(-2) - math.log(2.0) - torch.stack([torch.zeros_like(beta2), beta2], dim=-1)
    log_norm = torch.logsumexp(weight, dim=-1, keepdim=True)
    log_prob = torch.logsumexp(term1 - term2, dim=-1) - log_norm
    nll = -log_prob

    if valid_mask.shape[-1] == 1:
        valid_mask = valid_mask.expand_as(nll)

    finite_mask = torch.isfinite(nll).to(dtype=nll.dtype)
    final_mask = valid_mask * finite_mask
    denom = final_mask.sum().clamp_min(epsilon)
    return (nll * final_mask).sum() / denom


def sandbox_loss(pred, target, mask=None, **kwargs):
    pred_seq = [_as_bhwc(item) for item in _ensure_sequence(pred)]
    target_seq = [_as_bhwc(item) for item in _ensure_sequence(target)]

    if len(target_seq) == 1 and len(pred_seq) > 1:
        target_seq = target_seq * len(pred_seq)
    if len(target_seq) != len(pred_seq):
        raise ValueError("target sequence length must match pred sequence length")

    weight = _get_from_kwargs(kwargs, "weight")
    log_b = _get_from_kwargs(kwargs, "log_b")

    weight_seq = _prepare_param_sequence(weight, pred_seq, "weight") if weight is not None else None
    log_b_seq = _prepare_param_sequence(log_b, pred_seq, "log_b") if log_b is not None else None

    total_loss = pred_seq[0].new_zeros(())
    n_predictions = len(pred_seq)

    for i, (pred_item, target_item) in enumerate(zip(pred_seq, target_seq)):
        valid_mask = _expand_mask(mask, target_item)
        target_valid = torch.isfinite(target_item).all(dim=-1, keepdim=True).to(dtype=target_item.dtype)
        pred_valid = torch.isfinite(pred_item).all(dim=-1, keepdim=True).to(dtype=target_item.dtype)
        displacement = torch.sqrt(torch.sum(target_item * target_item, dim=-1, keepdim=True))
        flow_valid = (displacement < MAX_FLOW).to(dtype=target_item.dtype)
        valid_mask = valid_mask * target_valid * pred_valid * flow_valid

        if weight_seq is not None and log_b_seq is not None:
            weight_item = weight_seq[i]
            log_b_item = log_b_seq[i]
            param_valid = torch.isfinite(weight_item).all(dim=-1, keepdim=True).to(dtype=target_item.dtype)
            param_valid = param_valid * torch.isfinite(log_b_item).all(dim=-1, keepdim=True).to(dtype=target_item.dtype)
            item_loss = _mixture_laplace_nll(pred_item, target_item, weight_item, log_b_item, valid_mask * param_valid)
        else:
            item_loss = _masked_l1_fallback(pred_item, target_item, valid_mask)

        i_weight = gamma ** (n_predictions - i - 1)
        total_loss = total_loss + i_weight * item_loss

    return total_loss
