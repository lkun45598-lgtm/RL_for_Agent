import math
import torch
import torch.nn.functional as F

MAX_FLOW = 400
gamma = 0.85
use_var = True
var_min = 0
var_max = 10
epsilon = 1e-08


def _as_bhwc(x):
    if x is None:
        return None
    if not torch.is_tensor(x):
        raise TypeError("Expected tensor input")
    if x.dim() == 3:
        x = x.unsqueeze(-1)
    if x.dim() != 4:
        raise ValueError("Expected BHWC tensor")
    return x


def _ensure_sequence(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _reduce_channels(x):
    if x.shape[-1] == 1:
        return x[..., 0]
    return x.mean(dim=-1)


def _resolve_loss_input(name, kwargs, length=None, reference=None):
    value = kwargs.get(name, None)
    if value is None:
        loss_inputs = kwargs.get("loss_inputs", {})
        if isinstance(loss_inputs, dict):
            value = loss_inputs.get(name, None)
    if value is None:
        return None
    value = _ensure_sequence(value)
    value = [_as_bhwc(v) for v in value]
    if length is not None and len(value) == 1 and length > 1:
        value = value * length
    if length is not None and len(value) != length:
        raise ValueError(f"{name} sequence length mismatch")
    if reference is not None:
        aligned = []
        for item, ref in zip(value, reference):
            if item.shape[:3] != ref.shape[:3]:
                raise ValueError(f"{name} spatial shape mismatch")
            aligned.append(item)
        value = aligned
    return value


def _resolve_mask(mask, target):
    if mask is None:
        return torch.ones(target.shape[0], target.shape[1], target.shape[2], device=target.device, dtype=target.dtype)
    mask = _as_bhwc(mask)
    if mask.shape[:3] != target.shape[:3]:
        raise ValueError("mask spatial shape mismatch")
    if mask.shape[-1] > 1:
        mask = _reduce_channels(mask)
    else:
        mask = mask[..., 0]
    return (mask > 0.5).to(dtype=target.dtype)


def _prepare_weight(weight, pred):
    if weight is None:
        weight = torch.zeros(pred.shape[0], pred.shape[1], pred.shape[2], 2, device=pred.device, dtype=pred.dtype)
    else:
        weight = _as_bhwc(weight)
    if weight.shape[:3] != pred.shape[:3]:
        raise ValueError("weight spatial shape mismatch")
    if weight.shape[-1] == 1:
        weight = torch.cat([weight, torch.zeros_like(weight)], dim=-1)
    elif weight.shape[-1] < 2:
        raise ValueError("weight channel mismatch")
    elif weight.shape[-1] > 2:
        weight = weight[..., :2]
    return weight


def _prepare_log_b(log_b, pred):
    if log_b is None:
        log_b = torch.zeros(pred.shape[0], pred.shape[1], pred.shape[2], 2, device=pred.device, dtype=pred.dtype)
    else:
        log_b = _as_bhwc(log_b)
    if log_b.shape[:3] != pred.shape[:3]:
        raise ValueError("log_b spatial shape mismatch")
    if log_b.shape[-1] == 1:
        log_b = torch.cat([log_b, log_b], dim=-1)
    elif log_b.shape[-1] < 2:
        raise ValueError("log_b channel mismatch")
    elif log_b.shape[-1] > 2:
        log_b = log_b[..., :2]
    if use_var:
        log_b_0 = torch.clamp(log_b[..., 0], min=0.0, max=var_max)
        log_b_1 = torch.clamp(log_b[..., 1], min=var_min, max=0.0)
        log_b = torch.stack([log_b_0, log_b_1], dim=-1)
    else:
        log_b = torch.zeros_like(log_b)
    return log_b


def _masked_mix_laplace_nll(pred, target, mask, weight, log_b):
    pred = _as_bhwc(pred)
    target = _as_bhwc(target)
    if pred.shape != target.shape:
        raise ValueError("pred and target shape mismatch")
    if pred.shape[-1] != 2:
        raise ValueError("pred and target must have 2 channels in BHWC format")

    valid = _resolve_mask(mask, target)
    mag = torch.sqrt(torch.sum(target * target, dim=-1) + epsilon)
    valid = valid * (mag < MAX_FLOW).to(dtype=target.dtype)

    weight = _prepare_weight(weight, pred)
    log_b = _prepare_log_b(log_b, pred)

    finite_mask = torch.isfinite(pred).all(dim=-1)
    finite_mask = finite_mask & torch.isfinite(target).all(dim=-1)
    finite_mask = finite_mask & torch.isfinite(weight).all(dim=-1)
    finite_mask = finite_mask & torch.isfinite(log_b).all(dim=-1)

    error = torch.abs(target - pred)

    beta1 = pred.new_zeros(pred.shape[0], pred.shape[1], pred.shape[2], 1)
    beta2 = log_b[..., 1:2]
    log_alpha = weight[..., 0:1]
    log_one_minus_alpha = weight[..., 1:2]

    component0 = log_alpha - math.log(2.0) - beta1 - error * torch.exp(-beta1)
    component1 = log_one_minus_alpha - math.log(2.0) - beta2 - error * torch.exp(-beta2)

    stacked = torch.stack([component0, component1], dim=-1)
    mixture_log_prob = torch.logsumexp(stacked, dim=-1)
    nll = -mixture_log_prob.sum(dim=-1)

    valid = valid * finite_mask.to(dtype=target.dtype)
    valid = valid * torch.isfinite(nll).to(dtype=target.dtype)

    denom = valid.sum().clamp_min(1.0)
    if valid.sum() <= 0:
        return pred.new_zeros(())
    return (nll * valid).sum() / denom


def sandbox_loss(pred, target, mask=None, **kwargs):
    pred = _ensure_sequence(pred)
    target = _ensure_sequence(target)
    pred = [_as_bhwc(p) for p in pred]
    target = [_as_bhwc(t) for t in target]

    if len(target) == 1 and len(pred) > 1:
        target = target * len(pred)
    if len(pred) != len(target):
        raise ValueError("pred and target sequence length mismatch")

    weight = _resolve_loss_input("weight", kwargs, length=len(pred), reference=pred)
    log_b = _resolve_loss_input("log_b", kwargs, length=len(pred), reference=pred)

    total = pred[0].new_zeros(())
    n_predictions = len(pred)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        total = total + i_weight * _masked_mix_laplace_nll(
            pred=pred[i],
            target=target[i],
            mask=mask,
            weight=None if weight is None else weight[i],
            log_b=None if log_b is None else log_b[i],
        )
    return total
