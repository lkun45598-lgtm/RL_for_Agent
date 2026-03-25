import math
import torch
import torch.nn.functional as F


def _get_nested(kwargs, key):
    if key in kwargs and kwargs[key] is not None:
        return kwargs[key]
    loss_inputs = kwargs.get("loss_inputs", None)
    if isinstance(loss_inputs, dict):
        value = loss_inputs.get(key, None)
        if value is not None:
            return value
    return None


def _to_bhwc(x):
    if x is None:
        return None
    if not torch.is_tensor(x):
        return None
    if x.ndim == 3:
        x = x.unsqueeze(-1)
    return x


def _normalize_sequence(x):
    if x is None:
        return None, False
    if isinstance(x, (list, tuple)):
        return list(x), True
    return [x], False


def _broadcast_mask(mask, target):
    if mask is None:
        return torch.ones(target.shape[:-1] + (1,), dtype=target.dtype, device=target.device)

    mask = _to_bhwc(mask)
    if mask is None:
        return torch.ones(target.shape[:-1] + (1,), dtype=target.dtype, device=target.device)

    if mask.shape[0] != target.shape[0]:
        raise ValueError("mask batch dimension must match target")

    if mask.ndim != 4:
        raise ValueError("mask must be BHWC-compatible")

    if mask.shape[1] != target.shape[1] or mask.shape[2] != target.shape[2]:
        raise ValueError("mask spatial dimensions must match target")

    if mask.shape[-1] == 1:
        return mask.to(dtype=target.dtype)

    mask = mask.to(dtype=target.dtype)
    mask = torch.amax(mask, dim=-1, keepdim=True)
    return mask


def _match_last_dim(x, channels):
    if x is None:
        return None
    if x.shape[-1] == channels:
        return x
    if x.shape[-1] == 1:
        return x.expand(*x.shape[:-1], channels)
    raise ValueError("last dimension is incompatible with target channels")


def _reduce_weight_logits(weight):
    if weight is None:
        return None
    last_dim = weight.shape[-1]
    if last_dim == 1:
        ordinary_logit = weight
        ambiguous_logit = torch.zeros_like(weight)
        return ordinary_logit, ambiguous_logit
    if last_dim >= 2:
        ordinary_logit = weight[..., :1]
        ambiguous_logit = weight[..., 1:2]
        return ordinary_logit, ambiguous_logit
    raise ValueError("weight must have at least one channel")


def _magnitude_mask(target, max_flow):
    magnitude = torch.sqrt(torch.sum(target * target, dim=-1, keepdim=True))
    return (magnitude < max_flow).to(dtype=target.dtype)


def _safe_mean(values, valid_mask, eps):
    valid_mask = valid_mask.to(dtype=values.dtype)
    denom = valid_mask.sum()
    if denom.item() <= 0:
        return values.new_zeros(())
    return (values * valid_mask).sum() / denom.clamp_min(eps)


def _mol_nll_single(pred, target, mask, log_b, weight, gamma_weight, max_flow, epsilon, var_min, var_max):
    pred = _to_bhwc(pred)
    target = _to_bhwc(target)
    if pred is None or target is None:
        raise ValueError("pred and target must be torch tensors")
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError("pred and target must be BHWC tensors")
    if pred.shape != target.shape:
        raise ValueError("pred and target shapes must match")

    channels = target.shape[-1]
    if channels <= 0:
        raise ValueError("target must have a positive channel dimension")

    if log_b is None:
        log_b = torch.zeros(target.shape[:-1] + (1,), dtype=target.dtype, device=target.device)
    else:
        log_b = _to_bhwc(log_b)
        if log_b is None or log_b.ndim != 4:
            raise ValueError("log_b must be BHWC-compatible")
        if log_b.shape[:3] != target.shape[:3]:
            raise ValueError("log_b spatial dimensions must match target")

    if weight is None:
        weight = torch.zeros(target.shape[:-1] + (2,), dtype=target.dtype, device=target.device)
    else:
        weight = _to_bhwc(weight)
        if weight is None or weight.ndim != 4:
            raise ValueError("weight must be BHWC-compatible")
        if weight.shape[:3] != target.shape[:3]:
            raise ValueError("weight spatial dimensions must match target")

    valid_mask = _broadcast_mask(mask, target)
    valid_mask = valid_mask * _magnitude_mask(target, max_flow)

    finite_mask = torch.isfinite(pred).all(dim=-1, keepdim=True)
    finite_mask = finite_mask & torch.isfinite(target).all(dim=-1, keepdim=True)
    finite_mask = finite_mask & torch.isfinite(log_b).all(dim=-1, keepdim=True)
    finite_mask = finite_mask & torch.isfinite(weight).all(dim=-1, keepdim=True)
    valid_mask = valid_mask * finite_mask.to(dtype=target.dtype)

    log_b = torch.clamp(log_b, min=var_min, max=var_max)
    log_b = _match_last_dim(log_b, channels)

    ordinary_logit, ambiguous_logit = _reduce_weight_logits(weight)
    ordinary_logit = ordinary_logit.to(dtype=target.dtype)
    ambiguous_logit = ambiguous_logit.to(dtype=target.dtype)

    abs_error = torch.abs(target - pred)

    log_component_ordinary = ordinary_logit - math.log(2.0) - abs_error
    inv_scale = torch.exp(-log_b)
    log_component_ambiguous = ambiguous_logit - math.log(2.0) - log_b - abs_error * inv_scale

    stacked = torch.stack([log_component_ordinary, log_component_ambiguous], dim=-1)
    log_prob = torch.logsumexp(stacked, dim=-1)
    nll_per_dim = -log_prob

    per_pixel_nll = nll_per_dim.mean(dim=-1, keepdim=True)
    valid_values = torch.isfinite(per_pixel_nll)
    valid_mask = valid_mask * valid_values.to(dtype=target.dtype)
    per_pixel_nll = torch.where(valid_values, per_pixel_nll, torch.zeros_like(per_pixel_nll))

    return gamma_weight * _safe_mean(per_pixel_nll, valid_mask, epsilon)


def sandbox_loss(pred, target, mask=None, **kwargs):
    gamma = float(kwargs.get("gamma", 0.85))
    max_flow = float(kwargs.get("MAX_FLOW", kwargs.get("max_flow", 400.0)))
    epsilon = float(kwargs.get("epsilon", 1e-8))
    use_var = bool(kwargs.get("use_var", True))
    var_min = float(kwargs.get("var_min", 0.0 if use_var else 0.0))
    var_max = float(kwargs.get("var_max", 10.0 if use_var else 0.0))

    pred_list, _ = _normalize_sequence(pred)
    log_b_raw = _get_nested(kwargs, "log_b")
    weight_raw = _get_nested(kwargs, "weight")
    log_b_list, has_log_b = _normalize_sequence(log_b_raw)
    weight_list, has_weight = _normalize_sequence(weight_raw)

    if not has_log_b:
        log_b_list = [None] * len(pred_list)
    if not has_weight:
        weight_list = [None] * len(pred_list)

    if len(log_b_list) != len(pred_list):
        if len(log_b_list) == 1:
            log_b_list = log_b_list * len(pred_list)
        else:
            raise ValueError("log_b sequence length must match pred sequence length")

    if len(weight_list) != len(pred_list):
        if len(weight_list) == 1:
            weight_list = weight_list * len(pred_list)
        else:
            raise ValueError("weight sequence length must match pred sequence length")

    total = None
    n_predictions = len(pred_list)
    for index, pred_i in enumerate(pred_list):
        gamma_weight = gamma ** (n_predictions - index - 1)
        loss_i = _mol_nll_single(
            pred=pred_i,
            target=target,
            mask=mask,
            log_b=log_b_list[index],
            weight=weight_list[index],
            gamma_weight=gamma_weight,
            max_flow=max_flow,
            epsilon=epsilon,
            var_min=var_min,
            var_max=var_max,
        )
        total = loss_i if total is None else total + loss_i

    if total is None:
        target_tensor = _to_bhwc(target)
        if target_tensor is None:
            raise ValueError("target must be a torch tensor")
        return target_tensor.new_zeros(())

    return total
