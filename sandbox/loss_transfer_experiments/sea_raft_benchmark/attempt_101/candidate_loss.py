import math
import torch
import torch.nn.functional as F

MAX_FLOW = 400.0
GAMMA = 0.85
EPS = 1e-8
BETA_MAX = 10.0


def _to_bhwc(x):
    if x is None:
        return None
    if not torch.is_tensor(x):
        return x
    if x.dim() < 2:
        return x
    if x.dim() == 4:
        if x.shape[-1] <= 8:
            return x
        if x.shape[1] <= 8:
            return x.permute(0, 2, 3, 1)
        return x
    return x


def _ensure_tensor_on_ref(x, ref):
    if torch.is_tensor(x):
        return x.to(device=ref.device, dtype=ref.dtype)
    return torch.tensor(x, device=ref.device, dtype=ref.dtype)


def _as_sequence(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _extract_loss_input(kwargs, name):
    if name in kwargs and kwargs[name] is not None:
        return kwargs[name]
    loss_inputs = kwargs.get("loss_inputs", None)
    if isinstance(loss_inputs, dict):
        value = loss_inputs.get(name, None)
        if value is not None:
            return value
    return None


def _prepare_mask(mask, target, max_flow):
    ref = target
    if mask is None:
        valid = torch.ones(ref.shape[:-1], device=ref.device, dtype=torch.bool)
    else:
        mask = _to_bhwc(mask)
        if mask.dim() == 4 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        elif mask.dim() == 4:
            mask = mask[..., 0]
        valid = mask > 0.5
    mag = torch.sqrt(torch.sum(target * target, dim=-1))
    valid = valid & (mag < max_flow)
    return valid


def _normalize_component_channels(x, ref_channels):
    x = _to_bhwc(x)
    if x.dim() == 3:
        x = x.unsqueeze(-1)
    channels = x.shape[-1]
    if channels == ref_channels:
        return x
    if channels == 1:
        return x.expand(*x.shape[:-1], ref_channels)
    if channels == 2 and ref_channels == 1:
        return torch.logsumexp(x, dim=-1, keepdim=True)
    if channels > ref_channels:
        return x[..., :ref_channels]
    repeat_factor = int(math.ceil(float(ref_channels) / float(channels)))
    x = x.repeat(*([1] * (x.dim() - 1)), repeat_factor)
    return x[..., :ref_channels]


def _logsumexp_two(a, b):
    m = torch.maximum(a, b)
    return m + torch.log(torch.exp(a - m) + torch.exp(b - m) + EPS)


def _mol_nll_single(pred, target, weight, log_b, valid):
    pred = _to_bhwc(pred)
    target = _to_bhwc(target)
    channels = pred.shape[-1]

    weight = _normalize_component_channels(weight, 2)
    log_b = _normalize_component_channels(log_b, 1)
    log_b = torch.clamp(log_b, min=0.0, max=BETA_MAX)

    abs_err = torch.abs(target - pred)

    comp1 = weight[..., 0:1] - math.log(2.0) - abs_err
    inv_b2 = torch.exp(-log_b)
    comp2 = weight[..., 1:2] - math.log(2.0) - log_b - abs_err * inv_b2

    log_mix = _logsumexp_two(comp1, comp2)
    log_norm = torch.logsumexp(weight, dim=-1, keepdim=True)
    nll = log_norm - log_mix

    valid = valid.unsqueeze(-1)
    finite = torch.isfinite(nll)
    valid = valid & finite

    denom = valid.to(nll.dtype).sum()
    if denom <= 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return (nll * valid.to(nll.dtype)).sum() / denom


def sandbox_loss(pred, target, mask=None, **kwargs):
    target = _to_bhwc(target)
    pred_seq = _as_sequence(pred)

    weight_in = _extract_loss_input(kwargs, "weight")
    log_b_in = _extract_loss_input(kwargs, "log_b")
    weight_seq = _as_sequence(weight_in)
    log_b_seq = _as_sequence(log_b_in)

    gamma = kwargs.get("gamma", GAMMA)
    max_flow = kwargs.get("max_flow", kwargs.get("MAX_FLOW", MAX_FLOW))
    gamma = float(gamma)
    max_flow = float(max_flow)

    if weight_seq is None or log_b_seq is None:
        if torch.is_tensor(pred) and pred.dim() == 4 and pred.shape[-1] >= target.shape[-1] + 3:
            base_pred = pred[..., :target.shape[-1]]
            aux = pred[..., target.shape[-1]:]
            pred_seq = [base_pred]
            weight_seq = [aux[..., :2]]
            log_b_seq = [aux[..., 2:3]]
        else:
            diff = torch.abs(_to_bhwc(pred_seq[-1]) - target)
            valid = _prepare_mask(mask, target, max_flow)
            valid = valid.unsqueeze(-1) & torch.isfinite(diff)
            denom = valid.to(diff.dtype).sum()
            if denom <= 0:
                return torch.zeros((), device=target.device, dtype=target.dtype)
            return (diff * valid.to(diff.dtype)).sum() / denom

    n_predictions = min(len(pred_seq), len(weight_seq), len(log_b_seq))
    if n_predictions == 0:
        return torch.zeros((), device=target.device, dtype=target.dtype)

    valid = _prepare_mask(mask, target, max_flow)
    total = torch.zeros((), device=target.device, dtype=target.dtype)

    for i in range(n_predictions):
        step_pred = _to_bhwc(pred_seq[i])
        step_weight = _to_bhwc(weight_seq[i])
        step_log_b = _to_bhwc(log_b_seq[i])
        step_loss = _mol_nll_single(step_pred, target, step_weight, step_log_b, valid)
        step_weight_gamma = gamma ** (n_predictions - i - 1)
        total = total + _ensure_tensor_on_ref(step_weight_gamma, total) * step_loss

    return total
