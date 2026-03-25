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
    if x is None or not torch.is_tensor(x):
        return None
    if x.ndim == 3:
        x = x.unsqueeze(-1)
    if x.ndim != 4:
        return x
    if x.shape[-1] <= 8:
        return x
    if x.shape[1] <= 8 and x.shape[-1] > 8:
        return x.permute(0, 2, 3, 1)
    return x


def _flatten_predictions(x):
    if not torch.is_tensor(x):
        return []
    if x.ndim == 4:
        return [x]
    if x.ndim == 5:
        return [x[:, i] for i in range(x.shape[1])]
    return []


def _get_loss_input(name, kwargs):
    if name in kwargs and kwargs[name] is not None:
        return kwargs[name]
    loss_inputs = kwargs.get("loss_inputs", None)
    if isinstance(loss_inputs, dict):
        return loss_inputs.get(name, None)
    return None


def _as_sequence(x, n_predictions):
    if x is None:
        return [None] * n_predictions
    if isinstance(x, (list, tuple)):
        seq = list(x)
    elif torch.is_tensor(x):
        if x.ndim == 5:
            seq = [x[:, i] for i in range(x.shape[1])]
        else:
            seq = [x]
    else:
        seq = [None]
    if len(seq) == n_predictions:
        return seq
    if len(seq) == 1:
        return seq * n_predictions
    if len(seq) < n_predictions:
        return seq + [seq[-1]] * (n_predictions - len(seq))
    return seq[-n_predictions:]


def _prepare_target(target):
    target = _to_bhwc(target)
    if target is None:
        return None
    if target.shape[-1] > 2:
        target = target[..., :2]
    return target


def _prepare_mask(mask, target):
    dtype = target.dtype
    device = target.device
    base = torch.ones(target.shape[0], target.shape[1], target.shape[2], 1, device=device, dtype=dtype)
    if mask is None:
        return base
    mask = _to_bhwc(mask)
    if mask is None:
        return base
    if mask.shape[-1] != 1:
        mask = mask[..., :1]
    return (mask > 0.5).to(dtype)


def _safe_expand_last_dim(x, channels):
    if x is None:
        return None
    if x.shape[-1] == channels:
        return x
    if x.shape[-1] == 1:
        return x.expand(*x.shape[:-1], channels)
    if x.shape[-1] > channels:
        return x[..., :channels]
    return x[..., :1].expand(*x.shape[:-1], channels)


def _prepare_distribution_inputs(weight, log_b, pred):
    if weight is None or log_b is None:
        return None, None
    weight = _to_bhwc(weight)
    log_b = _to_bhwc(log_b)
    if weight is None or log_b is None:
        return None, None
    weight = _safe_expand_last_dim(weight, 2)
    log_b = _safe_expand_last_dim(log_b, 2)
    if weight.ndim != 4 or log_b.ndim != 4:
        return None, None
    if weight.shape[:3] != pred.shape[:3] or log_b.shape[:3] != pred.shape[:3]:
        return None, None
    return weight, log_b


def _fallback_robust_l1(pred, target, valid_mask):
    residual = torch.abs(pred - target)
    per_pixel = residual.sum(dim=-1, keepdim=True)
    weighted = per_pixel * valid_mask
    denom = valid_mask.sum().clamp_min(1.0)
    return weighted.sum() / denom


def _masked_mixture_laplace_nll(pred, target, mask, weight, log_b):
    pred = _to_bhwc(pred)
    if pred is None:
        return target.new_tensor(0.0)
    if pred.shape[-1] > 2:
        pred = pred[..., :2]

    finite_pred = torch.isfinite(pred).all(dim=-1, keepdim=True)
    finite_target = torch.isfinite(target).all(dim=-1, keepdim=True)
    mag = torch.sqrt(torch.sum(target * target, dim=-1, keepdim=True) + epsilon)
    valid_flow = mag < MAX_FLOW
    valid_mask = mask * finite_pred.to(mask.dtype) * finite_target.to(mask.dtype) * valid_flow.to(mask.dtype)

    weight, log_b = _prepare_distribution_inputs(weight, log_b, pred)
    if weight is None or log_b is None:
        return _fallback_robust_l1(pred, target, valid_mask)

    finite_weight = torch.isfinite(weight).all(dim=-1, keepdim=True)
    finite_log_b = torch.isfinite(log_b).all(dim=-1, keepdim=True)
    valid_mask = valid_mask * finite_weight.to(mask.dtype) * finite_log_b.to(mask.dtype)

    if use_var:
        beta1 = torch.zeros_like(log_b[..., 0:1])
        beta2 = torch.clamp(log_b[..., 1:2], min=var_min, max=var_max)
    else:
        beta1 = torch.zeros_like(log_b[..., 0:1])
        beta2 = torch.zeros_like(log_b[..., 1:2])
    log_b = torch.cat([beta1, beta2], dim=-1)

    residual = torch.abs(target - pred)
    log_two = math.log(2.0)

    log_alpha = F.logsigmoid(weight[..., 0:1])
    log_one_minus_alpha = F.logsigmoid(-weight[..., 0:1])

    component_log_prob_1 = log_alpha - log_two - beta1 - residual
    component_log_prob_2 = log_one_minus_alpha - log_two - beta2 - residual * torch.exp(-beta2)
    mixture_log_prob = torch.logsumexp(torch.cat([component_log_prob_1, component_log_prob_2], dim=-1), dim=-1)
    nll = -mixture_log_prob

    per_pixel = nll.mean(dim=-1, keepdim=True)
    denom = valid_mask.sum().clamp_min(1.0)
    return (per_pixel * valid_mask).sum() / denom


def sandbox_loss(pred, target, mask=None, **kwargs):
    target = _prepare_target(target)
    if target is None:
        if torch.is_tensor(pred):
            return pred.new_tensor(0.0)
        return torch.tensor(0.0)

    if isinstance(pred, (list, tuple)):
        pred_seq = list(pred)
    else:
        pred_seq = _flatten_predictions(pred)
    if len(pred_seq) == 0:
        pred_seq = [pred]

    mask = _prepare_mask(mask, target)
    weight = _get_loss_input("weight", kwargs)
    log_b = _get_loss_input("log_b", kwargs)
    weight_seq = _as_sequence(weight, len(pred_seq))
    log_b_seq = _as_sequence(log_b, len(pred_seq))

    total_loss = target.new_tensor(0.0)
    n_predictions = len(pred_seq)
    for i in range(n_predictions):
        stage_loss = _masked_mixture_laplace_nll(pred_seq[i], target, mask, weight_seq[i], log_b_seq[i])
        stage_weight = gamma ** (n_predictions - i - 1)
        total_loss = total_loss + stage_weight * stage_loss

    return total_loss
