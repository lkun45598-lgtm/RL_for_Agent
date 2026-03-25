import math
import torch
import torch.nn.functional as F

MAX_FLOW = 400.0
gamma = 0.85
var_min = 0.0
var_max = 10.0
epsilon = 1e-8


def _ensure_bhwc(x):
    if not torch.is_tensor(x):
        raise TypeError("tensor expected")
    if x.ndim != 4:
        raise ValueError("Expected 4D BHWC tensor")
    if x.shape[-1] <= 8:
        return x
    if x.shape[1] <= 8:
        return x.permute(0, 2, 3, 1).contiguous()
    raise ValueError("Unable to infer BHWC layout")


def _ensure_mask_bhw(mask, ref):
    if mask is None:
        return torch.ones(ref.shape[0], ref.shape[1], ref.shape[2], dtype=ref.dtype, device=ref.device)
    if not torch.is_tensor(mask):
        raise TypeError("mask must be tensor or None")
    if mask.ndim == 4:
        if mask.shape[-1] == 1:
            mask = mask[..., 0]
        elif mask.shape[1] == 1:
            mask = mask[:, 0]
        else:
            mask = mask.mean(dim=-1) if mask.shape[-1] <= 4 else mask.mean(dim=1)
    if mask.ndim != 3:
        raise ValueError("mask must broadcast to BHW")
    return mask.to(dtype=ref.dtype, device=ref.device)


def _fetch_loss_input(kwargs, name):
    if name in kwargs and torch.is_tensor(kwargs[name]):
        return kwargs[name]
    loss_inputs = kwargs.get("loss_inputs", {})
    if isinstance(loss_inputs, dict) and torch.is_tensor(loss_inputs.get(name)):
        return loss_inputs.get(name)
    return None


def _align_param(x, batch, height, width, channels, name):
    x = _ensure_bhwc(x)
    if x.shape[0] != batch or x.shape[1] != height or x.shape[2] != width:
        raise ValueError(name + " shape mismatch")
    if x.shape[-1] == 1 and channels > 1:
        x = x.expand(batch, height, width, channels)
    elif x.shape[-1] != channels:
        raise ValueError(name + " channel mismatch")
    return x


def _extract_prediction_sequence(pred):
    if torch.is_tensor(pred):
        return [_ensure_bhwc(pred)]
    if isinstance(pred, (list, tuple)):
        seq = []
        for item in pred:
            if torch.is_tensor(item):
                seq.append(_ensure_bhwc(item))
            elif isinstance(item, dict):
                candidate = item.get("pred", item.get("prediction", item.get("output")))
                if torch.is_tensor(candidate):
                    seq.append(_ensure_bhwc(candidate))
        if seq:
            return seq
    if isinstance(pred, dict):
        if isinstance(pred.get("flow"), (list, tuple)):
            seq = []
            for item in pred["flow"]:
                if torch.is_tensor(item):
                    seq.append(_ensure_bhwc(item))
            if seq:
                return seq
        candidate = pred.get("pred", pred.get("prediction", pred.get("output")))
        if torch.is_tensor(candidate):
            return [_ensure_bhwc(candidate)]
    raise TypeError("Unsupported pred container")


def _extract_sequence_param(source, name, length):
    if torch.is_tensor(source):
        return [source for _ in range(length)]
    if isinstance(source, (list, tuple)):
        tensors = [item for item in source if torch.is_tensor(item)]
        if len(tensors) == length:
            return tensors
        if len(tensors) == 1:
            return tensors * length
    if isinstance(source, dict):
        value = source.get(name)
        if torch.is_tensor(value):
            return [value for _ in range(length)]
        if isinstance(value, (list, tuple)):
            tensors = [item for item in value if torch.is_tensor(item)]
            if len(tensors) == length:
                return tensors
            if len(tensors) == 1:
                return tensors * length
    return None


def _mol_nll(pred, target, weight, log_b, valid_mask):
    abs_diff = (target - pred).abs()
    log_b = torch.clamp(log_b, min=var_min, max=var_max)
    alpha = torch.sigmoid(weight)
    ordinary = torch.log(alpha.clamp_min(epsilon)) - math.log(2.0) - abs_diff
    ambiguous = torch.log((1.0 - alpha).clamp_min(epsilon)) - math.log(2.0) - log_b - abs_diff * torch.exp(-log_b)
    component_log_prob = torch.logsumexp(torch.stack([ordinary, ambiguous], dim=-1), dim=-1)
    per_pixel = -component_log_prob.mean(dim=-1)
    masked = per_pixel * valid_mask
    denom = valid_mask.sum().clamp_min(1.0)
    return masked.sum() / denom


def sandbox_loss(pred, target, mask=None, **kwargs):
    pred_seq = _extract_prediction_sequence(pred)
    target = _ensure_bhwc(target)
    batch, height, width, channels = target.shape
    mag = torch.sqrt(torch.sum(target * target, dim=-1))
    valid_mask = _ensure_mask_bhw(mask, target)
    valid_mask = (valid_mask > 0.5).to(dtype=target.dtype) * (mag < MAX_FLOW).to(dtype=target.dtype)

    weight = _fetch_loss_input(kwargs, "weight")
    log_b = _fetch_loss_input(kwargs, "log_b")
    if weight is None or log_b is None:
        raise ValueError("weight and log_b are required")

    weight_seq = _extract_sequence_param(weight, "weight", len(pred_seq))
    log_b_seq = _extract_sequence_param(log_b, "log_b", len(pred_seq))
    if weight_seq is None or log_b_seq is None:
        raise ValueError("weight/log_b sequence structure mismatch")

    total = target.new_tensor(0.0)
    num_predictions = len(pred_seq)
    for index, pred_i in enumerate(pred_seq):
        if pred_i.shape != target.shape:
            raise ValueError("pred and target shape mismatch")
        weight_i = _align_param(weight_seq[index], batch, height, width, channels, "weight")
        log_b_i = _align_param(log_b_seq[index], batch, height, width, channels, "log_b")
        i_weight = gamma ** (num_predictions - index - 1)
        total = total + i_weight * _mol_nll(pred_i, target, weight_i, log_b_i, valid_mask)
    return total
