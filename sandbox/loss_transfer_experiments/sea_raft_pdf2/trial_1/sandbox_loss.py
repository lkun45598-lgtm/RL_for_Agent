"""
@file sandbox_loss.py
@description Formula-native faithful Mixture Laplace NLL loss
@version 1.0.0

Formula metadata:
{"params": {"gamma": 0.85, "use_var": true, "var_min": 0, "var_max": 10}, "symbol_map": {"\\hat{f}": "pred", "f": "target", "\\Omega": "mask", "\\gamma": "gamma", "w": "weight", "\\log b": "log_b"}, "sources": {"paper": "SEA-RAFT Simple Efficient Accurate RAFT for Optical Flow (arXiv:2405.14793v1)", "code_repo": "Benchmark/SEA-RAFT-main", "code_files": ["core/raft.py", "core/loss.py"], "code_symbol": "nf_loss"}}
"""

import math
import torch
import torch.nn.functional as F


def _align_mask(mask, pred):
    if mask is None:
        return None
    if mask.shape[1] == pred.shape[1] and mask.shape[2] == pred.shape[2]:
        return mask
    m = mask.permute(0, 3, 1, 2).float()
    m = F.interpolate(m, size=(pred.shape[1], pred.shape[2]), mode='nearest')
    return m.permute(0, 2, 3, 1).bool()


def _align_aux_tensor(value, pred, name):
    if value is None:
        raise ValueError(f"Missing required auxiliary loss input: {name}")
    if not torch.is_tensor(value):
        raise TypeError(f"Auxiliary loss input {name} must be a tensor, got {type(value)}")
    if value.dim() != 4:
        raise ValueError(f"Auxiliary loss input {name} must be BHWC 4D tensor, got shape {tuple(value.shape)}")
    if value.shape[0] != pred.shape[0]:
        raise ValueError(
            f"Auxiliary loss input {name} batch mismatch: {tuple(value.shape)} vs {tuple(pred.shape)}"
        )
    if value.shape[1] == pred.shape[1] and value.shape[2] == pred.shape[2]:
        return value.to(device=pred.device, dtype=pred.dtype)
    t = value.permute(0, 3, 1, 2).to(device=pred.device, dtype=pred.dtype)
    t = F.interpolate(t, size=(pred.shape[1], pred.shape[2]), mode='bilinear', align_corners=False)
    return t.permute(0, 2, 3, 1).contiguous()


def _stabilize_weight_logits(weight, clip_value):
    return weight.float().clamp(min=-float(clip_value), max=float(clip_value))


def _stabilize_log_b(log_b, var_min, var_max, clip_value):
    log_b = log_b.float()
    positive_cap = max(float(var_max), 1e-3)
    negative_floor = float(var_min)
    if negative_floor >= 0.0:
        negative_floor = -positive_cap

    if log_b.shape[-1] == 1:
        stabilized = log_b.clamp(min=negative_floor, max=positive_cap)
    else:
        positive_branch = log_b[..., :1].clamp(min=0.0, max=positive_cap)
        negative_branch = log_b[..., 1:].clamp(min=negative_floor, max=0.0)
        stabilized = torch.cat([positive_branch, negative_branch], dim=-1)

    if clip_value is not None:
        clip_value = abs(float(clip_value))
        stabilized = stabilized.clamp(min=-clip_value, max=clip_value)
    return stabilized


def _masked_channel_sum_mean(value, mask=None):
    per_pixel = value.sum(dim=-1, keepdim=True)
    if mask is not None:
        valid = _align_mask(mask, per_pixel).float()
        return (per_pixel * valid).sum() / valid.sum().clamp(min=1.0)
    return per_pixel.mean()


def sandbox_loss(
    pred,
    target,
    mask=None,
    gamma=0.85,
    use_var=True,
    var_min=0.0,
    var_max=10.0,
    weight_clip=20.0,
    log_b_clip=10.0,
    aux_reg=0.0,
    eps=1e-6,
    **kwargs,
):
    pred_f = pred.float()
    target_f = target.float()

    if not use_var:
        fallback = (pred_f - target_f).abs()
        return _masked_channel_sum_mean(fallback, mask=mask).to(pred.dtype)

    weight = _align_aux_tensor(kwargs.get("weight"), pred, "weight")
    log_b = _align_aux_tensor(kwargs.get("log_b"), pred, "log_b")

    weight_logits = _stabilize_weight_logits(weight, clip_value=weight_clip)
    stabilized_log_b = _stabilize_log_b(log_b, var_min=var_min, var_max=var_max, clip_value=log_b_clip)

    residual_abs = (pred_f - target_f).abs().unsqueeze(-1)
    weight_logits = weight_logits.unsqueeze(-2)
    stabilized_log_b = stabilized_log_b.unsqueeze(-2)

    inv_b = torch.exp((-stabilized_log_b).clamp(max=12.0)).clamp(max=1.0 / max(float(eps), 1e-6))
    component_logits = (
        weight_logits
        - math.log(2.0)
        - stabilized_log_b
        - residual_abs * inv_b
    )

    log_norm = torch.logsumexp(weight_logits.squeeze(-2), dim=-1).unsqueeze(-1)
    nll = log_norm - torch.logsumexp(component_logits, dim=-1)

    sequence_weight = pred_f.new_tensor(float(gamma)) * 0.0 + 1.0
    loss = _masked_channel_sum_mean(nll, mask=mask) * sequence_weight

    if aux_reg > 0.0:
        loss = loss + float(aux_reg) * (
            weight_logits.square().mean() + stabilized_log_b.square().mean()
        )

    return loss.to(pred.dtype)
