"""
@file sandbox_loss.py
@description Experiment #57: multi-scale rel L2 + relative gradient + residual FFT
    Replace absolute gradient loss with relative gradient loss:
    rel_grad = |grad(pred) - grad(target)| / (|grad(target)| + eps)
    This normalizes gradient errors by target gradient magnitude,
    making it scale-invariant like the rel L2 term.
    alpha=0.5 (rel L2), beta=0.3 (rel gradient), gamma=0.2 (residual FFT)
    scale_weights=[0.5,0.3,0.2]
@version 1.57.0
"""

import torch
import torch.nn.functional as F


def _align_mask(mask, pred):
    if mask is None:
        return None
    H, W = pred.shape[1], pred.shape[2]
    Hm, Wm = mask.shape[1], mask.shape[2]
    if Hm == H and Wm == W:
        return mask
    m = mask.permute(0, 3, 1, 2).float()
    m = F.interpolate(m, size=(H, W), mode='nearest')
    return m.permute(0, 2, 3, 1).bool()


def _downsample(x, scale):
    if scale == 1:
        return x
    B, H, W, C = x.shape
    t = x.permute(0, 3, 1, 2)
    t = F.avg_pool2d(t, kernel_size=scale, stride=scale)
    return t.permute(0, 2, 3, 1)


def _downsample_mask(mask, scale):
    if mask is None or scale == 1:
        return mask
    m = mask.permute(0, 3, 1, 2).float()
    m = F.avg_pool2d(m, kernel_size=scale, stride=scale)
    return (m > 0.5).permute(0, 2, 3, 1)


def _rel_l2(pred, target, mask=None):
    B = pred.size(0)
    pf = pred.reshape(B, -1)
    tf = target.reshape(B, -1)
    mf = mask.expand_as(pred).reshape(B, -1).float() if mask is not None else torch.ones_like(pf)
    diff = (pf - tf) * mf
    ym = tf * mf
    return (torch.norm(diff, 2, dim=1) / torch.norm(ym, 2, dim=1).clamp(min=1e-8)).sum()


def _rel_gradient_loss(pred, target, mask=None):
    """Relative gradient loss: normalize by target gradient magnitude."""
    B, H, W, C = pred.shape
    p = pred.permute(0, 3, 1, 2).reshape(B * C, 1, H, W)
    t = target.permute(0, 3, 1, 2).reshape(B * C, 1, H, W)
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    gp_x = F.conv2d(p, sx, padding=1)
    gp_y = F.conv2d(p, sy, padding=1)
    gt_x = F.conv2d(t, sx, padding=1)
    gt_y = F.conv2d(t, sy, padding=1)
    diff = torch.abs(gp_x - gt_x) + torch.abs(gp_y - gt_y)
    tgt_mag = (gt_x.abs() + gt_y.abs()).clamp(min=1e-6)
    rel_diff = diff / tgt_mag
    rel_diff = rel_diff.reshape(B, C, H, W).permute(0, 2, 3, 1)
    if mask is not None:
        mf = mask.expand_as(rel_diff).float()
        return (rel_diff * mf).sum() / mf.sum().clamp(min=1.0) * B
    return rel_diff.mean() * B


def _fft_loss(pred, target):
    """FFT of residual: spectral energy of the error signal."""
    residual = (pred - target).float().permute(0, 3, 1, 2)
    fft_r = torch.fft.rfft2(residual, norm='ortho')
    return fft_r.abs().mean().to(pred.dtype)


def sandbox_loss(pred, target, mask=None,
                 alpha=0.5, beta=0.3, gamma=0.2,
                 scale_weights=None, **kwargs):
    if scale_weights is None:
        scale_weights = [0.5, 0.3, 0.2]
    mask = _align_mask(mask, pred)
    B = pred.size(0)
    scales = [1, 2, 4]
    loss_rel = pred.new_zeros(1).squeeze()
    loss_grad = pred.new_zeros(1).squeeze()
    for s, sw in zip(scales, scale_weights):
        ps = _downsample(pred, s)
        ts = _downsample(target, s)
        ms = _downsample_mask(mask, s)
        loss_rel = loss_rel + sw * _rel_l2(ps, ts, ms)
        loss_grad = loss_grad + sw * _rel_gradient_loss(ps, ts, ms)
    loss_fft = _fft_loss(pred, target) * B
    return alpha * loss_rel + beta * loss_grad + gamma * loss_fft
