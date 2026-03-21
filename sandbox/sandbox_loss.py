"""
@file sandbox_loss.py
@description Experiment #52: multi-scale rel L2 + residual FFT (no gradient)
    Remove gradient loss from exp#41 to test if gradient is needed with residual FFT.
    alpha=0.8 (rel L2), gamma=0.2 (residual FFT), scale_weights=[0.5,0.3,0.2]
@version 1.52.0
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


def _fft_loss(pred, target):
    residual = (pred - target).float().permute(0, 3, 1, 2)
    return torch.fft.rfft2(residual, norm='ortho').abs().mean().to(pred.dtype)


def sandbox_loss(pred, target, mask=None,
                 alpha=0.8, beta=0.0, gamma=0.2,
                 scale_weights=None, **kwargs):
    if scale_weights is None:
        scale_weights = [0.5, 0.3, 0.2]
    mask = _align_mask(mask, pred)
    B = pred.size(0)
    scales = [1, 2, 4]
    loss_rel = pred.new_zeros(1).squeeze()
    for s, sw in zip(scales, scale_weights):
        ps = _downsample(pred, s)
        ts = _downsample(target, s)
        ms = _downsample_mask(mask, s)
        loss_rel = loss_rel + sw * _rel_l2(ps, ts, ms)
    loss_fft = _fft_loss(pred, target) * B
    return alpha * loss_rel + gamma * loss_fft
