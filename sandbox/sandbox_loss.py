"""
@file sandbox_loss.py
@description Experiment #50: multi-scale rel L2 + gradient + gradient-residual FFT
    Same base as exp#41 (scale_weights=[0.5,0.3,0.2], alpha=0.5, beta=0.3, gamma=0.2)
    FFT computed on Sobel gradient magnitude of residual:
    fft_loss = rfft2(|grad(pred-target)|).abs().mean()
    This captures spatial frequency of edge errors, not just pixel errors.
@version 1.50.0
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


def _gradient_loss(pred, target, mask=None):
    B, H, W, C = pred.shape
    p = pred.permute(0, 3, 1, 2).reshape(B * C, 1, H, W)
    t = target.permute(0, 3, 1, 2).reshape(B * C, 1, H, W)
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    diff = (torch.abs(F.conv2d(p, sx, padding=1) - F.conv2d(t, sx, padding=1)) +
            torch.abs(F.conv2d(p, sy, padding=1) - F.conv2d(t, sy, padding=1)))
    diff = diff.reshape(B, C, H, W).permute(0, 2, 3, 1)
    if mask is not None:
        mf = mask.expand_as(diff).float()
        return (diff * mf).sum() / mf.sum().clamp(min=1.0)
    return diff.mean()


def _fft_loss(pred, target):
    """FFT of gradient magnitude of residual."""
    B, H, W, C = pred.shape
    r = (pred - target).float().permute(0, 3, 1, 2).reshape(B * C, 1, H, W)
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    grad_mag = (F.conv2d(r, sx, padding=1).abs() + F.conv2d(r, sy, padding=1).abs())
    grad_mag = grad_mag.reshape(B, C, H, W)
    fft_r = torch.fft.rfft2(grad_mag, norm='ortho')
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
        loss_grad = loss_grad + sw * _gradient_loss(ps, ts, ms) * B
    loss_fft = _fft_loss(pred, target) * B
    return alpha * loss_rel + beta * loss_grad + gamma * loss_fft
