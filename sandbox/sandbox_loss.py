"""
@file sandbox_loss.py
@description Experiment #35: multi-scale rel L2 + gradient + low-freq weighted FFT
    Same as exp#13 (scale_weights=[0.5,0.3,0.2], alpha=0.5, beta=0.3, gamma=0.2)
    FFT loss weighted by 1/(1+r) where r is normalized radial frequency,
    emphasizing large-scale (low-frequency) spectral accuracy.
@version 1.35.0
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
    """FFT amplitude loss weighted by 1/(1+r), emphasizing low frequencies."""
    p = pred.float().permute(0, 3, 1, 2)
    t = target.float().permute(0, 3, 1, 2)
    B, C, H, W = p.shape
    Wh = W // 2 + 1
    fft_p = torch.fft.rfft2(p, norm='ortho')
    fft_t = torch.fft.rfft2(t, norm='ortho')
    # Build frequency weight grid: 1/(1+normalized_radius)
    fy = torch.fft.fftfreq(H, device=pred.device).unsqueeze(1)  # [H, 1]
    fx = torch.fft.rfftfreq(W, device=pred.device).unsqueeze(0)  # [1, Wh]
    r = torch.sqrt(fy ** 2 + fx ** 2)  # [H, Wh], range [0, ~0.7]
    weight = 1.0 / (1.0 + r * 10)  # normalize so DC=1, Nyquist~0.12
    weight = weight.unsqueeze(0).unsqueeze(0)  # [1, 1, H, Wh]
    amp_diff = torch.abs(fft_p.abs() - fft_t.abs())
    return (amp_diff * weight).mean().to(pred.dtype)


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
