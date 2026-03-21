"""
@file sandbox_loss.py
@description Experiment #58: multi-scale rel L2 + SSIM + residual FFT
    Replace gradient loss (exp#41) with SSIM loss.
    SSIM captures structural similarity better than Sobel gradients.
    ssim_loss = 1 - SSIM(pred, target) per channel, averaged.
    alpha=0.5 (rel L2), beta=0.3 (SSIM), gamma=0.2 (residual FFT)
    scale_weights=[0.5,0.3,0.2]
@version 1.58.0
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


def _ssim_loss(pred, target, window_size=7):
    """1 - SSIM, computed per channel then averaged."""
    B, H, W, C = pred.shape
    p = pred.float().permute(0, 3, 1, 2).reshape(B * C, 1, H, W)
    t = target.float().permute(0, 3, 1, 2).reshape(B * C, 1, H, W)

    # Gaussian window
    coords = torch.arange(window_size, dtype=p.dtype, device=p.device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)  # 1,1,k,k

    pad = window_size // 2
    mu_p = F.conv2d(p, window, padding=pad)
    mu_t = F.conv2d(t, window, padding=pad)
    mu_p2 = mu_p ** 2
    mu_t2 = mu_t ** 2
    mu_pt = mu_p * mu_t
    sigma_p2 = F.conv2d(p * p, window, padding=pad) - mu_p2
    sigma_t2 = F.conv2d(t * t, window, padding=pad) - mu_t2
    sigma_pt = F.conv2d(p * t, window, padding=pad) - mu_pt

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / (
        (mu_p2 + mu_t2 + C1) * (sigma_p2 + sigma_t2 + C2))
    return (1.0 - ssim_map.mean()).to(pred.dtype)


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
    loss_ssim = pred.new_zeros(1).squeeze()
    for s, sw in zip(scales, scale_weights):
        ps = _downsample(pred, s)
        ts = _downsample(target, s)
        ms = _downsample_mask(mask, s)
        loss_rel = loss_rel + sw * _rel_l2(ps, ts, ms)
        loss_ssim = loss_ssim + sw * _ssim_loss(ps, ts) * B
    loss_fft = _fft_loss(pred, target) * B
    return alpha * loss_rel + beta * loss_ssim + gamma * loss_fft
