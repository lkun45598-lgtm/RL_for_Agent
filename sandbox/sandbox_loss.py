"""
@file sandbox_loss.py
@description Experiment #64: rel L2 + gradient + residual FFT + vorticity loss (delta=0.05)
    Add vorticity loss to exp#41: penalize |curl(pred) - curl(target)| where
    curl = d(vo)/dx - d(uo)/dy (channel 0=uo, channel 1=vo)
    Vorticity captures rotational structure important in ocean currents.
    alpha=0.5, beta=0.3, gamma=0.2, delta=0.05 (vorticity)
    scale_weights=[0.5,0.3,0.2]
@version 1.64.0
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
    """FFT of residual: spectral energy of the error signal."""
    residual = (pred - target).float().permute(0, 3, 1, 2)
    fft_r = torch.fft.rfft2(residual, norm='ortho')
    return fft_r.abs().mean().to(pred.dtype)


def _vorticity_loss(pred, target, mask=None):
    """Vorticity: curl = d(vo)/dx - d(uo)/dy, channels: 0=uo, 1=vo."""
    B, H, W, C = pred.shape
    # Extract uo (ch0) and vo (ch1)
    uo_p = pred[..., 0:1].permute(0, 3, 1, 2)  # [B,1,H,W]
    vo_p = pred[..., 1:2].permute(0, 3, 1, 2)
    uo_t = target[..., 0:1].permute(0, 3, 1, 2)
    vo_t = target[..., 1:2].permute(0, 3, 1, 2)
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    curl_p = F.conv2d(vo_p, sx, padding=1) - F.conv2d(uo_p, sy, padding=1)
    curl_t = F.conv2d(vo_t, sx, padding=1) - F.conv2d(uo_t, sy, padding=1)
    diff = (curl_p - curl_t).abs().squeeze(1)  # [B,H,W]
    if mask is not None:
        mf = mask[..., 0].float()  # [1,H,W]
        return (diff * mf).sum() / mf.sum().clamp(min=1.0)
    return diff.mean()


def sandbox_loss(pred, target, mask=None,
                 alpha=0.5, beta=0.3, gamma=0.2, delta=0.05,
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
    loss_vort = _vorticity_loss(pred, target, mask) * B
    return alpha * loss_rel + beta * loss_grad + gamma * loss_fft + delta * loss_vort
