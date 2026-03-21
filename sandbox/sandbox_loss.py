"""
@file sandbox_loss.py

@description Experiment #25: multi-scale rel L2 + gradient L2-norm + FFT
    Replace MAE gradient (|gx|+|gy|) with L2-norm (sqrt(gx^2+gy^2)).
    Geometrically more accurate gradient magnitude.
    alpha=0.5, beta=0.3, gamma=0.2, scale_weights=[0.5,0.3,0.2]
@version 1.25.0
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
    m = F.interpolate(m, size=(H, W), mode="nearest")
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
    pred_flat = pred.reshape(B, -1)
    target_flat = target.reshape(B, -1)
    if mask is not None:
        mask_flat = mask.expand_as(pred).reshape(B, -1).float()
    else:
        mask_flat = torch.ones_like(pred_flat)
    diff = (pred_flat - target_flat) * mask_flat
    y_masked = target_flat * mask_flat
    diff_norms = torch.norm(diff, 2, dim=1)
    y_norms = torch.norm(y_masked, 2, dim=1).clamp(min=1e-8)
    return (diff_norms / y_norms).sum()


def _gradient_loss(pred, target, mask=None):
    B, H, W, C = pred.shape
    p = pred.permute(0, 3, 1, 2).reshape(B * C, 1, H, W)
    t = target.permute(0, 3, 1, 2).reshape(B * C, 1, H, W)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    gx_p = F.conv2d(p, sobel_x, padding=1)
    gy_p = F.conv2d(p, sobel_y, padding=1)
    gx_t = F.conv2d(t, sobel_x, padding=1)
    gy_t = F.conv2d(t, sobel_y, padding=1)
    # L2 gradient magnitude difference
    mag_p = torch.sqrt(gx_p ** 2 + gy_p ** 2 + 1e-8)
    mag_t = torch.sqrt(gx_t ** 2 + gy_t ** 2 + 1e-8)
    diff = torch.abs(mag_p - mag_t)
    diff = diff.reshape(B, C, H, W).permute(0, 2, 3, 1)
    if mask is not None:
        mask_f = mask.expand_as(diff).float()
        n_valid = mask_f.sum().clamp(min=1.0)
        return (diff * mask_f).sum() / n_valid
    return diff.mean()


def _fft_loss(pred, target):
    p = pred.float().permute(0, 3, 1, 2)
    t = target.float().permute(0, 3, 1, 2)
    fft_p = torch.fft.rfft2(p, norm="ortho")
    fft_t = torch.fft.rfft2(t, norm="ortho")
    amp_diff = torch.abs(fft_p.abs() - fft_t.abs())
    return amp_diff.mean().to(pred.dtype)


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
        p_s = _downsample(pred, s)
        t_s = _downsample(target, s)
        m_s = _downsample_mask(mask, s)
        loss_rel = loss_rel + sw * _rel_l2(p_s, t_s, m_s)
        loss_grad = loss_grad + sw * _gradient_loss(p_s, t_s, m_s) * B

    loss_fft = _fft_loss(pred, target) * B

    return alpha * loss_rel + beta * loss_grad + gamma * loss_fft
