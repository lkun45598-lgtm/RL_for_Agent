"""
@file sandbox_loss.py
@description Generated from TODO: 论文标题
@version 1.0.0
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


def _pixel_loss(pred, target, mask=None):
    diff = F.smooth_l1_loss(pred, target, reduction='none')
    if mask is not None:
        mf = mask.expand_as(diff).float()
        return (diff * mf).sum() / mf.sum().clamp(min=1.0)
    return diff.mean()


def _gradient_loss(pred, target, mask=None):
    B, H, W, C = pred.shape
    p = pred.permute(0, 3, 1, 2).reshape(B * C, 1, H, W)
    t = target.permute(0, 3, 1, 2).reshape(B * C, 1, H, W)
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    diff = (torch.abs(F.conv2d(p, sx, padding=1) - F.conv2d(t, sx, padding=1)) +
            torch.abs(F.conv2d(p, sy, padding=1) - F.conv2d(t, sy, padding=1)))
    diff = diff.reshape(B, C, H, W).permute(0, 2, 3, 1)
    if mask is not None:
        mf = mask.expand_as(diff).float()
        return (diff * mf).sum() / mf.sum().clamp(min=1.0)
    return diff.mean()


def _fft_loss(pred, target):
    p = pred.float().permute(0, 3, 1, 2)
    t = target.float().permute(0, 3, 1, 2)
    fft_p = torch.fft.rfft2(p, norm='ortho').abs()
    fft_t = torch.fft.rfft2(t, norm='ortho').abs()
    return (fft_p - fft_t).abs().mean().to(pred.dtype)



def sandbox_loss(pred, target, mask=None,
                 alpha=0.5, beta=0.3, gamma=0.2,
                 scale_weights=None, **kwargs):
    if scale_weights is None:
        scale_weights = [0.5, 0.3, 0.2]
    mask = _align_mask(mask, pred)
    B = pred.size(0)
    scales = [1, 2, 4]

    loss_pixel = pred.new_zeros(1).squeeze()
    loss_grad = pred.new_zeros(1).squeeze()

    for s, sw in zip(scales, scale_weights):
        ps = _downsample(pred, s)
        ts = _downsample(target, s)
        ms = _downsample_mask(mask, s)
        loss_pixel = loss_pixel + sw * _pixel_loss(ps, ts, ms)
        loss_grad = loss_grad + sw * _gradient_loss(ps, ts, ms) * B

    loss_fft = _fft_loss(pred, target) * B
    return alpha * loss_pixel + beta * loss_grad + gamma * loss_fft
