"""
@file sandbox_loss.py

@description Experiment #21: multi-scale rel L2 + (Sobel + Laplacian) + FFT
    Combine Sobel (first-order edges) and Laplacian (second-order sharpness)
    as complementary gradient terms.
    alpha=0.5, beta=0.15 (Sobel), delta=0.15 (Laplacian), gamma=0.2 (FFT)
    scale_weights=[0.5, 0.3, 0.2]
@version 1.21.0
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
    diff = (torch.abs(gx_p - gx_t) + torch.abs(gy_p - gy_t)).reshape(B, C, H, W).permute(0, 2, 3, 1)
    if mask is not None:
        mask_f = mask.expand_as(diff).float()
        return (diff * mask_f).sum() / mask_f.sum().clamp(min=1.0)
    return diff.mean()


def _laplacian_loss(pred, target, mask=None):
    B, H, W, C = pred.shape
    p = pred.permute(0, 3, 1, 2).reshape(B * C, 1, H, W)
    t = target.permute(0, 3, 1, 2).reshape(B * C, 1, H, W)
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                       dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    diff = torch.abs(F.conv2d(p, lap, padding=1) - F.conv2d(t, lap, padding=1))
    diff = diff.reshape(B, C, H, W).permute(0, 2, 3, 1)
    if mask is not None:
        mask_f = mask.expand_as(diff).float()
        return (diff * mask_f).sum() / mask_f.sum().clamp(min=1.0)
    return diff.mean()


def _fft_loss(pred, target):
    p = pred.float().permute(0, 3, 1, 2)
    t = target.float().permute(0, 3, 1, 2)
    fft_p = torch.fft.rfft2(p, norm='ortho')
    fft_t = torch.fft.rfft2(t, norm='ortho')
    return torch.abs(fft_p.abs() - fft_t.abs()).mean().to(pred.dtype)


def sandbox_loss(pred, target, mask=None,
                 alpha=0.5, beta=0.15, delta=0.15, gamma=0.2,
                 scale_weights=None, **kwargs):
    if scale_weights is None:
        scale_weights = [0.5, 0.3, 0.2]

    mask = _align_mask(mask, pred)
    B = pred.size(0)
    scales = [1, 2, 4]
    loss_rel = pred.new_zeros(1).squeeze()
    loss_grad = pred.new_zeros(1).squeeze()
    loss_lap = pred.new_zeros(1).squeeze()

    for s, sw in zip(scales, scale_weights):
        p_s = _downsample(pred, s)
        t_s = _downsample(target, s)
        m_s = _downsample_mask(mask, s)
        loss_rel = loss_rel + sw * _rel_l2(p_s, t_s, m_s)
        loss_grad = loss_grad + sw * _gradient_loss(p_s, t_s, m_s) * B
        loss_lap = loss_lap + sw * _laplacian_loss(p_s, t_s, m_s) * B

    loss_fft = _fft_loss(pred, target) * B
    return alpha * loss_rel + beta * loss_grad + delta * loss_lap + gamma * loss_fft
