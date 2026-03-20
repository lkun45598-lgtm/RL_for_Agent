"""
@file sandbox_loss.py

@description Experiment #7: rel L2 + gradient + FFT, stronger FFT
    alpha=0.5, beta=0.3, gamma=0.2 — 加大频域约束。
@version 1.7.0
"""

import torch
import torch.nn.functional as F


def _gradient_loss(pred, target, mask=None):
    """Sobel 梯度 L1 loss，BHWC"""
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

    diff_x = torch.abs(gx_p - gx_t)
    diff_y = torch.abs(gy_p - gy_t)
    grad_diff = (diff_x + diff_y).reshape(B, C, H, W).permute(0, 2, 3, 1)

    if mask is not None:
        mask_f = mask.expand_as(grad_diff).float()
        n_valid = mask_f.sum().clamp(min=1.0)
        return (grad_diff * mask_f).sum() / n_valid
    return grad_diff.mean()


def _fft_loss(pred, target, mask=None):
    """2D FFT 幅度谱 L1 loss，BHWC"""
    B, H, W, C = pred.shape
    p = pred.float().permute(0, 3, 1, 2)
    t = target.float().permute(0, 3, 1, 2)

    fft_p = torch.fft.rfft2(p, norm='ortho')
    fft_t = torch.fft.rfft2(t, norm='ortho')

    amp_diff = torch.abs(fft_p.abs() - fft_t.abs())
    loss = amp_diff.mean()
    return loss.to(pred.dtype)


def sandbox_loss(pred, target, mask=None, alpha=0.5, beta=0.3, gamma=0.2, **kwargs):
    """
    Relative L2 + Sobel gradient + FFT (stronger FFT).
    """
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
    loss_rel = (diff_norms / y_norms).sum()

    loss_grad = _gradient_loss(pred, target, mask) * B
    loss_fft = _fft_loss(pred, target, mask) * B

    return alpha * loss_rel + beta * loss_grad + gamma * loss_fft
