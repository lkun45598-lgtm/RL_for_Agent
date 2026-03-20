"""
@file sandbox_loss.py

@description Experiment #11: rel L2 + gradient + FFT + SSIM loss
    Directly optimize SSIM (the eval metric) as a 4th loss term.
    alpha=0.45, beta=0.3, gamma=0.15, eta=0.1 (SSIM)
@version 1.11.0
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
    diff_x = torch.abs(gx_p - gx_t)
    diff_y = torch.abs(gy_p - gy_t)
    grad_diff = (diff_x + diff_y).reshape(B, C, H, W).permute(0, 2, 3, 1)
    if mask is not None:
        mask_f = mask.expand_as(grad_diff).float()
        n_valid = mask_f.sum().clamp(min=1.0)
        return (grad_diff * mask_f).sum() / n_valid
    return grad_diff.mean()


def _fft_loss(pred, target):
    p = pred.float().permute(0, 3, 1, 2)
    t = target.float().permute(0, 3, 1, 2)
    fft_p = torch.fft.rfft2(p, norm='ortho')
    fft_t = torch.fft.rfft2(t, norm='ortho')
    amp_diff = torch.abs(fft_p.abs() - fft_t.abs())
    return amp_diff.mean().to(pred.dtype)


def _ssim_loss(pred, target, window_size=11):
    """
    1 - SSIM, averaged over batch and channels.
    pred/target: [B, H, W, C]
    """
    B, H, W, C = pred.shape
    p = pred.float().permute(0, 3, 1, 2).reshape(B * C, 1, H, W)
    t = target.float().permute(0, 3, 1, 2).reshape(B * C, 1, H, W)

    # Gaussian window
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    kernel = g.unsqueeze(0) * g.unsqueeze(1)  # [ws, ws]
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, ws, ws]

    pad = window_size // 2
    mu_p = F.conv2d(p, kernel, padding=pad)
    mu_t = F.conv2d(t, kernel, padding=pad)
    mu_p2 = mu_p * mu_p
    mu_t2 = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_p2 = F.conv2d(p * p, kernel, padding=pad) - mu_p2
    sigma_t2 = F.conv2d(t * t, kernel, padding=pad) - mu_t2
    sigma_pt = F.conv2d(p * t, kernel, padding=pad) - mu_pt

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p2 + mu_t2 + C1) * (sigma_p2 + sigma_t2 + C2))

    return (1.0 - ssim_map.mean()).to(pred.dtype)


def sandbox_loss(pred, target, mask=None,
                 alpha=0.45, beta=0.3, gamma=0.15, eta=0.1, **kwargs):
    """
    Relative L2 + Sobel gradient + FFT + SSIM loss.

    Args:
        pred:   [B, H, W, C=2]  (uo, vo)
        target: [B, H, W, C=2]
        mask:   [1, H_m, W_m, 1] bool
        alpha:  rel L2 weight
        beta:   gradient weight
        gamma:  FFT weight
        eta:    SSIM loss weight (1 - SSIM)
    """
    mask = _align_mask(mask, pred)

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
    loss_fft = _fft_loss(pred, target) * B
    loss_ssim = _ssim_loss(pred, target) * B

    return alpha * loss_rel + beta * loss_grad + gamma * loss_fft + eta * loss_ssim
