"""
@file sandbox_loss.py

@description Experiment #10: rel L2 + gradient + FFT + divergence constraint
    uo/vo 是海洋速度分量，近似满足不可压缩条件 div(u)=∂uo/∂x+∂vo/∂y≈0。
    在 pred 上加散度惩罚，引导模型生成物理一致的速度场。
    alpha=0.5, beta=0.25, gamma=0.15, delta=0.1 (divergence)
@version 1.10.0
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


def _divergence_loss(pred, mask=None):
    """
    不可压缩流体散度约束：div(u) = ∂uo/∂x + ∂vo/∂y ≈ 0
    pred: [B, H, W, C=2], channel 0=uo, channel 1=vo
    用有限差分近似偏导数。
    """
    B, H, W, C = pred.shape
    if C < 2:
        return pred.new_zeros(1).squeeze()

    uo = pred[..., 0:1]  # [B, H, W, 1]
    vo = pred[..., 1:2]  # [B, H, W, 1]

    # ∂uo/∂x: 沿 W 方向差分
    duo_dx = uo[:, :, 1:, :] - uo[:, :, :-1, :]   # [B, H, W-1, 1]
    # ∂vo/∂y: 沿 H 方向差分
    dvo_dy = vo[:, 1:, :, :] - vo[:, :-1, :, :]   # [B, H-1, W, 1]

    # 取公共区域 [B, H-1, W-1, 1]
    div = duo_dx[:, :-1, :, :] + dvo_dy[:, :, :-1, :]

    if mask is not None:
        # mask 对应公共区域
        mask_crop = mask[:, :-1, :-1, :]  # [1, H-1, W-1, 1]
        mask_f = mask_crop.expand_as(div).float()
        n_valid = mask_f.sum().clamp(min=1.0)
        return (div.abs() * mask_f).sum() / n_valid
    return div.abs().mean()


def sandbox_loss(pred, target, mask=None,
                 alpha=0.5, beta=0.25, gamma=0.15, delta=0.1, **kwargs):
    """
    Relative L2 + Sobel gradient + FFT + divergence constraint.

    Args:
        pred:   [B, H, W, C=2]  (uo, vo)
        target: [B, H, W, C=2]
        mask:   [1, H_m, W_m, 1] bool
        alpha:  rel L2 weight
        beta:   gradient weight
        gamma:  FFT weight
        delta:  divergence constraint weight
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
    loss_div = _divergence_loss(pred, mask) * B

    return alpha * loss_rel + beta * loss_grad + gamma * loss_fft + delta * loss_div
