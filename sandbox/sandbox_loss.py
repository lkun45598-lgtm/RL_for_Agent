"""
@file sandbox_loss.py

@description Experiment #3: relative L2 + gradient loss
    baseline 相对 L2 + Sobel 梯度约束，保持空间边缘结构。
@version 1.3.0
"""

import torch
import torch.nn.functional as F


def _gradient_loss(pred, target, mask=None):
    """Sobel 梯度 L1 loss，BHWC → 计算 H/W 方向梯度"""
    # pred/target: [B, H, W, C]
    # 转为 [B*C, 1, H, W] 做卷积
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

    diff_x = torch.abs(gx_p - gx_t)  # [B*C, 1, H, W]
    diff_y = torch.abs(gy_p - gy_t)
    grad_diff = (diff_x + diff_y).reshape(B, C, H, W).permute(0, 2, 3, 1)  # [B, H, W, C]

    if mask is not None:
        mask_f = mask.expand_as(grad_diff).float()
        n_valid = mask_f.sum().clamp(min=1.0)
        return (grad_diff * mask_f).sum() / n_valid
    return grad_diff.mean()


def sandbox_loss(pred, target, mask=None, alpha=0.8, beta=0.2, **kwargs):
    """
    Relative L2 (baseline) + Sobel gradient loss.

    Args:
        pred:   [B, H, W, C]
        target: [B, H, W, C]
        mask:   [1, H, W, 1] bool, True=海洋; 可能为 None
        alpha:  相对 L2 权重
        beta:   梯度 loss 权重

    Returns:
        标量 tensor
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
    rel_l2 = diff_norms / y_norms
    loss_rel = rel_l2.sum()

    # 梯度 loss（batch 平均，与 rel_l2 量纲对齐）
    loss_grad = _gradient_loss(pred, target, mask) * B

    return alpha * loss_rel + beta * loss_grad
