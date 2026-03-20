"""
@file sandbox_loss.py

@description Agent 唯一可修改的文件 — 自定义 loss 函数。
    签名与 MaskedLpLoss.__call__ 一致，返回标量 loss。
    初始实现 = baseline MaskedLpLoss（相对 L2）。
@author Leizheng
@date 2026-03-20
@version 1.0.0

@changelog
  - 2026-03-20 Leizheng: v1.0.0 初始 baseline（相对 L2 loss）
"""

import torch
import torch.nn.functional as F
import math


def sandbox_loss(pred, target, mask=None, **kwargs):
    """
    自定义 loss 函数（Agent 修改此函数来探索不同 loss）。

    Args:
        pred:   [B, H, W, C] 模型预测
        target: [B, H, W, C] 真值
        mask:   [1, H, W, 1] bool, True=海洋, False=陆地; 可能为 None

    Returns:
        标量 tensor（用于 backward）
        注意：BaseTrainer.train() 会调用 loss.sum().item() 和 loss.mean()，
        所以这里返回的标量同时兼容 .sum() 和 .mean()（标量的 sum/mean 返回自身）。
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
    rel_errors = diff_norms / y_norms

    return rel_errors.sum()
