"""
@file sandbox_loss.py
@description Experiment #31: multi-scale rel L2 + gradient + FFT, decoupled scale weights
    Decouple scale weights for rel L2 and gradient:
      rl2_scale_weights=[0.5, 0.3, 0.2]  (same as exp#13 best)
      grad_scale_weights=[0.7, 0.2, 0.1] (gradient is fine-scale: weight scale-1 more)
    Hypothesis: gradient edges are most meaningful at full resolution;
    multi-scale rel L2 still benefits from coarse supervision.
    alpha=0.5, beta=0.3, gamma=0.2, FFT at full scale only.
@version 1.31.0
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
    p = pred.float().permute(0, 3, 1, 2)
    t = target.float().permute(0, 3, 1, 2)
    fft_p = torch.fft.rfft2(p, norm='ortho')
    fft_t = torch.fft.rfft2(t, norm='ortho')
    return torch.abs(fft_p.abs() - fft_t.abs()).mean().to(pred.dtype)


def sandbox_loss(pred, target, mask=None,
                 alpha=0.5, beta=0.3, gamma=0.2,
                 rl2_scale_weights=None, grad_scale_weights=None, **kwargs):
    """
    Decoupled scale weights for rel L2 and gradient losses.
    rl2_scale_weights: per-scale weights for relative L2 at scales [1, 2, 4]
    grad_scale_weights: per-scale weights for gradient loss at scales [1, 2, 4]
    FFT applied only at full scale.
    """
    if rl2_scale_weights is None:
        rl2_scale_weights = [0.5, 0.3, 0.2]
    if grad_scale_weights is None:
        grad_scale_weights = [0.7, 0.2, 0.1]

    mask = _align_mask(mask, pred)
    B = pred.size(0)
    scales = [1, 2, 4]

    loss_rel = pred.new_zeros(1).squeeze()
    loss_grad = pred.new_zeros(1).squeeze()

    for s, rl2_w, grad_w in zip(scales, rl2_scale_weights, grad_scale_weights):
        ps = _downsample(pred, s)
        ts = _downsample(target, s)
        ms = _downsample_mask(mask, s)
        loss_rel = loss_rel + rl2_w * _rel_l2(ps, ts, ms)
        loss_grad = loss_grad + grad_w * _gradient_loss(ps, ts, ms) * B

    loss_fft = _fft_loss(pred, target) * B

    return alpha * loss_rel + beta * loss_grad + gamma * loss_fft
