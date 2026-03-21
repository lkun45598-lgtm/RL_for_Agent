"""
@file sandbox_loss.py
@description Experiment #29: single-scale rel L2 + gradient + FFT (replicate exp#7)
    Confirm exp#13 multi-scale gain is real. Same as exp#7:
    alpha=0.5, beta=0.3, gamma=0.2, NO multi-scale.
@version 1.29.0
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
    m = mask.permute(0,3,1,2).float()
    m = F.interpolate(m, size=(H,W), mode="nearest")
    return m.permute(0,2,3,1).bool()

def _gradient_loss(pred, target, mask=None):
    B, H, W, C = pred.shape
    p = pred.permute(0,3,1,2).reshape(B*C,1,H,W)
    t = target.permute(0,3,1,2).reshape(B*C,1,H,W)
    sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=pred.dtype, device=pred.device).view(1,1,3,3)
    sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=pred.dtype, device=pred.device).view(1,1,3,3)
    diff = (torch.abs(F.conv2d(p,sx,padding=1)-F.conv2d(t,sx,padding=1)) +
            torch.abs(F.conv2d(p,sy,padding=1)-F.conv2d(t,sy,padding=1)))
    diff = diff.reshape(B,C,H,W).permute(0,2,3,1)
    if mask is not None:
        mf = mask.expand_as(diff).float()
        return (diff*mf).sum() / mf.sum().clamp(min=1.0)
    return diff.mean()

def _fft_loss(pred, target):
    p = pred.float().permute(0,3,1,2)
    t = target.float().permute(0,3,1,2)
    fft_p = torch.fft.rfft2(p, norm="ortho")
    fft_t = torch.fft.rfft2(t, norm="ortho")
    return torch.abs(fft_p.abs() - fft_t.abs()).mean().to(pred.dtype)

def sandbox_loss(pred, target, mask=None,
                 alpha=0.5, beta=0.3, gamma=0.2, **kwargs):
    mask = _align_mask(mask, pred)
    B = pred.size(0)
    pf = pred.reshape(B,-1)
    tf = target.reshape(B,-1)
    mf = mask.expand_as(pred).reshape(B,-1).float() if mask is not None else torch.ones_like(pf)
    diff = (pf - tf) * mf
    ym = tf * mf
    loss_rel = (torch.norm(diff,2,dim=1) / torch.norm(ym,2,dim=1).clamp(min=1e-8)).sum()
    loss_grad = _gradient_loss(pred, target, mask) * B
    loss_fft = _fft_loss(pred, target) * B
    return alpha * loss_rel + beta * loss_grad + gamma * loss_fft
