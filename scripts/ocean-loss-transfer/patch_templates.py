"""
@file patch_templates.py
@description Patch 模板系统 - 从预定义模板生成 sandbox_loss.py 代码
@author Leizheng
@contributors kongzhiquan
@date 2026-03-22
@version 1.1.0

@changelog
  - 2026-03-22 Leizheng: v1.0.0 initial version
  - 2026-03-23 kongzhiquan: v1.1.0 refine type annotations
"""

from typing import Callable, Dict, List
from _types import PixelVariant, GradientVariant, FFTVariant


# ============ 辅助函数模板 ============

def template_align_mask() -> str:
    return '''def _align_mask(mask, pred):
    if mask is None:
        return None
    H, W = pred.shape[1], pred.shape[2]
    Hm, Wm = mask.shape[1], mask.shape[2]
    if Hm == H and Wm == W:
        return mask
    m = mask.permute(0, 3, 1, 2).float()
    m = F.interpolate(m, size=(H, W), mode='nearest')
    return m.permute(0, 2, 3, 1).bool()
'''


def template_downsample() -> str:
    return '''def _downsample(x, scale):
    if scale == 1:
        return x
    B, H, W, C = x.shape
    t = x.permute(0, 3, 1, 2)
    t = F.avg_pool2d(t, kernel_size=scale, stride=scale)
    return t.permute(0, 2, 3, 1)
'''


def template_downsample_mask() -> str:
    return '''def _downsample_mask(mask, scale):
    if mask is None or scale == 1:
        return mask
    m = mask.permute(0, 3, 1, 2).float()
    m = F.avg_pool2d(m, kernel_size=scale, stride=scale)
    return (m > 0.5).permute(0, 2, 3, 1)
'''


# ============ 像素级 Loss 模板 ============

def template_rel_l2() -> str:
    return '''def _pixel_loss(pred, target, mask=None):
    B = pred.size(0)
    pf = pred.reshape(B, -1)
    tf = target.reshape(B, -1)
    mf = mask.expand_as(pred).reshape(B, -1).float() if mask is not None else torch.ones_like(pf)
    diff = (pf - tf) * mf
    ym = tf * mf
    return (torch.norm(diff, 2, dim=1) / torch.norm(ym, 2, dim=1).clamp(min=1e-8)).sum()
'''


def template_abs_l1() -> str:
    return '''def _pixel_loss(pred, target, mask=None):
    diff = torch.abs(pred - target)
    if mask is not None:
        mf = mask.expand_as(diff).float()
        return (diff * mf).sum() / mf.sum().clamp(min=1.0)
    return diff.mean()
'''


def template_smooth_l1() -> str:
    return '''def _pixel_loss(pred, target, mask=None):
    diff = F.smooth_l1_loss(pred, target, reduction='none')
    if mask is not None:
        mf = mask.expand_as(diff).float()
        return (diff * mf).sum() / mf.sum().clamp(min=1.0)
    return diff.mean()
'''


# ============ 梯度算子模板 ============

def template_sobel_3x3() -> str:
    return '''def _gradient_loss(pred, target, mask=None):
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
'''


def template_scharr_3x3() -> str:
    return '''def _gradient_loss(pred, target, mask=None):
    B, H, W, C = pred.shape
    p = pred.permute(0, 3, 1, 2).reshape(B * C, 1, H, W)
    t = target.permute(0, 3, 1, 2).reshape(B * C, 1, H, W)
    sx = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sy = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    diff = (torch.abs(F.conv2d(p, sx, padding=1) - F.conv2d(t, sx, padding=1)) +
            torch.abs(F.conv2d(p, sy, padding=1) - F.conv2d(t, sy, padding=1)))
    diff = diff.reshape(B, C, H, W).permute(0, 2, 3, 1)
    if mask is not None:
        mf = mask.expand_as(diff).float()
        return (diff * mf).sum() / mf.sum().clamp(min=1.0)
    return diff.mean()
'''


# ============ 频域 Loss 模板 ============

def template_residual_fft() -> str:
    return '''def _fft_loss(pred, target):
    residual = (pred - target).float().permute(0, 3, 1, 2)
    fft_r = torch.fft.rfft2(residual, norm='ortho')
    return fft_r.abs().mean().to(pred.dtype)
'''


def template_amplitude_diff() -> str:
    return '''def _fft_loss(pred, target):
    p = pred.float().permute(0, 3, 1, 2)
    t = target.float().permute(0, 3, 1, 2)
    fft_p = torch.fft.rfft2(p, norm='ortho').abs()
    fft_t = torch.fft.rfft2(t, norm='ortho').abs()
    return (fft_p - fft_t).abs().mean().to(pred.dtype)
'''


# ============ 模板注册表 ============

# 模板函数类型: 无参数，返回代码字符串
TemplateFunc = Callable[[], str]

PIXEL_LOSS_TEMPLATES: Dict[PixelVariant, TemplateFunc] = {
    'rel_l2': template_rel_l2,
    'abs_l1': template_abs_l1,
    'smooth_l1': template_smooth_l1,
}

GRADIENT_TEMPLATES: Dict[GradientVariant, TemplateFunc] = {
    'sobel_3x3': template_sobel_3x3,
    'scharr_3x3': template_scharr_3x3,
}

FFT_TEMPLATES: Dict[FFTVariant, TemplateFunc] = {
    'residual_rfft2_abs': template_residual_fft,
    'amplitude_diff': template_amplitude_diff,
}


# ============ 主组装函数 ============

def assemble_sandbox_loss(
    pixel_variant: PixelVariant = 'rel_l2',
    gradient_variant: GradientVariant = 'sobel_3x3',
    fft_variant: FFTVariant = 'residual_rfft2_abs',
    scales: List[int] = [1, 2, 4],
    scale_weights: List[float] = [0.5, 0.3, 0.2],
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    description: str = "Generated loss"
) -> str:
    """组装完整的 sandbox_loss.py"""

    code = f'''"""
@file sandbox_loss.py
@description {description}
@version 1.0.0
"""

import torch
import torch.nn.functional as F


'''

    # 添加辅助函数
    code += template_align_mask() + '\n\n'
    code += template_downsample() + '\n\n'
    code += template_downsample_mask() + '\n\n'

    # 添加组件函数
    code += PIXEL_LOSS_TEMPLATES[pixel_variant]() + '\n\n'
    code += GRADIENT_TEMPLATES[gradient_variant]() + '\n\n'
    code += FFT_TEMPLATES[fft_variant]() + '\n\n'

    # 主函数
    code += f'''
def sandbox_loss(pred, target, mask=None,
                 alpha={alpha}, beta={beta}, gamma={gamma},
                 scale_weights=None, **kwargs):
    if scale_weights is None:
        scale_weights = {scale_weights}
    mask = _align_mask(mask, pred)
    B = pred.size(0)
    scales = {scales}

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
'''

    return code
