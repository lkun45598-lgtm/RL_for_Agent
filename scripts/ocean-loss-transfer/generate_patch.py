"""
@file generate_patch.py
@description 从 Loss IR 生成 sandbox_loss.py patch
@author Leizheng
@date 2026-03-22
@version 1.0.0
"""

import yaml
from pathlib import Path
from typing import Dict, Any
try:
    from .patch_templates import assemble_sandbox_loss
    from .loss_ir_schema import LossIR
except ImportError:
    from patch_templates import assemble_sandbox_loss
    from loss_ir_schema import LossIR


def generate_patch_from_ir(
    loss_ir: LossIR,
    patch_spec: Dict[str, Any],
    output_path: str = None
) -> Dict[str, Any]:
    """
    从 Loss IR + patch 规格生成新的 sandbox_loss.py
    
    Args:
        loss_ir: Loss IR 对象
        patch_spec: patch 规格 {pixel_variant, gradient_variant, fft_variant, ...}
        output_path: 输出路径
    
    Returns:
        {code, diff, summary}
    """
    
    # 提取参数
    pixel_variant = patch_spec.get('pixel_variant', 'rel_l2')
    gradient_variant = patch_spec.get('gradient_variant', 'sobel_3x3')
    fft_variant = patch_spec.get('fft_variant', 'residual_rfft2_abs')
    
    scales = patch_spec.get('scales', [1, 2, 4])
    scale_weights = patch_spec.get('scale_weights', [0.5, 0.3, 0.2])
    alpha = patch_spec.get('alpha', 0.5)
    beta = patch_spec.get('beta', 0.3)
    gamma = patch_spec.get('gamma', 0.2)
    
    description = f"Generated from {loss_ir.metadata.get('paper_title', 'unknown paper')}"
    
    # 生成代码
    code = assemble_sandbox_loss(
        pixel_variant=pixel_variant,
        gradient_variant=gradient_variant,
        fft_variant=fft_variant,
        scales=scales,
        scale_weights=scale_weights,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        description=description
    )
    
    # 保存
    if output_path:
        Path(output_path).write_text(code)
    
    return {
        'code': code,
        'summary': {
            'pixel_variant': pixel_variant,
            'gradient_variant': gradient_variant,
            'fft_variant': fft_variant,
            'scales': scales,
            'weights': {'alpha': alpha, 'beta': beta, 'gamma': gamma}
        }
    }
