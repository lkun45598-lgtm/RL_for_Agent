"""
@file generate_patch.py
@description 从 Loss IR 生成 sandbox_loss.py patch
@author Leizheng
@contributors kongzhiquan
@date 2026-03-22
@version 1.1.0

@changelog
  - 2026-03-22 Leizheng: v1.0.0 initial version
  - 2026-03-23 kongzhiquan: v1.1.0 refine type annotations
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union

from loss_transfer.generation.patch_templates import assemble_sandbox_loss
from loss_transfer.ir.loss_ir_schema import LossIR, LossIRLike
from loss_transfer.common._types import (
    PatchGenerateResult, PatchSummary, TemplatePatchSpec,
    PixelVariant, GradientVariant, FFTVariant,
)


def generate_patch_from_ir(
    loss_ir: LossIRLike,
    patch_spec: TemplatePatchSpec,
    output_path: Optional[str] = None
) -> PatchGenerateResult:
    """
    从 Loss IR + patch 规格生成新的 sandbox_loss.py

    Args:
        loss_ir: Loss IR 对象或 dict
        patch_spec: patch 规格
        output_path: 输出路径

    Returns:
        PatchGenerateResult
    """

    # 提取参数
    pixel_variant: PixelVariant = patch_spec.get('pixel_variant', 'rel_l2')
    gradient_variant: GradientVariant = patch_spec.get('gradient_variant', 'sobel_3x3')
    fft_variant: FFTVariant = patch_spec.get('fft_variant', 'residual_rfft2_abs')

    scales: List[int] = patch_spec.get('scales', [1, 2, 4])
    scale_weights: List[float] = patch_spec.get('scale_weights', [0.5, 0.3, 0.2])
    alpha: float = patch_spec.get('alpha', 0.5)
    beta: float = patch_spec.get('beta', 0.3)
    gamma: float = patch_spec.get('gamma', 0.2)

    # 获取 metadata
    metadata: Dict[str, Union[str, List[str]]]
    if hasattr(loss_ir, 'metadata'):
        metadata = loss_ir.metadata  # type: ignore[union-attr]
    elif isinstance(loss_ir, dict):
        metadata = loss_ir.get('metadata', {})
    else:
        metadata = {}

    description = f"Generated from {metadata.get('paper_title', 'unknown paper')}"

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

    summary: PatchSummary = {
        'pixel_variant': pixel_variant,
        'gradient_variant': gradient_variant,
        'fft_variant': fft_variant,
        'scales': scales,
        'weights': {'alpha': alpha, 'beta': beta, 'gamma': gamma}
    }

    return {
        'code': code,
        'summary': summary
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_ir_path', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--patch_type', default='faithful_core')
    args = parser.parse_args()

    with open(args.loss_ir_path) as f:
        ir_dict = yaml.safe_load(f)

    class SimpleIR:
        def __init__(self, d: dict) -> None:
            self.metadata: Dict[str, str] = d.get('metadata', {})

    loss_ir = SimpleIR(ir_dict)
    patch_spec: TemplatePatchSpec = {
        'pixel_variant': 'rel_l2',
        'gradient_variant': 'sobel_3x3',
        'fft_variant': 'residual_rfft2_abs',
    }
    result = generate_patch_from_ir(loss_ir, patch_spec, args.output_file)  # type: ignore[arg-type]
    print(f"Generated {len(result['code'])} chars")
