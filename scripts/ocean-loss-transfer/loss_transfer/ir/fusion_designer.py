"""
@file fusion_designer.py
@description 融合历史创新和新论文设计新 loss
@author Leizheng
@contributors kongzhiquan
@date 2026-03-22
@version 1.1.0

@changelog
  - 2026-03-22 Leizheng: v1.0.0 initial version
  - 2026-03-23 kongzhiquan: v1.1.0 refine type annotations
"""

from typing import Dict, List

from loss_transfer.ir.retrieval_engine import retrieve_innovations
from loss_transfer.common._types import Innovation, TemplatePatchSpec, LossIRDict


def design_fusion_loss(
    new_loss_ir: LossIRDict,
    task_description: str = ""
) -> List[TemplatePatchSpec]:
    """
    融合设计新 loss

    Args:
        new_loss_ir: 新论文的 Loss IR dict
        task_description: 任务描述

    Returns:
        融合后的 patch 规格列表
    """

    # 1. 检索相关创新
    innovations: List[Innovation] = retrieve_innovations(task_description, top_k=3)

    if not innovations:
        # 没有历史创新,使用新论文的配置
        return [{'pixel_variant': 'rel_l2', 'gradient_variant': 'sobel_3x3'}]

    # 2. 生成融合策略
    fusion_specs: List[TemplatePatchSpec] = []

    # Strategy 1: 使用最佳历史创新
    best_inn = innovations[0]
    spec1: TemplatePatchSpec = {
        'name': f'Best Historical ({best_inn["paper"]})',
        'pixel_variant': 'rel_l2',
        'gradient_variant': 'sobel_3x3',
        'fft_variant': 'residual_rfft2_abs' if 'fft' in best_inn.get('tags', []) else 'amplitude_diff'
    }
    fusion_specs.append(spec1)

    # Strategy 2: 融合 Top-2 创新
    if len(innovations) >= 2:
        spec2: TemplatePatchSpec = {
            'name': 'Fusion Top-2',
            'pixel_variant': 'rel_l2',
            'gradient_variant': 'scharr_3x3',
            'fft_variant': 'residual_rfft2_abs'
        }
        fusion_specs.append(spec2)

    return fusion_specs
