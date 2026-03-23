"""
@file check_compatibility.py
@description 检查 Loss IR 与目标接口的兼容性
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
from typing import List

from loss_ir_schema import LossIR
from _types import (
    CompatibilityResult, HardIncompatibilityResult,
    BlockedPatternResult, BlockedPatternsConfig,
)


def load_blocked_patterns() -> BlockedPatternsConfig:
    """加载已知失败模式"""
    blocked_file = Path(__file__).parent.parent.parent / 'workflow/loss_transfer/blocked_patterns.yaml'
    if blocked_file.exists():
        return yaml.safe_load(blocked_file.read_text())
    return {'blocked_components': []}


def check_hard_incompatibility(loss_ir: LossIR) -> HardIncompatibilityResult:
    """检查硬性不兼容"""
    flags = loss_ir.incompatibility_flags
    issues: List[str] = []

    if flags.get('requires_model_features'):
        issues.append('需要模型内部特征 (不支持)')
    if flags.get('requires_pretrained_network'):
        issues.append('需要预训练网络如 VGG (不支持)')
    if flags.get('requires_adversarial'):
        issues.append('需要对抗训练 (不支持)')
    if flags.get('requires_multiple_forward_passes'):
        issues.append('需要多次 forward (不支持)')

    return {
        'compatible': len(issues) == 0,
        'issues': issues
    }


def check_blocked_patterns(loss_ir: LossIR) -> BlockedPatternResult:
    """检查是否匹配已知失败模式"""
    warnings: List[str] = []

    for comp in loss_ir.components:
        comp_type = comp.type
        comp_name = comp.name.lower()

        # 检查 SSIM
        if 'ssim' in comp_name or comp_type == 'structural_loss':
            warnings.append(f"SSIM loss 已知会崩溃 (exp#11)")

        # 检查 Laplacian
        if 'laplacian' in comp_name:
            warnings.append(f"Laplacian 已知会崩溃 (exp#20,#40,#66)")

    return {'warnings': warnings}


def check_compatibility(loss_ir: LossIR) -> CompatibilityResult:
    """主兼容性检查函数"""

    # 硬性不兼容
    hard_check = check_hard_incompatibility(loss_ir)
    if not hard_check['compatible']:
        return {
            'status': 'incompatible',
            'reason': 'hard_incompatibility',
            'issues': hard_check['issues']
        }

    # 已知失败模式
    pattern_check = check_blocked_patterns(loss_ir)

    if pattern_check['warnings']:
        return {
            'status': 'partially_compatible',
            'warnings': pattern_check['warnings']
        }

    return {
        'status': 'fully_compatible',
        'warnings': []
    }
