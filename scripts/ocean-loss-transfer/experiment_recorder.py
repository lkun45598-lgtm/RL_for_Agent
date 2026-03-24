"""
@file experiment_recorder.py
@description 结构化记录实验结果
@author Leizheng
@contributors kongzhiquan
@date 2026-03-22
@version 1.2.0

@changelog
  - 2026-03-22 Leizheng: v1.0.0 initial version
  - 2026-03-23 kongzhiquan: v1.1.0 refine type annotations
  - 2026-03-24 Leizheng: v1.2.0 enhanced recording: failure classification,
    training curve, baseline delta, fix attempts
"""

import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from _types import PatchSpec, ValidationResult, FailureCategory, FixAttempt


def classify_failure(validation_results: Dict[str, ValidationResult]) -> Optional[str]:
    """
    从验证结果中分类失败原因。

    按层级顺序检查第一个失败的 layer，提取其 error 字段。
    """
    for layer_key in ('layer1', 'layer2', 'layer3', 'layer4', 'llm_generate'):
        result = validation_results.get(layer_key)
        if result and not result.get('passed', True):
            return result.get('error', 'unknown')
    return None


def record_trial(
    paper_slug: str,
    trial_id: int,
    patch_spec: PatchSpec,
    validation_results: Dict[str, ValidationResult],
    loss_file_path: Optional[str] = None,
    baseline_ssim: Optional[float] = None,
    fix_attempts: Optional[List[FixAttempt]] = None,
) -> str:
    """
    记录单次 trial 结果

    Args:
        paper_slug: 论文标识符
        trial_id: Trial 编号
        patch_spec: Patch 规格
        validation_results: 各层验证结果
        loss_file_path: 生成的 loss 文件路径
        baseline_ssim: 基线 SSIM（用于计算 delta）
        fix_attempts: 自动修复尝试记录

    Returns:
        trial 目录路径
    """
    base_dir = Path(__file__).parent.parent.parent / 'sandbox/loss_transfer_experiments'
    trial_dir = base_dir / paper_slug / f'trial_{trial_id}'
    trial_dir.mkdir(parents=True, exist_ok=True)

    # 复制生成的 loss 文件
    if loss_file_path and Path(loss_file_path).exists():
        shutil.copy(loss_file_path, trial_dir / 'sandbox_loss.py')

    # 失败归因分类
    failure_category = classify_failure(validation_results)

    # 基线对比
    baseline_delta = None
    if baseline_ssim is not None:
        # 从最深层的验证结果中提取 SSIM
        for layer_key in ('layer4', 'layer3'):
            vr = validation_results.get(layer_key, {})
            metrics = vr.get('metrics', {})
            trial_ssim = metrics.get('val_ssim') or metrics.get('swinir')
            if trial_ssim:
                baseline_delta = round(float(trial_ssim) - baseline_ssim, 6)
                break

    # 训练曲线（由改进6的超时恢复或正常训练填充）
    training_curve = None
    for layer_key in ('layer3', 'layer4'):
        vr = validation_results.get(layer_key, {})
        if vr.get('training_curve'):
            training_curve = vr['training_curve']
            break

    # 构建结果
    result = {
        'trial_id': trial_id,
        'timestamp': datetime.now().isoformat(),
        'patch_spec': dict(patch_spec),
        'validation': {k: dict(v) for k, v in validation_results.items()},
        'failure_category': failure_category,
    }

    if baseline_delta is not None:
        result['baseline_delta'] = baseline_delta

    if training_curve:
        result['training_curve'] = training_curve

    if fix_attempts:
        result['fix_attempts'] = [dict(a) for a in fix_attempts]

    # 保存
    (trial_dir / 'result.yaml').write_text(yaml.dump(result, allow_unicode=True, default_flow_style=False))

    return str(trial_dir)
