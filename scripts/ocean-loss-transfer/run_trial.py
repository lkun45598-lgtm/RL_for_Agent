"""
@file run_trial.py
@description 执行单次 trial: 生成 → 验证 → 记录
@author Leizheng
@contributors kongzhiquan
@date 2026-03-22
@version 1.3.0

@changelog
  - 2026-03-22 Leizheng: v1.0.0 initial version
  - 2026-03-23 kongzhiquan: v1.1.0 support dual mode: template / llm_generate
  - 2026-03-23 kongzhiquan: v1.2.0 refine type annotations
  - 2026-03-23 kongzhiquan: v1.3.0 use find_first_python_path instead of hardcoded path
"""

import json
import subprocess
from pathlib import Path
import sys
from typing import Dict, List, Optional
from loss_ir_schema import LossIR, LossIRLike
from generate_patch import generate_patch_from_ir
from experiment_recorder import record_trial
from llm_code_generator import generate_loss_code
sys.path.append(str(Path(__file__).parent.parent))  # 添加上层目录（scripts）到路径，以便导入 python_manager
from python_manager import find_first_python_path
from _types import (
    TrialResult, PatchSpec, ValidationResult, CodeSnippet,
    LLMGenerateValidation, TemplatePatchSpec,
)

_PYTHON = find_first_python_path() or 'python3'


def run_single_trial(
    loss_ir: LossIRLike,
    patch_spec: PatchSpec,
    trial_id: int,
    paper_slug: str
) -> TrialResult:
    """
    执行单次 trial

    Returns:
        TrialResult
    """

    temp_loss_file = Path('/tmp') / f'trial_{trial_id}_loss.py'
    validation_results: Dict[str, ValidationResult] = {}
    layer_stopped: Optional[str] = None
    validator_script = Path(__file__).parent / 'validate_loss.py'

    # ===== 根据 mode 分支 =====
    if patch_spec.get('mode') == 'llm_generate':
        # LLM 直接生成模式
        code_snippets: List[CodeSnippet] = []
        if hasattr(loss_ir, '_raw_code_snippets'):
            code_snippets = loss_ir._raw_code_snippets  # type: ignore[union-attr]
        elif isinstance(loss_ir, dict):
            code_snippets = loss_ir.get('_raw_code_snippets', [])

        strategy = patch_spec.get('strategy', 'faithful')  # type: ignore[call-overload]
        print(f'  LLM generating code (strategy={strategy})...')

        gen_result = generate_loss_code(
            loss_ir=loss_ir,
            code_snippets=code_snippets,
            strategy=strategy,
        )

        llm_validation: LLMGenerateValidation = {
            'passed_static': gen_result['passed_static'],
            'passed_smoke': gen_result['passed_smoke'],
            'repair_rounds': gen_result['repair_rounds'],
            'error': gen_result.get('error'),
        }
        validation_results['llm_generate'] = llm_validation  # type: ignore[assignment]

        if not gen_result['passed_smoke']:
            # LLM 生成的代码未通过 static+smoke，降级为模板模式
            print(f'  LLM generation failed, falling back to template mode')
            fallback_spec: TemplatePatchSpec = {
                'pixel_variant': 'rel_l2',
                'gradient_variant': 'sobel_3x3',
                'fft_variant': 'residual_rfft2_abs',
                'scales': [1, 2, 4],
                'scale_weights': [0.5, 0.3, 0.2],
                'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2,
            }
            generate_patch_from_ir(loss_ir, fallback_spec, str(temp_loss_file))
            # 走完整 4 层验证
        else:
            # LLM 代码已通过 static+smoke，写入文件，跳到 Layer 3
            temp_loss_file.write_text(gen_result['code'])
            # 直接跳到 Layer 3
            return _run_layer3_and_layer4(
                temp_loss_file, validator_script, validation_results,
                paper_slug, trial_id, patch_spec
            )
    else:
        # 模板模式
        generate_patch_from_ir(loss_ir, patch_spec, str(temp_loss_file))

    # ===== 模板路径：完整 4 层验证 =====

    # Layer 1: Static
    cmd = f'{_PYTHON} {validator_script} --loss_file {temp_loss_file} --mode static'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if not result.stdout.strip():
        return {'passed': False, 'layer_stopped': 'layer1', 'error': 'No output from validator'}

    try:
        layer1: ValidationResult = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {'passed': False, 'layer_stopped': 'layer1', 'error': f'Invalid JSON: {result.stdout[:200]}'}

    validation_results['layer1'] = layer1

    if not layer1.get('passed'):
        layer_stopped = 'layer1'
        record_trial(paper_slug, trial_id, patch_spec, validation_results, str(temp_loss_file))
        return {'passed': False, 'layer_stopped': layer_stopped, 'validation': validation_results}

    # Layer 2: Smoke
    cmd = f'{_PYTHON} {validator_script} --loss_file {temp_loss_file} --mode smoke'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    layer2: ValidationResult = json.loads(result.stdout)
    validation_results['layer2'] = layer2

    if not layer2.get('passed'):
        layer_stopped = 'layer2'
        record_trial(paper_slug, trial_id, patch_spec, validation_results, str(temp_loss_file))
        return {'passed': False, 'layer_stopped': layer_stopped, 'validation': validation_results}

    # Layer 3 + 4
    return _run_layer3_and_layer4(
        temp_loss_file, validator_script, validation_results,
        paper_slug, trial_id, patch_spec
    )


def _run_layer3_and_layer4(
    temp_loss_file: Path,
    validator_script: Path,
    validation_results: Dict[str, ValidationResult],
    paper_slug: str,
    trial_id: int,
    patch_spec: PatchSpec,
) -> TrialResult:
    """执行 Layer 3 (Single Model) 和 Layer 4 (Full Run) 验证"""

    # Layer 3: Single Model
    cmd = f'{_PYTHON} {validator_script} --loss_file {temp_loss_file} --mode single'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    layer3: ValidationResult = json.loads(result.stdout)
    validation_results['layer3'] = layer3

    if not layer3.get('passed') or layer3.get('metrics', {}).get('val_ssim', 0) < 0.3:
        record_trial(paper_slug, trial_id, patch_spec, validation_results, str(temp_loss_file))
        return {'passed': False, 'layer_stopped': 'layer3', 'validation': validation_results}

    # Layer 4: Full Run
    cmd = f'{_PYTHON} scripts/ocean-loss-transfer/validate_loss.py --loss_file {temp_loss_file} --mode full'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
    layer4: ValidationResult = json.loads(result.stdout)
    validation_results['layer4'] = layer4

    # 记录
    trial_dir = record_trial(paper_slug, trial_id, patch_spec, validation_results, str(temp_loss_file))

    return {
        'passed': layer4.get('passed', False),
        'layer_stopped': None if layer4.get('passed') else 'layer4',
        'validation': validation_results,
        'metrics': layer4.get('metrics', {}),
        'trial_dir': trial_dir
    }
