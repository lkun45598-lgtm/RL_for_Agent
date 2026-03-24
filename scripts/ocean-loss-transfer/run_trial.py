"""
@file run_trial.py
@description 执行单次 trial: 生成 → 验证 → 记录
@author Leizheng
@contributors kongzhiquan
@date 2026-03-22
@version 1.5.0

@changelog
  - 2026-03-22 Leizheng: v1.0.0 initial version
  - 2026-03-23 kongzhiquan: v1.1.0 support dual mode: template / llm_generate
  - 2026-03-23 kongzhiquan: v1.2.0 refine type annotations
  - 2026-03-23 kongzhiquan: v1.3.0 use find_first_python_path instead of hardcoded path
  - 2026-03-24 Leizheng: v1.4.0 add exec-fix loop for template mode Layer 2 failures
    (max 2 auto-repair attempts based on error type)
  - 2026-03-24 Leizheng: v1.5.0 load optional loss_formula.json and apply formula
    alignment checks to agent_generate / llm_generate flow
"""

import json
import subprocess
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from loss_ir_schema import LossIR, LossIRLike
from generate_patch import generate_patch_from_ir
from experiment_recorder import record_trial
from formula_interface_analysis import analyze_formula_interface
from llm_code_generator import generate_loss_code
sys.path.append(str(Path(__file__).parent.parent))  # 添加上层目录（scripts）到路径，以便导入 python_manager
from python_manager import find_first_python_path
from _types import (
    TrialResult, PatchSpec, ValidationResult, CodeSnippet,
    LLMGenerateValidation, TemplatePatchSpec, FixAttempt,
)

_PYTHON = find_first_python_path() or 'python3'

# 最大自动修复尝试次数
_MAX_FIX_ATTEMPTS = 2


def _formula_spec_path(paper_slug: str) -> Path:
    return (
        Path(__file__).parent.parent.parent
        / 'sandbox'
        / 'loss_transfer_experiments'
        / paper_slug
        / 'loss_formula.json'
    )


def _load_formula_spec(paper_slug: str) -> Optional[Dict[str, Any]]:
    """加载论文公式中间表示（若存在）。"""
    formula_path = _formula_spec_path(paper_slug)
    if not formula_path.exists():
        return None

    try:
        data = json.loads(formula_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None

    return data if isinstance(data, dict) else None


def _run_validator(
    validator_script: Path,
    loss_file: Path,
    mode: str,
    formula_spec_path: Optional[Path] = None,
    dataset_root: Optional[str] = None,
    timeout: Optional[int] = None,
) -> ValidationResult:
    cmd = [_PYTHON, str(validator_script), '--loss_file', str(loss_file), '--mode', mode]
    if formula_spec_path and formula_spec_path.exists():
        cmd.extend(['--formula_spec', str(formula_spec_path)])
    if dataset_root:
        cmd.extend(['--dataset_root', dataset_root])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if not result.stdout.strip():
        detail = result.stderr[:300] if result.stderr else 'No output from validator'
        return {'passed': False, 'error': 'validator_no_output', 'detail': detail}

    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            'passed': False,
            'error': 'validator_invalid_json',
            'detail': result.stdout[:300],
        }

    return parsed


def _check_formula_runtime_compatibility(
    formula_spec: Optional[Dict[str, Any]],
) -> Optional[ValidationResult]:
    if not formula_spec:
        return None

    interface_analysis = analyze_formula_interface(formula_spec)
    if interface_analysis.get('status') != 'incompatible':
        return None

    extra_vars = interface_analysis.get('extra_required_variables', [])
    detail = '; '.join(interface_analysis.get('issues', []))
    if extra_vars:
        detail += (
            f". The sandbox runtime can consume model-provided loss_inputs via a sandbox "
            f"model adapter, but the current loss-only transfer pipeline does not create these model outputs automatically: "
            f"{', '.join(extra_vars)}"
        )

    return {
        'passed': False,
        'error': 'formula_interface_incompatible',
        'detail': detail,
        'fix_hint': (
            '将模型改为返回 {"pred": ..., "loss_inputs": {...}}，或将论文 loss '
            '改写为仅依赖 pred/target/mask/params 的形式'
        ),
    }


def _auto_fix_patch_spec(
    error: str,
    current_spec: TemplatePatchSpec,
    attempt: int,
) -> Optional[TemplatePatchSpec]:
    """
    根据错误类型尝试自动修复 patch spec。

    Returns:
        修复后的新 spec，或 None 表示无可用修复。
    """
    new_spec: TemplatePatchSpec = dict(current_spec)  # type: ignore[assignment]
    base_name = current_spec.get('name', '')

    if error in ('nan_in_forward', 'nan_in_gradient', 'nan_during_training'):
        if attempt == 1:
            new_spec['pixel_variant'] = 'smooth_l1'
            new_spec['name'] = f'{base_name} [fix: smooth_l1]'
            return new_spec
        elif attempt == 2:
            new_spec['pixel_variant'] = 'smooth_l1'
            new_spec['fft_variant'] = 'amplitude_diff'
            new_spec['name'] = f'{base_name} [fix: smooth_l1+amp_diff]'
            return new_spec

    if error == 'inf_in_forward':
        if attempt == 1:
            new_spec['pixel_variant'] = 'abs_l1'
            new_spec['name'] = f'{base_name} [fix: abs_l1]'
            return new_spec

    if error == 'gradient_vanish':
        if attempt == 1:
            new_beta = min(current_spec.get('beta', 0.3) * 2, 0.6)
            new_spec['beta'] = round(new_beta, 2)
            new_spec['alpha'] = round(1.0 - new_beta - current_spec.get('gamma', 0.2), 2)
            new_spec['name'] = f'{base_name} [fix: beta*2]'
            return new_spec

    if error == 'gradient_explode':
        if attempt == 1:
            new_beta = current_spec.get('beta', 0.3) * 0.5
            new_gamma = current_spec.get('gamma', 0.2) * 0.5
            new_spec['beta'] = round(new_beta, 2)
            new_spec['gamma'] = round(new_gamma, 2)
            new_spec['alpha'] = round(1.0 - new_beta - new_gamma, 2)
            new_spec['name'] = f'{base_name} [fix: reduce beta/gamma]'
            return new_spec

    # 以下错误不可自动修复:
    # runtime_error (shape mismatch), no_gradient, syntax_error, import_error,
    # blocked_pattern, forbidden_import, boundary_*
    return None


def run_single_trial(
    loss_ir: LossIRLike,
    patch_spec: PatchSpec,
    trial_id: int,
    paper_slug: str,
    dataset_root: Optional[str] = None,
) -> TrialResult:
    """
    执行单次 trial

    Returns:
        TrialResult
    """

    temp_loss_file = Path('/tmp') / f'trial_{trial_id}_loss.py'
    validation_results: Dict[str, ValidationResult] = {}
    layer_stopped: Optional[str] = None
    validator_script = Path(__file__).resolve().parent / 'validate_loss.py'
    formula_spec = _load_formula_spec(paper_slug)
    formula_spec_path = _formula_spec_path(paper_slug)
    formula_interface_issue = _check_formula_runtime_compatibility(formula_spec)

    if formula_interface_issue is not None:
        validation_results['formula_interface'] = formula_interface_issue
        record_trial(paper_slug, trial_id, patch_spec, validation_results, None)
        return {
            'passed': False,
            'layer_stopped': 'layer1',
            'validation': validation_results,
            'error': formula_interface_issue['detail'],
        }

    # ===== 根据 mode 分支 =====
    if patch_spec.get('mode') == 'agent_generate':
        # Agent-Native 模式：Agent 直接提供代码，只做验证
        agent_code = patch_spec.get('code', '')
        if not agent_code:
            return {'passed': False, 'error': 'agent_generate mode requires "code" in patch_spec'}

        strategy = patch_spec.get('strategy', 'faithful')
        print(f'  Agent-Native generating code (strategy={strategy})...')

        gen_result = generate_loss_code(
            loss_ir=loss_ir,
            code_snippets=[],
            strategy=strategy,
            code=agent_code,  # Agent 直接提供的代码
            formula_spec=formula_spec,
        )

        llm_validation: LLMGenerateValidation = {
            'passed_static': gen_result['passed_static'],
            'passed_smoke': gen_result['passed_smoke'],
            'repair_rounds': gen_result['repair_rounds'],
            'error': gen_result.get('error'),
            'passed_formula_alignment': gen_result.get('passed_formula_alignment'),
            'formula_alignment_error': gen_result.get('formula_alignment_error'),
            'formula_alignment_warnings': gen_result.get('formula_alignment_warnings', []),
        }
        validation_results['agent_generate'] = llm_validation  # type: ignore[assignment]

        formula_failed = gen_result.get('passed_formula_alignment') is False
        if not gen_result['passed_smoke'] or formula_failed:
            # Agent 代码未通过验证，返回错误让 Agent 自行修复
            record_trial(paper_slug, trial_id, patch_spec, validation_results, None)
            return {
                'passed': False,
                'layer_stopped': 'layer2' if gen_result['passed_static'] else 'layer1',
                'validation': validation_results,
                'error': gen_result.get('error'),
            }

        # 通过 static + smoke，写入文件，直接跳到 Layer 3
        temp_loss_file.write_text(gen_result['code'])
        return _run_layer3_and_layer4(
            temp_loss_file, validator_script, validation_results,
            paper_slug, trial_id, patch_spec,
            formula_spec_path=formula_spec_path,
            dataset_root=dataset_root,
        )

    elif patch_spec.get('mode') == 'llm_generate':
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
            formula_spec=formula_spec,
        )

        llm_validation: LLMGenerateValidation = {
            'passed_static': gen_result['passed_static'],
            'passed_smoke': gen_result['passed_smoke'],
            'repair_rounds': gen_result['repair_rounds'],
            'error': gen_result.get('error'),
            'passed_formula_alignment': gen_result.get('passed_formula_alignment'),
            'formula_alignment_error': gen_result.get('formula_alignment_error'),
            'formula_alignment_warnings': gen_result.get('formula_alignment_warnings', []),
        }
        validation_results['llm_generate'] = llm_validation  # type: ignore[assignment]

        formula_failed = gen_result.get('passed_formula_alignment') is False
        if not gen_result['passed_smoke'] or formula_failed:
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
                paper_slug, trial_id, patch_spec,
                formula_spec_path=formula_spec_path,
                dataset_root=dataset_root,
            )
    else:
        # 模板模式
        generate_patch_from_ir(loss_ir, patch_spec, str(temp_loss_file))

    # ===== 模板路径：完整 4 层验证 =====

    # Layer 1: Static
    layer1 = _run_validator(
        validator_script,
        temp_loss_file,
        'static',
        formula_spec_path=formula_spec_path,
        dataset_root=dataset_root,
    )

    validation_results['layer1'] = layer1

    if not layer1.get('passed'):
        layer_stopped = 'layer1'
        record_trial(paper_slug, trial_id, patch_spec, validation_results, str(temp_loss_file))
        return {'passed': False, 'layer_stopped': layer_stopped, 'validation': validation_results}

    # Layer 2: Smoke
    layer2 = _run_validator(
        validator_script,
        temp_loss_file,
        'smoke',
        formula_spec_path=formula_spec_path,
        dataset_root=dataset_root,
    )
    validation_results['layer2'] = layer2

    fix_attempts: List[FixAttempt] = []

    if not layer2.get('passed'):
        # === 改进4: 执行修复循环（仅模板模式） ===
        error_type = layer2.get('error', 'unknown')
        fixed = False

        for attempt in range(1, _MAX_FIX_ATTEMPTS + 1):
            new_spec = _auto_fix_patch_spec(error_type, patch_spec, attempt)
            if new_spec is None:
                break  # 该错误类型无可用修复

            fix_attempt: FixAttempt = {
                'attempt_num': attempt,
                'original_error': error_type,
                'fix_applied': new_spec.get('name', 'unknown'),
            }

            print(f'  Auto-fix attempt {attempt}: {fix_attempt["fix_applied"]}')

            # 重新生成代码
            generate_patch_from_ir(loss_ir, new_spec, str(temp_loss_file))

            # 重新跑 Layer 1
            retry_l1 = _run_validator(
                validator_script,
                temp_loss_file,
                'static',
                formula_spec_path=formula_spec_path,
                dataset_root=dataset_root,
            )
            if retry_l1.get('error') in ('validator_no_output', 'validator_invalid_json'):
                fix_attempt['result'] = 'different_error'
                fix_attempts.append(fix_attempt)
                continue
            if not retry_l1.get('passed'):
                fix_attempt['result'] = 'different_error'
                fix_attempts.append(fix_attempt)
                continue

            # 重新跑 Layer 2
            retry_l2 = _run_validator(
                validator_script,
                temp_loss_file,
                'smoke',
                formula_spec_path=formula_spec_path,
                dataset_root=dataset_root,
            )

            if retry_l2.get('passed'):
                fix_attempt['result'] = 'fixed'
                fix_attempts.append(fix_attempt)
                validation_results['layer1'] = retry_l1
                validation_results['layer2'] = retry_l2
                patch_spec = new_spec  # 使用修复后的 spec
                fixed = True
                print(f'  Auto-fix succeeded!')
                break
            else:
                new_error = retry_l2.get('error', 'unknown')
                fix_attempt['result'] = 'same_error' if new_error == error_type else 'different_error'
                fix_attempts.append(fix_attempt)
                error_type = new_error  # 尝试修复新错误

        if not fixed:
            layer_stopped = 'layer2'
            record_trial(paper_slug, trial_id, patch_spec, validation_results,
                         str(temp_loss_file), fix_attempts=fix_attempts)
            return {
                'passed': False,
                'layer_stopped': layer_stopped,
                'validation': validation_results,
                'fix_attempts': fix_attempts,
            }

    # Layer 3 + 4
    return _run_layer3_and_layer4(
        temp_loss_file, validator_script, validation_results,
        paper_slug, trial_id, patch_spec,
        formula_spec_path=formula_spec_path,
        dataset_root=dataset_root,
        fix_attempts=fix_attempts,
    )


def _run_layer3_and_layer4(
    temp_loss_file: Path,
    validator_script: Path,
    validation_results: Dict[str, ValidationResult],
    paper_slug: str,
    trial_id: int,
    patch_spec: PatchSpec,
    formula_spec_path: Optional[Path] = None,
    dataset_root: Optional[str] = None,
    fix_attempts: Optional[List[FixAttempt]] = None,
) -> TrialResult:
    """执行 Layer 3 (Single Model) 和 Layer 4 (Full Run) 验证"""

    # Layer 3: Single Model
    layer3 = _run_validator(
        validator_script,
        temp_loss_file,
        'single',
        formula_spec_path=formula_spec_path,
        dataset_root=dataset_root,
        timeout=600,
    )
    validation_results['layer3'] = layer3

    if not layer3.get('passed') or layer3.get('metrics', {}).get('val_ssim', 0) < 0.3:
        record_trial(paper_slug, trial_id, patch_spec, validation_results,
                     str(temp_loss_file), fix_attempts=fix_attempts)
        return {'passed': False, 'layer_stopped': 'layer3', 'validation': validation_results,
                'fix_attempts': fix_attempts or []}

    # Layer 4: Full Run
    layer4 = _run_validator(
        validator_script,
        temp_loss_file,
        'full',
        formula_spec_path=formula_spec_path,
        dataset_root=dataset_root,
        timeout=900,
    )
    validation_results['layer4'] = layer4

    # 记录
    trial_dir = record_trial(paper_slug, trial_id, patch_spec, validation_results,
                             str(temp_loss_file), fix_attempts=fix_attempts)

    return {
        'passed': layer4.get('passed', False),
        'layer_stopped': None if layer4.get('passed') else 'layer4',
        'validation': validation_results,
        'metrics': layer4.get('metrics', {}),
        'trial_dir': trial_dir,
        'fix_attempts': fix_attempts or [],
    }
