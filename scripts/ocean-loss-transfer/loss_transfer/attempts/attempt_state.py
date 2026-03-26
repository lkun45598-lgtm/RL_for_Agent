"""
@file attempt_state.py
@description Pure helpers for repair-record bookkeeping and final attempt result assembly.
@author OpenAI Codex
@date 2026-03-26
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_STAGE_SCORE = {
    'code_generation': 0,
    'formula_interface': 0,
    'layer1': 1,
    'layer2': 2,
    'formula_alignment': 3,
    'layer3': 4,
    'layer4': 5,
    None: 6,
}


def layer_rank(stop_layer: Optional[str], layer_order: Dict[Optional[str], int]) -> int:
    return layer_order.get(stop_layer, -1)


def snapshot_path(attempt_dir: Path, name: str, round_number: int, suffix: str) -> Path:
    return attempt_dir / f'{name}_round_{round_number}{suffix}'


def build_code_generation_failure_result(
    *,
    attempt_id: int,
    attempt_spec: Dict[str, Any],
    attempt_dir: Path,
    code_path: Path,
    baseline: Dict[str, Any],
    max_agent_repair_rounds: int,
    error_text: str,
) -> Dict[str, Any]:
    return {
        'attempt_id': attempt_id,
        'name': attempt_spec.get('name', f'attempt_{attempt_id}'),
        'kind': str(attempt_spec.get('kind', 'agent_code')),
        'status': 'failed',
        'passed': False,
        'run_training': bool(attempt_spec.get('run_training', True)),
        'passed_static': False,
        'passed_smoke': False,
        'passed_formula_alignment': None,
        'stop_layer': 'code_generation',
        'error': error_text,
        'validation': {},
        'metrics': {},
        'baseline_delta': None,
        'repair_rounds': [],
        'reward_summary': {
            'primary_metric_name': None,
            'primary_metric': None,
            'baseline_delta': None,
            'passed': False,
            'stop_layer': 'code_generation',
            'stage_score': _STAGE_SCORE['code_generation'],
            'repair_rounds_used': 0,
        },
        'paths': {
            'attempt_dir': str(attempt_dir),
            'code_path': str(code_path),
            'result_path': str(attempt_dir / 'result.json'),
        },
        'strategy_delta': attempt_spec.get('strategy_delta'),
        'metadata': {
            'baseline': baseline,
            'notes': attempt_spec.get('notes'),
            'strategy_delta': attempt_spec.get('strategy_delta'),
            'max_agent_repair_rounds': max_agent_repair_rounds,
        },
    }


def build_initial_repair_record(
    *,
    round_number: int,
    trigger_stop_layer: Optional[str],
    failure_feedback: Dict[str, Any],
    repair_info: Optional[Dict[str, Any]],
    pre_repair_code_path: Path,
) -> Dict[str, Any]:
    return {
        'round': round_number,
        'trigger_stop_layer': trigger_stop_layer,
        'failure_feedback': failure_feedback,
        'repair': repair_info,
        'artifacts': {
            'pre_repair_code_path': str(pre_repair_code_path),
        },
    }


def attach_repair_artifact(repair_record: Dict[str, Any], *, key: str, path: Path) -> None:
    artifacts = repair_record.setdefault('artifacts', {})
    if isinstance(artifacts, dict):
        artifacts[key] = str(path)


def annotate_repair_post_validation(
    repair_record: Dict[str, Any],
    *,
    validation: Dict[str, Any],
    stop_layer: Optional[str],
    metrics: Dict[str, Any],
    baseline_delta: Optional[float],
    error_text: Optional[str],
) -> None:
    repair_record.update(
        {
            'post_validation': validation,
            'post_stop_layer': stop_layer,
            'post_error': error_text,
            'post_metrics': metrics,
            'post_baseline_delta': baseline_delta,
        }
    )


def should_revert_repair(
    *,
    trigger_stop_layer: Optional[str],
    post_stop_layer: Optional[str],
    layer_order: Dict[Optional[str], int],
) -> bool:
    return layer_rank(post_stop_layer, layer_order) < layer_rank(trigger_stop_layer, layer_order)


def mark_repair_reverted(
    repair_record: Dict[str, Any],
    *,
    restored_code_path: Path,
) -> None:
    trigger_stop_layer = repair_record.get('trigger_stop_layer')
    post_stop_layer = repair_record.get('post_stop_layer')
    repair_record.update(
        {
            'status': 'reverted_regression',
            'reverted': True,
            'reversion_reason': (
                f'Repair regressed validation from {trigger_stop_layer} '
                f'to {post_stop_layer}; restored previous candidate.'
            ),
        }
    )
    attach_repair_artifact(repair_record, key='restored_code_path', path=restored_code_path)


def build_reward_summary(
    *,
    metric_name: Optional[str],
    metric_value: Optional[float],
    baseline_delta: Optional[float],
    passed: bool,
    stop_layer: Optional[str],
    validation: Dict[str, Any],
    metrics: Dict[str, Any],
    repair_rounds: List[Dict[str, Any]],
    passed_static: bool,
    passed_smoke: bool,
    passed_formula_alignment: Optional[bool],
) -> Dict[str, Any]:
    val_ssim = None
    for key in ('val_ssim', 'swinir'):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            val_ssim = float(value)
            break

    val_psnr = metrics.get('val_psnr')
    val_psnr_value = float(val_psnr) if isinstance(val_psnr, (int, float)) else None

    curve_source = None
    for layer_key in ('layer4', 'layer3'):
        layer_result = validation.get(layer_key)
        if isinstance(layer_result, dict):
            training_curve = layer_result.get('training_curve')
            if isinstance(training_curve, dict):
                curve_source = training_curve
                break

    return {
        'primary_metric_name': metric_name,
        'primary_metric': metric_value,
        'baseline_delta': baseline_delta,
        'passed': passed,
        'stop_layer': stop_layer,
        'stage_score': _STAGE_SCORE.get(stop_layer, -1),
        'passed_static': passed_static,
        'passed_smoke': passed_smoke,
        'passed_formula_alignment': passed_formula_alignment,
        'passed_single_model': bool(validation.get('layer3', {}).get('passed')) if isinstance(validation.get('layer3'), dict) else False,
        'passed_full_run': bool(validation.get('layer4', {}).get('passed')) if isinstance(validation.get('layer4'), dict) else False,
        'val_ssim': val_ssim,
        'val_psnr': val_psnr_value,
        'training_curve_trend': curve_source.get('trend') if isinstance(curve_source, dict) else None,
        'training_curve_last_epoch': curve_source.get('last_epoch') if isinstance(curve_source, dict) else None,
        'repair_rounds_used': len(repair_rounds),
        'reverted_repair_rounds': sum(
            1
            for round_info in repair_rounds
            if isinstance(round_info, dict) and bool(round_info.get('reverted', False))
        ),
    }


def build_attempt_result(
    *,
    attempt_id: int,
    attempt_spec: Dict[str, Any],
    source_kind: str,
    attempt_dir: Path,
    code_path: Path,
    validation: Dict[str, Any],
    stop_layer: Optional[str],
    metrics: Dict[str, Any],
    baseline: Dict[str, Any],
    repair_rounds: List[Dict[str, Any]],
    run_training: bool,
    formula_spec_path: Optional[str],
    generation_info: Optional[Dict[str, Any]],
    repair_info: Optional[Dict[str, Any]],
    max_agent_repair_rounds: int,
    validation_error_text_fn: Callable[[Optional[str], Dict[str, Any]], Optional[str]],
    compute_baseline_delta_fn: Callable[[Optional[Dict[str, Any]], Dict[str, Any]], Optional[float]],
    extract_primary_metric_fn: Callable[[Optional[Dict[str, Any]]], tuple[Optional[str], Optional[float]]],
) -> Dict[str, Any]:
    passed_static = bool(validation.get('layer1', {}).get('passed', stop_layer != 'layer1'))
    passed_smoke = bool(validation.get('layer2', {}).get('passed', stop_layer not in ('layer1', 'layer2')))
    passed_formula_alignment = None
    if formula_spec_path:
        passed_formula_alignment = bool(
            validation.get('formula_alignment', {}).get('passed', stop_layer != 'formula_alignment')
        )

    passed = stop_layer is None
    baseline_delta = compute_baseline_delta_fn(metrics, baseline)
    metric_name, metric_value = extract_primary_metric_fn(metrics)

    return {
        'attempt_id': attempt_id,
        'name': attempt_spec.get('name', f'attempt_{attempt_id}'),
        'kind': source_kind,
        'status': 'passed' if passed else 'failed',
        'passed': passed,
        'run_training': run_training,
        'passed_static': passed_static,
        'passed_smoke': passed_smoke,
        'passed_formula_alignment': passed_formula_alignment,
        'stop_layer': stop_layer,
        'error': validation_error_text_fn(stop_layer, validation),
        'validation': validation,
        'metrics': metrics,
        'baseline_delta': baseline_delta,
        'repair_rounds': repair_rounds,
        'reward_summary': build_reward_summary(
            metric_name=metric_name,
            metric_value=metric_value,
            baseline_delta=baseline_delta,
            passed=passed,
            stop_layer=stop_layer,
            validation=validation,
            metrics=metrics,
            repair_rounds=repair_rounds,
            passed_static=passed_static,
            passed_smoke=passed_smoke,
            passed_formula_alignment=passed_formula_alignment,
        ),
        'paths': {
            'attempt_dir': str(attempt_dir),
            'code_path': str(code_path),
            'result_path': str(attempt_dir / 'result.json'),
        },
        'strategy_delta': attempt_spec.get('strategy_delta'),
        'metadata': {
            'baseline': baseline,
            'notes': attempt_spec.get('notes'),
            'strategy_delta': attempt_spec.get('strategy_delta'),
            'agent_generation': generation_info,
            'agent_repair': repair_info,
            'max_agent_repair_rounds': max_agent_repair_rounds,
        },
    }
