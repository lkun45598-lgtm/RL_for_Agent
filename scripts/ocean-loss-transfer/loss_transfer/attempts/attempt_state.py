"""
@file attempt_state.py
@description Pure helpers for repair-record bookkeeping and final attempt result assembly.
@author OpenAI Codex
@date 2026-03-26
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


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
        },
        'paths': {
            'attempt_dir': str(attempt_dir),
            'code_path': str(code_path),
            'result_path': str(attempt_dir / 'result.json'),
        },
        'metadata': {
            'baseline': baseline,
            'notes': attempt_spec.get('notes'),
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
) -> Dict[str, Any]:
    return {
        'primary_metric_name': metric_name,
        'primary_metric': metric_value,
        'baseline_delta': baseline_delta,
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
        ),
        'paths': {
            'attempt_dir': str(attempt_dir),
            'code_path': str(code_path),
            'result_path': str(attempt_dir / 'result.json'),
        },
        'metadata': {
            'baseline': baseline,
            'notes': attempt_spec.get('notes'),
            'agent_generation': generation_info,
            'agent_repair': repair_info,
            'max_agent_repair_rounds': max_agent_repair_rounds,
        },
    }
