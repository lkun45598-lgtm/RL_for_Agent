"""
@file attempt_feedback.py
@description Pure helpers for summarizing validation feedback and repair history.
@author OpenAI Codex
@date 2026-03-26
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def extract_primary_metric(metrics: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[float]]:
    if not metrics:
        return None, None

    for key in ('swinir', 'val_ssim'):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)

    return None, None


def compute_baseline_delta(
    metrics: Optional[Dict[str, Any]],
    baseline: Dict[str, Any],
) -> Optional[float]:
    _, value = extract_primary_metric(metrics)
    baseline_center = baseline.get('ssim_mean')
    if value is None or not isinstance(baseline_center, (int, float)):
        return None
    return round(float(value) - float(baseline_center), 6)


def validation_error_text(stop_layer: Optional[str], validation: Dict[str, Any]) -> Optional[str]:
    if not stop_layer:
        return None

    result = validation.get(stop_layer)
    if not isinstance(result, dict):
        return None

    if stop_layer == 'formula_alignment':
        errors = result.get('errors', [])
        if isinstance(errors, list) and errors:
            return '; '.join(str(item) for item in errors)

    detail = result.get('detail')
    if isinstance(detail, str) and detail:
        return detail

    error = result.get('error')
    return str(error) if error is not None else None


def summarize_repair_rounds(repair_rounds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for round_info in repair_rounds[-3:]:
        repair = round_info.get('repair', {})
        repair_payload = repair if isinstance(repair, dict) else {}
        summary.append(
            {
                'round': round_info.get('round'),
                'status': round_info.get('status'),
                'trigger_stop_layer': round_info.get('trigger_stop_layer'),
                'post_stop_layer': round_info.get('post_stop_layer'),
                'post_error': round_info.get('post_error'),
                'post_baseline_delta': round_info.get('post_baseline_delta'),
                'reverted': bool(round_info.get('reverted', False)),
                'agent_response_path': repair_payload.get('agent_response_path'),
            }
        )
    return summary


def _build_performance_target(
    metrics: Optional[Dict[str, Any]],
    baseline: Dict[str, Any],
) -> Dict[str, Any]:
    metric_name, metric_value = extract_primary_metric(metrics)
    viable_threshold = baseline.get('viable_threshold')
    improvement_threshold = baseline.get('improvement_threshold')
    return {
        'primary_metric_name': metric_name or baseline.get('model') or 'swinir',
        'current_value': metric_value,
        'baseline_delta': compute_baseline_delta(metrics, baseline),
        'viable_threshold': float(viable_threshold) if isinstance(viable_threshold, (int, float)) else None,
        'improvement_threshold': (
            float(improvement_threshold) if isinstance(improvement_threshold, (int, float)) else None
        ),
        'goal': (
            'Improve validation SSIM above the viable threshold while preserving formula '
            'alignment, runtime stability, and trainability.'
        ),
    }


def build_failure_feedback(
    *,
    stop_layer: Optional[str],
    validation: Dict[str, Any],
    metrics: Optional[Dict[str, Any]],
    baseline: Dict[str, Any],
    repair_rounds: List[Dict[str, Any]],
    runtime_routing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    feedback: Dict[str, Any] = {
        'stop_layer': stop_layer,
        'validation': validation,
    }

    error_text = validation_error_text(stop_layer, validation)
    if error_text:
        feedback['error'] = error_text
    if metrics:
        feedback['metrics'] = metrics
    if repair_rounds:
        feedback['previous_repair_rounds'] = summarize_repair_rounds(repair_rounds)
    if runtime_routing is not None:
        feedback['runtime_routing'] = runtime_routing
    if stop_layer == 'layer4':
        feedback['performance_target'] = _build_performance_target(metrics, baseline)
    return feedback
