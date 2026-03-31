"""
@file case_memory_retriever.py

@description Shared case-memory retrieval and prompt formatting helpers for agent workflows.
@author kongzhiquan
@contributors kongzhiquan
@date 2026-03-28
@version 1.1.0

@changelog
  - 2026-03-28 kongzhiquan: v1.0.0 extract shared case-memory retrieval and prompt-formatting helpers
  - 2026-03-30 kongzhiquan: v1.1.0 add success-aware scoring on failure signatures and repair effectiveness
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

from loss_transfer.memory.case_memory_store import (
    DEFAULT_CASE_MEMORY_PATH,
    build_failure_signature,
    load_case_memory_records,
    normalize_string_list,
    safe_dict,
    safe_list,
)


_MEMORY_TOKEN_RE = re.compile(r'[a-z0-9_]{3,}')
_MEMORY_STOPWORDS = {
    'the', 'and', 'for', 'with', 'from', 'that', 'this', 'into', 'when', 'then', 'than',
    'after', 'before', 'while', 'where', 'which', 'have', 'has', 'had', 'were', 'was',
    'are', 'but', 'not', 'did', 'does', 'done', 'can', 'could', 'should', 'would',
    'loss', 'attempt', 'repair', 'candidate', 'objective', 'validation', 'result',
    'strategy', 'previous', 'current', 'should', 'improve', 'metric', 'signal',
}


def _stringify_memory_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        return ' '.join(_stringify_memory_value(item) for item in value.values())
    if isinstance(value, list):
        return ' '.join(_stringify_memory_value(item) for item in value)
    return ''


def _keyword_tokens(*values: Any) -> set[str]:
    text = ' '.join(_stringify_memory_value(value).lower() for value in values if value is not None)
    return {
        match.group(0)
        for match in _MEMORY_TOKEN_RE.finditer(text)
        if match.group(0) not in _MEMORY_STOPWORDS
    }


def _string_set(value: Any) -> set[str]:
    return set(normalize_string_list(value))


def _build_memory_query_context(
    *,
    task_context: Dict[str, Any],
    attempt_spec: Optional[Dict[str, Any]] = None,
    failure_feedback: Optional[Dict[str, Any]] = None,
    latest_attempt_result: Optional[Dict[str, Any]] = None,
    latest_repair_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    integration = safe_dict(task_context.get('integration_assessment'))
    attempt_payload = safe_dict(attempt_spec)
    latest_attempt_payload = safe_dict(latest_attempt_result)
    failure_payload = safe_dict(failure_feedback)
    strategy_delta = safe_dict(
        attempt_payload.get('strategy_delta')
        or latest_attempt_payload.get('strategy_delta')
    )
    repair_plan = safe_dict(latest_repair_plan)
    current_stop_layer = failure_payload.get('stop_layer') or latest_attempt_payload.get('stop_layer')
    current_error = failure_payload.get('error') or latest_attempt_payload.get('error')
    current_metrics = failure_payload.get('metrics') or latest_attempt_payload.get('metrics')
    required_edit_paths = normalize_string_list(
        attempt_payload.get('required_edit_paths')
        or latest_attempt_payload.get('required_edit_paths')
    )
    files_to_edit = normalize_string_list(
        attempt_payload.get('files_to_edit')
        or latest_attempt_payload.get('files_to_edit')
    )
    tokens = _keyword_tokens(
        attempt_payload.get('objective'),
        current_error,
        strategy_delta,
        repair_plan.get('failure_hypothesis'),
        repair_plan.get('fallback_plan'),
        current_stop_layer,
        build_failure_signature(
            stop_layer=current_stop_layer,
            error=current_error,
            metrics=current_metrics,
        ),
        required_edit_paths,
        files_to_edit,
    )

    return {
        'paper_slug': task_context.get('paper_slug'),
        'integration_path': integration.get('recommended_path'),
        'kind': attempt_payload.get('kind') or latest_attempt_payload.get('kind'),
        'stop_layer': current_stop_layer,
        'failure_signature': build_failure_signature(
            stop_layer=current_stop_layer,
            error=current_error,
            metrics=current_metrics,
        ),
        'required_edit_paths': _string_set(required_edit_paths),
        'files_to_edit': _string_set(files_to_edit),
        'tokens': tokens,
        'attempt_id': latest_attempt_payload.get('attempt_id'),
    }


def _score_memory_case(case: Dict[str, Any], current_context: Dict[str, Any]) -> float:
    score = 0.0

    integration_path = current_context.get('integration_path')
    if integration_path and case.get('integration_path') == integration_path:
        score += 5.0

    kind = current_context.get('kind')
    if kind and case.get('kind') == kind:
        score += 2.0

    current_failure_signature = safe_dict(current_context.get('failure_signature'))
    case_failure_signature = safe_dict(case.get('failure_signature'))
    stop_layer = current_context.get('stop_layer')
    case_failure_stop_layer = case_failure_signature.get('stop_layer') or case.get('trigger_stop_layer') or case.get('stop_layer')
    if stop_layer and case_failure_stop_layer == stop_layer:
        score += 5.0
    current_error_family = current_failure_signature.get('error_family')
    if current_error_family and case_failure_signature.get('error_family') == current_error_family:
        score += 4.0

    current_required_edit_paths = current_context.get('required_edit_paths')
    if isinstance(current_required_edit_paths, set) and current_required_edit_paths:
        required_overlap = current_required_edit_paths & _string_set(case.get('required_edit_paths'))
        score += min(len(required_overlap), 2) * 2.0

    current_files_to_edit = current_context.get('files_to_edit')
    if isinstance(current_files_to_edit, set) and current_files_to_edit:
        file_overlap = current_files_to_edit & _string_set(case.get('files_to_edit'))
        score += min(len(file_overlap), 2) * 1.0

    current_tokens = current_context.get('tokens')
    if isinstance(current_tokens, set) and current_tokens:
        overlap = current_tokens & _keyword_tokens(
            case.get('objective'),
            case.get('key_idea'),
            case.get('why_works'),
            case.get('error'),
            case.get('trigger_error'),
            safe_dict(case.get('strategy_delta')),
            case_failure_signature,
            case.get('repair_hypothesis'),
            case.get('post_error'),
            case.get('stop_layer'),
            case.get('post_stop_layer'),
            case.get('component_type'),
            safe_list(case.get('tags')),
        )
        score += min(len(overlap), 6) * 1.5

    repair_outcome = safe_dict(case.get('repair_outcome'))
    if repair_outcome.get('effective') is True:
        score += 5.0
    if repair_outcome.get('resolved_failure') is True:
        score += 2.0
    if repair_outcome.get('improved') is True:
        score += 1.5
    if repair_outcome.get('reverted') is True:
        score -= 6.0

    if case.get('passed') is True:
        score += 1.5

    baseline_delta = case.get('baseline_delta')
    if isinstance(baseline_delta, (int, float)):
        clipped_delta = max(min(float(baseline_delta), 0.1), -0.1)
        score += clipped_delta * 20.0

    stage_score = case.get('stage_score')
    if isinstance(stage_score, (int, float)):
        score += max(float(stage_score), 0.0) * 0.1

    return score


def load_similar_case_memories(
    *,
    task_context: Dict[str, Any],
    attempt_spec: Optional[Dict[str, Any]] = None,
    failure_feedback: Optional[Dict[str, Any]] = None,
    latest_attempt_result: Optional[Dict[str, Any]] = None,
    latest_repair_plan: Optional[Dict[str, Any]] = None,
    top_k: int = 3,
    case_memory_path: Optional[Path] = None,
) -> list[Dict[str, Any]]:
    current_context = _build_memory_query_context(
        task_context=task_context,
        attempt_spec=attempt_spec,
        failure_feedback=failure_feedback,
        latest_attempt_result=latest_attempt_result,
        latest_repair_plan=latest_repair_plan,
    )
    scored: list[tuple[float, Dict[str, Any]]] = []
    current_paper_slug = current_context.get('paper_slug')
    current_attempt_id = current_context.get('attempt_id')
    resolved_case_memory_path = case_memory_path or DEFAULT_CASE_MEMORY_PATH

    for case in load_case_memory_records(
        task_context,
        case_memory_path=resolved_case_memory_path,
    ):
        if (
            current_paper_slug is not None
            and current_attempt_id is not None
            and case.get('paper_slug') == current_paper_slug
            and case.get('attempt_id') == current_attempt_id
        ):
            continue

        score = _score_memory_case(case, current_context)
        if score <= 0:
            continue
        scored.append((score, case))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [case for _, case in scored[:top_k]]


def format_case_memory_block(cases: list[Dict[str, Any]]) -> str:
    if not cases:
        return ''

    lines = [
        '相似历史案例参考（只吸收失败原因和策略变化，不要机械复用旧方案）:',
    ]
    for index, case in enumerate(cases, start=1):
        strategy_delta = safe_dict(case.get('strategy_delta'))
        failure_signature = safe_dict(case.get('failure_signature'))
        repair_outcome = safe_dict(case.get('repair_outcome'))
        strategy_changes = [
            str(item)
            for item in safe_list(strategy_delta.get('what_changes_now'))
            if isinstance(item, str) and item.strip()
        ]
        required_edit_paths = normalize_string_list(case.get('required_edit_paths'))
        metric_name = case.get('primary_metric_name') or 'primary_metric'
        metric_value = case.get('primary_metric')
        metric_text = (
            f'{metric_name}={float(metric_value):.6f}'
            if isinstance(metric_value, (int, float))
            else f'{metric_name}=n/a'
        )
        baseline_delta = case.get('baseline_delta')
        baseline_delta_text = (
            f'{float(baseline_delta):+.6f}'
            if isinstance(baseline_delta, (int, float))
            else 'n/a'
        )
        lines.extend(
            [
                f'案例 {index}: {case.get("paper_slug")}#{case.get("attempt_id")}',
                (
                    f'- integration_path={case.get("integration_path")} | kind={case.get("kind")} '
                    f'| stop_layer={failure_signature.get("stop_layer") or case.get("trigger_stop_layer") or case.get("stop_layer")} '
                    f'| error_family={failure_signature.get("error_family") or "other"} '
                    f'| passed={case.get("passed")}'
                ),
                f'- error: {case.get("error") or "n/a"}',
                f'- objective: {case.get("objective") or case.get("key_idea") or "n/a"}',
                f'- strategy change: {"; ".join(strategy_changes[:2]) if strategy_changes else "n/a"}',
                (
                    f'- edit scope: {"; ".join(required_edit_paths[:2]) if required_edit_paths else "n/a"} '
                    f'| repair_effective={repair_outcome.get("effective")} | reverted={repair_outcome.get("reverted")}'
                ),
                f'- repair hypothesis: {case.get("repair_hypothesis") or case.get("why_works") or "n/a"}',
                (
                    f'- post-repair stop_layer: {case.get("post_stop_layer") or "n/a"} '
                    f'| post-error: {case.get("post_error") or "n/a"}'
                ),
                (
                    f'- outcome: {metric_text}, baseline_delta={baseline_delta_text}, '
                    f'stage_score={case.get("stage_score")}, repair_rounds_used={case.get("repair_rounds_used")}'
                ),
            ]
        )
    return '\n'.join(lines)


def append_memory_block(prompt: str, memory_block: str) -> str:
    if not memory_block:
        return prompt
    return prompt.rstrip() + '\n\n' + memory_block + '\n'
