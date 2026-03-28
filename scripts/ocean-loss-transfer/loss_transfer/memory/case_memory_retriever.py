"""
@file case_memory_retriever.py
@description Shared case-memory retrieval and prompt formatting helpers for agent workflows.
@author kongzhiquan
@date 2026-03-28
@version 1.0.0

@changelog
  - 2026-03-28 kongzhiquan: v1.0.0 extract shared case-memory retrieval and prompt-formatting helpers
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

from loss_transfer.memory.case_memory_store import (
    DEFAULT_CASE_MEMORY_PATH,
    load_case_memory_records,
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
    tokens = _keyword_tokens(
        attempt_payload.get('objective'),
        latest_attempt_payload.get('error'),
        failure_payload.get('error'),
        strategy_delta,
        repair_plan.get('failure_hypothesis'),
        repair_plan.get('fallback_plan'),
        failure_payload.get('stop_layer'),
        latest_attempt_payload.get('stop_layer'),
    )

    return {
        'paper_slug': task_context.get('paper_slug'),
        'integration_path': integration.get('recommended_path'),
        'kind': attempt_payload.get('kind') or latest_attempt_payload.get('kind'),
        'stop_layer': failure_payload.get('stop_layer') or latest_attempt_payload.get('stop_layer'),
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

    stop_layer = current_context.get('stop_layer')
    if stop_layer and case.get('stop_layer') == stop_layer:
        score += 7.0

    current_tokens = current_context.get('tokens')
    if isinstance(current_tokens, set) and current_tokens:
        overlap = current_tokens & _keyword_tokens(
            case.get('objective'),
            case.get('key_idea'),
            case.get('why_works'),
            case.get('error'),
            safe_dict(case.get('strategy_delta')),
            case.get('repair_hypothesis'),
            case.get('post_error'),
            case.get('stop_layer'),
            case.get('post_stop_layer'),
            case.get('component_type'),
            safe_list(case.get('tags')),
        )
        score += min(len(overlap), 6) * 1.5

    if case.get('passed') is True:
        score += 1.5

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
        strategy_changes = [
            str(item)
            for item in safe_list(strategy_delta.get('what_changes_now'))
            if isinstance(item, str) and item.strip()
        ]
        metric_name = case.get('primary_metric_name') or 'primary_metric'
        metric_value = case.get('primary_metric')
        metric_text = (
            f'{metric_name}={float(metric_value):.6f}'
            if isinstance(metric_value, (int, float))
            else f'{metric_name}=n/a'
        )
        lines.extend(
            [
                f'案例 {index}: {case.get("paper_slug")}#{case.get("attempt_id")}',
                f'- integration_path={case.get("integration_path")} | kind={case.get("kind")} | stop_layer={case.get("stop_layer")} | passed={case.get("passed")}',
                f'- error: {case.get("error") or "n/a"}',
                f'- objective: {case.get("objective") or case.get("key_idea") or "n/a"}',
                f'- strategy change: {"; ".join(strategy_changes[:2]) if strategy_changes else "n/a"}',
                f'- repair hypothesis: {case.get("repair_hypothesis") or case.get("why_works") or "n/a"}',
                f'- post-repair stop_layer: {case.get("post_stop_layer") or "n/a"} | post-error: {case.get("post_error") or "n/a"}',
                f'- outcome: {metric_text}, stage_score={case.get("stage_score")}, repair_rounds_used={case.get("repair_rounds_used")}',
            ]
        )
    return '\n'.join(lines)


def append_memory_block(prompt: str, memory_block: str) -> str:
    if not memory_block:
        return prompt
    return prompt.rstrip() + '\n\n' + memory_block + '\n'
