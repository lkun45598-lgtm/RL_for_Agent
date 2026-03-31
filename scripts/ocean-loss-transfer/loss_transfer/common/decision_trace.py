"""
@file decision_trace.py

@description Build RL-friendly decision-trace samples from executed loss-transfer attempts.
@author OpenAI Codex
@contributors kongzhiquan
@date 2026-03-26
@version 1.3.0

@changelog
  - 2026-03-26 OpenAI Codex: v1.0.0 initial version
  - 2026-03-28 kongzhiquan: v1.1.0 reuse shared case-memory store for persistence
  - 2026-03-28 kongzhiquan: v1.2.0 merge routing-audit provenance with shared case-memory export
  - 2026-03-30 kongzhiquan: v1.3.0 export case_memory.v2 records with richer failure and repair summaries
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loss_transfer.memory.case_memory_store import (
    DEFAULT_CASE_MEMORY_PATH,
    build_failure_signature,
    build_repair_outcome_summary,
    merge_case_memory_records,
)


_DEFAULT_CASE_MEMORY_PATH = DEFAULT_CASE_MEMORY_PATH


def _safe_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _as_bool(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _latest_repair_round(attempt: Dict[str, Any]) -> Dict[str, Any]:
    for round_info in reversed(_safe_list(attempt.get('repair_rounds'))):
        if isinstance(round_info, dict):
            return round_info
    return {}


def _normalize_string_list(value: Any) -> List[str]:
    items = value if isinstance(value, list) else []
    normalized: List[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if not candidate or candidate in normalized:
            continue
        normalized.append(candidate)
    return normalized


def build_case_memory_record(
    *,
    trace_record: Dict[str, Any],
    attempt: Dict[str, Any],
) -> Dict[str, Any]:
    state = _safe_dict(trace_record.get('state'))
    action = _safe_dict(trace_record.get('action'))
    reward = _safe_dict(trace_record.get('reward'))
    outcome = _safe_dict(trace_record.get('outcome'))
    provenance = _safe_dict(trace_record.get('provenance'))
    latest_repair_round = _latest_repair_round(attempt)
    failure_feedback = _safe_dict(latest_repair_round.get('failure_feedback'))
    repair_payload = _safe_dict(latest_repair_round.get('repair'))
    repair_plan_summary = _safe_dict(repair_payload.get('repair_plan_summary'))
    metrics = _safe_dict(outcome.get('metrics'))
    repair_rounds = _safe_list(attempt.get('repair_rounds'))
    reverted_repair_rounds = sum(
        1
        for round_info in repair_rounds
        if isinstance(round_info, dict) and bool(round_info.get('reverted', False))
    )
    trigger_stop_layer = latest_repair_round.get('trigger_stop_layer') or outcome.get('stop_layer')
    trigger_error = failure_feedback.get('error') or outcome.get('error')
    trigger_metrics = _safe_dict(failure_feedback.get('metrics')) or metrics

    return {
        'schema_version': 'case_memory.v2',
        'paper_slug': trace_record.get('paper_slug'),
        'attempt_id': trace_record.get('attempt_id'),
        'integration_path': state.get('integration_path'),
        'kind': action.get('kind'),
        'name': action.get('name'),
        'objective': action.get('objective'),
        'strategy_delta': _safe_dict(action.get('strategy_delta')),
        'files_to_edit': _normalize_string_list(action.get('files_to_edit')),
        'required_edit_paths': _normalize_string_list(action.get('required_edit_paths')),
        'evidence_refs': _normalize_string_list(action.get('evidence_refs')),
        'stop_layer': outcome.get('stop_layer'),
        'error': outcome.get('error'),
        'trigger_stop_layer': trigger_stop_layer,
        'trigger_error': trigger_error,
        'passed': outcome.get('passed'),
        'primary_metric_name': reward.get('primary_metric_name'),
        'primary_metric': reward.get('primary_metric'),
        'stage_score': reward.get('stage_score'),
        'repair_rounds_used': reward.get('repair_rounds_used'),
        'baseline_delta': reward.get('baseline_delta'),
        'val_ssim': reward.get('val_ssim'),
        'val_psnr': reward.get('val_psnr'),
        'training_curve_trend': reward.get('training_curve_trend'),
        'training_curve_last_epoch': reward.get('training_curve_last_epoch'),
        'reverted_repair_rounds': reverted_repair_rounds,
        'requires_model_changes': state.get('requires_model_changes'),
        'loss_only_pipeline_viable': state.get('loss_only_pipeline_viable'),
        'formula_requires_model_changes': state.get('formula_requires_model_changes'),
        'metrics': metrics,
        'repair_hypothesis': repair_plan_summary.get('failure_hypothesis'),
        'post_stop_layer': latest_repair_round.get('post_stop_layer'),
        'post_error': latest_repair_round.get('post_error'),
        'failure_signature': build_failure_signature(
            stop_layer=trigger_stop_layer,
            error=trigger_error,
            metrics=trigger_metrics,
        ),
        'repair_outcome': build_repair_outcome_summary(
            stop_layer=trigger_stop_layer,
            post_stop_layer=latest_repair_round.get('post_stop_layer'),
            post_error=latest_repair_round.get('post_error'),
            passed=outcome.get('passed'),
            repair_rounds_used=reward.get('repair_rounds_used'),
            baseline_delta=reward.get('baseline_delta'),
            reverted=reverted_repair_rounds > 0,
            status=latest_repair_round.get('status'),
        ),
        'provenance': {
            'result_path': provenance.get('result_path'),
            'analysis_plan_path': provenance.get('analysis_plan_path'),
            'trajectory_path': provenance.get('trajectory_path'),
        },
    }


def build_decision_trace_record(
    *,
    paper_slug: str,
    task_context: Dict[str, Any],
    attempt: Dict[str, Any],
    analysis_plan_path: Optional[str],
    trajectory_path: Optional[str],
    routing_audit: Optional[Dict[str, Any]] = None,
    previous_attempt: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    paths = _safe_dict(task_context.get('paths'))
    integration = _safe_dict(task_context.get('integration_assessment'))
    audit_paths = _safe_dict(_safe_dict(routing_audit).get('paths'))
    effective_route = _safe_dict(_safe_dict(_safe_dict(routing_audit).get('routes')).get('effective'))
    formula_interface = _safe_dict(task_context.get('formula_interface'))
    reward_summary = _safe_dict(attempt.get('reward_summary'))
    metrics = _safe_dict(attempt.get('metrics'))
    previous_attempt_payload = _safe_dict(previous_attempt)
    previous_reward = _safe_dict(previous_attempt_payload.get('reward_summary'))
    strategy_delta = _safe_dict(attempt.get('strategy_delta'))
    attempt_paths = _safe_dict(attempt.get('paths'))

    return {
        'schema_version': 'decision_trace.v1',
        'paper_slug': paper_slug,
        'attempt_id': attempt.get('attempt_id'),
        'state': {
            'integration_path': effective_route.get('canonical_path') or integration.get('recommended_path'),
            'integration_path_raw': effective_route.get('raw_path') or integration.get('recommended_path_raw'),
            'integration_path_status': effective_route.get('status') or integration.get('recommended_path_status'),
            'integration_path_source': effective_route.get('selected_from'),
            'requires_model_changes': integration.get('requires_model_changes'),
            'loss_only_pipeline_viable': integration.get('loss_only_pipeline_viable'),
            'formula_requires_model_changes': formula_interface.get('requires_model_changes'),
            'previous_attempt_id': strategy_delta.get('previous_attempt_id'),
            'previous_stop_layer': previous_attempt_payload.get('stop_layer'),
            'previous_error': previous_attempt_payload.get('error'),
            'previous_primary_metric': previous_reward.get('primary_metric'),
            'previous_stage_score': previous_reward.get('stage_score'),
        },
        'action': {
            'name': attempt.get('name'),
            'kind': attempt.get('kind'),
            'variant': attempt.get('variant'),
            'objective': attempt.get('objective'),
            'run_training': attempt.get('run_training'),
            'files_to_edit': _safe_list(attempt.get('files_to_edit')),
            'required_edit_paths': _safe_list(attempt.get('required_edit_paths')),
            'evidence_refs': _safe_list(attempt.get('evidence_refs')),
            'strategy_delta': strategy_delta,
        },
        'reward': reward_summary,
        'outcome': {
            'status': attempt.get('status'),
            'passed': attempt.get('passed'),
            'stop_layer': attempt.get('stop_layer'),
            'error': attempt.get('error'),
            'metrics': metrics,
            'passed_static': attempt.get('passed_static'),
            'passed_smoke': attempt.get('passed_smoke'),
            'passed_formula_alignment': attempt.get('passed_formula_alignment'),
        },
        'provenance': {
            'task_context_path': paths.get('task_context_path'),
            'analysis_plan_path': analysis_plan_path,
            'trajectory_path': trajectory_path,
            'routing_audit_path': audit_paths.get('routing_audit_path') or paths.get('routing_audit_path'),
            'attempt_dir': attempt_paths.get('attempt_dir'),
            'result_path': attempt_paths.get('result_path'),
            'code_path': attempt_paths.get('code_path'),
        },
    }


def build_rl_decision_dataset_record(
    *,
    trace_record: Dict[str, Any],
    is_terminal: bool,
    next_attempt_id: Optional[int],
) -> Dict[str, Any]:
    state = _safe_dict(trace_record.get('state'))
    action = _safe_dict(trace_record.get('action'))
    reward = _safe_dict(trace_record.get('reward'))
    outcome = _safe_dict(trace_record.get('outcome'))
    strategy_delta = _safe_dict(action.get('strategy_delta'))
    files_to_edit = _safe_list(action.get('files_to_edit'))
    required_edit_paths = _safe_list(action.get('required_edit_paths'))

    return {
        'schema_version': 'rl_decision_dataset.v1',
        'paper_slug': trace_record.get('paper_slug'),
        'attempt_id': trace_record.get('attempt_id'),
        'terminal': is_terminal,
        'next_attempt_id': next_attempt_id,
        'state_features': {
            'integration_path': state.get('integration_path'),
            'requires_model_changes': _as_bool(state.get('requires_model_changes')),
            'loss_only_pipeline_viable': _as_bool(state.get('loss_only_pipeline_viable')),
            'formula_requires_model_changes': _as_bool(state.get('formula_requires_model_changes')),
            'previous_attempt_id': state.get('previous_attempt_id'),
            'previous_stop_layer': state.get('previous_stop_layer'),
            'previous_primary_metric': state.get('previous_primary_metric'),
            'previous_stage_score': state.get('previous_stage_score'),
        },
        'action_features': {
            'name': action.get('name'),
            'kind': action.get('kind'),
            'variant': action.get('variant'),
            'run_training': _as_bool(action.get('run_training')),
            'files_to_edit_count': len(files_to_edit),
            'required_edit_paths_count': len(required_edit_paths),
            'strategy_has_delta': bool(strategy_delta),
            'strategy_delta': strategy_delta,
        },
        'reward': {
            'passed': _as_bool(reward.get('passed')),
            'primary_metric_name': reward.get('primary_metric_name'),
            'primary_metric': reward.get('primary_metric'),
            'stage_score': reward.get('stage_score'),
            'val_ssim': reward.get('val_ssim'),
            'val_psnr': reward.get('val_psnr'),
            'repair_rounds_used': reward.get('repair_rounds_used'),
        },
        'outcome': {
            'status': outcome.get('status'),
            'stop_layer': outcome.get('stop_layer'),
            'error': outcome.get('error'),
        },
        'provenance': _safe_dict(trace_record.get('provenance')),
    }


def _build_rl_records_from_trace_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rl_records: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        next_attempt_id = None
        if index + 1 < len(records):
            candidate_next_id = records[index + 1].get('attempt_id')
            if isinstance(candidate_next_id, int):
                next_attempt_id = candidate_next_id

        outcome = _safe_dict(record.get('outcome'))
        passed = outcome.get('passed')
        is_terminal = bool(index == len(records) - 1 or passed is True)
        rl_records.append(
            build_rl_decision_dataset_record(
                trace_record=record,
                is_terminal=is_terminal,
                next_attempt_id=next_attempt_id,
            )
        )
    return rl_records


def export_rl_dataset_from_decision_trace(
    decision_trace_path: Path,
    *,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    source_path = decision_trace_path.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f'decision trace file does not exist: {source_path}')

    records: List[Dict[str, Any]] = []
    for line in source_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)

    target_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else source_path.with_name('rl_decision_dataset.jsonl')
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)

    rl_records = _build_rl_records_from_trace_records(records)
    with target_path.open('w', encoding='utf-8') as handle:
        for record in rl_records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    return {
        'decision_trace_path': str(source_path),
        'decision_trace_count': len(records),
        'rl_dataset_path': str(target_path),
        'rl_dataset_count': len(rl_records),
    }


def write_decision_trace(
    *,
    experiment_dir: Path,
    paper_slug: str,
    task_context: Dict[str, Any],
    analysis_plan_path: Optional[str],
    trajectory_path: Optional[str],
    attempts: List[Dict[str, Any]],
    routing_audit: Optional[Dict[str, Any]] = None,
    case_memory_path: Optional[Path] = None,
) -> Dict[str, Any]:
    decision_trace_path = experiment_dir / 'decision_trace.jsonl'
    rl_dataset_path = experiment_dir / 'rl_decision_dataset.jsonl'
    attempts_by_id = {
        int(item['attempt_id']): item
        for item in attempts
        if isinstance(item, dict) and isinstance(item.get('attempt_id'), int)
    }

    records: List[Dict[str, Any]] = []
    case_memory_records: List[Dict[str, Any]] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        strategy_delta = _safe_dict(attempt.get('strategy_delta'))
        previous_attempt = None
        previous_attempt_id = strategy_delta.get('previous_attempt_id')
        if isinstance(previous_attempt_id, int):
            previous_attempt = attempts_by_id.get(previous_attempt_id)
        trace_record = build_decision_trace_record(
            paper_slug=paper_slug,
            task_context=task_context,
            attempt=attempt,
            analysis_plan_path=analysis_plan_path,
            trajectory_path=trajectory_path,
            previous_attempt=previous_attempt,
        )
        records.append(trace_record)
        case_memory_records.append(
            build_case_memory_record(
                trace_record=trace_record,
                attempt=attempt,
            )
        )

    decision_trace_path.parent.mkdir(parents=True, exist_ok=True)
    with decision_trace_path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    rl_records = _build_rl_records_from_trace_records(records)

    with rl_dataset_path.open('w', encoding='utf-8') as handle:
        for record in rl_records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    resolved_case_memory_path = (
        case_memory_path.expanduser().resolve()
        if case_memory_path is not None
        else _DEFAULT_CASE_MEMORY_PATH
    )
    case_memory_count = merge_case_memory_records(
        case_memory_path=resolved_case_memory_path,
        records=case_memory_records,
    )

    return {
        'decision_trace_path': str(decision_trace_path),
        'decision_trace_count': len(records),
        'rl_dataset_path': str(rl_dataset_path),
        'rl_dataset_count': len(rl_records),
        'case_memory_path': str(resolved_case_memory_path),
        'case_memory_count': case_memory_count,
    }
