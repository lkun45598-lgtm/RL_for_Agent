"""
@file agent_repair_loop.py
@description Agentic execution loop driven by task_context.json and analysis_plan.json.
@author Leizheng
@date 2026-03-25
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loss_transfer.formula.formula_code_generator import supports_formula_codegen
from loss_transfer.attempts.integration_policy import merge_attempt_with_edit_policy, resolve_recommended_integration_path
from loss_transfer.common.trajectory_logger import append_trajectory_event, ensure_experiment_dir, write_json
from loss_transfer.agent.validate_analysis_plan import validate_analysis_plan


def _load_analysis_plan(path: str) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('analysis_plan.json must contain a JSON object')
    validation = validate_analysis_plan(data)
    if validation['status'] == 'error':
        raise ValueError(
            'analysis_plan.json validation failed:\n  - '
            + '\n  - '.join(validation.get('errors', []))
        )

    normalized = validation.get('normalized_plan')
    if not isinstance(normalized, dict):
        raise ValueError('analysis_plan.json validation did not return a normalized plan')
    if validation.get('warnings'):
        normalized['validation_warnings'] = validation['warnings']
    return normalized


def _normalize_attempts(
    raw_attempts: Any,
    *,
    task_context: Dict[str, Any],
    analysis_plan: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(raw_attempts, list):
        return []

    integration_path = resolve_recommended_integration_path(task_context, analysis_plan)
    attempts: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_attempts, start=1):
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        normalized.setdefault('name', f'Attempt {idx}')
        normalized.setdefault('kind', 'agent_code')
        normalized.setdefault('run_training', True)
        attempts.append(merge_attempt_with_edit_policy(normalized, integration_path=integration_path))
    return attempts


def _build_bootstrap_attempts(task_context: Dict[str, Any], max_attempts: int) -> List[Dict[str, Any]]:
    formula_spec = task_context.get('formula_spec')
    integration_path = resolve_recommended_integration_path(task_context)
    if integration_path == 'loss_only' and isinstance(formula_spec, dict) and supports_formula_codegen(formula_spec):
        attempts = [
            {
                'name': 'Formula Native Faithful',
                'kind': 'formula_variant',
                'variant': 'faithful',
                'run_training': True,
                'notes': 'Bootstrap candidate generated directly from loss_formula.json',
            },
            {
                'name': 'Formula Native Stabilized',
                'kind': 'formula_variant',
                'variant': 'stabilized',
                'run_training': True,
                'notes': 'Bootstrap candidate with conservative numeric stabilization',
            },
        ]
        return [
            merge_attempt_with_edit_policy(attempt, integration_path=integration_path)
            for attempt in attempts[:max_attempts]
        ]

    paper_slug = str(task_context.get('paper_slug', 'paper'))
    attempts = [
        {
            'name': f'Bootstrap {integration_path} faithful',
            'kind': 'agent_code',
            'objective': (
                f'For {paper_slug}, implement the faithful loss-transfer path `{integration_path}` using '
                'paper_analysis + code_analysis + formula_spec together. Keep candidate_loss.py aligned '
                'with loss_formula.json, and if extra model-provided loss inputs are required, modify the '
                'attempt-scoped sandbox override files instead of repo-root code.'
            ),
            'run_training': True,
            'notes': 'Auto bootstrap for papers that need adapter/model-output integration rather than formula-only codegen.',
        },
        {
            'name': f'Bootstrap {integration_path} stabilized',
            'kind': 'agent_code',
            'objective': (
                f'For {paper_slug}, implement a numerically stabilized version of the `{integration_path}` path. '
                'Preserve formula alignment, BHWC/mask semantics, and if the paper couples auxiliary tensors to '
                'model.forward, repair the attempt-scoped copied model path instead of forcing a loss-only workaround.'
            ),
            'run_training': True,
            'notes': 'Auto bootstrap with stronger numeric safeguards for the same integration path.',
        },
    ]
    return [
        merge_attempt_with_edit_policy(attempt, integration_path=integration_path)
        for attempt in attempts[:max_attempts]
    ]


def _summarize_attempts(
    paper_slug: str,
    task_context: Dict[str, Any],
    attempts: List[Dict[str, Any]],
    analysis_plan_path: Optional[str],
    output_dir: Optional[str],
) -> Dict[str, Any]:
    best_attempt_id: Optional[int] = None
    best_metric_name: Optional[str] = None
    best_metric_value: Optional[float] = None

    for attempt in attempts:
        reward = attempt.get('reward_summary', {})
        metric_name = reward.get('primary_metric_name')
        metric_value = reward.get('primary_metric')
        if not isinstance(metric_name, str) or not isinstance(metric_value, (int, float)):
            continue
        if best_metric_value is None or float(metric_value) > best_metric_value:
            best_attempt_id = attempt.get('attempt_id')
            best_metric_name = metric_name
            best_metric_value = float(metric_value)

    experiment_dir = ensure_experiment_dir(paper_slug, output_dir=output_dir)
    summary: Dict[str, Any] = {
        'status': 'completed' if attempts else 'analysis_required',
        'paper_slug': paper_slug,
        'task_context_path': task_context.get('paths', {}).get('task_context_path'),
        'analysis_plan_path': analysis_plan_path,
        'trajectory_path': str(experiment_dir / 'trajectory.jsonl'),
        'attempts': attempts,
        'best_attempt_id': best_attempt_id,
        'best_metric_name': best_metric_name,
        'best_metric_value': best_metric_value,
    }
    if attempts and all(not bool(attempt.get('passed')) for attempt in attempts):
        summary['status'] = 'completed_with_failures'
    return summary


def run_agent_repair_loop(
    task_context: Dict[str, Any],
    *,
    analysis_plan_path: Optional[str] = None,
    max_attempts: int = 4,
    bootstrap_formula: bool = True,
    dataset_root: Optional[str] = None,
    output_dir: Optional[str] = None,
    agent_service_url: Optional[str] = None,
    agent_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    paper_slug = str(task_context['paper_slug'])
    experiment_dir = ensure_experiment_dir(paper_slug, output_dir=output_dir)

    append_trajectory_event(
        paper_slug,
        'task_context_ready',
        {
            'task_context_path': task_context.get('paths', {}).get('task_context_path'),
            'analysis_plan_path': analysis_plan_path,
            'bootstrap_formula': bootstrap_formula,
            'max_attempts': max_attempts,
        },
        output_dir=output_dir,
    )

    analysis_plan: Dict[str, Any] = {}
    attempts: List[Dict[str, Any]]
    if analysis_plan_path:
        analysis_plan = _load_analysis_plan(analysis_plan_path)
        write_json(experiment_dir / 'analysis_plan.json', analysis_plan)
        attempts = _normalize_attempts(
            analysis_plan.get('attempts'),
            task_context=task_context,
            analysis_plan=analysis_plan,
        )
    elif bootstrap_formula:
        attempts = _build_bootstrap_attempts(task_context, max_attempts=max_attempts)
        analysis_plan = {
            'summary': 'Auto-generated bootstrap plan from formula codegen',
            'stop_on_first_pass': False,
            'attempts': attempts,
        }
        if attempts:
            write_json(experiment_dir / 'analysis_plan.json', analysis_plan)
    else:
        attempts = []

    if not attempts:
        result = {
            'status': 'analysis_required',
            'paper_slug': paper_slug,
            'task_context_path': task_context.get('paths', {}).get('task_context_path'),
            'suggested_next_step': (
                'Read task_context.json, write analysis_plan.json, then rerun with --analysis_plan_json'
            ),
        }
        write_json(experiment_dir / 'agent_loop_summary.json', result)
        return result

    stop_on_first_pass = bool(analysis_plan.get('stop_on_first_pass', False))
    executed_attempts: List[Dict[str, Any]] = []
    from loss_transfer.attempts.attempt_executor import execute_attempt

    for index, attempt in enumerate(attempts[:max_attempts], start=1):
        append_trajectory_event(
            paper_slug,
            'attempt_planned',
            {
                'attempt_id': index,
                'name': attempt.get('name'),
                'kind': attempt.get('kind'),
            },
            output_dir=output_dir,
        )
        result = execute_attempt(
            paper_slug=paper_slug,
            attempt_id=index,
            attempt_spec=attempt,
            dataset_root=dataset_root or task_context.get('inputs', {}).get('dataset_root'),
            output_dir=output_dir,
            task_context_path=task_context.get('paths', {}).get('task_context_path'),
            analysis_plan_path=str(experiment_dir / 'analysis_plan.json') if (experiment_dir / 'analysis_plan.json').exists() else analysis_plan_path,
            agent_service_url=agent_service_url,
            agent_api_key=agent_api_key,
        )
        executed_attempts.append(result)
        if stop_on_first_pass and result.get('passed'):
            break

    summary = _summarize_attempts(
        paper_slug=paper_slug,
        task_context=task_context,
        attempts=executed_attempts,
        analysis_plan_path=str(experiment_dir / 'analysis_plan.json') if (experiment_dir / 'analysis_plan.json').exists() else analysis_plan_path,
        output_dir=output_dir,
    )
    write_json(experiment_dir / 'agent_loop_summary.json', summary)
    return summary
