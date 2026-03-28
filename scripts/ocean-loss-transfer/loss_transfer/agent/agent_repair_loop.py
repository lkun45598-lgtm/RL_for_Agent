"""
@file agent_repair_loop.py
@description Agentic execution loop driven by task_context.json and analysis_plan.json.
@author Leizheng
@date 2026-03-25
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loss_transfer.agent.agent_artifact_generator import generate_followup_attempt
from loss_transfer.attempts.integration_policy import merge_attempt_with_edit_policy, resolve_recommended_integration_path
from loss_transfer.common.contract_validation import write_contract_validation
from loss_transfer.common.decision_trace import write_decision_trace
from loss_transfer.common.run_manifest import write_run_manifest
from loss_transfer.common.trajectory_logger import append_trajectory_event, ensure_experiment_dir, write_json
from loss_transfer.formula.formula_code_generator import supports_formula_codegen


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
        'attempt_count': len(attempts),
        'best_attempt_id': best_attempt_id,
        'best_metric_name': best_metric_name,
        'best_metric_value': best_metric_value,
    }
    if best_attempt_id is not None:
        for attempt in attempts:
            if isinstance(attempt, dict) and attempt.get('attempt_id') == best_attempt_id:
                summary['best_reward_summary'] = attempt.get('reward_summary')
                summary['best_strategy_delta'] = attempt.get('strategy_delta')
                break
    if attempts and all(not bool(attempt.get('passed')) for attempt in attempts):
        summary['status'] = 'completed_with_failures'
    return summary


def _attempt_signature(attempt: Dict[str, Any]) -> str:
    keys = (
        'kind',
        'variant',
        'objective',
        'code_path',
        'run_training',
        'files_to_edit',
        'required_edit_paths',
        'notes',
        'strategy_delta',
    )
    payload = {key: attempt.get(key) for key in keys}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _build_contract_error_result(
    *,
    paper_slug: str,
    task_context: Dict[str, Any],
    analysis_plan_path: Optional[str],
    output_dir: Optional[str],
    bootstrap_formula: bool,
    max_attempts: int,
    agent_service_url: Optional[str],
    initial_manifest: Dict[str, Any],
    contract_validation: Dict[str, Any],
) -> Dict[str, Any]:
    experiment_dir = ensure_experiment_dir(paper_slug, output_dir=output_dir)
    result = {
        'status': 'contract_error',
        'paper_slug': paper_slug,
        'task_context_path': task_context.get('paths', {}).get('task_context_path'),
        'analysis_plan_path': analysis_plan_path,
        'routing_audit_path': contract_validation.get('paths', {}).get('routing_audit_path'),
        'contract_validation_path': contract_validation.get('contract_validation_path'),
        'run_manifest_path': initial_manifest.get('run_manifest_path'),
        'contract_validation_errors': contract_validation.get('errors', []),
        'contract_validation_warnings': contract_validation.get('warnings', []),
    }
    write_run_manifest(
        experiment_dir=experiment_dir,
        paper_slug=paper_slug,
        task_context=task_context,
        mode='agent_loop',
        bootstrap_formula=bootstrap_formula,
        max_attempts=max_attempts,
        auto_generate_plan=analysis_plan_path is None,
        service_url=agent_service_url,
        analysis_plan_path=analysis_plan_path,
        loop_summary=result,
    )
    write_json(experiment_dir / 'agent_loop_summary.json', result)
    return result


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
    initial_manifest = write_run_manifest(
        experiment_dir=experiment_dir,
        paper_slug=paper_slug,
        task_context=task_context,
        mode='agent_loop',
        bootstrap_formula=bootstrap_formula,
        max_attempts=max_attempts,
        auto_generate_plan=analysis_plan_path is None,
        service_url=agent_service_url,
        analysis_plan_path=analysis_plan_path,
    )

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
    contract_validation_result: Dict[str, Any]
    if analysis_plan_path:
        contract_validation_result = write_contract_validation(
            experiment_dir=experiment_dir,
            paper_slug=paper_slug,
            task_context=task_context,
            analysis_plan_path=analysis_plan_path,
        )
        if contract_validation_result.get('status') == 'error':
            return _build_contract_error_result(
                paper_slug=paper_slug,
                task_context=task_context,
                analysis_plan_path=analysis_plan_path,
                output_dir=output_dir,
                bootstrap_formula=bootstrap_formula,
                max_attempts=max_attempts,
                agent_service_url=agent_service_url,
                initial_manifest=initial_manifest,
                contract_validation=contract_validation_result,
            )
        normalized_analysis_plan = contract_validation_result.get('normalized_analysis_plan')
        analysis_plan = normalized_analysis_plan if isinstance(normalized_analysis_plan, dict) else {}
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
        contract_validation_result = write_contract_validation(
            experiment_dir=experiment_dir,
            paper_slug=paper_slug,
            task_context=task_context,
            analysis_plan=analysis_plan if analysis_plan else None,
            analysis_plan_path=str(experiment_dir / 'analysis_plan.json') if (experiment_dir / 'analysis_plan.json').exists() else None,
        )
        if contract_validation_result.get('status') == 'error':
            return _build_contract_error_result(
                paper_slug=paper_slug,
                task_context=task_context,
                analysis_plan_path=str(experiment_dir / 'analysis_plan.json') if (experiment_dir / 'analysis_plan.json').exists() else None,
                output_dir=output_dir,
                bootstrap_formula=bootstrap_formula,
                max_attempts=max_attempts,
                agent_service_url=agent_service_url,
                initial_manifest=initial_manifest,
                contract_validation=contract_validation_result,
            )
    else:
        attempts = []
        contract_validation_result = write_contract_validation(
            experiment_dir=experiment_dir,
            paper_slug=paper_slug,
            task_context=task_context,
        )
        if contract_validation_result.get('status') == 'error':
            return _build_contract_error_result(
                paper_slug=paper_slug,
                task_context=task_context,
                analysis_plan_path=analysis_plan_path,
                output_dir=output_dir,
                bootstrap_formula=bootstrap_formula,
                max_attempts=max_attempts,
                agent_service_url=agent_service_url,
                initial_manifest=initial_manifest,
                contract_validation=contract_validation_result,
            )

    resolved_analysis_plan_path = (
        str(experiment_dir / 'analysis_plan.json')
        if (experiment_dir / 'analysis_plan.json').exists()
        else analysis_plan_path
    )
    routing_audit_result = {
        'routing_audit_path': contract_validation_result.get('paths', {}).get('routing_audit_path'),
        'routing_audit': contract_validation_result.get('routing_audit'),
        'effective_integration_path': contract_validation_result.get('effective_integration_path'),
        'effective_integration_path_source': contract_validation_result.get('effective_integration_path_source'),
    }

    if not attempts:
        result = {
            'status': 'analysis_required',
            'paper_slug': paper_slug,
            'task_context_path': task_context.get('paths', {}).get('task_context_path'),
            'routing_audit_path': routing_audit_result.get('routing_audit_path'),
            'contract_validation_path': contract_validation_result.get('contract_validation_path'),
            'run_manifest_path': initial_manifest.get('run_manifest_path'),
            'suggested_next_step': (
                'Read task_context.json, write analysis_plan.json, then rerun with --analysis_plan_json'
            ),
        }
        write_run_manifest(
            experiment_dir=experiment_dir,
            paper_slug=paper_slug,
            task_context=task_context,
            mode='agent_loop',
            bootstrap_formula=bootstrap_formula,
            max_attempts=max_attempts,
            auto_generate_plan=analysis_plan_path is None,
            service_url=agent_service_url,
            analysis_plan_path=resolved_analysis_plan_path,
            loop_summary=result,
        )
        write_json(experiment_dir / 'agent_loop_summary.json', result)
        return result

    stop_on_first_pass = bool(analysis_plan.get('stop_on_first_pass', False))
    executed_attempts: List[Dict[str, Any]] = []
    from loss_transfer.attempts.attempt_executor import execute_attempt
    integration_path = resolve_recommended_integration_path(task_context, analysis_plan)
    seen_attempt_signatures = {_attempt_signature(attempt) for attempt in attempts}
    from loss_transfer.attempts.attempt_executor import execute_attempt

    index = 0
    while index < len(attempts) and len(executed_attempts) < max_attempts:
        attempt = attempts[index]
        attempt_id = index + 1
        append_trajectory_event(
            paper_slug,
            'attempt_planned',
            {
                'attempt_id': attempt_id,
                'name': attempt.get('name'),
                'kind': attempt.get('kind'),
                'strategy_delta': attempt.get('strategy_delta'),
            },
            output_dir=output_dir,
        )
        result = execute_attempt(
            paper_slug=paper_slug,
            attempt_id=attempt_id,
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
        index += 1

        if index < len(attempts):
            continue
        if len(attempts) >= max_attempts:
            continue
        if result.get('passed'):
            continue
        if not agent_service_url:
            continue
        task_context_path = task_context.get('paths', {}).get('task_context_path')
        if not isinstance(task_context_path, str) or not task_context_path:
            continue
        result_path = ((result.get('paths') or {}) if isinstance(result.get('paths'), dict) else {}).get('result_path')
        if not isinstance(result_path, str) or not result_path:
            continue

        followup_attempt_path = experiment_dir / f'followup_attempt_{len(attempts) + 1}.json'
        replan = generate_followup_attempt(
            result_path,
            task_context_path=task_context_path,
            output_attempt_path=str(followup_attempt_path),
            analysis_plan_path=str(experiment_dir / 'analysis_plan.json') if (experiment_dir / 'analysis_plan.json').exists() else analysis_plan_path,
            service_url=agent_service_url,
            api_key=agent_api_key,
            max_attempts=max_attempts,
            next_attempt_id=len(attempts) + 1,
        )
        if replan.get('status') != 'success':
            append_trajectory_event(
                paper_slug,
                'attempt_replan_failed',
                {
                    'attempt_id': attempt_id,
                    'next_attempt_id': len(attempts) + 1,
                    'error': replan.get('error'),
                    'attempt_path': replan.get('attempt_path'),
                },
                output_dir=output_dir,
            )
            continue

        new_attempt_spec = merge_attempt_with_edit_policy(
            dict(replan['attempt_spec']),
            integration_path=integration_path,
        )
        signature = _attempt_signature(new_attempt_spec)
        if signature in seen_attempt_signatures:
            append_trajectory_event(
                paper_slug,
                'attempt_replan_skipped_duplicate',
                {
                    'attempt_id': attempt_id,
                    'next_attempt_id': len(attempts) + 1,
                    'attempt_path': replan.get('attempt_path'),
                },
                output_dir=output_dir,
            )
            continue

        seen_attempt_signatures.add(signature)
        attempts.append(new_attempt_spec)
        append_trajectory_event(
            paper_slug,
            'attempt_replanned',
            {
                'source_attempt_id': attempt_id,
                'next_attempt_id': len(attempts),
                'attempt_path': replan.get('attempt_path'),
                'repair_plan_path': replan.get('latest_repair_plan_path'),
                'name': new_attempt_spec.get('name'),
                'kind': new_attempt_spec.get('kind'),
                'strategy_delta': new_attempt_spec.get('strategy_delta'),
            },
            output_dir=output_dir,
        )

    summary = _summarize_attempts(
        paper_slug=paper_slug,
        task_context=task_context,
        attempts=executed_attempts,
        analysis_plan_path=str(experiment_dir / 'analysis_plan.json') if (experiment_dir / 'analysis_plan.json').exists() else analysis_plan_path,
        output_dir=output_dir,
    )
    summary.update(
        routing_audit_result
    )
    summary['contract_validation_path'] = contract_validation_result.get('contract_validation_path')
    summary.update(
        write_decision_trace(
            experiment_dir=experiment_dir,
            paper_slug=paper_slug,
            task_context=task_context,
            analysis_plan_path=str(experiment_dir / 'analysis_plan.json') if (experiment_dir / 'analysis_plan.json').exists() else analysis_plan_path,
            trajectory_path=str(experiment_dir / 'trajectory.jsonl'),
            attempts=executed_attempts,
            routing_audit=routing_audit_result.get('routing_audit'),
        )
    )
    summary.update(
        write_run_manifest(
            experiment_dir=experiment_dir,
            paper_slug=paper_slug,
            task_context=task_context,
            mode='agent_loop',
            bootstrap_formula=bootstrap_formula,
            max_attempts=max_attempts,
            auto_generate_plan=analysis_plan_path is None,
            service_url=agent_service_url,
            analysis_plan_path=str(experiment_dir / 'analysis_plan.json') if (experiment_dir / 'analysis_plan.json').exists() else analysis_plan_path,
            loop_summary=summary,
        )
    )
    write_json(experiment_dir / 'agent_loop_summary.json', summary)
    return summary
