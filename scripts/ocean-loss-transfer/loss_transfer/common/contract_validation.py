"""
@file contract_validation.py
@description Fail-fast contract validation for task_context, analysis_plan, and routing audit.
@author OpenAI Codex
@date 2026-03-27
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loss_transfer.agent.validate_analysis_plan import validate_analysis_plan
from loss_transfer.common.integration_path import (
    IntegrationPathContractError,
    normalize_integration_path_or_error,
)
from loss_transfer.common.routing_audit import write_routing_audit
from loss_transfer.common.trajectory_logger import write_json


def _safe_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _safe_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _record_check(
    *,
    checks: List[Dict[str, Any]],
    errors: List[str],
    warnings: List[str],
    name: str,
    ok: bool,
    severity: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    checks.append(
        {
            'name': name,
            'status': 'pass' if ok else severity,
            'message': message,
            'details': details or {},
        }
    )
    if ok:
        return
    if severity == 'warning':
        warnings.append(message)
        return
    errors.append(message)


def load_analysis_plan_or_raise(
    path: str | Path,
    *,
    task_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    candidate = Path(path).expanduser().resolve()
    try:
        data = json.loads(candidate.read_text(encoding='utf-8'))
    except OSError as exc:
        raise ValueError(f'analysis_plan.json could not be read: {exc}') from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f'analysis_plan.json is not valid JSON: {exc}') from exc

    if not isinstance(data, dict):
        raise ValueError('analysis_plan.json must contain a JSON object')

    validation = validate_analysis_plan(data, task_context=task_context)
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


def _resolve_analysis_plan(
    *,
    analysis_plan: Optional[Dict[str, Any]],
    analysis_plan_path: Optional[str],
    task_context: Dict[str, Any],
    preflight_errors: List[str],
) -> Optional[Dict[str, Any]]:
    if isinstance(analysis_plan, dict):
        return analysis_plan
    if not analysis_plan_path:
        return None
    try:
        return load_analysis_plan_or_raise(analysis_plan_path, task_context=task_context)
    except ValueError as exc:
        preflight_errors.append(str(exc))
        return None


def _validate_contract_paths(
    *,
    task_context: Dict[str, Any],
    analysis_plan: Optional[Dict[str, Any]],
    routing_audit: Dict[str, Any],
    checks: List[Dict[str, Any]],
    errors: List[str],
    warnings: List[str],
) -> Dict[str, Optional[str]]:
    task_context_expected: Optional[str] = None
    analysis_plan_expected: Optional[str] = None
    effective_expected: Optional[str] = None

    integration_assessment = _safe_dict(task_context.get('integration_assessment'))
    raw_task_context_path = integration_assessment.get('recommended_path')
    try:
        task_context_expected = normalize_integration_path_or_error(
            raw_task_context_path,
            field_name='task_context.integration_assessment.recommended_path',
        )
        _record_check(
            checks=checks,
            errors=errors,
            warnings=warnings,
            name='task_context.recommended_path',
            ok=True,
            severity='error',
            message='task_context.integration_assessment.recommended_path matches the canonical contract.',
            details={
                'raw_path': raw_task_context_path,
                'canonical_path': task_context_expected,
            },
        )
    except IntegrationPathContractError as exc:
        _record_check(
            checks=checks,
            errors=errors,
            warnings=warnings,
            name='task_context.recommended_path',
            ok=False,
            severity='error',
            message=str(exc),
            details={'raw_path': raw_task_context_path},
        )

    task_context_route = _safe_dict(_safe_dict(routing_audit.get('routes')).get('task_context'))
    _record_check(
        checks=checks,
        errors=errors,
        warnings=warnings,
        name='routing_audit.task_context_route',
        ok=bool(task_context_route),
        severity='error',
        message='routing_audit.routes.task_context must exist after task_context build.',
        details={'route': task_context_route},
    )
    if task_context_route and task_context_expected is not None:
        actual = _safe_str(task_context_route.get('canonical_path'))
        _record_check(
            checks=checks,
            errors=errors,
            warnings=warnings,
            name='routing_audit.task_context_route_consistency',
            ok=actual == task_context_expected,
            severity='error',
            message=(
                'routing_audit.routes.task_context.canonical_path must match '
                'task_context.integration_assessment.recommended_path.'
            ),
            details={
                'expected': task_context_expected,
                'actual': actual,
            },
        )

    if isinstance(analysis_plan, dict):
        integration_decision = _safe_dict(analysis_plan.get('integration_decision'))
        raw_plan_path = integration_decision.get('path')
        has_analysis_path = _safe_str(raw_plan_path) is not None
        if not has_analysis_path:
            raw_plan_path = None
        if has_analysis_path:
            try:
                analysis_plan_expected = normalize_integration_path_or_error(
                    raw_plan_path,
                    field_name='analysis_plan.integration_decision.path',
                )
                _record_check(
                    checks=checks,
                    errors=errors,
                    warnings=warnings,
                    name='analysis_plan.integration_decision.path',
                    ok=True,
                    severity='error',
                    message='analysis_plan.integration_decision.path matches the canonical contract.',
                    details={
                        'raw_path': raw_plan_path,
                        'canonical_path': analysis_plan_expected,
                    },
                )
            except IntegrationPathContractError as exc:
                _record_check(
                    checks=checks,
                    errors=errors,
                    warnings=warnings,
                    name='analysis_plan.integration_decision.path',
                    ok=False,
                    severity='error',
                    message=str(exc),
                    details={'raw_path': raw_plan_path},
                )

        analysis_plan_route = _safe_dict(_safe_dict(routing_audit.get('routes')).get('analysis_plan'))
        if has_analysis_path:
            _record_check(
                checks=checks,
                errors=errors,
                warnings=warnings,
                name='routing_audit.analysis_plan_route',
                ok=bool(analysis_plan_route),
                severity='error',
                message='routing_audit.routes.analysis_plan must exist when analysis_plan supplies a routing path.',
                details={'route': analysis_plan_route},
            )
        if analysis_plan_route and analysis_plan_expected is not None:
            actual = _safe_str(analysis_plan_route.get('canonical_path'))
            _record_check(
                checks=checks,
                errors=errors,
                warnings=warnings,
                name='routing_audit.analysis_plan_route_consistency',
                ok=actual == analysis_plan_expected,
                severity='error',
                message=(
                    'routing_audit.routes.analysis_plan.canonical_path must match '
                    'analysis_plan.integration_decision.path.'
                ),
                details={
                    'expected': analysis_plan_expected,
                    'actual': actual,
                },
            )

    effective_expected = analysis_plan_expected or task_context_expected
    effective_route = _safe_dict(_safe_dict(routing_audit.get('routes')).get('effective'))
    _record_check(
        checks=checks,
        errors=errors,
        warnings=warnings,
        name='routing_audit.effective_route',
        ok=bool(effective_route),
        severity='error',
        message='routing_audit.routes.effective must exist before attempt execution.',
        details={'route': effective_route},
    )
    if effective_route and effective_expected is not None:
        actual = _safe_str(effective_route.get('canonical_path'))
        selected_from = _safe_str(effective_route.get('selected_from'))
        expected_source = 'analysis_plan' if analysis_plan_expected is not None else 'task_context'
        _record_check(
            checks=checks,
            errors=errors,
            warnings=warnings,
            name='routing_audit.effective_route_consistency',
            ok=actual == effective_expected,
            severity='error',
            message='routing_audit.routes.effective.canonical_path must match the selected routing source.',
            details={
                'expected': effective_expected,
                'actual': actual,
            },
        )
        _record_check(
            checks=checks,
            errors=errors,
            warnings=warnings,
            name='routing_audit.effective_source',
            ok=selected_from == expected_source,
            severity='error',
            message='routing_audit.routes.effective.selected_from must match the active routing source.',
            details={
                'expected': expected_source,
                'actual': selected_from,
            },
        )

    return {
        'task_context_path': task_context_expected,
        'analysis_plan_path': analysis_plan_expected,
        'effective_path': effective_expected,
    }


def _validate_runtime_preconditions(
    *,
    task_context: Dict[str, Any],
    effective_integration_path: Optional[str],
    checks: List[Dict[str, Any]],
    errors: List[str],
    warnings: List[str],
) -> None:
    if effective_integration_path is None:
        return

    inputs = _safe_dict(task_context.get('inputs'))
    prepared_context = _safe_dict(task_context.get('prepared_context'))
    code_analysis = _safe_dict(task_context.get('code_analysis'))
    code_repo_path = _safe_str(inputs.get('code_repo_path'))
    code_repo_exists = False
    if code_repo_path:
        code_repo_exists = Path(code_repo_path).expanduser().exists()

    primary_files = _safe_list(prepared_context.get('primary_files'))
    focus_files = _safe_list(code_analysis.get('focus_files'))
    needs_repo = effective_integration_path in {
        'adapter_wrapper',
        'extend_model_outputs',
        'model_surgery',
    }
    needs_richer_context = effective_integration_path in {
        'extend_model_outputs',
        'model_surgery',
    }

    if needs_repo:
        _record_check(
            checks=checks,
            errors=errors,
            warnings=warnings,
            name='runtime_preconditions.code_repo_path',
            ok=code_repo_path is not None,
            severity='error',
            message=(
                f'Integration path {effective_integration_path!r} requires inputs.code_repo_path '
                'so the agent can edit attempt-scoped adapter/model copies.'
            ),
            details={'effective_integration_path': effective_integration_path},
        )
        _record_check(
            checks=checks,
            errors=errors,
            warnings=warnings,
            name='runtime_preconditions.code_repo_exists',
            ok=bool(code_repo_exists),
            severity='error',
            message=(
                f'Integration path {effective_integration_path!r} requires a readable code repository.'
            ),
            details={'code_repo_path': code_repo_path},
        )

    if needs_richer_context:
        _record_check(
            checks=checks,
            errors=errors,
            warnings=warnings,
            name='runtime_preconditions.code_context_available',
            ok=bool(primary_files or focus_files),
            severity='warning',
            message=(
                f'Integration path {effective_integration_path!r} has no scanned code-context evidence yet; '
                'attempt routing may still work, but agent guidance will be weak.'
            ),
            details={
                'primary_files_count': len(primary_files),
                'focus_files_count': len(focus_files),
            },
        )


def build_contract_validation(
    *,
    experiment_dir: Path,
    paper_slug: str,
    task_context: Dict[str, Any],
    analysis_plan: Optional[Dict[str, Any]] = None,
    analysis_plan_path: Optional[str] = None,
    preflight_errors: Optional[List[str]] = None,
    preflight_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    errors: List[str] = list(preflight_errors or [])
    warnings: List[str] = list(preflight_warnings or [])
    checks: List[Dict[str, Any]] = []

    resolved_analysis_plan = _resolve_analysis_plan(
        analysis_plan=analysis_plan,
        analysis_plan_path=analysis_plan_path,
        task_context=task_context,
        preflight_errors=errors,
    )
    routing_audit_result = write_routing_audit(
        experiment_dir=experiment_dir,
        paper_slug=paper_slug,
        task_context=task_context,
        analysis_plan=resolved_analysis_plan,
        analysis_plan_path=analysis_plan_path,
    )
    routing_audit = routing_audit_result.get('routing_audit', {})
    resolved_paths = _validate_contract_paths(
        task_context=task_context,
        analysis_plan=resolved_analysis_plan,
        routing_audit=routing_audit if isinstance(routing_audit, dict) else {},
        checks=checks,
        errors=errors,
        warnings=warnings,
    )
    _validate_runtime_preconditions(
        task_context=task_context,
        effective_integration_path=resolved_paths.get('effective_path'),
        checks=checks,
        errors=errors,
        warnings=warnings,
    )

    status = 'ok'
    if errors:
        status = 'error'
    elif warnings:
        status = 'warning'

    paths = _safe_dict(task_context.get('paths'))
    contract_validation_path = (
        _safe_str(paths.get('contract_validation_path'))
        or str((experiment_dir / 'contract_validation.json').resolve())
    )
    return {
        'schema_version': 'contract_validation.v1',
        'paper_slug': paper_slug,
        'status': status,
        'paths': {
            'task_context_path': _safe_str(paths.get('task_context_path')),
            'analysis_plan_path': analysis_plan_path or _safe_str(paths.get('analysis_plan_path')),
            'routing_audit_path': routing_audit_result.get('routing_audit_path'),
            'contract_validation_path': contract_validation_path,
        },
        'routing': _safe_dict(routing_audit.get('routes')),
        'checks': checks,
        'errors': errors,
        'warnings': warnings,
        'effective_integration_path': resolved_paths.get('effective_path'),
        'effective_integration_path_source': _safe_str(
            _safe_dict(_safe_dict(routing_audit.get('routes')).get('effective')).get('selected_from')
        ),
        'normalized_analysis_plan': resolved_analysis_plan,
        'routing_audit': routing_audit,
    }


def write_contract_validation(
    *,
    experiment_dir: Path,
    paper_slug: str,
    task_context: Dict[str, Any],
    analysis_plan: Optional[Dict[str, Any]] = None,
    analysis_plan_path: Optional[str] = None,
    preflight_errors: Optional[List[str]] = None,
    preflight_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    validation = build_contract_validation(
        experiment_dir=experiment_dir,
        paper_slug=paper_slug,
        task_context=task_context,
        analysis_plan=analysis_plan,
        analysis_plan_path=analysis_plan_path,
        preflight_errors=preflight_errors,
        preflight_warnings=preflight_warnings,
    )
    contract_validation_path = Path(validation['paths']['contract_validation_path']).expanduser().resolve()
    write_json(contract_validation_path, {
        key: value
        for key, value in validation.items()
        if key not in {'normalized_analysis_plan', 'routing_audit'}
    })
    validation['contract_validation_path'] = str(contract_validation_path)
    return validation
