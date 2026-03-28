"""
@file routing_audit.py
@description Structured routing-audit artifact for loss-transfer integration decisions.
@author OpenAI Codex
@date 2026-03-27
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from loss_transfer.common.integration_path import describe_integration_path
from loss_transfer.common.trajectory_logger import write_json


def _safe_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _safe_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _safe_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and str(item).strip()]


def _normalization_reason(status: Optional[str], raw_path: Optional[str], canonical_path: Optional[str]) -> Optional[str]:
    if status == 'alias_mapped' and raw_path and canonical_path:
        return f'Alias path {raw_path!r} was normalized to canonical path {canonical_path!r}.'
    if status == 'exact':
        return 'Path already matched the canonical integration-path contract.'
    if status == 'missing':
        return 'No integration path was available for this routing decision.'
    if status == 'error':
        return 'Routing path did not match the canonical integration-path contract.'
    return None


def _build_route_record(
    *,
    scope: str,
    source: str,
    raw_path: Optional[str],
    canonical_path: Optional[str],
    status: Optional[str],
    rationale: Optional[str],
    evidence_refs: List[str],
) -> Optional[Dict[str, Any]]:
    reference_path = raw_path if raw_path is not None else canonical_path
    description = describe_integration_path(reference_path)
    normalized_path = description.get('normalized_path')
    resolved_canonical = canonical_path or description.get('canonical_path')
    resolved_status = status or description.get('status')

    if raw_path is None and canonical_path is None:
        return None

    return {
        'scope': scope,
        'source': source,
        'raw_path': raw_path,
        'normalized_path': normalized_path,
        'canonical_path': resolved_canonical,
        'status': resolved_status,
        'normalization_reason': _normalization_reason(resolved_status, raw_path, resolved_canonical),
        'rationale': rationale,
        'evidence_refs': evidence_refs,
    }


def build_task_context_route_record(task_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    integration = _safe_dict(task_context.get('integration_assessment'))
    raw_path = _safe_str(integration.get('recommended_path_raw'))
    canonical_path = _safe_str(integration.get('recommended_path'))
    status = _safe_str(integration.get('recommended_path_status'))
    source = _safe_str(integration.get('recommended_path_source')) or 'task_context'
    rationale = _safe_str(integration.get('recommended_path_reason'))
    evidence_refs = _safe_string_list(integration.get('recommended_path_evidence_refs'))

    return _build_route_record(
        scope='task_context.integration_assessment',
        source=source,
        raw_path=raw_path or canonical_path,
        canonical_path=canonical_path,
        status=status,
        rationale=rationale,
        evidence_refs=evidence_refs,
    )


def build_analysis_plan_route_record(analysis_plan: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(analysis_plan, dict):
        return None

    integration_decision = _safe_dict(analysis_plan.get('integration_decision'))
    if not integration_decision:
        return None

    raw_path = _safe_str(integration_decision.get('path_raw'))
    canonical_path = _safe_str(integration_decision.get('path'))
    status = _safe_str(integration_decision.get('path_status'))
    source = _safe_str(integration_decision.get('path_source')) or 'analysis_plan.integration_decision'
    rationale = _safe_str(integration_decision.get('rationale'))
    evidence_refs = _safe_string_list(integration_decision.get('evidence_refs'))

    return _build_route_record(
        scope='analysis_plan.integration_decision',
        source=source,
        raw_path=raw_path or canonical_path,
        canonical_path=canonical_path,
        status=status,
        rationale=rationale,
        evidence_refs=evidence_refs,
    )


def build_routing_audit(
    *,
    paper_slug: str,
    task_context: Dict[str, Any],
    analysis_plan: Optional[Dict[str, Any]] = None,
    routing_audit_path: Optional[str] = None,
    analysis_plan_path: Optional[str] = None,
) -> Dict[str, Any]:
    paths = _safe_dict(task_context.get('paths'))
    task_context_path = _safe_str(paths.get('task_context_path'))
    resolved_analysis_plan_path = analysis_plan_path or _safe_str(paths.get('analysis_plan_path'))
    task_context_route = build_task_context_route_record(task_context)
    analysis_plan_route = build_analysis_plan_route_record(analysis_plan)

    effective_route: Optional[Dict[str, Any]] = None
    warnings: List[str] = []
    if analysis_plan_route is not None:
        effective_route = dict(analysis_plan_route)
        effective_route['scope'] = 'effective_runtime_routing'
        effective_route['selected_from'] = 'analysis_plan'
        if task_context_route is None:
            effective_route['selection_reason'] = 'analysis_plan_only'
        elif analysis_plan_route.get('canonical_path') != task_context_route.get('canonical_path'):
            effective_route['selection_reason'] = 'analysis_plan_overrides_task_context'
        else:
            effective_route['selection_reason'] = 'analysis_plan_confirms_task_context'
    elif task_context_route is not None:
        effective_route = dict(task_context_route)
        effective_route['scope'] = 'effective_runtime_routing'
        effective_route['selected_from'] = 'task_context'
        effective_route['selection_reason'] = 'task_context_default'
    else:
        warnings.append('No routing decision was available from task_context or analysis_plan.')

    return {
        'schema_version': 'routing_audit.v1',
        'paper_slug': paper_slug,
        'paths': {
            'task_context_path': task_context_path,
            'analysis_plan_path': resolved_analysis_plan_path,
            'routing_audit_path': routing_audit_path,
        },
        'routes': {
            'task_context': task_context_route,
            'analysis_plan': analysis_plan_route,
            'effective': effective_route,
        },
        'warnings': warnings,
    }


def write_routing_audit(
    *,
    experiment_dir: Path,
    paper_slug: str,
    task_context: Dict[str, Any],
    analysis_plan: Optional[Dict[str, Any]] = None,
    analysis_plan_path: Optional[str] = None,
) -> Dict[str, Any]:
    routing_audit_path = experiment_dir / 'routing_audit.json'
    audit = build_routing_audit(
        paper_slug=paper_slug,
        task_context=task_context,
        analysis_plan=analysis_plan,
        routing_audit_path=str(routing_audit_path),
        analysis_plan_path=analysis_plan_path,
    )
    write_json(routing_audit_path, audit)
    effective_route = _safe_dict(_safe_dict(audit.get('routes')).get('effective'))
    return {
        'routing_audit_path': str(routing_audit_path),
        'routing_audit': audit,
        'effective_integration_path': effective_route.get('canonical_path'),
        'effective_integration_path_source': effective_route.get('selected_from'),
    }
