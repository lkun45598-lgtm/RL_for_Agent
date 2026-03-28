"""
@file run_manifest.py
@description Experiment-level run manifest for reproducibility and agent-session auditing.
@author OpenAI Codex
@date 2026-03-27
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loss_transfer.agent.agent_service_client import fetch_service_health, resolve_service_descriptor
from loss_transfer.common.paths import PROJECT_ROOT
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_existing_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _hash_file(path_str: Optional[str]) -> Optional[str]:
    if not isinstance(path_str, str) or not path_str.strip():
        return None
    path = Path(path_str).expanduser().resolve()
    if not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_git_sha() -> Optional[str]:
    try:
        completed = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    sha = completed.stdout.strip()
    return sha or None


def _extract_path_bundle(
    task_context: Dict[str, Any],
    *,
    analysis_plan_path: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    paths = _safe_dict(task_context.get('paths'))
    resolved_analysis_plan_path = _safe_str(analysis_plan_path) or _safe_str(paths.get('analysis_plan_path'))
    return {
        'experiment_dir': _safe_str(paths.get('experiment_dir')),
        'task_context_path': _safe_str(paths.get('task_context_path')),
        'analysis_plan_path': resolved_analysis_plan_path,
        'analysis_evidence_probe_request_path': _safe_str(paths.get('analysis_evidence_probe_request_path')),
        'analysis_evidence_probe_script_path': _safe_str(paths.get('analysis_evidence_probe_script_path')),
        'analysis_evidence_probe_result_path': _safe_str(paths.get('analysis_evidence_probe_result_path')),
        'loss_formula_path': _safe_str(paths.get('loss_formula_path')),
        'loss_ir_path': _safe_str(paths.get('loss_ir_path')),
        'routing_audit_path': _safe_str(paths.get('routing_audit_path')),
        'contract_validation_path': _safe_str(paths.get('contract_validation_path')),
        'decision_trace_path': _safe_str(paths.get('decision_trace_path')),
        'rl_dataset_path': _safe_str(paths.get('rl_dataset_path')),
        'run_manifest_path': _safe_str(paths.get('run_manifest_path')),
    }


def _build_hash_bundle(paths: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    return {
        'task_context_sha256': _hash_file(paths.get('task_context_path')),
        'analysis_plan_sha256': _hash_file(paths.get('analysis_plan_path')),
        'analysis_evidence_probe_request_sha256': _hash_file(paths.get('analysis_evidence_probe_request_path')),
        'analysis_evidence_probe_result_sha256': _hash_file(paths.get('analysis_evidence_probe_result_path')),
        'loss_formula_sha256': _hash_file(paths.get('loss_formula_path')),
        'routing_audit_sha256': _hash_file(paths.get('routing_audit_path')),
        'contract_validation_sha256': _hash_file(paths.get('contract_validation_path')),
    }


def _build_service_snapshot(
    *,
    service_url: Optional[str],
    probe_timeout_sec: int,
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    descriptor = resolve_service_descriptor(service_url)
    existing_health = _safe_dict(existing.get('health'))
    health = fetch_service_health(
        service_url=service_url,
        timeout_sec=probe_timeout_sec,
    )
    if not health and existing_health:
        health = existing_health
    return {
        'resolved_url': descriptor['resolved_url'],
        'source': descriptor['source'],
        'health': health,
    }


def _build_execution_snapshot(
    *,
    existing: Dict[str, Any],
    mode: Optional[str],
    bootstrap_formula: Optional[bool],
    max_attempts: Optional[int],
    auto_generate_plan: Optional[bool],
    session_policy: Optional[str],
) -> Dict[str, Any]:
    existing_execution = _safe_dict(existing.get('execution'))
    return {
        'mode': mode if isinstance(mode, str) else existing_execution.get('mode'),
        'bootstrap_formula': (
            bootstrap_formula
            if isinstance(bootstrap_formula, bool)
            else existing_execution.get('bootstrap_formula')
        ),
        'max_attempts': (
            int(max_attempts)
            if isinstance(max_attempts, int)
            else existing_execution.get('max_attempts')
        ),
        'auto_generate_plan': (
            auto_generate_plan
            if isinstance(auto_generate_plan, bool)
            else existing_execution.get('auto_generate_plan')
        ),
        'session_policy': session_policy or existing_execution.get('session_policy') or 'new_request_session_per_call',
        'python_executable': sys.executable,
    }


def _build_plan_generation_snapshot(plan_generation: Optional[Dict[str, Any]], existing: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(plan_generation, dict):
        return _safe_dict(existing.get('plan_generation')) or None

    return {
        'status': plan_generation.get('status'),
        'analysis_plan_path': plan_generation.get('analysis_plan_path'),
        'analysis_evidence_probe_status': plan_generation.get('analysis_evidence_probe_status'),
        'analysis_evidence_probe_request_path': plan_generation.get('analysis_evidence_probe_request_path'),
        'analysis_evidence_probe_result_path': plan_generation.get('analysis_evidence_probe_result_path'),
        'agent_response_path': plan_generation.get('agent_response_path'),
        'agent_id': plan_generation.get('agent_id'),
        'session_scope': plan_generation.get('session_scope'),
        'service_url': plan_generation.get('service_url'),
        'service_url_source': plan_generation.get('service_url_source'),
        'error': plan_generation.get('error'),
    }


def _build_loop_snapshot(loop_summary: Optional[Dict[str, Any]], existing: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(loop_summary, dict):
        return _safe_dict(existing.get('loop')) or None

    return {
        'status': loop_summary.get('status'),
        'attempt_count': loop_summary.get('attempt_count'),
        'best_attempt_id': loop_summary.get('best_attempt_id'),
        'best_metric_name': loop_summary.get('best_metric_name'),
        'best_metric_value': loop_summary.get('best_metric_value'),
        'decision_trace_path': loop_summary.get('decision_trace_path'),
        'rl_dataset_path': loop_summary.get('rl_dataset_path'),
        'routing_audit_path': loop_summary.get('routing_audit_path'),
        'contract_validation_path': loop_summary.get('contract_validation_path'),
    }


def write_run_manifest(
    *,
    experiment_dir: Path,
    paper_slug: str,
    task_context: Dict[str, Any],
    mode: Optional[str] = None,
    bootstrap_formula: Optional[bool] = None,
    max_attempts: Optional[int] = None,
    auto_generate_plan: Optional[bool] = None,
    service_url: Optional[str] = None,
    analysis_plan_path: Optional[str] = None,
    session_policy: Optional[str] = None,
    plan_generation: Optional[Dict[str, Any]] = None,
    loop_summary: Optional[Dict[str, Any]] = None,
    probe_timeout_sec: int = 2,
) -> Dict[str, Any]:
    manifest_path = experiment_dir / 'run_manifest.json'
    existing = _load_existing_manifest(manifest_path)
    paths = _extract_path_bundle(task_context, analysis_plan_path=analysis_plan_path)
    paths['run_manifest_path'] = str(manifest_path)

    manifest = {
        'schema_version': 'run_manifest.v1',
        'paper_slug': paper_slug,
        'started_at': existing.get('started_at') or _now_iso(),
        'updated_at': _now_iso(),
        'code_version': {
            'git_commit_sha': _resolve_git_sha(),
        },
        'execution': _build_execution_snapshot(
            existing=existing,
            mode=mode,
            bootstrap_formula=bootstrap_formula,
            max_attempts=max_attempts,
            auto_generate_plan=auto_generate_plan,
            session_policy=session_policy,
        ),
        'service': _build_service_snapshot(
            service_url=service_url,
            probe_timeout_sec=probe_timeout_sec,
            existing=_safe_dict(existing.get('service')),
        ),
        'paths': paths,
        'hashes': _build_hash_bundle(paths),
        'plan_generation': _build_plan_generation_snapshot(plan_generation, existing),
        'loop': _build_loop_snapshot(loop_summary, existing),
        'agent_calls': _safe_list(existing.get('agent_calls')),
    }
    write_json(manifest_path, manifest)
    return {
        'run_manifest_path': str(manifest_path),
        'run_manifest': manifest,
    }


def append_run_manifest_agent_call(
    run_manifest_path: str | Path,
    call_record: Dict[str, Any],
) -> Dict[str, Any]:
    manifest_path = Path(run_manifest_path).expanduser().resolve()
    existing = _load_existing_manifest(manifest_path)
    agent_calls = _safe_list(existing.get('agent_calls'))
    agent_calls.append(
        {
            'timestamp': _now_iso(),
            **call_record,
        }
    )
    existing['agent_calls'] = agent_calls
    existing['updated_at'] = _now_iso()
    write_json(manifest_path, existing)
    return {
        'run_manifest_path': str(manifest_path),
        'agent_call_count': len(agent_calls),
    }
