"""
@file evidence_probe.py
@description Validation and execution helpers for agent-authored analysis evidence probes.
@author OpenAI Codex
@date 2026-03-27
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


_ALLOWED_PROBE_STATUSES = {'not_needed', 'probe_needed'}


def _as_non_empty_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _validate_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and str(item).strip()]


def load_json_object(path: str | Path) -> Dict[str, Any]:
    candidate = Path(path).expanduser().resolve()
    data = json.loads(candidate.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Expected JSON object at {candidate}')
    return data


def validate_evidence_probe_request(request: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(request, dict):
        return {
            'status': 'error',
            'errors': ['evidence probe request must be a JSON object'],
            'warnings': [],
            'normalized_request': None,
        }

    status = _as_non_empty_str(request.get('status'))
    reason = _as_non_empty_str(request.get('reason'))
    evidence_refs = _validate_string_list(request.get('evidence_refs'))
    probe_goal = _as_non_empty_str(request.get('probe_goal'))
    expected_output_keys = _validate_string_list(request.get('expected_output_keys'))

    if status not in _ALLOWED_PROBE_STATUSES:
        errors.append('evidence probe request.status must be one of: not_needed, probe_needed')
    if reason is None:
        errors.append('evidence probe request.reason must be a non-empty string')
    if not evidence_refs:
        warnings.append('evidence probe request.evidence_refs is empty; traceability is weak')

    if status == 'probe_needed':
        if probe_goal is None:
            errors.append('evidence probe request.probe_goal must be a non-empty string when status=probe_needed')
        if not expected_output_keys:
            warnings.append('evidence probe request.expected_output_keys is empty; result schema is underspecified')

    normalized_request = None
    if not errors:
        normalized_request = {
            'status': status,
            'reason': reason,
            'evidence_refs': evidence_refs,
            'probe_goal': probe_goal,
            'expected_output_keys': expected_output_keys,
        }

    result_status = 'ok' if not errors else 'error'
    if result_status == 'ok' and warnings:
        result_status = 'warning'
    return {
        'status': result_status,
        'errors': errors,
        'warnings': warnings,
        'normalized_request': normalized_request,
    }


def execute_evidence_probe(
    *,
    script_path: str | Path,
    code_repo_path: str,
    task_context_path: str | Path,
    output_path: str | Path,
    timeout_sec: int = 120,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    resolved_script_path = Path(script_path).expanduser().resolve()
    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_task_context_path = Path(task_context_path).expanduser().resolve()
    resolved_code_repo_path = Path(code_repo_path).expanduser().resolve()

    if not resolved_script_path.exists():
        return {
            'status': 'error',
            'error': f'Evidence probe script not found: {resolved_script_path}',
            'output_path': str(resolved_output_path),
        }
    if not resolved_code_repo_path.exists():
        return {
            'status': 'error',
            'error': f'Code repository not found for evidence probe: {resolved_code_repo_path}',
            'output_path': str(resolved_output_path),
        }

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(resolved_script_path),
        '--code_repo',
        str(resolved_code_repo_path),
        '--task_context',
        str(resolved_task_context_path),
    ]
    if extra_args:
        command.extend(str(item) for item in extra_args if isinstance(item, str) and item.strip())
    command.extend(
        [
            '--output',
            str(resolved_output_path),
        ]
    )

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )

    if completed.returncode != 0:
        return {
            'status': 'error',
            'error': f'Evidence probe exited with code {completed.returncode}',
            'stdout': completed.stdout[-4000:],
            'stderr': completed.stderr[-4000:],
            'output_path': str(resolved_output_path),
        }

    if not resolved_output_path.exists():
        return {
            'status': 'error',
            'error': f'Evidence probe completed but did not write output to {resolved_output_path}',
            'stdout': completed.stdout[-4000:],
            'stderr': completed.stderr[-4000:],
            'output_path': str(resolved_output_path),
        }

    try:
        payload = load_json_object(resolved_output_path)
    except Exception as exc:
        return {
            'status': 'error',
            'error': f'Evidence probe wrote invalid JSON: {exc}',
            'stdout': completed.stdout[-4000:],
            'stderr': completed.stderr[-4000:],
            'output_path': str(resolved_output_path),
        }

    return {
        'status': 'success',
        'output_path': str(resolved_output_path),
        'result': payload,
        'stdout': completed.stdout[-4000:],
        'stderr': completed.stderr[-4000:],
    }
