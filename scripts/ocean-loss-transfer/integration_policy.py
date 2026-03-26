"""
@file integration_policy.py
@description Shared helpers for integration-path routing and attempt edit policies.
@author OpenAI Codex
@date 2026-03-25
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


_DEFAULT_INTEGRATION_PATH = 'agent_decides'
_ATTEMPT_EDIT_POLICIES: Dict[str, Dict[str, List[str]]] = {
    'loss_only': {
        'files_to_edit': ['candidate_loss.py'],
        'required_edit_paths': [],
    },
    'adapter_wrapper': {
        'files_to_edit': [
            'candidate_loss.py',
            'sandbox model adapter files exposing extra loss inputs',
            'sandbox trainer files',
        ],
        'required_edit_paths': ['sandbox_model_adapter.py'],
    },
    'extend_model_outputs': {
        'files_to_edit': [
            'candidate_loss.py',
            'sandbox model adapter files exposing extra loss inputs',
            'sandbox trainer files',
            'models',
        ],
        'required_edit_paths': ['models'],
    },
    'model_surgery': {
        'files_to_edit': [
            'candidate_loss.py',
            'sandbox trainer files',
            'models',
        ],
        'required_edit_paths': ['models'],
    },
    _DEFAULT_INTEGRATION_PATH: {
        'files_to_edit': ['candidate_loss.py'],
        'required_edit_paths': [],
    },
}


def normalize_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and str(item).strip()]


def normalize_integration_path(path: Optional[str]) -> str:
    if not isinstance(path, str):
        return _DEFAULT_INTEGRATION_PATH
    normalized = path.strip().lower()
    return normalized or _DEFAULT_INTEGRATION_PATH


def resolve_recommended_integration_path(
    task_context: Dict[str, Any],
    analysis_plan: Optional[Dict[str, Any]] = None,
) -> str:
    if isinstance(analysis_plan, dict):
        integration_decision = analysis_plan.get('integration_decision')
        if isinstance(integration_decision, dict):
            path = integration_decision.get('path')
            normalized = normalize_integration_path(path if isinstance(path, str) else None)
            if normalized != _DEFAULT_INTEGRATION_PATH:
                return normalized

    integration_assessment = (
        task_context.get('integration_assessment', {})
        if isinstance(task_context.get('integration_assessment'), dict)
        else {}
    )
    path = integration_assessment.get('recommended_path')
    return normalize_integration_path(path if isinstance(path, str) else None)


def build_attempt_edit_policy(integration_path: str) -> Dict[str, List[str]]:
    normalized = normalize_integration_path(integration_path)
    policy = _ATTEMPT_EDIT_POLICIES.get(normalized, _ATTEMPT_EDIT_POLICIES[_DEFAULT_INTEGRATION_PATH])
    return {
        'files_to_edit': list(policy['files_to_edit']),
        'required_edit_paths': list(policy['required_edit_paths']),
    }


def merge_attempt_with_edit_policy(
    attempt: Dict[str, Any],
    *,
    integration_path: str,
) -> Dict[str, Any]:
    normalized_attempt = dict(attempt)
    policy = build_attempt_edit_policy(integration_path)

    existing_files = normalize_string_list(normalized_attempt.get('files_to_edit'))
    merged_files = list(existing_files)
    for item in policy['files_to_edit']:
        if item not in merged_files:
            merged_files.append(item)
    normalized_attempt['files_to_edit'] = merged_files

    existing_required = normalize_string_list(normalized_attempt.get('required_edit_paths'))
    normalized_attempt['required_edit_paths'] = (
        existing_required if existing_required else list(policy['required_edit_paths'])
    )
    return normalized_attempt


def integration_path_needs_adapter_overrides(integration_path: str) -> bool:
    return normalize_integration_path(integration_path) in {'adapter_wrapper', 'extend_model_outputs'}


def integration_path_needs_model_tree(integration_path: str) -> bool:
    return normalize_integration_path(integration_path) in {'extend_model_outputs', 'model_surgery'}
