"""
@file integration_path.py
@description Canonical integration-path contract shared by producers and consumers.
@author OpenAI Codex
@date 2026-03-27
"""

from __future__ import annotations

from typing import Any, Dict, Optional


CANONICAL_INTEGRATION_PATHS = (
    'loss_only',
    'adapter_wrapper',
    'extend_model_outputs',
    'model_surgery',
)
CANONICAL_INTEGRATION_PATH_SET = set(CANONICAL_INTEGRATION_PATHS)

INTEGRATION_PATH_ALIASES = {
    'reuse_existing_loss_config': 'loss_only',
    'add_spec_driven_recipe': 'loss_only',
    'add_loss_inputs_adapter': 'adapter_wrapper',
    'loss_inputs_adapter': 'adapter_wrapper',
    'model_output_extension': 'extend_model_outputs',
}


class IntegrationPathContractError(ValueError):
    """Raised when an integration-path value violates the shared contract."""


def describe_integration_path(path: Any) -> Dict[str, Optional[str]]:
    raw_path = path if isinstance(path, str) else None
    if raw_path is None:
        return {
            'raw_path': None,
            'normalized_path': None,
            'canonical_path': None,
            'status': 'missing',
        }

    normalized = raw_path.strip().lower()
    if not normalized:
        return {
            'raw_path': raw_path,
            'normalized_path': None,
            'canonical_path': None,
            'status': 'missing',
        }

    if normalized in CANONICAL_INTEGRATION_PATH_SET:
        return {
            'raw_path': raw_path,
            'normalized_path': normalized,
            'canonical_path': normalized,
            'status': 'exact',
        }

    alias_target = INTEGRATION_PATH_ALIASES.get(normalized)
    if alias_target is not None:
        return {
            'raw_path': raw_path,
            'normalized_path': normalized,
            'canonical_path': alias_target,
            'status': 'alias_mapped',
        }

    return {
        'raw_path': raw_path,
        'normalized_path': normalized,
        'canonical_path': None,
        'status': 'error',
    }


def format_allowed_integration_paths() -> str:
    return ', '.join(CANONICAL_INTEGRATION_PATHS)


def normalize_integration_path_or_error(
    path: Any,
    *,
    field_name: str,
) -> str:
    description = describe_integration_path(path)
    canonical_path = description.get('canonical_path')
    if isinstance(canonical_path, str):
        return canonical_path

    status = description.get('status')
    if status == 'missing':
        raise IntegrationPathContractError(
            f'{field_name} is missing; expected one of: {format_allowed_integration_paths()}'
        )

    raw_path = description.get('raw_path')
    raise IntegrationPathContractError(
        f'{field_name}={raw_path!r} is invalid; expected one of: {format_allowed_integration_paths()}'
    )
