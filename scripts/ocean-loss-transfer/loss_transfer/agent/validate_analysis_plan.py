"""
@file validate_analysis_plan.py
@description Validate analysis_plan.json before executing the agentic loss-transfer loop.
@author OpenAI Codex
@date 2026-03-25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


_ALLOWED_INTEGRATION_PATHS = {
    'loss_only',
    'adapter_wrapper',
    'extend_model_outputs',
    'model_surgery',
}
_ALLOWED_ATTEMPT_KINDS = {'agent_code', 'formula_variant'}
_ALLOWED_FORMULA_VARIANTS = {'faithful', 'stabilized'}


def _as_non_empty_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _validate_string_list(
    value: Any,
    *,
    field_name: str,
    errors: List[str],
) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        errors.append(f'{field_name} must be a list of strings')
        return []

    result: List[str] = []
    for idx, item in enumerate(value):
        normalized = _as_non_empty_str(item)
        if normalized is None:
            errors.append(f'{field_name}[{idx}] must be a non-empty string')
            continue
        result.append(normalized)
    return result


def _validate_integration_decision(
    value: Any,
    *,
    errors: List[str],
    warnings: List[str],
) -> Optional[Dict[str, Any]]:
    if value is None:
        warnings.append('integration_decision is missing; Agent rationale will be hard to audit')
        return None
    if not isinstance(value, dict):
        errors.append('integration_decision must be an object')
        return None

    path = _as_non_empty_str(value.get('path'))
    rationale = _as_non_empty_str(value.get('rationale'))
    evidence_refs = _validate_string_list(
        value.get('evidence_refs'),
        field_name='integration_decision.evidence_refs',
        errors=errors,
    )

    if path is None:
        errors.append('integration_decision.path must be a non-empty string')
    elif path not in _ALLOWED_INTEGRATION_PATHS:
        errors.append(
            'integration_decision.path must be one of: '
            + ', '.join(sorted(_ALLOWED_INTEGRATION_PATHS))
        )

    if rationale is None:
        errors.append('integration_decision.rationale must be a non-empty string')
    if not evidence_refs:
        warnings.append('integration_decision.evidence_refs is empty; plan traceability is weak')

    return {
        'path': path,
        'rationale': rationale,
        'evidence_refs': evidence_refs,
    }


def _validate_attempt(
    attempt: Any,
    *,
    index: int,
    errors: List[str],
    warnings: List[str],
) -> Optional[Dict[str, Any]]:
    prefix = f'attempts[{index}]'
    if not isinstance(attempt, dict):
        errors.append(f'{prefix} must be an object')
        return None

    name = _as_non_empty_str(attempt.get('name'))
    kind = _as_non_empty_str(attempt.get('kind')) or 'agent_code'
    variant = _as_non_empty_str(attempt.get('variant'))
    code = attempt.get('code')
    code_path = _as_non_empty_str(attempt.get('code_path'))
    objective = _as_non_empty_str(attempt.get('objective'))
    files_to_edit = _validate_string_list(
        attempt.get('files_to_edit'),
        field_name=f'{prefix}.files_to_edit',
        errors=errors,
    )
    required_edit_paths = _validate_string_list(
        attempt.get('required_edit_paths'),
        field_name=f'{prefix}.required_edit_paths',
        errors=errors,
    )
    evidence_refs = _validate_string_list(
        attempt.get('evidence_refs'),
        field_name=f'{prefix}.evidence_refs',
        errors=errors,
    )
    notes = attempt.get('notes')
    run_training = attempt.get('run_training', True)

    if name is None:
        errors.append(f'{prefix}.name must be a non-empty string')
    if kind not in _ALLOWED_ATTEMPT_KINDS:
        errors.append(
            f'{prefix}.kind must be one of: ' + ', '.join(sorted(_ALLOWED_ATTEMPT_KINDS))
        )

    if notes is not None and not isinstance(notes, str):
        errors.append(f'{prefix}.notes must be a string when provided')
    if not isinstance(run_training, bool):
        errors.append(f'{prefix}.run_training must be a boolean')

    normalized_code: Optional[str] = None
    if code is not None:
        if not isinstance(code, str) or not code.strip():
            errors.append(f'{prefix}.code must be a non-empty string when provided')
        else:
            normalized_code = code

    if kind == 'formula_variant':
        if variant not in _ALLOWED_FORMULA_VARIANTS:
            errors.append(
                f'{prefix}.variant must be one of: ' + ', '.join(sorted(_ALLOWED_FORMULA_VARIANTS))
            )
        if normalized_code is not None or code_path is not None:
            warnings.append(f'{prefix} is formula_variant; inline code/code_path will be ignored')
    elif kind == 'agent_code':
        if normalized_code is None and code_path is None and objective is None:
            errors.append(
                f'{prefix} with kind=agent_code must provide code, code_path, or objective'
            )
        if variant is not None:
            warnings.append(f'{prefix}.variant is only used for formula_variant attempts')
        if objective is None and normalized_code is None and code_path is None:
            warnings.append(f'{prefix} lacks an implementation objective')

    if not evidence_refs:
        warnings.append(f'{prefix}.evidence_refs is empty; this attempt is not well grounded')

    return {
        'name': name,
        'kind': kind,
        'variant': variant,
        'code': normalized_code,
        'code_path': code_path,
        'objective': objective,
        'files_to_edit': files_to_edit,
        'required_edit_paths': required_edit_paths,
        'evidence_refs': evidence_refs,
        'run_training': run_training,
        'notes': notes,
    }


def validate_analysis_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(plan, dict):
        return {
            'status': 'error',
            'errors': ['analysis_plan must be a JSON object'],
            'warnings': [],
            'normalized_plan': None,
        }

    summary = _as_non_empty_str(plan.get('summary'))
    if summary is None:
        warnings.append('summary is missing or empty')

    stop_on_first_pass = plan.get('stop_on_first_pass', False)
    if not isinstance(stop_on_first_pass, bool):
        errors.append('stop_on_first_pass must be a boolean')

    integration_decision = _validate_integration_decision(
        plan.get('integration_decision'),
        errors=errors,
        warnings=warnings,
    )

    raw_attempts = plan.get('attempts')
    if not isinstance(raw_attempts, list):
        errors.append('attempts must be a list')
        raw_attempts = []

    normalized_attempts: List[Dict[str, Any]] = []
    for idx, attempt in enumerate(raw_attempts):
        normalized = _validate_attempt(
            attempt,
            index=idx,
            errors=errors,
            warnings=warnings,
        )
        if normalized is not None:
            normalized_attempts.append(normalized)

    if not normalized_attempts:
        errors.append('attempts must contain at least one executable attempt')

    status = 'ok' if not errors else 'error'
    if status == 'ok' and warnings:
        status = 'warning'

    return {
        'status': status,
        'errors': errors,
        'warnings': warnings,
        'normalized_plan': {
            'summary': summary,
            'stop_on_first_pass': stop_on_first_pass if isinstance(stop_on_first_pass, bool) else False,
            'integration_decision': integration_decision,
            'attempts': normalized_attempts,
        } if not errors else None,
    }


def _load_json(path: str) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('analysis_plan must contain a JSON object')
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate analysis_plan.json')
    parser.add_argument('--analysis_plan', required=True)
    args = parser.parse_args()

    result = validate_analysis_plan(_load_json(args.analysis_plan))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
