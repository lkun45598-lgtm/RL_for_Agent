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
from typing import Any, Dict, List, Optional, Set

from loss_transfer.common.integration_path import (
    describe_integration_path,
    format_allowed_integration_paths,
)
_ALLOWED_ATTEMPT_KINDS = {'agent_code', 'formula_variant'}
_ALLOWED_FORMULA_VARIANTS = {'faithful', 'stabilized'}
_EVIDENCE_ROOT_ALIASES = {
    'paper': 'paper_analysis',
    'code': 'code_analysis',
    'formula': 'formula_spec',
    'formula_interface': 'formula_interface',
    'loss_spec': 'loss_spec',
    'compatibility': 'compatibility',
    'integration': 'integration_assessment',
    'prepared_context': 'prepared_context',
    'analysis_evidence_probe_request': 'analysis_evidence_probe_request',
    'analysis_evidence_probe_result': 'analysis_evidence_probe_result',
    'task_context': 'task_context',
    'result': 'runtime_result',
    'repair_plan': 'repair_plan',
}


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


def _load_optional_json_object(path: Optional[str]) -> Optional[Dict[str, Any]]:
    candidate = _as_non_empty_str(path)
    if candidate is None:
        return None
    file_path = Path(candidate).expanduser().resolve()
    if not file_path.exists():
        return None
    try:
        data = json.loads(file_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _build_available_evidence_roots(task_context: Optional[Dict[str, Any]]) -> Set[str]:
    if not isinstance(task_context, dict):
        return set()

    available_roots: Set[str] = {'task_context'}
    for field_name in (
        'paper_analysis',
        'code_analysis',
        'formula_spec',
        'formula_interface',
        'loss_spec',
        'compatibility',
        'integration_assessment',
        'prepared_context',
        'legacy_loss_ir_status',
    ):
        if isinstance(task_context.get(field_name), dict) and task_context.get(field_name):
            available_roots.add(field_name)

    paths = task_context.get('paths')
    if isinstance(paths, dict):
        probe_request = _load_optional_json_object(paths.get('analysis_evidence_probe_request_path'))
        probe_result = _load_optional_json_object(paths.get('analysis_evidence_probe_result_path'))
        if probe_request:
            available_roots.add('analysis_evidence_probe_request')
        if probe_result:
            available_roots.add('analysis_evidence_probe_result')

    return available_roots


def _extract_evidence_root(reference: str) -> Optional[str]:
    stripped = reference.strip()
    if not stripped:
        return None

    for separator in ('.', '['):
        if separator in stripped:
            stripped = stripped.split(separator, 1)[0]
            break
    return stripped or None


def _resolve_evidence_root(reference: str, available_roots: Set[str]) -> Optional[str]:
    root = _extract_evidence_root(reference)
    if root is None:
        return None
    if root in available_roots:
        return root

    canonical_root = _EVIDENCE_ROOT_ALIASES.get(root)
    if canonical_root is None:
        return None
    if canonical_root == 'task_context':
        return canonical_root
    if canonical_root == 'runtime_result':
        return canonical_root
    if canonical_root == 'repair_plan':
        return canonical_root
    if canonical_root in available_roots:
        return canonical_root
    return None


def _probe_result_required(task_context: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(task_context, dict):
        return False
    paths = task_context.get('paths')
    if not isinstance(paths, dict):
        return False
    request_payload = _load_optional_json_object(paths.get('analysis_evidence_probe_request_path'))
    result_payload = _load_optional_json_object(paths.get('analysis_evidence_probe_result_path'))
    if not request_payload or not result_payload:
        return False
    return _as_non_empty_str(request_payload.get('status')) == 'probe_needed'


def _validate_evidence_ref_bundle(
    *,
    evidence_refs: List[str],
    field_name: str,
    available_roots: Set[str],
    errors: List[str],
    warnings: List[str],
    resolved_probe_refs: List[str],
) -> None:
    if not evidence_refs:
        return

    for evidence_ref in evidence_refs:
        resolved_root = _resolve_evidence_root(evidence_ref, available_roots)
        if resolved_root is None:
            warnings.append(
                f'{field_name} contains an unknown evidence ref {evidence_ref!r}; '
                'Agent traceability may be weak'
            )
            continue
        if resolved_root == 'analysis_evidence_probe_result':
            resolved_probe_refs.append(evidence_ref)


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
        path_raw = None
        path_status = None
    else:
        path_info = describe_integration_path(path)
        path_raw = path
        path_status = path_info.get('status')
        canonical_path = path_info.get('canonical_path')
        if canonical_path is None:
            errors.append(
                'integration_decision.path must be one of: '
                + format_allowed_integration_paths()
            )
        else:
            if path_info.get('status') == 'alias_mapped':
                warnings.append(
                    f'integration_decision.path normalized from {path!r} to {canonical_path!r}'
                )
            path = canonical_path

    if rationale is None:
        errors.append('integration_decision.rationale must be a non-empty string')
    if not evidence_refs:
        warnings.append('integration_decision.evidence_refs is empty; plan traceability is weak')

    return {
        'path': path,
        'path_raw': path_raw,
        'path_status': path_status,
        'path_source': 'analysis_plan.integration_decision',
        'rationale': rationale,
        'evidence_refs': evidence_refs,
    }


def _validate_strategy_delta(
    value: Any,
    *,
    prefix: str,
    errors: List[str],
    warnings: List[str],
) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        errors.append(f'{prefix} must be an object')
        return None

    previous_attempt_id = value.get('previous_attempt_id')
    if previous_attempt_id is not None and not isinstance(previous_attempt_id, int):
        errors.append(f'{prefix}.previous_attempt_id must be an integer when provided')

    why_previous_failed = _as_non_empty_str(value.get('why_previous_failed'))
    what_changes_now = _validate_string_list(
        value.get('what_changes_now'),
        field_name=f'{prefix}.what_changes_now',
        errors=errors,
    )
    why_not_repeat_previous = _as_non_empty_str(value.get('why_not_repeat_previous'))
    expected_signal = _as_non_empty_str(value.get('expected_signal'))

    if why_previous_failed is None:
        warnings.append(f'{prefix}.why_previous_failed is missing; replan rationale is weak')
    if not what_changes_now:
        warnings.append(f'{prefix}.what_changes_now is empty; strategy delta is not explicit')
    if why_not_repeat_previous is None:
        warnings.append(f'{prefix}.why_not_repeat_previous is missing')
    if expected_signal is None:
        warnings.append(f'{prefix}.expected_signal is missing')

    return {
        'previous_attempt_id': previous_attempt_id,
        'why_previous_failed': why_previous_failed,
        'what_changes_now': what_changes_now,
        'why_not_repeat_previous': why_not_repeat_previous,
        'expected_signal': expected_signal,
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
    strategy_delta = _validate_strategy_delta(
        attempt.get('strategy_delta'),
        prefix=f'{prefix}.strategy_delta',
        errors=errors,
        warnings=warnings,
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
        'strategy_delta': strategy_delta,
        'run_training': run_training,
        'notes': notes,
    }


def validate_attempt_spec(attempt: Any) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    normalized_attempt = _validate_attempt(
        attempt,
        index=0,
        errors=errors,
        warnings=warnings,
    )
    status = 'ok' if not errors else 'error'
    if status == 'ok' and warnings:
        status = 'warning'
    return {
        'status': status,
        'errors': errors,
        'warnings': warnings,
        'normalized_attempt': normalized_attempt if not errors else None,
    }


def validate_analysis_plan(
    plan: Dict[str, Any],
    *,
    task_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    available_evidence_roots = _build_available_evidence_roots(task_context)
    probe_result_required = _probe_result_required(task_context)
    resolved_probe_refs: List[str] = []

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
    if isinstance(integration_decision, dict):
        _validate_evidence_ref_bundle(
            evidence_refs=integration_decision.get('evidence_refs', []),
            field_name='integration_decision.evidence_refs',
            available_roots=available_evidence_roots,
            errors=errors,
            warnings=warnings,
            resolved_probe_refs=resolved_probe_refs,
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
            _validate_evidence_ref_bundle(
                evidence_refs=normalized.get('evidence_refs', []),
                field_name=f'attempts[{idx}].evidence_refs',
                available_roots=available_evidence_roots,
                errors=errors,
                warnings=warnings,
                resolved_probe_refs=resolved_probe_refs,
            )
            normalized_attempts.append(normalized)

    if not normalized_attempts:
        errors.append('attempts must contain at least one executable attempt')

    if probe_result_required and not resolved_probe_refs:
        errors.append(
            'analysis_plan must reference analysis_evidence_probe_result in integration_decision.evidence_refs '
            'or attempts[*].evidence_refs when a probe was requested and executed'
        )

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
            'evidence_validation': {
                'available_roots': sorted(available_evidence_roots),
                'probe_result_required': probe_result_required,
                'probe_result_refs': resolved_probe_refs,
            },
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
