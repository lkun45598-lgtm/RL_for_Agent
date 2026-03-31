"""
@file case_memory_store.py

@description Shared case-memory persistence and compatibility helpers for historical experience records.
@author kongzhiquan
@contributors kongzhiquan
@date 2026-03-28
@version 1.1.0

@changelog
  - 2026-03-28 kongzhiquan: v1.0.0 extract shared case-memory storage and innovation compatibility helpers
  - 2026-03-30 kongzhiquan: v1.1.0 add case_memory.v2 normalization helpers and richer failure/repair summaries
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from loss_transfer.common._types import Innovation
from loss_transfer.common.paths import PROJECT_ROOT


DEFAULT_CASE_MEMORY_PATH = PROJECT_ROOT / 'workflow' / 'loss_transfer' / 'knowledge_base' / 'case_memories.jsonl'
_DEFAULT_SCHEMA_VERSION = 'case_memory.v2'
_LEGACY_SCHEMA_VERSION = 'case_memory.v1'
_KNOWLEDGE_RESULT_PATH_RE = re.compile(r'^knowledge_db:(?P<identifier>inn_(?P<number>\d+))$')
_LAYER_RANKS: Dict[Any, int] = {
    'code_generation': 0,
    'formula_interface': 0,
    'layer1': 1,
    'layer2': 2,
    'formula_alignment': 3,
    'layer3': 4,
    'layer4': 5,
    None: 6,
}
_ERROR_FAMILY_PATTERNS: Sequence[tuple[str, Sequence[str]]] = (
    ('oom', ('cuda out of memory', 'out of memory', ' oom')),
    ('timeout', ('timeout', 'timed out')),
    ('nan', (' nan', 'nan ', 'not a number')),
    ('shape_mismatch', ('shape mismatch', 'size mismatch', 'dimension mismatch')),
    ('missing_input', ('loss_inputs missing', 'required positional argument', 'unexpected keyword')),
    ('syntax_error', ('syntaxerror', 'syntax error', 'indentationerror', 'indentation error')),
    ('formula_alignment', ('formula alignment', 'symbol map', 'symbol mismatch')),
    ('ssim_collapse', ('ssim collapse', 'ssim collapsed', 'ssim too low', 'below threshold', 'below_baseline', 'model_collapse')),
    ('parse_failed', ('parse_failed', 'parse failed')),
)


def safe_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def safe_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def normalize_string_list(value: Any) -> List[str]:
    items = value if isinstance(value, list) else []
    normalized: List[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if not candidate or candidate in normalized:
            continue
        normalized.append(candidate)
    return normalized


def layer_rank(stop_layer: Any) -> int:
    return _LAYER_RANKS.get(stop_layer, -1)


def _safe_optional_str(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    return candidate or None


def _coerce_optional_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_primary_metric(metrics: Any) -> tuple[Optional[str], Optional[float]]:
    payload = safe_dict(metrics)
    for key in ('swinir', 'val_ssim', 'val_psnr'):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    return None, None


def infer_error_family(error_text: Any, stop_layer: Any = None) -> Optional[str]:
    text = _safe_optional_str(error_text)
    lowered = f' {text.lower()} ' if text else ''
    if stop_layer == 'formula_alignment':
        return 'formula_alignment'
    if not lowered:
        return 'other' if stop_layer is not None else None

    for family, patterns in _ERROR_FAMILY_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return family

    if stop_layer == 'layer4' and 'ssim' in lowered:
        return 'ssim_collapse'
    if 'parse' in lowered:
        return 'parse_failed'
    return 'other'


def build_failure_signature(
    *,
    stop_layer: Any,
    error: Any,
    metrics: Any = None,
) -> Dict[str, Any]:
    metric_name, metric_value = _extract_primary_metric(metrics)
    return {
        'stop_layer': stop_layer,
        'error_family': infer_error_family(error, stop_layer),
        'error_excerpt': _safe_optional_str(error),
        'metric_name': metric_name,
        'metric_value': metric_value,
    }


def build_repair_outcome_summary(
    *,
    stop_layer: Any,
    post_stop_layer: Any,
    post_error: Any,
    passed: Any,
    repair_rounds_used: Any,
    baseline_delta: Any,
    reverted: Any = False,
    status: Any = None,
) -> Dict[str, Any]:
    repair_rounds_used_value = int(repair_rounds_used) if isinstance(repair_rounds_used, int) else 0
    reverted_value = bool(reverted)
    initial_rank = layer_rank(stop_layer)
    post_rank = layer_rank(post_stop_layer)
    positive_baseline_delta = isinstance(baseline_delta, (int, float)) and float(baseline_delta) > 0
    improved = False
    if repair_rounds_used_value > 0 and not reverted_value:
        if passed is True:
            improved = True
        elif initial_rank >= 0 and post_rank >= 0 and post_rank > initial_rank:
            improved = True

    return {
        'status': _safe_optional_str(status),
        'post_stop_layer': post_stop_layer,
        'post_error': _safe_optional_str(post_error),
        'reverted': reverted_value,
        'improved': improved,
        'effective': bool(not reverted_value and (improved or positive_baseline_delta)),
        'resolved_failure': bool(not reverted_value and (passed is True or improved)),
        'positive_baseline_delta': positive_baseline_delta,
    }


def _overlay_non_none(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if value is None:
            continue
        merged[key] = value
    return merged


def _normalize_case_memory_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    strategy_delta = safe_dict(payload.get('strategy_delta'))
    metrics = safe_dict(payload.get('metrics'))
    baseline_delta = _coerce_optional_float(payload.get('baseline_delta'))
    val_ssim = _coerce_optional_float(payload.get('val_ssim'))
    val_psnr = _coerce_optional_float(payload.get('val_psnr'))
    training_curve_last_epoch = _coerce_optional_float(payload.get('training_curve_last_epoch'))
    reverted_repair_rounds = payload.get('reverted_repair_rounds')
    reverted_repair_rounds_value = int(reverted_repair_rounds) if isinstance(reverted_repair_rounds, int) else 0
    repair_rounds_used = payload.get('repair_rounds_used')
    repair_rounds_used_value = int(repair_rounds_used) if isinstance(repair_rounds_used, int) else 0
    trigger_stop_layer = payload.get('trigger_stop_layer') or payload.get('stop_layer')
    trigger_error = payload.get('trigger_error') or payload.get('error')
    repair_outcome_payload = safe_dict(payload.get('repair_outcome'))
    reverted_flag = reverted_repair_rounds_value > 0 or bool(repair_outcome_payload.get('reverted'))
    default_failure_signature = build_failure_signature(
        stop_layer=trigger_stop_layer,
        error=trigger_error,
        metrics=metrics,
    )
    failure_signature = _overlay_non_none(
        default_failure_signature,
        safe_dict(payload.get('failure_signature')),
    )
    default_repair_outcome = build_repair_outcome_summary(
        stop_layer=trigger_stop_layer,
        post_stop_layer=payload.get('post_stop_layer'),
        post_error=payload.get('post_error'),
        passed=payload.get('passed'),
        repair_rounds_used=repair_rounds_used_value,
        baseline_delta=baseline_delta,
        reverted=reverted_flag,
        status=repair_outcome_payload.get('status'),
    )
    repair_outcome = _overlay_non_none(
        default_repair_outcome,
        repair_outcome_payload,
    )

    normalized = dict(payload)
    normalized['schema_version'] = _DEFAULT_SCHEMA_VERSION
    normalized['strategy_delta'] = strategy_delta
    normalized['files_to_edit'] = normalize_string_list(payload.get('files_to_edit'))
    normalized['required_edit_paths'] = normalize_string_list(payload.get('required_edit_paths'))
    normalized['evidence_refs'] = normalize_string_list(payload.get('evidence_refs'))
    normalized['trigger_stop_layer'] = trigger_stop_layer
    normalized['trigger_error'] = _safe_optional_str(trigger_error)
    normalized['baseline_delta'] = baseline_delta
    normalized['val_ssim'] = val_ssim
    normalized['val_psnr'] = val_psnr
    normalized['training_curve_trend'] = _safe_optional_str(payload.get('training_curve_trend'))
    normalized['training_curve_last_epoch'] = training_curve_last_epoch
    normalized['reverted_repair_rounds'] = reverted_repair_rounds_value
    normalized['requires_model_changes'] = payload.get('requires_model_changes')
    normalized['loss_only_pipeline_viable'] = payload.get('loss_only_pipeline_viable')
    normalized['formula_requires_model_changes'] = payload.get('formula_requires_model_changes')
    normalized['metrics'] = metrics
    normalized['tags'] = _normalize_tags(safe_list(payload.get('tags')))
    normalized['failure_signature'] = failure_signature
    normalized['repair_outcome'] = repair_outcome
    normalized['provenance'] = safe_dict(payload.get('provenance'))
    return normalized


def read_jsonl_dicts(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def normalize_case_memory_record(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    schema_version = payload.get('schema_version')
    if schema_version == _DEFAULT_SCHEMA_VERSION:
        return _normalize_case_memory_payload(payload)

    if schema_version == _LEGACY_SCHEMA_VERSION:
        return _normalize_case_memory_payload(payload)

    if schema_version != 'decision_trace.v1':
        return None

    state = safe_dict(payload.get('state'))
    action = safe_dict(payload.get('action'))
    reward = safe_dict(payload.get('reward'))
    outcome = safe_dict(payload.get('outcome'))

    return _normalize_case_memory_payload(
        {
        'paper_slug': payload.get('paper_slug'),
        'attempt_id': payload.get('attempt_id'),
        'integration_path': state.get('integration_path'),
        'kind': action.get('kind'),
        'name': action.get('name'),
        'objective': action.get('objective'),
        'strategy_delta': safe_dict(action.get('strategy_delta')),
        'files_to_edit': normalize_string_list(action.get('files_to_edit')),
        'required_edit_paths': normalize_string_list(action.get('required_edit_paths')),
        'evidence_refs': normalize_string_list(action.get('evidence_refs')),
        'stop_layer': outcome.get('stop_layer'),
        'error': outcome.get('error'),
        'passed': outcome.get('passed'),
        'primary_metric_name': reward.get('primary_metric_name'),
        'primary_metric': reward.get('primary_metric'),
        'stage_score': reward.get('stage_score'),
        'repair_rounds_used': reward.get('repair_rounds_used'),
        'baseline_delta': reward.get('baseline_delta'),
        'val_ssim': reward.get('val_ssim'),
        'val_psnr': reward.get('val_psnr'),
        'training_curve_trend': reward.get('training_curve_trend'),
        'training_curve_last_epoch': reward.get('training_curve_last_epoch'),
        'reverted_repair_rounds': reward.get('reverted_repair_rounds'),
        'requires_model_changes': state.get('requires_model_changes'),
        'loss_only_pipeline_viable': state.get('loss_only_pipeline_viable'),
        'formula_requires_model_changes': state.get('formula_requires_model_changes'),
        'metrics': safe_dict(outcome.get('metrics')),
        'provenance': safe_dict(payload.get('provenance')),
        }
    )


def list_case_memory_sources(
    task_context: Optional[Dict[str, Any]] = None,
    *,
    case_memory_path: Optional[Path] = None,
) -> List[Path]:
    resolved_case_memory_path = (case_memory_path or DEFAULT_CASE_MEMORY_PATH).expanduser().resolve()
    if resolved_case_memory_path.exists():
        return [resolved_case_memory_path]

    if not isinstance(task_context, dict):
        return []

    paths = safe_dict(task_context.get('paths'))
    experiment_dir_value = paths.get('experiment_dir')
    if not isinstance(experiment_dir_value, str) or not experiment_dir_value.strip():
        return []

    experiment_dir = Path(experiment_dir_value).expanduser().resolve()
    experiments_root = experiment_dir.parent
    if not experiments_root.exists():
        return []

    return sorted(
        path
        for path in experiments_root.glob('*/decision_trace.jsonl')
        if path.is_file()
    )


def case_memory_key(record: Dict[str, Any]) -> str:
    provenance = safe_dict(record.get('provenance'))
    result_path = provenance.get('result_path')
    if isinstance(result_path, str) and result_path.strip():
        return f'result:{result_path}'
    return f'{record.get("paper_slug")}:{record.get("attempt_id")}:{record.get("name")}'


def load_case_memory_records(
    task_context: Optional[Dict[str, Any]] = None,
    *,
    case_memory_path: Optional[Path] = None,
    sources: Optional[Sequence[Path]] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    resolved_sources = (
        [Path(path).expanduser().resolve() for path in sources]
        if sources is not None
        else list_case_memory_sources(task_context, case_memory_path=case_memory_path)
    )

    for source in resolved_sources:
        for payload in read_jsonl_dicts(source):
            normalized = normalize_case_memory_record(payload)
            if not isinstance(normalized, dict):
                continue
            key = case_memory_key(normalized)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            records.append(normalized)

    return records


def merge_case_memory_records(
    *,
    case_memory_path: Path,
    records: List[Dict[str, Any]],
) -> int:
    resolved_case_memory_path = case_memory_path.expanduser().resolve()
    existing_by_key = {
        case_memory_key(item): item
        for item in load_case_memory_records(
            sources=[resolved_case_memory_path],
            case_memory_path=resolved_case_memory_path,
        )
        if isinstance(item, dict)
    }
    for record in records:
        normalized = normalize_case_memory_record(record)
        if not isinstance(normalized, dict):
            continue
        existing_by_key[case_memory_key(normalized)] = normalized

    resolved_case_memory_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_case_memory_path.open('w', encoding='utf-8') as handle:
        for record in existing_by_key.values():
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    return len(existing_by_key)


def _coerce_float(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _normalize_tags(values: Iterable[Any]) -> List[str]:
    tags: List[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        candidate = value.strip()
        if not candidate or candidate in tags:
            continue
        tags.append(candidate)
    return tags


def case_memory_record_to_innovation(
    record: Dict[str, Any],
    *,
    innovation_id: Optional[str] = None,
) -> Innovation:
    strategy_delta = safe_dict(record.get('strategy_delta'))
    strategy_changes = [
        str(item).strip()
        for item in safe_list(strategy_delta.get('what_changes_now'))
        if isinstance(item, str) and item.strip()
    ]
    record_tags = _normalize_tags(safe_list(record.get('tags')))
    tags = _normalize_tags(
        [
            *record_tags,
            str(record.get('component_type') or ''),
            str(record.get('integration_path') or ''),
            *strategy_changes,
        ]
    )
    if not tags and isinstance(record.get('paper_slug'), str):
        tags = [str(record['paper_slug'])]

    primary_metric = _coerce_float(record.get('primary_metric'))
    stage_score = _coerce_float(record.get('stage_score'))
    improvement = primary_metric if primary_metric > 0 else stage_score / 10.0
    confidence = min(max(stage_score / 10.0, 0.0), 1.0)
    if confidence == 0.0 and record.get('passed') is True:
        confidence = 0.5

    key_idea = str(
        record.get('key_idea')
        or record.get('objective')
        or record.get('name')
        or 'Historical experience'
    ).strip()
    why_works = str(
        record.get('why_works')
        or record.get('repair_hypothesis')
        or strategy_delta.get('expected_signal')
        or record.get('post_error')
        or 'Derived from case memory'
    ).strip()
    component_type = str(
        record.get('component_type')
        or record.get('integration_path')
        or record.get('kind')
        or 'unknown'
    ).strip()
    evidence = safe_dict(record.get('evidence'))
    baseline_ssim = _coerce_float(evidence.get('baseline_ssim'))
    new_ssim = _coerce_float(evidence.get('new_ssim'), default=primary_metric)
    if new_ssim == 0.0 and improvement > 0.0:
        new_ssim = improvement

    innovation: Innovation = {
        'paper': str(record.get('paper_slug') or 'unknown-paper'),
        'component_type': component_type or 'unknown',
        'key_idea': key_idea,
        'why_works': why_works,
        'improvement': improvement,
        'confidence': confidence,
        'evidence': {
            'baseline_ssim': baseline_ssim,
            'new_ssim': new_ssim,
        },
        'tags': tags,
    }
    resolved_innovation_id = innovation_id or str(record.get('attempt_id') or '').strip()
    if resolved_innovation_id:
        innovation['id'] = resolved_innovation_id
    return innovation


def load_case_memory_innovations(
    *,
    case_memory_path: Optional[Path] = None,
    task_context: Optional[Dict[str, Any]] = None,
) -> List[Innovation]:
    innovations: List[Innovation] = []
    records = load_case_memory_records(task_context, case_memory_path=case_memory_path)
    for index, record in enumerate(records, start=1):
        provenance = safe_dict(record.get('provenance'))
        result_path = provenance.get('result_path')
        innovation_id: Optional[str] = None
        if isinstance(result_path, str):
            match = _KNOWLEDGE_RESULT_PATH_RE.match(result_path)
            if match:
                innovation_id = match.group('identifier')
        if innovation_id is None:
            innovation_id = str(record.get('attempt_id') or f'case_{index:03d}')
        innovations.append(case_memory_record_to_innovation(record, innovation_id=innovation_id))
    return innovations


def _next_innovation_identifier(existing_records: List[Dict[str, Any]]) -> str:
    max_number = 0
    for record in existing_records:
        provenance = safe_dict(record.get('provenance'))
        result_path = provenance.get('result_path')
        if not isinstance(result_path, str):
            continue
        match = _KNOWLEDGE_RESULT_PATH_RE.match(result_path)
        if match:
            max_number = max(max_number, int(match.group('number')))
    return f'inn_{max_number + 1:03d}'


def innovation_to_case_memory_record(
    innovation: Innovation,
    *,
    innovation_id: str,
) -> Dict[str, Any]:
    tags = _normalize_tags(safe_list(innovation.get('tags')))
    why_works = str(innovation.get('why_works') or '').strip()
    component_type = str(innovation.get('component_type') or 'unknown').strip()
    key_idea = str(innovation.get('key_idea') or f'Imported innovation {innovation_id}').strip()
    improvement = _coerce_float(innovation.get('improvement'))
    confidence = min(max(_coerce_float(innovation.get('confidence')), 0.0), 1.0)
    evidence = safe_dict(innovation.get('evidence'))

    return _normalize_case_memory_payload(
        {
        'schema_version': _DEFAULT_SCHEMA_VERSION,
        'paper_slug': str(innovation.get('paper') or 'unknown-paper'),
        'attempt_id': innovation_id,
        'integration_path': component_type or 'unknown',
        'kind': 'knowledge_innovation',
        'name': innovation_id,
        'objective': key_idea,
        'strategy_delta': {
            'what_changes_now': tags,
            'expected_signal': why_works,
        },
        'stop_layer': None,
        'error': None,
        'passed': True,
        'primary_metric_name': 'improvement',
        'primary_metric': improvement,
        'stage_score': round(confidence * 10.0, 4),
        'repair_rounds_used': 0,
        'repair_hypothesis': why_works,
        'post_stop_layer': None,
        'post_error': None,
        'component_type': component_type or 'unknown',
        'key_idea': key_idea,
        'why_works': why_works,
        'tags': tags,
        'confidence': confidence,
        'evidence': {
            'baseline_ssim': _coerce_float(evidence.get('baseline_ssim')),
            'new_ssim': _coerce_float(evidence.get('new_ssim'), default=improvement),
        },
        'files_to_edit': [],
        'required_edit_paths': [],
        'evidence_refs': [],
        'baseline_delta': None,
        'val_ssim': None,
        'val_psnr': None,
        'training_curve_trend': None,
        'training_curve_last_epoch': None,
        'reverted_repair_rounds': 0,
        'metrics': {},
        'provenance': {
            'result_path': f'knowledge_db:{innovation_id}',
        },
        }
    )


def add_innovation_to_case_memory(
    innovation: Innovation,
    *,
    case_memory_path: Optional[Path] = None,
) -> str:
    resolved_case_memory_path = (case_memory_path or DEFAULT_CASE_MEMORY_PATH).expanduser().resolve()
    existing_records = load_case_memory_records(
        case_memory_path=resolved_case_memory_path,
        sources=[resolved_case_memory_path],
    )
    innovation_id = str(innovation.get('id') or '').strip() or _next_innovation_identifier(existing_records)
    record = innovation_to_case_memory_record(innovation, innovation_id=innovation_id)
    merge_case_memory_records(
        case_memory_path=resolved_case_memory_path,
        records=[record],
    )
    return innovation_id
