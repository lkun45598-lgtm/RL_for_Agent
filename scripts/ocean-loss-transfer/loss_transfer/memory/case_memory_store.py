"""
@file case_memory_store.py
@description Shared case-memory persistence and compatibility helpers for historical experience records.
@author kongzhiquan
@date 2026-03-28
@version 1.0.0

@changelog
  - 2026-03-28 kongzhiquan: v1.0.0 extract shared case-memory storage and innovation compatibility helpers
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from loss_transfer.common._types import Innovation
from loss_transfer.common.paths import PROJECT_ROOT


DEFAULT_CASE_MEMORY_PATH = PROJECT_ROOT / 'workflow' / 'loss_transfer' / 'knowledge_base' / 'case_memories.jsonl'
_DEFAULT_SCHEMA_VERSION = 'case_memory.v1'
_KNOWLEDGE_RESULT_PATH_RE = re.compile(r'^knowledge_db:(?P<identifier>inn_(?P<number>\d+))$')


def safe_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def safe_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


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
        normalized = dict(payload)
        normalized.setdefault('schema_version', _DEFAULT_SCHEMA_VERSION)
        return normalized

    if schema_version != 'decision_trace.v1':
        return None

    state = safe_dict(payload.get('state'))
    action = safe_dict(payload.get('action'))
    reward = safe_dict(payload.get('reward'))
    outcome = safe_dict(payload.get('outcome'))

    return {
        'schema_version': _DEFAULT_SCHEMA_VERSION,
        'paper_slug': payload.get('paper_slug'),
        'attempt_id': payload.get('attempt_id'),
        'integration_path': state.get('integration_path'),
        'kind': action.get('kind'),
        'name': action.get('name'),
        'objective': action.get('objective'),
        'strategy_delta': safe_dict(action.get('strategy_delta')),
        'stop_layer': outcome.get('stop_layer'),
        'error': outcome.get('error'),
        'passed': outcome.get('passed'),
        'primary_metric_name': reward.get('primary_metric_name'),
        'primary_metric': reward.get('primary_metric'),
        'stage_score': reward.get('stage_score'),
        'repair_rounds_used': reward.get('repair_rounds_used'),
        'provenance': safe_dict(payload.get('provenance')),
    }


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

    return {
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
        'provenance': {
            'result_path': f'knowledge_db:{innovation_id}',
        },
    }


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
