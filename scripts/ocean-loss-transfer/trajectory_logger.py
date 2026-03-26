"""
@file trajectory_logger.py
@description Structured logging helpers for the agentic loss-transfer loop.
@author Leizheng
@date 2026-03-25
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_EXPERIMENT_ROOT = _PROJECT_ROOT / 'sandbox' / 'loss_transfer_experiments'


def ensure_experiment_dir(paper_slug: str, output_dir: Optional[str] = None) -> Path:
    base_dir = Path(output_dir) if output_dir else (_DEFAULT_EXPERIMENT_ROOT / paper_slug)
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, '__dict__') and not isinstance(value, type):
        return _to_jsonable(vars(value))
    return value


def write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    return str(path)


def append_trajectory_event(
    paper_slug: str,
    event_type: str,
    payload: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> str:
    base_dir = ensure_experiment_dir(paper_slug, output_dir=output_dir)
    log_path = base_dir / 'trajectory.jsonl'
    event = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'event_type': event_type,
        'payload': _to_jsonable(payload),
    }
    with log_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(event, ensure_ascii=False) + '\n')
    return str(log_path)


def write_attempt_artifacts(
    base_dir: Path,
    attempt_id: int,
    *,
    attempt_spec: Optional[Dict[str, Any]] = None,
    code: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
) -> Path:
    attempt_dir = base_dir / f'attempt_{attempt_id}'
    attempt_dir.mkdir(parents=True, exist_ok=True)

    if attempt_spec is not None:
        write_json(attempt_dir / 'attempt_spec.json', attempt_spec)
    if code is not None:
        (attempt_dir / 'candidate_loss.py').write_text(code, encoding='utf-8')
    if result is not None:
        write_json(attempt_dir / 'result.json', result)

    return attempt_dir
