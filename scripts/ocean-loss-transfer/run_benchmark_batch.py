"""
@file run_benchmark_batch.py
@description Batch entry point for benchmark-driven loss-transfer experiments
@author OpenAI Codex
@date 2026-03-26
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from agent_artifact_generator import generate_analysis_plan
from agent_repair_loop import run_agent_repair_loop
from build_benchmark_catalog import build_benchmark_catalog
from context_builder import build_task_context
from materialize_benchmark_entry import materialize_benchmark_entry
from trajectory_logger import write_json


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_BATCH_ROOT = _PROJECT_ROOT / 'sandbox' / 'benchmark_batch_runs'


def _load_catalog(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Catalog at {path} must be a JSON object')
    return data


def _normalize_filters(values: Optional[Sequence[str]]) -> set[str]:
    if not values:
        return set()
    return {str(value).strip() for value in values if str(value).strip()}


def _resolve_batch_run_dir(output_root: Optional[str], run_id: Optional[str]) -> Path:
    root = Path(output_root).expanduser().resolve() if output_root else _DEFAULT_BATCH_ROOT
    resolved_run_id = run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir = root / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_catalog(
    *,
    benchmark_root: Optional[str],
    catalog_path: Optional[str],
    run_dir: Path,
) -> Dict[str, Any]:
    if catalog_path:
        catalog = _load_catalog(Path(catalog_path).expanduser().resolve())
    else:
        catalog = build_benchmark_catalog(benchmark_root or 'Benchmark', max_depth=2)
    write_json(run_dir / 'catalog_snapshot.json', catalog)
    return catalog


def select_benchmark_entries(
    catalog: Dict[str, Any],
    *,
    entry_ids: Optional[Sequence[str]] = None,
    paper_slugs: Optional[Sequence[str]] = None,
    categories: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    entries = catalog.get('entries')
    if not isinstance(entries, list):
        raise ValueError('Catalog is missing entries[]')

    entry_id_filter = _normalize_filters(entry_ids)
    paper_slug_filter = _normalize_filters(paper_slugs)
    category_filter = _normalize_filters(categories)

    selected: List[Dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        if item.get('status') != 'ready':
            continue
        if entry_id_filter and str(item.get('entry_id')) not in entry_id_filter:
            continue
        if paper_slug_filter and str(item.get('paper_slug')) not in paper_slug_filter:
            continue
        if category_filter and str(item.get('category') or '') not in category_filter:
            continue
        selected.append(item)

    selected.sort(key=lambda item: (str(item.get('category') or ''), str(item.get('relative_dir') or '')))
    if limit is not None:
        return selected[: max(0, int(limit))]
    return selected


def _formula_status(task_context: Dict[str, Any]) -> str:
    formula_draft_status = task_context.get('formula_draft_status')
    if isinstance(formula_draft_status, dict):
        status = formula_draft_status.get('status')
        if isinstance(status, str) and status:
            return status

    paths = task_context.get('paths')
    if isinstance(paths, dict) and paths.get('loss_formula_path'):
        return 'available'
    return 'missing'


def _select_representative_attempt(loop_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    attempts = loop_result.get('attempts')
    if not isinstance(attempts, list) or not attempts:
        return None

    best_attempt_id = loop_result.get('best_attempt_id')
    if best_attempt_id is not None:
        for attempt in attempts:
            if isinstance(attempt, dict) and attempt.get('attempt_id') == best_attempt_id:
                return attempt

    for attempt in attempts:
        if isinstance(attempt, dict) and attempt.get('passed'):
            return attempt
    return attempts[-1] if isinstance(attempts[-1], dict) else None


def _extract_loop_error(loop_result: Dict[str, Any]) -> Optional[str]:
    representative = _select_representative_attempt(loop_result)
    if representative is None:
        return None
    error = representative.get('error')
    return str(error) if error else None


def _derive_overall_status(entry_summary: Dict[str, Any], *, mode: str) -> str:
    error_summary = entry_summary.get('error_summary')
    if isinstance(error_summary, str) and error_summary:
        return 'error'

    if mode == 'agent_loop':
        return str(entry_summary.get('loop_status') or 'unknown')
    if mode == 'plan_only':
        return str(entry_summary.get('plan_status') or entry_summary.get('context_status') or 'unknown')
    return str(entry_summary.get('context_status') or 'unknown')


def _run_single_entry(
    entry: Dict[str, Any],
    *,
    benchmark_root: Optional[str],
    catalog_path: Optional[str],
    cache_root: Optional[str],
    dataset_root: Optional[str],
    mode: str,
    max_attempts: int,
    auto_generate_plan: bool,
    bootstrap_formula: bool,
    service_url: Optional[str],
    service_api_key: Optional[str],
    timeout_sec: int,
    entry_output_dir: Path,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        'entry_id': entry.get('entry_id'),
        'paper_slug': entry.get('paper_slug'),
        'title': entry.get('title'),
        'category': entry.get('category'),
        'relative_dir': entry.get('relative_dir'),
        'catalog_status': entry.get('status'),
        'context_status': 'pending',
        'formula_status': 'pending',
        'plan_status': 'skipped',
        'loop_status': 'skipped',
        'best_attempt_id': None,
        'best_metric_name': None,
        'best_metric_value': None,
        'stop_layer': None,
        'error_summary': None,
    }

    try:
        materialized = materialize_benchmark_entry(
            benchmark_root=benchmark_root,
            catalog_path=catalog_path,
            entry_id=str(entry['entry_id']),
            cache_root=cache_root,
        )
        summary.update(
            {
                'paper_pdf_path': materialized.get('paper_pdf_path'),
                'code_repo_path': materialized.get('code_repo_path'),
                'source_code_path': materialized.get('source_code_path'),
                'materialized': bool(materialized.get('materialized', False)),
                'cache_dir': materialized.get('cache_dir'),
            }
        )

        task_context = build_task_context(
            paper_slug=str(entry['paper_slug']),
            paper_pdf_path=materialized.get('paper_pdf_path'),
            code_repo_path=materialized.get('code_repo_path'),
            dataset_root=dataset_root,
            output_dir=str(entry_output_dir),
        )
        paths = task_context.get('paths', {}) if isinstance(task_context.get('paths'), dict) else {}
        summary.update(
            {
                'context_status': str(task_context.get('status', 'unknown')),
                'formula_status': _formula_status(task_context),
                'experiment_dir': paths.get('experiment_dir'),
                'task_context_path': paths.get('task_context_path'),
                'loss_formula_path': paths.get('loss_formula_path'),
                'analysis_plan_path': paths.get('analysis_plan_path'),
                'trajectory_path': paths.get('trajectory_path'),
            }
        )

        analysis_plan_path = paths.get('analysis_plan_path') if isinstance(paths.get('analysis_plan_path'), str) else None
        if mode in {'plan_only', 'agent_loop'} and auto_generate_plan:
            if not isinstance(paths.get('task_context_path'), str):
                raise ValueError('task_context_path is missing from task_context')
            plan_result = generate_analysis_plan(
                paths['task_context_path'],
                max_attempts=max_attempts,
                service_url=service_url,
                api_key=service_api_key,
                timeout_sec=timeout_sec,
            )
            summary['plan_status'] = str(plan_result.get('status', 'unknown'))
            if plan_result.get('analysis_plan_path'):
                analysis_plan_path = str(plan_result['analysis_plan_path'])
                summary['analysis_plan_path'] = analysis_plan_path
            if plan_result.get('status') != 'success':
                summary['error_summary'] = str(plan_result.get('error') or 'analysis plan generation failed')
                summary['overall_status'] = _derive_overall_status(summary, mode=mode)
                return summary

        existing_plan_path = (
            analysis_plan_path
            if isinstance(analysis_plan_path, str) and Path(analysis_plan_path).exists()
            else None
        )

        if mode == 'agent_loop':
            loop_result = run_agent_repair_loop(
                task_context,
                analysis_plan_path=existing_plan_path,
                max_attempts=max_attempts,
                bootstrap_formula=bootstrap_formula,
                dataset_root=dataset_root,
                output_dir=str(entry_output_dir),
                agent_service_url=service_url,
                agent_api_key=service_api_key,
            )
            representative_attempt = _select_representative_attempt(loop_result)
            summary.update(
                {
                    'loop_status': str(loop_result.get('status', 'unknown')),
                    'best_attempt_id': loop_result.get('best_attempt_id'),
                    'best_metric_name': loop_result.get('best_metric_name'),
                    'best_metric_value': loop_result.get('best_metric_value'),
                    'stop_layer': representative_attempt.get('stop_layer') if representative_attempt else None,
                    'trajectory_path': loop_result.get('trajectory_path', summary.get('trajectory_path')),
                }
            )
            if summary.get('loop_status') == 'completed_with_failures':
                summary['error_summary'] = _extract_loop_error(loop_result)

        summary['overall_status'] = _derive_overall_status(summary, mode=mode)
        return summary

    except Exception as exc:
        summary['error_summary'] = str(exc)
        summary['overall_status'] = 'error'
        return summary


def run_benchmark_batch(
    *,
    benchmark_root: Optional[str] = None,
    catalog_path: Optional[str] = None,
    entry_ids: Optional[Sequence[str]] = None,
    paper_slugs: Optional[Sequence[str]] = None,
    categories: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    output_root: Optional[str] = None,
    run_id: Optional[str] = None,
    cache_root: Optional[str] = None,
    dataset_root: Optional[str] = None,
    mode: str = 'context_only',
    max_attempts: int = 4,
    auto_generate_plan: bool = False,
    bootstrap_formula: bool = True,
    service_url: Optional[str] = None,
    service_api_key: Optional[str] = None,
    timeout_sec: int = 900,
) -> Dict[str, Any]:
    if mode not in {'context_only', 'plan_only', 'agent_loop'}:
        raise ValueError(f'Unsupported mode: {mode}')

    run_dir = _resolve_batch_run_dir(output_root, run_id)
    catalog = _resolve_catalog(
        benchmark_root=benchmark_root,
        catalog_path=catalog_path,
        run_dir=run_dir,
    )
    selected_entries = select_benchmark_entries(
        catalog,
        entry_ids=entry_ids,
        paper_slugs=paper_slugs,
        categories=categories,
        limit=limit,
    )

    results: List[Dict[str, Any]] = []
    for index, entry in enumerate(selected_entries, start=1):
        entry_output_dir = run_dir / 'entries' / str(entry['entry_id'])
        entry_output_dir.mkdir(parents=True, exist_ok=True)
        entry_summary = _run_single_entry(
            entry,
            benchmark_root=benchmark_root,
            catalog_path=catalog_path,
            cache_root=cache_root,
            dataset_root=dataset_root,
            mode=mode,
            max_attempts=max_attempts,
            auto_generate_plan=auto_generate_plan,
            bootstrap_formula=bootstrap_formula,
            service_url=service_url,
            service_api_key=service_api_key,
            timeout_sec=timeout_sec,
            entry_output_dir=entry_output_dir,
        )
        entry_summary['batch_index'] = index
        results.append(entry_summary)

    overall_status_counts: Dict[str, int] = {}
    for item in results:
        status = str(item.get('overall_status', 'unknown'))
        overall_status_counts[status] = overall_status_counts.get(status, 0) + 1

    summary: Dict[str, Any] = {
        'status': 'completed',
        'run_id': run_dir.name,
        'mode': mode,
        'auto_generate_plan': auto_generate_plan,
        'bootstrap_formula': bootstrap_formula,
        'run_dir': str(run_dir),
        'catalog_path': str(run_dir / 'catalog_snapshot.json'),
        'benchmark_root': catalog.get('benchmark_root') or benchmark_root,
        'selected_count': len(selected_entries),
        'overall_status_counts': overall_status_counts,
        'results': results,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_dir / 'benchmark_run_summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a batch of benchmark-driven loss-transfer experiments')
    parser.add_argument('--benchmark_root', default=None)
    parser.add_argument('--catalog_path', default=None)
    parser.add_argument('--entry_id', action='append', dest='entry_ids', default=None)
    parser.add_argument('--paper_slug', action='append', dest='paper_slugs', default=None)
    parser.add_argument('--category', action='append', dest='categories', default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--output_root', default=None)
    parser.add_argument('--run_id', default=None)
    parser.add_argument('--cache_root', default=None)
    parser.add_argument('--dataset_root', default=None)
    parser.add_argument('--mode', choices=['context_only', 'plan_only', 'agent_loop'], default='context_only')
    parser.add_argument('--max_attempts', type=int, default=4)
    parser.add_argument('--auto_generate_plan', action='store_true', default=False)
    parser.add_argument('--bootstrap_formula', action='store_true', default=True)
    parser.add_argument('--no_bootstrap_formula', action='store_false', dest='bootstrap_formula')
    parser.add_argument('--service_url', default=None)
    parser.add_argument('--service_api_key', default=None)
    parser.add_argument('--timeout_sec', type=int, default=900)
    args = parser.parse_args()

    result = run_benchmark_batch(
        benchmark_root=args.benchmark_root,
        catalog_path=args.catalog_path,
        entry_ids=args.entry_ids,
        paper_slugs=args.paper_slugs,
        categories=args.categories,
        limit=args.limit,
        output_root=args.output_root,
        run_id=args.run_id,
        cache_root=args.cache_root,
        dataset_root=args.dataset_root,
        mode=args.mode,
        max_attempts=args.max_attempts,
        auto_generate_plan=args.auto_generate_plan,
        bootstrap_formula=args.bootstrap_formula,
        service_url=args.service_url,
        service_api_key=args.service_api_key,
        timeout_sec=args.timeout_sec,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
