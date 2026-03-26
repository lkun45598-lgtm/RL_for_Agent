"""
@file materialize_benchmark_entry.py
@description Resolve one benchmark catalog entry into a ready-to-run paper/code bundle
@author OpenAI Codex
@date 2026-03-26
"""

from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from build_benchmark_catalog import build_benchmark_catalog


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CACHE_ROOT = _PROJECT_ROOT / 'sandbox' / 'benchmark_code_cache'


def _load_catalog(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Catalog at {path} must be a JSON object')
    return data


def _resolve_entry(catalog: Dict[str, Any], *, entry_id: Optional[str], paper_slug: Optional[str]) -> Dict[str, Any]:
    entries = catalog.get('entries')
    if not isinstance(entries, list):
        raise ValueError('Catalog is missing entries[]')

    matches = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry_id and entry.get('entry_id') == entry_id:
            matches.append(entry)
        elif paper_slug and entry.get('paper_slug') == paper_slug:
            matches.append(entry)

    if not matches:
        raise ValueError('No benchmark entry matched the requested entry_id/paper_slug')
    if len(matches) > 1:
        raise ValueError('Multiple benchmark entries matched; please use entry_id')
    return matches[0]


def _extract_archive(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    lower_name = archive_path.name.lower()
    if lower_name.endswith('.zip'):
        with zipfile.ZipFile(archive_path) as zip_file:
            zip_file.extractall(target_dir)
        return
    if lower_name.endswith(('.tar', '.tgz', '.tar.gz', '.gz')):
        with tarfile.open(archive_path) as tar_file:
            tar_file.extractall(target_dir)
        return
    raise ValueError(f'Unsupported archive format: {archive_path}')


def _looks_like_repo_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    if any(item.is_file() and item.suffix == '.py' for item in path.iterdir()):
        return True
    if any((path / name).is_dir() for name in ('src', 'core', 'models', 'config', 'scripts')):
        return True
    if any((path / name).is_file() for name in ('requirements.txt', 'pyproject.toml', 'setup.py', 'README.md')):
        return True
    return False


def _resolve_extracted_repo_root(extraction_dir: Path) -> Path:
    if _looks_like_repo_root(extraction_dir):
        return extraction_dir

    children = [item for item in extraction_dir.iterdir() if item.is_dir()]
    if len(children) == 1 and _looks_like_repo_root(children[0]):
        return children[0]

    repo_candidates = [item for item in children if _looks_like_repo_root(item)]
    if len(repo_candidates) == 1:
        return repo_candidates[0]

    raise ValueError(f'Unable to determine extracted repo root under {extraction_dir}')


def materialize_benchmark_entry(
    *,
    benchmark_root: Optional[str] = None,
    catalog_path: Optional[str] = None,
    entry_id: Optional[str] = None,
    paper_slug: Optional[str] = None,
    cache_root: Optional[str] = None,
) -> Dict[str, Any]:
    if not entry_id and not paper_slug:
        raise ValueError('Either entry_id or paper_slug must be provided')

    if catalog_path:
        catalog = _load_catalog(Path(catalog_path).expanduser().resolve())
    else:
        catalog = build_benchmark_catalog(benchmark_root or 'Benchmark', max_depth=2)

    entry = _resolve_entry(catalog, entry_id=entry_id, paper_slug=paper_slug)
    if entry.get('status') != 'ready':
        raise ValueError(f'Benchmark entry is not ready: status={entry.get("status")}')

    paper_pdf_path = entry.get('suggested_paper_pdf_path')
    code_path = entry.get('suggested_code_path')
    if not isinstance(paper_pdf_path, str) or not isinstance(code_path, str):
        raise ValueError('Ready benchmark entry is missing suggested paper/code paths')

    code_candidates = entry.get('code_candidates')
    if not isinstance(code_candidates, list) or not code_candidates:
        raise ValueError('Benchmark entry has no code candidates')
    primary_candidate = code_candidates[0]
    if not isinstance(primary_candidate, dict):
        raise ValueError('Benchmark entry code candidate is malformed')

    candidate_type = str(primary_candidate.get('type', 'repo'))
    if candidate_type == 'repo':
        resolved_code_repo = str(Path(code_path).expanduser().resolve())
        return {
            'status': 'ready',
            'entry_id': entry.get('entry_id'),
            'paper_slug': entry.get('paper_slug'),
            'paper_pdf_path': str(Path(paper_pdf_path).expanduser().resolve()),
            'code_repo_path': resolved_code_repo,
            'source_code_path': resolved_code_repo,
            'materialized': False,
        }

    if candidate_type != 'archive':
        raise ValueError(f'Unsupported code candidate type: {candidate_type}')

    archive_path = Path(code_path).expanduser().resolve()
    cache_dir = (Path(cache_root).expanduser().resolve() if cache_root else _DEFAULT_CACHE_ROOT) / str(entry['entry_id'])
    marker_path = cache_dir / '.materialized_from'
    source_marker = str(archive_path)

    if not marker_path.exists() or marker_path.read_text(encoding='utf-8') != source_marker:
        if cache_dir.exists():
            for child in sorted(cache_dir.iterdir(), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink()
                else:
                    import shutil
                    shutil.rmtree(child)
        _extract_archive(archive_path, cache_dir)
        marker_path.write_text(source_marker, encoding='utf-8')

    repo_root = _resolve_extracted_repo_root(cache_dir)
    return {
        'status': 'ready',
        'entry_id': entry.get('entry_id'),
        'paper_slug': entry.get('paper_slug'),
        'paper_pdf_path': str(Path(paper_pdf_path).expanduser().resolve()),
        'code_repo_path': str(repo_root),
        'source_code_path': str(archive_path),
        'materialized': True,
        'cache_dir': str(cache_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Materialize one benchmark entry into a ready-to-run bundle')
    parser.add_argument('--benchmark_root', default=None)
    parser.add_argument('--catalog_path', default=None)
    parser.add_argument('--entry_id', default=None)
    parser.add_argument('--paper_slug', default=None)
    parser.add_argument('--cache_root', default=None)
    args = parser.parse_args()

    result = materialize_benchmark_entry(
        benchmark_root=args.benchmark_root,
        catalog_path=args.catalog_path,
        entry_id=args.entry_id,
        paper_slug=args.paper_slug,
        cache_root=args.cache_root,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
