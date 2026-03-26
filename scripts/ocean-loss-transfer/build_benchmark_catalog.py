"""
@file build_benchmark_catalog.py
@description Scan a benchmark root and build a normalized catalog for loss-transfer experiments
@author OpenAI Codex
@date 2026-03-26
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_ARCHIVE_SUFFIXES = ('.zip', '.tar', '.tgz', '.gz', '.tar.gz')
_CODE_HINT_FILES = {'requirements.txt', 'setup.py', 'pyproject.toml', 'README.md'}
_CODE_HINT_DIRS = {'src', 'core', 'models', 'config', 'scripts'}


def _iter_dirs(root: Path, *, max_depth: int) -> Iterable[Path]:
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        yield current
        if depth >= max_depth:
            continue
        children = sorted((item for item in current.iterdir() if item.is_dir()), key=lambda p: p.name, reverse=True)
        for child in children:
            stack.append((child, depth + 1))


def _looks_like_code_repo(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(('-main', '-master', '-code', '-repo')):
        return True
    if any(item.is_file() and item.suffix == '.py' for item in path.iterdir()):
        return True
    if any((path / hint).is_file() for hint in _CODE_HINT_FILES):
        return True
    if any((path / hint).is_dir() for hint in _CODE_HINT_DIRS):
        return True
    return False


def _normalize_slug(text: str, *, fallback_seed: str) -> str:
    normalized = re.sub(r'[^a-zA-Z0-9]+', '-', text.lower()).strip('-')
    normalized = re.sub(r'-+', '-', normalized)
    if normalized:
        return normalized[:80]
    digest = hashlib.sha1(fallback_seed.encode('utf-8')).hexdigest()[:12]
    return f'benchmark-{digest}'


def _collect_direct_pdfs(directory: Path) -> list[Path]:
    return sorted(
        [item for item in directory.iterdir() if item.is_file() and item.suffix.lower() == '.pdf'],
        key=lambda p: p.name,
    )


def _collect_direct_archives(directory: Path) -> list[Path]:
    return sorted(
        [
            item
            for item in directory.iterdir()
            if item.is_file() and any(item.name.lower().endswith(suffix) for suffix in _ARCHIVE_SUFFIXES)
        ],
        key=lambda p: p.name,
    )


def _collect_direct_code_dirs(directory: Path) -> list[Path]:
    return sorted(
        [item for item in directory.iterdir() if item.is_dir() and _looks_like_code_repo(item)],
        key=lambda p: p.name,
    )


def _status_for(pdfs: list[Path], code_candidates: list[Dict[str, Any]]) -> str:
    if not pdfs or not code_candidates:
        return 'incomplete'
    if len(pdfs) == 1 and len(code_candidates) == 1:
        return 'ready'
    return 'ambiguous'


def _entry_from_directory(root: Path, directory: Path) -> Optional[Dict[str, Any]]:
    pdfs = _collect_direct_pdfs(directory)
    archives = _collect_direct_archives(directory)
    repos = _collect_direct_code_dirs(directory)

    if directory != root and _looks_like_code_repo(directory) and not pdfs and not archives:
        return None

    if not pdfs and not archives and not repos:
        return None

    relative_dir = '.' if directory == root else str(directory.relative_to(root))
    code_candidates: list[Dict[str, Any]] = []
    code_candidates.extend(
        {
            'path': str(path.resolve()),
            'type': 'repo',
            'extract_required': False,
        }
        for path in repos
    )
    code_candidates.extend(
        {
            'path': str(path.resolve()),
            'type': 'archive',
            'extract_required': True,
        }
        for path in archives
    )

    title = directory.name if directory != root else (pdfs[0].stem if len(pdfs) == 1 else root.name)
    category = None
    if directory != root:
        parts = directory.relative_to(root).parts
        category = parts[0] if len(parts) >= 2 else None

    slug_seed = relative_dir if relative_dir != '.' else title
    paper_slug = _normalize_slug(title, fallback_seed=slug_seed)
    status = _status_for(pdfs, code_candidates)
    suggested_pdf = str(pdfs[0].resolve()) if len(pdfs) == 1 else None
    suggested_code = str(Path(code_candidates[0]['path']).resolve()) if len(code_candidates) == 1 else None

    notes: list[str] = []
    if not pdfs:
        notes.append('missing_pdf')
    if not code_candidates:
        notes.append('missing_code')
    if len(pdfs) > 1:
        notes.append('multiple_pdfs')
    if len(code_candidates) > 1:
        notes.append('multiple_code_candidates')

    return {
        'entry_id': _normalize_slug(relative_dir if relative_dir != '.' else title, fallback_seed=slug_seed),
        'paper_slug': paper_slug,
        'title': title,
        'category': category,
        'benchmark_dir': str(directory.resolve()),
        'relative_dir': relative_dir,
        'paper_pdf_candidates': [str(path.resolve()) for path in pdfs],
        'code_candidates': code_candidates,
        'status': status,
        'suggested_paper_pdf_path': suggested_pdf,
        'suggested_code_path': suggested_code,
        'notes': notes,
    }


def build_benchmark_catalog(benchmark_root: str, *, max_depth: int = 2) -> Dict[str, Any]:
    root = Path(benchmark_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f'Benchmark root does not exist: {benchmark_root}')

    entries: list[Dict[str, Any]] = []
    for directory in _iter_dirs(root, max_depth=max_depth):
        entry = _entry_from_directory(root, directory)
        if entry is not None:
            entries.append(entry)

    entries.sort(key=lambda item: (item.get('relative_dir') != '.', str(item.get('relative_dir'))))
    status_counts: Dict[str, int] = {}
    for entry in entries:
        status = str(entry.get('status', 'unknown'))
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        'benchmark_root': str(root),
        'entry_count': len(entries),
        'status_counts': status_counts,
        'entries': entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a normalized benchmark catalog')
    parser.add_argument('--benchmark_root', default='Benchmark')
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--max_depth', type=int, default=2)
    args = parser.parse_args()

    catalog = build_benchmark_catalog(args.benchmark_root, max_depth=args.max_depth)
    if args.output_path:
        output_path = Path(args.output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(catalog, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
