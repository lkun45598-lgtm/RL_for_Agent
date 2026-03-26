"""
@file agent_edit_workspace.py
@description Attempt-scoped editable workspace and manifest helpers for agent-authored loss transfers.
@author OpenAI Codex
@date 2026-03-26
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from loss_transfer.attempts.integration_policy import (
    integration_path_needs_adapter_overrides,
    integration_path_needs_model_tree,
    resolve_recommended_integration_path,
)
from loss_transfer.common.paths import PROJECT_ROOT, SANDBOX_DIR, TRAINING_PIPELINE_DIR


_PROJECT_ROOT = PROJECT_ROOT
_SANDBOX_ROOT = SANDBOX_DIR
_PIPELINE_ROOT = TRAINING_PIPELINE_DIR
_DEFAULT_OVERRIDE_FILE_SOURCES = {
    'sandbox_model_adapter.py': _SANDBOX_ROOT / 'sandbox_model_adapter.py',
    'sandbox_trainer.py': _SANDBOX_ROOT / 'sandbox_trainer.py',
    '_run_once.py': _SANDBOX_ROOT / '_run_once.py',
}
_DEFAULT_OVERRIDE_TREE_SOURCES = {
    'models': _PIPELINE_ROOT / 'models',
}
_DEFAULT_OVERRIDE_FILE_ALIASES = {
    'sandbox model adapter files exposing extra loss inputs': ['sandbox_model_adapter.py'],
    'sandbox adapter/model-output layer': ['sandbox_model_adapter.py', 'sandbox_trainer.py'],
    'sandbox trainer files': ['sandbox_trainer.py'],
    'sandbox runtime entrypoint': ['_run_once.py'],
    'sandbox/sandbox_model_adapter.py': ['sandbox_model_adapter.py'],
    'sandbox/sandbox_trainer.py': ['sandbox_trainer.py'],
    'sandbox/_run_once.py': ['_run_once.py'],
}
_DEFAULT_OVERRIDE_TREE_ALIASES = {
    'model files': ['models'],
    'copied model files': ['models'],
    'sandbox copied model files': ['models'],
    'training model files': ['models'],
    'scripts/ocean-sr-training-masked/models': ['models'],
    'scripts/ocean-SR-training-masked/models': ['models'],
    'models': ['models'],
}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and str(item).strip()]


def normalize_required_edit_paths(attempt_spec: Dict[str, Any]) -> list[str]:
    return as_string_list(attempt_spec.get('required_edit_paths'))


def resolve_requested_override_files(
    task_context: Dict[str, Any],
    attempt_spec: Dict[str, Any],
    *,
    override_file_sources: Optional[Dict[str, Path]] = None,
    override_tree_sources: Optional[Dict[str, Path]] = None,
    override_file_aliases: Optional[Dict[str, list[str]]] = None,
    override_tree_aliases: Optional[Dict[str, list[str]]] = None,
) -> Dict[str, list[str]]:
    file_sources = override_file_sources or _DEFAULT_OVERRIDE_FILE_SOURCES
    tree_sources = override_tree_sources or _DEFAULT_OVERRIDE_TREE_SOURCES
    file_aliases = override_file_aliases or _DEFAULT_OVERRIDE_FILE_ALIASES
    tree_aliases = override_tree_aliases or _DEFAULT_OVERRIDE_TREE_ALIASES

    resolved_files: list[str] = []
    resolved_trees: list[str] = []
    requested = as_string_list(attempt_spec.get('files_to_edit'))

    for item in requested:
        normalized = item.strip()
        if normalized == 'candidate_loss.py':
            continue
        alias_matches = file_aliases.get(normalized.lower())
        if alias_matches:
            for name in alias_matches:
                if name not in resolved_files:
                    resolved_files.append(name)
            continue
        tree_alias_matches = tree_aliases.get(normalized.lower())
        if tree_alias_matches:
            for name in tree_alias_matches:
                if name not in resolved_trees:
                    resolved_trees.append(name)
            continue
        candidate_name = Path(normalized).name
        if candidate_name in file_sources and candidate_name not in resolved_files:
            resolved_files.append(candidate_name)
        if candidate_name in tree_sources and candidate_name not in resolved_trees:
            resolved_trees.append(candidate_name)

    recommended_path = resolve_recommended_integration_path(task_context)
    if integration_path_needs_adapter_overrides(recommended_path) and 'sandbox_model_adapter.py' not in resolved_files:
        resolved_files.append('sandbox_model_adapter.py')
    if integration_path_needs_adapter_overrides(recommended_path) and 'sandbox_trainer.py' not in resolved_files:
        resolved_files.append('sandbox_trainer.py')
    if integration_path_needs_model_tree(recommended_path) and 'models' not in resolved_trees:
        resolved_trees.append('models')

    return {
        'files': resolved_files,
        'trees': resolved_trees,
    }


def prepare_attempt_edit_workspace(
    *,
    task_context: Dict[str, Any],
    attempt_spec: Dict[str, Any],
    output_code_path: Path,
    override_file_sources: Optional[Dict[str, Path]] = None,
    override_tree_sources: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
    file_sources = override_file_sources or _DEFAULT_OVERRIDE_FILE_SOURCES
    tree_sources = override_tree_sources or _DEFAULT_OVERRIDE_TREE_SOURCES

    attempt_dir = output_code_path.parent
    attempt_dir.mkdir(parents=True, exist_ok=True)
    if not output_code_path.exists():
        output_code_path.write_text(
            '# Agent will replace this placeholder with the attempt-specific candidate loss.\n',
            encoding='utf-8',
        )
    override_dir = attempt_dir / 'sandbox_overrides'
    integration = task_context.get('integration_assessment', {}) if isinstance(task_context.get('integration_assessment'), dict) else {}
    recommended_path = resolve_recommended_integration_path(task_context)
    editable_targets = [
        {
            'path': str(output_code_path),
            'kind': 'candidate_loss',
            'description': 'Primary sandbox loss entrypoint for this attempt.',
        }
    ]

    override_targets = resolve_requested_override_files(
        task_context,
        attempt_spec,
        override_file_sources=file_sources,
        override_tree_sources=tree_sources,
    )
    override_files = override_targets['files']
    override_trees = override_targets['trees']
    if override_files or override_trees:
        override_dir.mkdir(parents=True, exist_ok=True)

    for file_name in override_files:
        source_path = file_sources[file_name]
        target_path = override_dir / file_name
        if not target_path.exists():
            shutil.copy2(source_path, target_path)
        editable_targets.append(
            {
                'path': str(target_path),
                'kind': 'sandbox_override',
                'source_path': str(source_path),
                'description': (
                    'Attempt-scoped sandbox override. Validators load this file via '
                    'SANDBOX_OVERRIDE_DIR instead of editing repo-root sandbox modules.'
                ),
            }
        )

    for tree_name in override_trees:
        source_dir = tree_sources[tree_name]
        target_dir = override_dir / tree_name
        if not target_dir.exists():
            shutil.copytree(source_dir, target_dir)
        editable_targets.append(
            {
                'path': str(target_dir),
                'kind': 'sandbox_override_tree',
                'source_path': str(source_dir),
                'description': (
                    'Attempt-scoped copy of the original training model package. '
                    'Edit files under this directory only when model-level changes are required.'
                ),
            }
        )

    manifest = {
        'candidate_loss_path': str(output_code_path),
        'sandbox_override_dir': str(override_dir) if (override_files or override_trees) else None,
        'routing_policy': {
            'recommended_path': recommended_path,
            'requires_model_changes': bool(integration.get('requires_model_changes')),
            'validator_behavior': (
                'For formulas that need model-provided loss inputs, validators prefer '
                'attempt-scoped model-output extension when the copied model constructor '
                'supports output_aux_loss_inputs; otherwise they fall back to sandbox_adapter heads.'
            ),
        },
        'editable_targets': editable_targets,
        'notes': [
            'Edit only the files listed here.',
            'Do not modify repo-root sandbox/, training/, or data-processing files during loss transfer attempts.',
            'If sandbox_override_dir is present, validators will load same-named Python modules from it first.',
            'Directory targets mean the whole copied tree is editable, but only inside that attempt-scoped copy.',
            'If recommended_path is extend_model_outputs or model_surgery, prefer editing the copied models/ tree instead of forcing a loss-only hack.',
        ],
    }
    manifest_path = attempt_dir / 'editable_files.json'
    _write_json(manifest_path, manifest)
    return {
        'manifest_path': manifest_path,
        'editable_targets': editable_targets,
        'sandbox_override_dir': override_dir if (override_files or override_trees) else None,
    }


def format_editable_targets(editable_targets: list[Dict[str, Any]]) -> str:
    lines = []
    for item in editable_targets:
        path = item.get('path')
        description = item.get('description')
        if isinstance(path, str):
            if isinstance(description, str) and description:
                lines.append(f'- {path}  # {description}')
            else:
                lines.append(f'- {path}')
    return '\n'.join(lines)


def _path_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    if not path.exists():
        hasher.update(b'missing')
        return hasher.hexdigest()

    if path.is_file():
        hasher.update(b'file')
        hasher.update(path.read_bytes())
        return hasher.hexdigest()

    hasher.update(b'dir')
    for child in sorted(p for p in path.rglob('*') if p.is_file()):
        hasher.update(str(child.relative_to(path)).encode('utf-8'))
        hasher.update(child.read_bytes())
    return hasher.hexdigest()


def snapshot_editable_targets(editable_targets: list[Dict[str, Any]]) -> Dict[str, str]:
    snapshot: Dict[str, str] = {}
    for item in editable_targets:
        path = item.get('path')
        if isinstance(path, str) and path.strip():
            snapshot[path] = _path_digest(Path(path))
    return snapshot


def detect_touched_paths(
    before_snapshot: Dict[str, str],
    after_snapshot: Dict[str, str],
) -> list[str]:
    touched: list[str] = []
    for path, before_digest in before_snapshot.items():
        after_digest = after_snapshot.get(path)
        if after_digest is None or after_digest != before_digest:
            touched.append(path)
    return touched


def load_existing_touched_paths(attempt_dir: Path) -> list[str]:
    touched: list[str] = []
    for log_name in ('agent_code_generation_response.json', 'agent_code_repair_response.json'):
        log_path = attempt_dir / log_name
        if not log_path.exists():
            continue
        try:
            payload = json.loads(log_path.read_text(encoding='utf-8'))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        logged_paths = payload.get('touched_paths') if isinstance(payload, dict) else None
        if not isinstance(logged_paths, list):
            continue
        for path in logged_paths:
            if isinstance(path, str) and path.strip() and path not in touched:
                touched.append(path)
    return touched


def path_matches_requirement(path: str, requirement: str) -> bool:
    normalized_req = requirement.strip().rstrip('/').lower()
    candidate = path.strip().rstrip('/').lower()
    if not normalized_req or not candidate:
        return False
    if normalized_req in candidate:
        return True
    return Path(candidate).name == Path(normalized_req).name


def check_required_edit_paths(
    *,
    required_edit_paths: list[str],
    touched_paths: list[str],
) -> Optional[Dict[str, Any]]:
    if not required_edit_paths:
        return None

    unmet = [
        requirement
        for requirement in required_edit_paths
        if not any(path_matches_requirement(path, requirement) for path in touched_paths)
    ]
    if unmet:
        return {
            'status': 'error',
            'error': 'required_edit_paths_not_modified',
            'detail': (
                'Agent did not modify the required attempt-scoped paths: '
                + ', '.join(unmet)
            ),
            'touched_paths': touched_paths,
            'required_edit_paths': required_edit_paths,
        }
    return None
