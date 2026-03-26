"""
@file context_builder.py
@description Build a structured task_context.json for the agentic loss-transfer loop.
@author Leizheng
@date 2026-03-25
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from check_compatibility import check_compatibility
from extract_loss_formula import extract_loss_formula_draft
from formula_interface_analysis import analyze_formula_interface
from loss_ir_schema import LossIR
from prepare_context import prepare_context
from trajectory_logger import ensure_experiment_dir, write_json


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _load_yaml(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def _load_loss_ir(path: Path) -> Optional[LossIR]:
    if not path.exists():
        return None
    try:
        return LossIR.from_yaml(str(path))
    except Exception:
        return None


def _collapse_text(text: str, max_chars: int = 800) -> str:
    normalized = re.sub(r'\s+', ' ', text or '').strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + '...'


def _pick_paper_sections(paper: Dict[str, Any], limit: int = 6) -> List[Dict[str, str]]:
    sections = paper.get('sections', [])
    if not isinstance(sections, list):
        return []

    preferred: List[Dict[str, str]] = []
    fallback: List[Dict[str, str]] = []
    keywords = (
        'loss', 'objective', 'training', 'method',
        'architecture', 'probabil', 'uncertainty', 'implementation',
    )

    for item in sections:
        if not isinstance(item, dict):
            continue
        heading = str(item.get('heading', '')).strip()
        text = _collapse_text(str(item.get('text', '')), max_chars=900)
        if not heading or not text:
            continue
        record = {'heading': heading, 'text': text}
        heading_lower = heading.lower()
        if any(keyword in heading_lower for keyword in keywords):
            preferred.append(record)
        else:
            fallback.append(record)

    return (preferred + fallback)[:limit]


def _pick_loss_snippets(paper: Dict[str, Any], limit: int = 8) -> List[Dict[str, str]]:
    snippets = paper.get('loss_snippets', [])
    if not isinstance(snippets, list):
        return []

    result: List[Dict[str, str]] = []
    for item in snippets[:limit]:
        if not isinstance(item, dict):
            continue
        snippet = _collapse_text(str(item.get('snippet', '')), max_chars=700)
        tag = str(item.get('tag', 'loss_snippet'))
        if snippet:
            result.append({'tag': tag, 'snippet': snippet})
    return result


def _summarize_paper_context(prepared_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    paper = ((prepared_context or {}).get('paper') or {}) if isinstance(prepared_context, dict) else {}
    if not isinstance(paper, dict) or not paper:
        return {
            'available': False,
            'notes': ['No parsed paper context available'],
        }

    metadata = paper.get('metadata', {}) if isinstance(paper.get('metadata'), dict) else {}
    return {
        'available': bool(paper.get('success')),
        'title': metadata.get('title'),
        'abstract': _collapse_text(str(paper.get('abstract', '')), max_chars=1200),
        'sections': _pick_paper_sections(paper),
        'loss_snippets': _pick_loss_snippets(paper),
        'full_text_path': paper.get('full_text_path'),
        'notes': [
            'Use paper sections and loss snippets together with loss_formula.json; formula alone is not enough.',
            'Prefer direct evidence from Implementation Details / Method / Loss-related sections when deciding integration depth.',
        ],
    }


def _summarize_code_context(prepared_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    primary_files = ((prepared_context or {}).get('primary_files') or []) if isinstance(prepared_context, dict) else []
    if not isinstance(primary_files, list) or not primary_files:
        return {
            'available': False,
            'notes': ['No primary loss-related code files were found'],
        }

    focus_files: List[Dict[str, Any]] = []
    for item in primary_files[:6]:
        if not isinstance(item, dict):
            continue
        focus_files.append({
            'path': item.get('path'),
            'priority': item.get('priority'),
            'functions': item.get('functions', []),
            'imports': item.get('imports', []),
            'content_preview': _collapse_text(str(item.get('content', '')), max_chars=1200),
        })

    return {
        'available': True,
        'focus_files': focus_files,
        'notes': [
            'Agent should inspect where the paper loss is computed: standalone loss file, training loop, or model.forward.',
            'If the paper computes auxiliary tensors inside model.forward, do not force a loss-only migration.',
        ],
    }


def _build_integration_assessment(
    formula_interface: Optional[Dict[str, Any]],
    loss_spec: Optional[Dict[str, Any]],
    compatibility: Dict[str, Any],
    has_loss_ir: bool,
) -> Dict[str, Any]:
    assessment: Dict[str, Any] = {
        'primary_source_of_truth': 'paper + code evidence + loss_formula',
        'loss_ir_available': has_loss_ir,
        'loss_ir_role': 'optional_reference' if has_loss_ir else 'not_required_for_agent_analysis',
        'loss_only_pipeline_viable': None,
        'recommended_path': 'agent_decides',
        'change_level': None,
        'requires_model_changes': None,
        'notes': [],
    }

    if formula_interface:
        assessment['loss_only_pipeline_viable'] = formula_interface.get('loss_only_pipeline_compatible')
        assessment['recommended_path'] = formula_interface.get('recommended_integration_path', 'agent_decides')
        assessment['change_level'] = formula_interface.get('change_level')
        assessment['change_level_label'] = formula_interface.get('change_level_label')
        assessment['requires_model_changes'] = formula_interface.get('requires_model_changes')
        assessment['extra_required_variables'] = formula_interface.get('extra_required_variables', [])
        assessment['issues'] = formula_interface.get('issues', [])
        assessment['notes'].append(
            'Use interface_analysis to decide whether the SR model needs loss_inputs / adapter heads / output extension.'
        )

    if loss_spec:
        assessment['loss_recipe'] = ((loss_spec.get('loss') or {}) if isinstance(loss_spec.get('loss'), dict) else {}).get('recipe')
        assessment['recipe_status'] = loss_spec.get('recipe_status')

    compat_status = compatibility.get('status', 'unknown')
    assessment['compatibility_status'] = compat_status
    if compat_status != 'fully_compatible':
        assessment['notes'].append(
            'Loss IR compatibility is secondary here; when it conflicts with paper/code evidence, trust the richer evidence bundle.'
        )

    return assessment


def build_task_context(
    paper_slug: str,
    *,
    paper_pdf_path: Optional[str] = None,
    code_repo_path: Optional[str] = None,
    loss_ir_yaml: Optional[str] = None,
    dataset_root: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    experiment_dir = ensure_experiment_dir(paper_slug, output_dir=output_dir)
    task_context_path = experiment_dir / 'task_context.json'
    formula_path = experiment_dir / 'loss_formula.json'
    loss_spec_path = experiment_dir / 'loss_spec.yaml'
    loss_ir_path = Path(loss_ir_yaml).expanduser().resolve() if loss_ir_yaml else (experiment_dir / 'loss_ir.yaml')

    prepared_context: Optional[Dict[str, Any]] = None
    if code_repo_path:
        prepared_context = prepare_context(
            code_repo_path=code_repo_path,
            paper_slug=paper_slug,
            output_dir=str(experiment_dir),
            paper_pdf_path=paper_pdf_path,
        )

    formula_draft_status: Dict[str, Any] = {'status': 'skipped'}
    if code_repo_path and not formula_path.exists():
        try:
            formula_draft = extract_loss_formula_draft(
                code_repo_path=code_repo_path,
                paper_slug=paper_slug,
                paper_pdf_path=paper_pdf_path,
                output_path=str(formula_path),
            )
            formula_draft_status = {
                'status': formula_draft.get('status', 'unknown'),
                'written_path': formula_draft.get('written_path'),
                'loss_spec_written_path': formula_draft.get('loss_spec_written_path'),
                'validation': formula_draft.get('validation'),
            }
        except Exception as exc:
            formula_draft_status = {
                'status': 'error',
                'error': str(exc),
            }

    loss_ir_raw = _load_yaml(loss_ir_path)
    loss_ir_obj = _load_loss_ir(loss_ir_path)
    compatibility = (
        check_compatibility(loss_ir_obj)
        if loss_ir_obj is not None
        else {
            'status': 'partially_compatible',
            'warnings': ['Loss IR is missing or not parseable yet; Agent should rely on paper/code/formula evidence.'],
        }
    )

    formula_spec = _load_json(formula_path)
    formula_interface = analyze_formula_interface(formula_spec) if formula_spec else None
    loss_spec = _load_yaml(loss_spec_path)
    paper_analysis = _summarize_paper_context(prepared_context)
    code_analysis = _summarize_code_context(prepared_context)
    integration_assessment = _build_integration_assessment(
        formula_interface=formula_interface,
        loss_spec=loss_spec,
        compatibility=compatibility,
        has_loss_ir=loss_ir_obj is not None,
    )
    legacy_loss_ir_status = {
        'auto_extraction_enabled': False,
        'status': 'loaded' if loss_ir_obj is not None else 'skipped',
        'notes': [
            'Legacy LLM-driven extract_loss_ir is no longer part of the main agentic path.',
            'If needed, provide --loss_ir_yaml explicitly as a manual or offline artifact.',
        ],
    }

    task_context: Dict[str, Any] = {
        'status': 'context_ready',
        'paper_slug': paper_slug,
        'inputs': {
            'paper_pdf_path': paper_pdf_path,
            'code_repo_path': code_repo_path,
            'loss_ir_yaml': str(loss_ir_path) if loss_ir_path.exists() else None,
            'dataset_root': dataset_root,
        },
        'paths': {
            'experiment_dir': str(experiment_dir),
            'task_context_path': str(task_context_path),
            'loss_ir_path': str(loss_ir_path) if loss_ir_path.exists() else None,
            'loss_formula_path': str(formula_path) if formula_path.exists() else None,
            'loss_spec_path': str(loss_spec_path) if loss_spec_path.exists() else None,
            'trajectory_path': str(experiment_dir / 'trajectory.jsonl'),
            'analysis_plan_path': str(experiment_dir / 'analysis_plan.json'),
        },
        'prepared_context': prepared_context,
        'paper_analysis': paper_analysis,
        'code_analysis': code_analysis,
        'integration_assessment': integration_assessment,
        'loss_ir': loss_ir_raw,
        'legacy_loss_ir_status': legacy_loss_ir_status,
        'compatibility': compatibility,
        'formula_spec': formula_spec,
        'formula_interface': formula_interface,
        'loss_spec': loss_spec,
        'formula_draft_status': formula_draft_status,
        'analysis_plan_schema': {
            'summary': 'Short explanation of the integration decision using paper + code + formula evidence.',
            'stop_on_first_pass': False,
            'integration_decision': {
                'path': 'loss_only | adapter_wrapper | extend_model_outputs | model_surgery',
                'rationale': 'Why this path is required for the paper.',
                'evidence_refs': ['paper_analysis.loss_snippets[0]', 'code_analysis.focus_files[0]'],
            },
            'attempts': [
                {
                    'name': 'Attempt name',
                    'kind': 'agent_code | formula_variant',
                    'variant': 'faithful | stabilized',
                    'code': 'optional inline candidate_loss.py content for agent_code',
                    'code_path': 'optional path to candidate_loss.py',
                    'objective': 'implementation goal for agent-generated candidate code',
                    'files_to_edit': ['optional file paths the Agent expects to edit'],
                    'required_edit_paths': ['optional paths that must actually be modified in this attempt'],
                    'evidence_refs': ['task_context references supporting this attempt'],
                    'run_training': True,
                    'notes': 'Why this attempt should work',
                }
            ],
        },
        'agent_guidance': [
            'Do not analyze from formula only. Combine paper_analysis, code_analysis, formula_spec, and interface constraints.',
            'Use loss_formula.json as the formula source of truth for latex, params, and symbol_map, but validate it against paper/code evidence.',
            'Keep sandbox_loss(pred, target, mask=None, **kwargs) as the stable integration surface when the paper allows loss-only migration.',
            'If integration_assessment indicates adapter_wrapper or extend_model_outputs, modify the adapter/model path intentionally instead of hacking the loss.',
            'Treat loss_ir as optional supporting material, not the main brain of the system.',
            'If you output an agent_code attempt, include executable code, a concrete code_path, or a precise objective that a later Agent pass can turn into candidate_loss.py.',
            'Record every attempt as analysis_plan.json + candidate_loss.py + result.json for later RL training.',
        ],
    }

    write_json(task_context_path, task_context)
    return task_context
