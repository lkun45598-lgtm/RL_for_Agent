"""
@file agent_artifact_generator.py
@description Use the local KODE agent service to generate analysis plans and candidate loss code.
@author OpenAI Codex
@date 2026-03-25
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from agent_service_client import run_agent_chat
from validate_analysis_plan import validate_analysis_plan


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SANDBOX_ROOT = _PROJECT_ROOT / 'sandbox'
_PIPELINE_ROOT = _PROJECT_ROOT / 'scripts' / 'ocean-SR-training-masked'
_OVERRIDE_FILE_SOURCES = {
    'sandbox_model_adapter.py': _SANDBOX_ROOT / 'sandbox_model_adapter.py',
    'sandbox_trainer.py': _SANDBOX_ROOT / 'sandbox_trainer.py',
    '_run_once.py': _SANDBOX_ROOT / '_run_once.py',
}
_OVERRIDE_TREE_SOURCES = {
    'models': _PIPELINE_ROOT / 'models',
}
_OVERRIDE_FILE_ALIASES = {
    'sandbox model adapter files exposing extra loss inputs': ['sandbox_model_adapter.py'],
    'sandbox adapter/model-output layer': ['sandbox_model_adapter.py', 'sandbox_trainer.py'],
    'sandbox trainer files': ['sandbox_trainer.py'],
    'sandbox runtime entrypoint': ['_run_once.py'],
    'sandbox/sandbox_model_adapter.py': ['sandbox_model_adapter.py'],
    'sandbox/sandbox_trainer.py': ['sandbox_trainer.py'],
    'sandbox/_run_once.py': ['_run_once.py'],
}
_OVERRIDE_TREE_ALIASES = {
    'model files': ['models'],
    'copied model files': ['models'],
    'sandbox copied model files': ['models'],
    'training model files': ['models'],
    'scripts/ocean-sr-training-masked/models': ['models'],
    'scripts/ocean-SR-training-masked/models': ['models'],
    'models': ['models'],
}


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Expected JSON object at {path}')
    return data


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def _resolve_working_dir(task_context: Dict[str, Any]) -> Path:
    code_repo = ((task_context.get('inputs') or {}) if isinstance(task_context.get('inputs'), dict) else {}).get('code_repo_path')
    if isinstance(code_repo, str) and code_repo.strip():
        repo_path = Path(code_repo).expanduser()
        if repo_path.exists():
            return _PROJECT_ROOT if _PROJECT_ROOT in repo_path.parents or repo_path == _PROJECT_ROOT else repo_path.parent
    return _PROJECT_ROOT


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and str(item).strip()]


def _resolve_requested_override_files(
    task_context: Dict[str, Any],
    attempt_spec: Dict[str, Any],
) -> Dict[str, list[str]]:
    resolved_files: list[str] = []
    resolved_trees: list[str] = []
    requested = _as_string_list(attempt_spec.get('files_to_edit'))

    for item in requested:
        normalized = item.strip()
        if normalized == 'candidate_loss.py':
            continue
        alias_matches = _OVERRIDE_FILE_ALIASES.get(normalized.lower())
        if alias_matches:
            for name in alias_matches:
                if name not in resolved_files:
                    resolved_files.append(name)
            continue
        tree_alias_matches = _OVERRIDE_TREE_ALIASES.get(normalized.lower())
        if tree_alias_matches:
            for name in tree_alias_matches:
                if name not in resolved_trees:
                    resolved_trees.append(name)
            continue
        candidate_name = Path(normalized).name
        if candidate_name in _OVERRIDE_FILE_SOURCES and candidate_name not in resolved_files:
            resolved_files.append(candidate_name)
        if candidate_name in _OVERRIDE_TREE_SOURCES and candidate_name not in resolved_trees:
            resolved_trees.append(candidate_name)

    integration = task_context.get('integration_assessment', {}) if isinstance(task_context.get('integration_assessment'), dict) else {}
    recommended_path = str(integration.get('recommended_path', '')).strip().lower()
    if recommended_path in {'adapter_wrapper', 'extend_model_outputs'} and 'sandbox_model_adapter.py' not in resolved_files:
        resolved_files.append('sandbox_model_adapter.py')
    if recommended_path in {'adapter_wrapper', 'extend_model_outputs'} and 'sandbox_trainer.py' not in resolved_files:
        resolved_files.append('sandbox_trainer.py')
    if recommended_path in {'extend_model_outputs', 'model_surgery'} and 'models' not in resolved_trees:
        resolved_trees.append('models')
    if recommended_path == 'model_surgery' and 'models' not in resolved_trees:
        resolved_trees.append('models')

    return {
        'files': resolved_files,
        'trees': resolved_trees,
    }


def _prepare_attempt_edit_workspace(
    *,
    task_context: Dict[str, Any],
    attempt_spec: Dict[str, Any],
    output_code_path: Path,
) -> Dict[str, Any]:
    attempt_dir = output_code_path.parent
    attempt_dir.mkdir(parents=True, exist_ok=True)
    if not output_code_path.exists():
        output_code_path.write_text(
            '# Agent will replace this placeholder with the attempt-specific candidate loss.\n',
            encoding='utf-8',
        )
    override_dir = attempt_dir / 'sandbox_overrides'
    integration = task_context.get('integration_assessment', {}) if isinstance(task_context.get('integration_assessment'), dict) else {}
    recommended_path = str(integration.get('recommended_path', 'agent_decides')).strip().lower()
    editable_targets = [
        {
            'path': str(output_code_path),
            'kind': 'candidate_loss',
            'description': 'Primary sandbox loss entrypoint for this attempt.',
        }
    ]

    override_targets = _resolve_requested_override_files(task_context, attempt_spec)
    override_files = override_targets['files']
    override_trees = override_targets['trees']
    if override_files:
        override_dir.mkdir(parents=True, exist_ok=True)
    if override_trees:
        override_dir.mkdir(parents=True, exist_ok=True)

    for file_name in override_files:
        source_path = _OVERRIDE_FILE_SOURCES[file_name]
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
        source_dir = _OVERRIDE_TREE_SOURCES[tree_name]
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


def _format_editable_targets(editable_targets: list[Dict[str, Any]]) -> str:
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


def _normalize_required_edit_paths(attempt_spec: Dict[str, Any]) -> list[str]:
    return _as_string_list(attempt_spec.get('required_edit_paths'))


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


def _snapshot_editable_targets(editable_targets: list[Dict[str, Any]]) -> Dict[str, str]:
    snapshot: Dict[str, str] = {}
    for item in editable_targets:
        path = item.get('path')
        if isinstance(path, str) and path.strip():
            snapshot[path] = _path_digest(Path(path))
    return snapshot


def _detect_touched_paths(
    before_snapshot: Dict[str, str],
    after_snapshot: Dict[str, str],
) -> list[str]:
    touched: list[str] = []
    for path, before_digest in before_snapshot.items():
        after_digest = after_snapshot.get(path)
        if after_digest is None or after_digest != before_digest:
            touched.append(path)
    return touched


def _load_existing_touched_paths(attempt_dir: Path) -> list[str]:
    touched: list[str] = []
    for log_name in ('agent_code_generation_response.json', 'agent_code_repair_response.json'):
        log_path = attempt_dir / log_name
        if not log_path.exists():
            continue
        try:
            payload = _load_json(log_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        logged_paths = payload.get('touched_paths') if isinstance(payload, dict) else None
        if not isinstance(logged_paths, list):
            continue
        for path in logged_paths:
            if isinstance(path, str) and path.strip() and path not in touched:
                touched.append(path)
    return touched


def _path_matches_requirement(path: str, requirement: str) -> bool:
    normalized_req = requirement.strip().rstrip('/').lower()
    candidate = path.strip().rstrip('/').lower()
    if not normalized_req or not candidate:
        return False
    if normalized_req in candidate:
        return True
    return Path(candidate).name == Path(normalized_req).name


def _check_required_edit_paths(
    *,
    required_edit_paths: list[str],
    touched_paths: list[str],
) -> Optional[Dict[str, Any]]:
    if not required_edit_paths:
        return None

    unmet = [
        requirement
        for requirement in required_edit_paths
        if not any(_path_matches_requirement(path, requirement) for path in touched_paths)
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


def _build_analysis_plan_prompt(
    task_context: Dict[str, Any],
    *,
    task_context_path: Path,
    analysis_plan_path: Path,
    max_attempts: int,
) -> str:
    integration = task_context.get('integration_assessment', {}) if isinstance(task_context.get('integration_assessment'), dict) else {}
    recommended_path = integration.get('recommended_path', 'agent_decides')
    requires_model_changes = integration.get('requires_model_changes')

    return f"""你在执行 loss-transfer 的分析阶段。

请先读取这个文件：
- task_context.json: {task_context_path}

然后写出：
- analysis_plan.json: {analysis_plan_path}

要求：
1. 必须基于 paper_analysis、code_analysis、formula_spec、integration_assessment 综合分析，不能只看公式。
2. 输出文件必须是单个 JSON object，并匹配 task_context.analysis_plan_schema。
3. integration_decision 必须填写 path、rationale、evidence_refs。
4. attempts 最多 {max_attempts} 个。
5. 每个 attempt 都必须填写 evidence_refs。
6. formula_variant 只能用 faithful 或 stabilized。
7. 如果 kind=agent_code，但你还不想直接内嵌完整代码，请填写 objective，后续自动代码生成器会继续接管。
8. 如果你已经非常确定，也可以直接在 attempt 里给 code 或 code_path。
9. 优先最小改动；但如果 integration_assessment 明确要求更深改动，就要诚实写出来。

当前 task_context 给出的信号：
- recommended_path = {recommended_path}
- requires_model_changes = {requires_model_changes}

写完文件后，只回复一行简短确认：
plan_written:{analysis_plan_path}
"""


def _build_candidate_loss_prompt(
    *,
    task_context_path: Path,
    analysis_plan_path: Optional[Path],
    loss_formula_path: Optional[Path],
    editable_manifest_path: Path,
    editable_targets: list[Dict[str, Any]],
    output_code_path: Path,
    attempt_spec: Dict[str, Any],
) -> str:
    attempt_json = json.dumps(attempt_spec, indent=2, ensure_ascii=False)
    analysis_plan_line = f"- analysis_plan.json: {analysis_plan_path}" if analysis_plan_path else "- analysis_plan.json: (not provided)"
    formula_line = f"- loss_formula.json: {loss_formula_path}" if loss_formula_path else "- loss_formula.json: (read it from task_context if present)"
    editable_targets_block = _format_editable_targets(editable_targets)
    required_edit_paths = _normalize_required_edit_paths(attempt_spec)
    required_paths_line = (
        '\n必须实际修改这些路径中的至少一个：\n- ' + '\n- '.join(required_edit_paths)
        if required_edit_paths
        else ''
    )

    return f"""你在执行 loss-transfer 的代码生成阶段。

请先读取：
- task_context.json: {task_context_path}
{analysis_plan_line}
{formula_line}
- editable_files.json: {editable_manifest_path}

当前要实现的 attempt 如下：
```json
{attempt_json}
```

本轮允许你写入的路径只有这些（文件或目录树，必须严格限制在这个白名单内）：
{editable_targets_block}
{required_paths_line}

硬约束：
1. 默认必须写 candidate_loss.py；只有当 editable_files.json/attempt_spec 明确列出了 attempt-scoped sandbox override 路径时，才允许同步修改这些 override 文件或目录树。
2. 函数签名必须是 sandbox_loss(pred, target, mask=None, **kwargs)。
3. 只允许 import: torch, torch.nn.functional, math。
4. pred/target/mask 都按 BHWC 处理。
5. 必须正确处理 mask=None。
6. 返回标量 tensor。
7. 如果需要额外变量，比如 weight/log_b，优先支持：
   - kwargs["weight"] / kwargs["log_b"]
   - kwargs.get("loss_inputs", {{}}).get("weight") / .get("log_b")
8. 必须对齐 loss_formula.json 中的 symbol_map 与 params。symbol_map 映射到的变量名、以及 params 中的关键参数名，必须以“同名标识符”出现在代码中，不要把 epsilon 改写成 EPS 之类无法对齐校验的别名。
9. 如果论文需要更深的模型改动，优先在 attempt-scoped sandbox override 路径里修改复制出来的 adapter/model 文件，不要直接修改 repo-root 的 sandbox/ 或训练代码。
10. 原始仓库代码在这里是模板/经验库；凡是模型层改动，必须落在 copied models 目录或其它 attempt-scoped override 路径里。
11. 如果 editable_files.json 里的 routing_policy.recommended_path 是 extend_model_outputs 或 model_surgery，不要只改 loss；需要时直接修改白名单里的 copied models/ 或 sandbox_trainer/sandbox_model_adapter。
12. 验证器会自动为支持该能力的 copied model 打开 `output_aux_loss_inputs`；如果当前 copied model 还不支持，就在白名单内把它补上。
13. 不要输出 markdown 代码块，不要只在聊天里贴代码，必须把文件真正写到目标路径。

写完之后，只回复一行简短确认：
code_written:{output_code_path}
"""


def _build_candidate_loss_repair_prompt(
    *,
    task_context_path: Path,
    analysis_plan_path: Optional[Path],
    loss_formula_path: Optional[Path],
    editable_manifest_path: Path,
    editable_targets: list[Dict[str, Any]],
    current_code_path: Path,
    output_code_path: Path,
    attempt_spec: Dict[str, Any],
    failure_feedback: Dict[str, Any],
) -> str:
    attempt_json = json.dumps(attempt_spec, indent=2, ensure_ascii=False)
    feedback_json = json.dumps(failure_feedback, indent=2, ensure_ascii=False)
    analysis_plan_line = f"- analysis_plan.json: {analysis_plan_path}" if analysis_plan_path else "- analysis_plan.json: (not provided)"
    formula_line = f"- loss_formula.json: {loss_formula_path}" if loss_formula_path else "- loss_formula.json: (read it from task_context if present)"
    editable_targets_block = _format_editable_targets(editable_targets)
    required_edit_paths = _normalize_required_edit_paths(attempt_spec)
    required_paths_line = (
        '\n必须实际修改这些路径中的至少一个：\n- ' + '\n- '.join(required_edit_paths)
        if required_edit_paths
        else ''
    )

    return f"""你在执行 loss-transfer 的代码修复阶段。

请先读取：
- task_context.json: {task_context_path}
{analysis_plan_line}
{formula_line}
- editable_files.json: {editable_manifest_path}
- current candidate_loss.py: {current_code_path}

当前 attempt 如下：
```json
{attempt_json}
```

上一轮验证失败反馈如下：
```json
{feedback_json}
```

本轮允许你写入的路径只有这些（文件或目录树，必须严格限制在这个白名单内）：
{editable_targets_block}
{required_paths_line}

硬约束：
1. 默认只修改 candidate_loss.py；只有当 editable_files.json 明确列出了 attempt-scoped sandbox override 路径时，才允许同步修改这些 override 文件或目录树。
2. 保持函数签名 `sandbox_loss(pred, target, mask=None, **kwargs)`。
3. 只允许 import: torch, torch.nn.functional, math。
4. 必须保留 BHWC 语义、mask=None 兼容、标量 tensor 返回。
5. 必须修复上面验证反馈中指出的问题，而不是只做无关改写。
6. 必须对齐 loss_formula.json 中的 symbol_map 与 params；尤其是校验器报缺失的变量名，必须以同名标识符真正出现在代码里。
7. 额外 loss 输入优先继续支持：
   - kwargs["weight"] / kwargs["log_b"]
   - kwargs.get("loss_inputs", {{}}).get("weight") / .get("log_b")
8. 如果 failure_feedback.stop_layer=layer4，或者 failure_feedback 里有 performance_target，你的目标不是“仅仅不报错”，而是提升主要验证指标（优先 swinir/val_ssim）超过给出的 viable_threshold；不要只做语法或数值稳定层面的表面修补。
9. 如果 failure_feedback 暗示问题来自 adapter/model-output 路径，可以在白名单内同步修复 attempt-scoped sandbox override 文件，但禁止直接改 repo-root sandbox/ 或训练代码。
10. 原始仓库代码在这里是模板/经验库；凡是模型层改动，必须落在 copied models 目录或其它 attempt-scoped override 路径里。
11. 如果 editable_files.json 里的 routing_policy.recommended_path 是 extend_model_outputs 或 model_surgery，不要只修 loss；优先检查 copied models/ 是否真的产出了所需的 loss_inputs。
12. 如果 failure_feedback.runtime_routing 给出了各模型的 output-extension 支持状态，要利用这些信息决定该修 copied models 还是保留 adapter fallback。
13. 不要输出 markdown 代码块，不要只在聊天里贴代码，必须把文件真正写回目标路径。

写完之后，只回复一行简短确认：
code_written:{output_code_path}
"""


def generate_analysis_plan(
    task_context_path: str,
    *,
    max_attempts: int = 4,
    service_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_sec: int = 900,
) -> Dict[str, Any]:
    task_context_file = Path(task_context_path).expanduser().resolve()
    task_context = _load_json(task_context_file)
    paths = task_context.get('paths', {}) if isinstance(task_context.get('paths'), dict) else {}
    analysis_plan_path = Path(str(paths.get('analysis_plan_path') or task_context_file.parent / 'analysis_plan.json')).expanduser().resolve()
    experiment_dir = Path(str(paths.get('experiment_dir') or task_context_file.parent)).expanduser().resolve()
    working_dir = _resolve_working_dir(task_context)
    notebook_path = experiment_dir / 'analysis_plan_agent.ipynb'

    response = run_agent_chat(
        message=_build_analysis_plan_prompt(
            task_context,
            task_context_path=task_context_file,
            analysis_plan_path=analysis_plan_path,
            max_attempts=max_attempts,
        ),
        mode='edit',
        working_dir=str(working_dir),
        outputs_path=str(experiment_dir),
        notebook_path=str(notebook_path),
        files=[str(task_context_file)],
        service_url=service_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
    )

    log_path = experiment_dir / 'analysis_plan_agent_response.json'
    _write_json(log_path, response if isinstance(response, dict) else {'status': 'error'})

    if response.get('status') == 'error':
        return {
            'status': 'error',
            'error': response.get('error'),
            'analysis_plan_path': str(analysis_plan_path),
            'agent_response_path': str(log_path),
        }

    if not analysis_plan_path.exists():
        return {
            'status': 'error',
            'error': f'Agent finished but did not write analysis_plan.json to {analysis_plan_path}',
            'analysis_plan_path': str(analysis_plan_path),
            'agent_response_path': str(log_path),
            'agent_text': response.get('text', ''),
        }

    plan = _load_json(analysis_plan_path)
    validation = validate_analysis_plan(plan)
    result = {
        'status': 'success' if validation['status'] != 'error' else 'error',
        'analysis_plan_path': str(analysis_plan_path),
        'agent_response_path': str(log_path),
        'agent_id': response.get('agent_id'),
        'validation': validation,
        'agent_text': response.get('text', ''),
    }
    _write_json(log_path, {**response, 'validation': validation})
    return result


def generate_candidate_loss(
    *,
    task_context_path: str,
    attempt_spec: Dict[str, Any],
    output_code_path: str,
    analysis_plan_path: Optional[str] = None,
    service_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_sec: int = 900,
) -> Dict[str, Any]:
    task_context_file = Path(task_context_path).expanduser().resolve()
    output_path = Path(output_code_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_context = _load_json(task_context_file)
    paths = task_context.get('paths', {}) if isinstance(task_context.get('paths'), dict) else {}
    experiment_dir = Path(str(paths.get('experiment_dir') or task_context_file.parent)).expanduser().resolve()
    working_dir = _resolve_working_dir(task_context)
    notebook_path = output_path.parent / 'candidate_loss_agent.ipynb'
    edit_workspace = _prepare_attempt_edit_workspace(
        task_context=task_context,
        attempt_spec=attempt_spec,
        output_code_path=output_path,
    )
    before_snapshot = _snapshot_editable_targets(edit_workspace['editable_targets'])
    resolved_loss_formula = (
        Path(str(paths.get('loss_formula_path'))).expanduser().resolve()
        if paths.get('loss_formula_path')
        else None
    )
    resolved_analysis_plan = (
        Path(analysis_plan_path).expanduser().resolve()
        if analysis_plan_path
        else None
    )

    response = run_agent_chat(
        message=_build_candidate_loss_prompt(
            task_context_path=task_context_file,
            analysis_plan_path=resolved_analysis_plan,
            loss_formula_path=resolved_loss_formula,
            editable_manifest_path=Path(edit_workspace['manifest_path']),
            editable_targets=edit_workspace['editable_targets'],
            output_code_path=output_path,
            attempt_spec=attempt_spec,
        ),
        mode='edit',
        working_dir=str(working_dir),
        outputs_path=str(experiment_dir),
        notebook_path=str(notebook_path),
        files=[
            str(task_context_file),
            str(edit_workspace['manifest_path']),
            *([str(resolved_loss_formula)] if resolved_loss_formula else []),
            *([str(resolved_analysis_plan)] if resolved_analysis_plan else []),
            *[
                str(item['path'])
                for item in edit_workspace['editable_targets']
                if isinstance(item, dict)
                and isinstance(item.get('path'), str)
                and Path(str(item['path'])).is_file()
            ],
        ],
        service_url=service_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
    )

    log_path = output_path.parent / 'agent_code_generation_response.json'
    _write_json(log_path, response if isinstance(response, dict) else {'status': 'error'})

    if not output_path.exists():
        return {
            'status': 'error',
            'error': response.get('error') or f'Agent finished but did not write candidate code to {output_path}',
            'code_path': str(output_path),
            'agent_response_path': str(log_path),
            'agent_text': response.get('text', ''),
        }

    code = output_path.read_text(encoding='utf-8')
    after_snapshot = _snapshot_editable_targets(edit_workspace['editable_targets'])
    touched_paths = _detect_touched_paths(before_snapshot, after_snapshot)
    required_edit_error = _check_required_edit_paths(
        required_edit_paths=_normalize_required_edit_paths(attempt_spec),
        touched_paths=touched_paths,
    )
    if required_edit_error is not None:
        _write_json(log_path, {**response, 'code_chars': len(code), **required_edit_error})
        return {
            'status': 'error',
            'error': required_edit_error['detail'],
            'code_path': str(output_path),
            'agent_response_path': str(log_path),
            'editable_manifest_path': str(edit_workspace['manifest_path']),
            'sandbox_override_dir': (
                str(edit_workspace['sandbox_override_dir'])
                if edit_workspace.get('sandbox_override_dir') is not None
                else None
            ),
            'touched_paths': touched_paths,
            'required_edit_paths': _normalize_required_edit_paths(attempt_spec),
        }
    result = {
        'status': 'success',
        'code_path': str(output_path),
        'agent_response_path': str(log_path),
        'agent_id': response.get('agent_id'),
        'agent_text': response.get('text', ''),
        'agent_status': response.get('status'),
        'agent_error': response.get('error'),
        'code_chars': len(code),
        'editable_manifest_path': str(edit_workspace['manifest_path']),
        'sandbox_override_dir': (
            str(edit_workspace['sandbox_override_dir'])
            if edit_workspace.get('sandbox_override_dir') is not None
            else None
        ),
        'touched_paths': touched_paths,
    }
    _write_json(log_path, {**response, 'code_chars': len(code), 'touched_paths': touched_paths})
    return result


def repair_candidate_loss(
    *,
    task_context_path: str,
    attempt_spec: Dict[str, Any],
    output_code_path: str,
    failure_feedback: Dict[str, Any],
    analysis_plan_path: Optional[str] = None,
    service_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_sec: int = 900,
) -> Dict[str, Any]:
    task_context_file = Path(task_context_path).expanduser().resolve()
    output_path = Path(output_code_path).expanduser().resolve()
    if not output_path.exists():
        raise FileNotFoundError(f'Candidate code not found for repair: {output_path}')

    task_context = _load_json(task_context_file)
    paths = task_context.get('paths', {}) if isinstance(task_context.get('paths'), dict) else {}
    experiment_dir = Path(str(paths.get('experiment_dir') or task_context_file.parent)).expanduser().resolve()
    working_dir = _resolve_working_dir(task_context)
    notebook_path = output_path.parent / 'candidate_loss_repair_agent.ipynb'
    edit_workspace = _prepare_attempt_edit_workspace(
        task_context=task_context,
        attempt_spec=attempt_spec,
        output_code_path=output_path,
    )
    before_snapshot = _snapshot_editable_targets(edit_workspace['editable_targets'])
    resolved_loss_formula = (
        Path(str(paths.get('loss_formula_path'))).expanduser().resolve()
        if paths.get('loss_formula_path')
        else None
    )
    resolved_analysis_plan = (
        Path(analysis_plan_path).expanduser().resolve()
        if analysis_plan_path
        else None
    )

    response = run_agent_chat(
        message=_build_candidate_loss_repair_prompt(
            task_context_path=task_context_file,
            analysis_plan_path=resolved_analysis_plan,
            loss_formula_path=resolved_loss_formula,
            editable_manifest_path=Path(edit_workspace['manifest_path']),
            editable_targets=edit_workspace['editable_targets'],
            current_code_path=output_path,
            output_code_path=output_path,
            attempt_spec=attempt_spec,
            failure_feedback=failure_feedback,
        ),
        mode='edit',
        working_dir=str(working_dir),
        outputs_path=str(experiment_dir),
        notebook_path=str(notebook_path),
        files=[
            str(task_context_file),
            str(edit_workspace['manifest_path']),
            str(output_path),
            *([str(resolved_loss_formula)] if resolved_loss_formula else []),
            *([str(resolved_analysis_plan)] if resolved_analysis_plan else []),
            *[
                str(item['path'])
                for item in edit_workspace['editable_targets']
                if isinstance(item, dict)
                and isinstance(item.get('path'), str)
                and Path(str(item['path'])).is_file()
            ],
        ],
        service_url=service_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
    )

    log_path = output_path.parent / 'agent_code_repair_response.json'
    _write_json(log_path, response if isinstance(response, dict) else {'status': 'error'})

    code = output_path.read_text(encoding='utf-8')
    after_snapshot = _snapshot_editable_targets(edit_workspace['editable_targets'])
    touched_paths = _detect_touched_paths(before_snapshot, after_snapshot)
    historical_touched_paths = _load_existing_touched_paths(output_path.parent)
    effective_touched_paths = list(dict.fromkeys([*historical_touched_paths, *touched_paths]))
    required_edit_error = _check_required_edit_paths(
        required_edit_paths=_normalize_required_edit_paths(attempt_spec),
        touched_paths=effective_touched_paths,
    )
    if required_edit_error is not None:
        _write_json(
            log_path,
            {
                **response,
                'code_chars': len(code),
                'failure_feedback': failure_feedback,
                'touched_paths': touched_paths,
                'historical_touched_paths': historical_touched_paths,
                **required_edit_error,
            },
        )
        return {
            'status': 'error',
            'error': required_edit_error['detail'],
            'code_path': str(output_path),
            'agent_response_path': str(log_path),
            'editable_manifest_path': str(edit_workspace['manifest_path']),
            'sandbox_override_dir': (
                str(edit_workspace['sandbox_override_dir'])
                if edit_workspace.get('sandbox_override_dir') is not None
                else None
            ),
            'touched_paths': touched_paths,
            'historical_touched_paths': historical_touched_paths,
            'required_edit_paths': _normalize_required_edit_paths(attempt_spec),
        }
    result = {
        'status': 'success',
        'code_path': str(output_path),
        'agent_response_path': str(log_path),
        'agent_id': response.get('agent_id'),
        'agent_text': response.get('text', ''),
        'agent_status': response.get('status'),
        'agent_error': response.get('error'),
        'code_chars': len(code),
        'editable_manifest_path': str(edit_workspace['manifest_path']),
        'sandbox_override_dir': (
            str(edit_workspace['sandbox_override_dir'])
            if edit_workspace.get('sandbox_override_dir') is not None
            else None
        ),
        'touched_paths': touched_paths,
        'historical_touched_paths': historical_touched_paths,
    }
    _write_json(
        log_path,
        {
            **response,
            'code_chars': len(code),
            'failure_feedback': failure_feedback,
            'touched_paths': touched_paths,
            'historical_touched_paths': historical_touched_paths,
        },
    )
    return result
