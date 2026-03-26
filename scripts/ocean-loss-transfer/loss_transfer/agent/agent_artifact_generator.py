"""
@file agent_artifact_generator.py
@description Use the local KODE agent service to generate analysis plans and candidate loss code.
@author OpenAI Codex
@date 2026-03-25
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from loss_transfer.agent.agent_edit_workspace import (
    check_required_edit_paths as _check_required_edit_paths,
    detect_touched_paths as _detect_touched_paths,
    format_editable_targets as _format_editable_targets,
    load_existing_touched_paths as _load_existing_touched_paths,
    normalize_required_edit_paths as _normalize_required_edit_paths,
    prepare_attempt_edit_workspace as _prepare_attempt_edit_workspace,
    snapshot_editable_targets as _snapshot_editable_targets,
)
from loss_transfer.agent.agent_service_client import run_agent_chat
from loss_transfer.agent.validate_analysis_plan import validate_analysis_plan
from loss_transfer.common.paths import PROJECT_ROOT


_PROJECT_ROOT = PROJECT_ROOT


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
13. 如果某个分支在没有有效 mask / 没有有效样本时需要返回零损失，必须返回与 pred 保持计算图连接的零值，例如 `pred.sum() * 0.0`，不要返回 detached 常量 tensor。
14. 不要输出 markdown 代码块，不要只在聊天里贴代码，必须把文件真正写到目标路径。

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
13. 如果修复分支里存在“没有有效 mask / 没有有效样本”的情况，零损失也必须与 pred 保持计算图连接，例如 `pred.sum() * 0.0`；否则 layer3 backward 会报 “does not require grad”。
14. 不要输出 markdown 代码块，不要只在聊天里贴代码，必须把文件真正写回目标路径。

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


def _resolve_candidate_support_paths(
    *,
    paths: Dict[str, Any],
    analysis_plan_path: Optional[str],
) -> Dict[str, Optional[Path]]:
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
    return {
        'loss_formula_path': resolved_loss_formula,
        'analysis_plan_path': resolved_analysis_plan,
    }


def _build_agent_edit_input_files(
    *,
    task_context_file: Path,
    editable_manifest_path: Path,
    editable_targets: list[Dict[str, Any]],
    resolved_loss_formula: Optional[Path],
    resolved_analysis_plan: Optional[Path],
    current_code_path: Optional[Path] = None,
) -> list[str]:
    files = [
        str(task_context_file),
        str(editable_manifest_path),
    ]
    if current_code_path is not None:
        files.append(str(current_code_path))
    if resolved_loss_formula is not None:
        files.append(str(resolved_loss_formula))
    if resolved_analysis_plan is not None:
        files.append(str(resolved_analysis_plan))
    files.extend(
        str(item['path'])
        for item in editable_targets
        if isinstance(item, dict)
        and isinstance(item.get('path'), str)
        and Path(str(item['path'])).is_file()
    )
    return files


def _prepare_candidate_edit_context(
    *,
    task_context_path: str,
    attempt_spec: Dict[str, Any],
    output_code_path: str,
    analysis_plan_path: Optional[str],
    notebook_name: str,
) -> Dict[str, Any]:
    task_context_file = Path(task_context_path).expanduser().resolve()
    output_path = Path(output_code_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_context = _load_json(task_context_file)
    paths = task_context.get('paths', {}) if isinstance(task_context.get('paths'), dict) else {}
    experiment_dir = Path(str(paths.get('experiment_dir') or task_context_file.parent)).expanduser().resolve()
    working_dir = _resolve_working_dir(task_context)
    notebook_path = output_path.parent / notebook_name
    edit_workspace = _prepare_attempt_edit_workspace(
        task_context=task_context,
        attempt_spec=attempt_spec,
        output_code_path=output_path,
    )
    before_snapshot = _snapshot_editable_targets(edit_workspace['editable_targets'])
    support_paths = _resolve_candidate_support_paths(
        paths=paths,
        analysis_plan_path=analysis_plan_path,
    )
    return {
        'task_context_file': task_context_file,
        'task_context': task_context,
        'paths': paths,
        'output_path': output_path,
        'experiment_dir': experiment_dir,
        'working_dir': working_dir,
        'notebook_path': notebook_path,
        'edit_workspace': edit_workspace,
        'before_snapshot': before_snapshot,
        'resolved_loss_formula': support_paths['loss_formula_path'],
        'resolved_analysis_plan': support_paths['analysis_plan_path'],
    }


def _build_edit_workspace_result_fields(edit_workspace: Dict[str, Any]) -> Dict[str, Optional[str]]:
    return {
        'editable_manifest_path': str(edit_workspace['manifest_path']),
        'sandbox_override_dir': (
            str(edit_workspace['sandbox_override_dir'])
            if edit_workspace.get('sandbox_override_dir') is not None
            else None
        ),
    }


def _finalize_candidate_edit_result(
    *,
    output_path: Path,
    response: Dict[str, Any],
    log_path: Path,
    edit_workspace: Dict[str, Any],
    before_snapshot: Dict[str, str],
    attempt_spec: Dict[str, Any],
    missing_output_error: str,
    failure_feedback: Optional[Dict[str, Any]] = None,
    include_history: bool = False,
) -> Dict[str, Any]:
    if not output_path.exists():
        return {
            'status': 'error',
            'error': response.get('error') or missing_output_error,
            'code_path': str(output_path),
            'agent_response_path': str(log_path),
            'agent_text': response.get('text', ''),
        }

    code = output_path.read_text(encoding='utf-8')
    after_snapshot = _snapshot_editable_targets(edit_workspace['editable_targets'])
    touched_paths = _detect_touched_paths(before_snapshot, after_snapshot)
    historical_touched_paths = _load_existing_touched_paths(output_path.parent) if include_history else []
    effective_touched_paths = (
        list(dict.fromkeys([*historical_touched_paths, *touched_paths]))
        if include_history
        else touched_paths
    )
    required_edit_paths = _normalize_required_edit_paths(attempt_spec)
    required_edit_error = _check_required_edit_paths(
        required_edit_paths=required_edit_paths,
        touched_paths=effective_touched_paths,
    )
    log_payload: Dict[str, Any] = {
        **response,
        'code_chars': len(code),
        'touched_paths': touched_paths,
    }
    if include_history:
        log_payload['historical_touched_paths'] = historical_touched_paths
    if failure_feedback is not None:
        log_payload['failure_feedback'] = failure_feedback

    workspace_fields = _build_edit_workspace_result_fields(edit_workspace)
    if required_edit_error is not None:
        _write_json(log_path, {**log_payload, **required_edit_error})
        result: Dict[str, Any] = {
            'status': 'error',
            'error': required_edit_error['detail'],
            'code_path': str(output_path),
            'agent_response_path': str(log_path),
            **workspace_fields,
            'touched_paths': touched_paths,
            'required_edit_paths': required_edit_paths,
        }
        if include_history:
            result['historical_touched_paths'] = historical_touched_paths
        return result

    result = {
        'status': 'success',
        'code_path': str(output_path),
        'agent_response_path': str(log_path),
        'agent_id': response.get('agent_id'),
        'agent_text': response.get('text', ''),
        'agent_status': response.get('status'),
        'agent_error': response.get('error'),
        'code_chars': len(code),
        **workspace_fields,
        'touched_paths': touched_paths,
    }
    if include_history:
        result['historical_touched_paths'] = historical_touched_paths
    _write_json(log_path, log_payload)
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
    context = _prepare_candidate_edit_context(
        task_context_path=task_context_path,
        attempt_spec=attempt_spec,
        output_code_path=output_code_path,
        analysis_plan_path=analysis_plan_path,
        notebook_name='candidate_loss_agent.ipynb',
    )

    response = run_agent_chat(
        message=_build_candidate_loss_prompt(
            task_context_path=context['task_context_file'],
            analysis_plan_path=context['resolved_analysis_plan'],
            loss_formula_path=context['resolved_loss_formula'],
            editable_manifest_path=Path(context['edit_workspace']['manifest_path']),
            editable_targets=context['edit_workspace']['editable_targets'],
            output_code_path=context['output_path'],
            attempt_spec=attempt_spec,
        ),
        mode='edit',
        working_dir=str(context['working_dir']),
        outputs_path=str(context['experiment_dir']),
        notebook_path=str(context['notebook_path']),
        files=_build_agent_edit_input_files(
            task_context_file=context['task_context_file'],
            editable_manifest_path=Path(context['edit_workspace']['manifest_path']),
            editable_targets=context['edit_workspace']['editable_targets'],
            resolved_loss_formula=context['resolved_loss_formula'],
            resolved_analysis_plan=context['resolved_analysis_plan'],
        ),
        service_url=service_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
    )

    log_path = context['output_path'].parent / 'agent_code_generation_response.json'
    _write_json(log_path, response if isinstance(response, dict) else {'status': 'error'})
    return _finalize_candidate_edit_result(
        output_path=context['output_path'],
        response=response,
        log_path=log_path,
        edit_workspace=context['edit_workspace'],
        before_snapshot=context['before_snapshot'],
        attempt_spec=attempt_spec,
        missing_output_error=f'Agent finished but did not write candidate code to {context["output_path"]}',
    )


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
    output_path = Path(output_code_path).expanduser().resolve()
    if not output_path.exists():
        raise FileNotFoundError(f'Candidate code not found for repair: {output_path}')
    context = _prepare_candidate_edit_context(
        task_context_path=task_context_path,
        attempt_spec=attempt_spec,
        output_code_path=output_code_path,
        analysis_plan_path=analysis_plan_path,
        notebook_name='candidate_loss_repair_agent.ipynb',
    )

    response = run_agent_chat(
        message=_build_candidate_loss_repair_prompt(
            task_context_path=context['task_context_file'],
            analysis_plan_path=context['resolved_analysis_plan'],
            loss_formula_path=context['resolved_loss_formula'],
            editable_manifest_path=Path(context['edit_workspace']['manifest_path']),
            editable_targets=context['edit_workspace']['editable_targets'],
            current_code_path=context['output_path'],
            output_code_path=context['output_path'],
            attempt_spec=attempt_spec,
            failure_feedback=failure_feedback,
        ),
        mode='edit',
        working_dir=str(context['working_dir']),
        outputs_path=str(context['experiment_dir']),
        notebook_path=str(context['notebook_path']),
        files=_build_agent_edit_input_files(
            task_context_file=context['task_context_file'],
            editable_manifest_path=Path(context['edit_workspace']['manifest_path']),
            editable_targets=context['edit_workspace']['editable_targets'],
            resolved_loss_formula=context['resolved_loss_formula'],
            resolved_analysis_plan=context['resolved_analysis_plan'],
            current_code_path=context['output_path'],
        ),
        service_url=service_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
    )

    log_path = context['output_path'].parent / 'agent_code_repair_response.json'
    _write_json(log_path, response if isinstance(response, dict) else {'status': 'error'})
    return _finalize_candidate_edit_result(
        output_path=context['output_path'],
        response=response,
        log_path=log_path,
        edit_workspace=context['edit_workspace'],
        before_snapshot=context['before_snapshot'],
        attempt_spec=attempt_spec,
        missing_output_error=f'Agent finished but did not write repaired code to {context["output_path"]}',
        failure_feedback=failure_feedback,
        include_history=True,
    )
