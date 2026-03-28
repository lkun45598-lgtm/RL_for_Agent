"""
@file agent_artifact_generator.py
@description Use the local KODE agent service to generate analysis plans and candidate loss code.
@author OpenAI Codex
@contributors kongzhiquan
@date 2026-03-25
@version 1.2.0

@changelog
  - 2026-03-25 OpenAI Codex: v1.0.0 initial version
  - 2026-03-28 kongzhiquan: v1.1.0 extract shared case-memory retrieval helpers
  - 2026-03-28 kongzhiquan: v1.2.0 merge evidence-probe orchestration with shared case-memory retrieval
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
from loss_transfer.agent.evidence_probe import (
    execute_evidence_probe,
    load_json_object as _load_probe_json_object,
    validate_evidence_probe_request,
)
from loss_transfer.agent.agent_service_client import run_agent_chat
from loss_transfer.agent.validate_analysis_plan import validate_analysis_plan, validate_attempt_spec
from loss_transfer.common.paths import PROJECT_ROOT
from loss_transfer.common.run_manifest import append_run_manifest_agent_call, write_run_manifest
from loss_transfer.common.routing_audit import write_routing_audit
from loss_transfer.memory.case_memory_retriever import (
    append_memory_block as _append_memory_block,
    format_case_memory_block as _format_case_memory_block,
    load_similar_case_memories as _load_similar_case_memories,
)
from loss_transfer.memory.case_memory_store import DEFAULT_CASE_MEMORY_PATH


_PROJECT_ROOT = PROJECT_ROOT
_CASE_MEMORY_PATH = DEFAULT_CASE_MEMORY_PATH


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Expected JSON object at {path}')
    return data


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def _append_agent_call_manifest(
    *,
    task_context: Dict[str, Any],
    stage: str,
    response: Dict[str, Any],
    agent_response_path: Path,
    mode: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    paths = task_context.get('paths', {}) if isinstance(task_context.get('paths'), dict) else {}
    run_manifest_path = paths.get('run_manifest_path')
    if not isinstance(run_manifest_path, str) or not run_manifest_path.strip():
        experiment_dir = paths.get('experiment_dir')
        if isinstance(experiment_dir, str) and experiment_dir.strip():
            run_manifest_path = str(Path(experiment_dir).expanduser().resolve() / 'run_manifest.json')
        else:
            return None

    call_record: Dict[str, Any] = {
        'stage': stage,
        'mode': mode,
        'status': response.get('status'),
        'error': response.get('error'),
        'service_url': response.get('service_url'),
        'service_url_source': response.get('service_url_source'),
        'session_scope': response.get('session_scope') or 'new_request_session',
        'requested_agent_id': response.get('requested_agent_id'),
        'resolved_agent_id': response.get('agent_id'),
        'agent_response_path': str(agent_response_path),
    }
    if extra:
        call_record.update(extra)
    return append_run_manifest_agent_call(run_manifest_path, call_record)


def _safe_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _safe_load_json_object(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        return _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _build_repair_plan_placeholder(failure_feedback: Dict[str, Any]) -> Dict[str, Any]:
    performance_target = (
        failure_feedback.get('performance_target', {})
        if isinstance(failure_feedback.get('performance_target'), dict)
        else {}
    )
    target_metric = performance_target.get('primary_metric_name')
    if not isinstance(target_metric, str) or not target_metric.strip():
        target_metric = 'val_ssim'

    return {
        'failure_hypothesis': '',
        'planned_changes': [],
        'target_metric': target_metric,
        'success_criteria': '',
        'fallback_plan': '',
        'evidence_refs': [],
    }


def _validate_repair_plan(plan: Dict[str, Any]) -> Optional[str]:
    string_fields = (
        'failure_hypothesis',
        'target_metric',
        'success_criteria',
        'fallback_plan',
    )
    for field in string_fields:
        value = plan.get(field)
        if not isinstance(value, str) or not value.strip():
            return f'repair plan field `{field}` must be a non-empty string'

    planned_changes = plan.get('planned_changes')
    if not isinstance(planned_changes, list) or not any(
        isinstance(item, str) and item.strip() for item in planned_changes
    ):
        return 'repair plan field `planned_changes` must contain at least one non-empty string item'

    evidence_refs = plan.get('evidence_refs')
    if not isinstance(evidence_refs, list) or not any(
        isinstance(item, str) and item.strip() for item in evidence_refs
    ):
        return 'repair plan field `evidence_refs` must contain at least one non-empty string item'

    return None


def _evidence_refs_contain_prefix(evidence_refs: Any, prefix: str) -> bool:
    if not isinstance(evidence_refs, list):
        return False
    normalized_prefix = prefix.strip()
    if not normalized_prefix:
        return False
    return any(
        isinstance(item, str) and item.strip().startswith(normalized_prefix)
        for item in evidence_refs
    )


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
    evidence_probe_result_path: Optional[Path] = None,
) -> str:
    integration = task_context.get('integration_assessment', {}) if isinstance(task_context.get('integration_assessment'), dict) else {}
    recommended_path = integration.get('recommended_path')
    requires_model_changes = integration.get('requires_model_changes')
    paths = task_context.get('paths', {}) if isinstance(task_context.get('paths'), dict) else {}
    evidence_graph_path = paths.get('evidence_graph_path')
    evidence_graph_line = (
        f"- evidence_graph.json: {evidence_graph_path}"
        if isinstance(evidence_graph_path, str) and evidence_graph_path.strip()
        else "- evidence_graph.json: (not provided)"
    )
    probe_result_line = (
        f"- analysis_evidence_probe_result.json: {evidence_probe_result_path}"
        if evidence_probe_result_path is not None
        else "- analysis_evidence_probe_result.json: (no extra evidence probe was needed)"
    )

    return f"""你在执行 loss-transfer 的分析阶段。

请先读取这个文件：
- task_context.json: {task_context_path}
{evidence_graph_line}
{probe_result_line}

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
10. 如果 analysis_evidence_probe_result.json 存在，必须吸收里面的发现；不要忽略你自己补采集出来的证据。

当前 task_context 给出的信号：
- recommended_path = {recommended_path}
- requires_model_changes = {requires_model_changes}

写完文件后，只回复一行简短确认：
plan_written:{analysis_plan_path}
"""


def _build_analysis_evidence_probe_prompt(
    *,
    task_context_path: Path,
    probe_request_path: Path,
    probe_script_path: Path,
    probe_result_path: Path,
) -> str:
    return f"""你在执行 loss-transfer 的分析补证据阶段。

请先读取：
- task_context.json: {task_context_path}

然后判断：仅凭当前 task_context / paper / code evidence 是否已经足够生成 analysis_plan。

你必须先写出：
- analysis_evidence_probe_request.json: {probe_request_path}

如果你认为证据已经足够：
1. 把 request.json 写成单个 JSON object，字段至少包含：
   - status: "not_needed"
   - reason: 为什么现有证据已经足够
   - evidence_refs: 你依赖的 task_context/paper/code 引用
2. 不要写 probe 脚本。

如果你认为证据还不够：
1. 把 request.json 写成单个 JSON object，字段至少包含：
   - status: "probe_needed"
   - reason: 为什么现有证据还不够
   - evidence_refs: 已有但仍不足的依据
   - probe_goal: 你还需要验证什么
   - expected_output_keys: 你希望 probe 产出的关键 JSON 字段名
2. 同时写一个只读 Python 脚本：
   - analysis_evidence_probe.py: {probe_script_path}
3. 脚本运行方式固定为：
   `python analysis_evidence_probe.py --code_repo <repo> --task_context <task_context> --output <json>`
4. 脚本必须只做“读取仓库 + 解析信息 + 写 JSON 到输出路径 {probe_result_path}”。
5. 禁止修改 code_repo 内任何文件，禁止训练、禁止联网、禁止长时间阻塞。
6. 优先使用标准库做静态分析，例如 `ast/json/re/pathlib`；这是分析工具，不是实现补丁。

写完之后，只回复一行简短确认：
probe_written:{probe_request_path}
"""


def _run_analysis_evidence_probe(
    *,
    task_context: Dict[str, Any],
    task_context_file: Path,
    experiment_dir: Path,
    working_dir: Path,
    service_url: Optional[str],
    api_key: Optional[str],
    timeout_sec: int,
) -> Dict[str, Any]:
    paths = task_context.get('paths', {}) if isinstance(task_context.get('paths'), dict) else {}
    probe_request_path = Path(
        str(paths.get('analysis_evidence_probe_request_path') or experiment_dir / 'analysis_evidence_probe_request.json')
    ).expanduser().resolve()
    probe_script_path = Path(
        str(paths.get('analysis_evidence_probe_script_path') or experiment_dir / 'analysis_evidence_probe.py')
    ).expanduser().resolve()
    probe_result_path = Path(
        str(paths.get('analysis_evidence_probe_result_path') or experiment_dir / 'analysis_evidence_probe_result.json')
    ).expanduser().resolve()
    notebook_path = experiment_dir / 'analysis_evidence_probe_agent.ipynb'
    files = [str(task_context_file)]
    evidence_graph_path = paths.get('evidence_graph_path')
    if isinstance(evidence_graph_path, str) and evidence_graph_path.strip():
        evidence_graph_file = Path(evidence_graph_path).expanduser().resolve()
        if evidence_graph_file.exists():
            files.append(str(evidence_graph_file))

    response = run_agent_chat(
        message=_build_analysis_evidence_probe_prompt(
            task_context_path=task_context_file,
            probe_request_path=probe_request_path,
            probe_script_path=probe_script_path,
            probe_result_path=probe_result_path,
        ),
        mode='edit',
        working_dir=str(working_dir),
        outputs_path=str(experiment_dir),
        notebook_path=str(notebook_path),
        files=files,
        service_url=service_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
    )

    log_path = experiment_dir / 'analysis_evidence_probe_agent_response.json'
    _write_json(log_path, response if isinstance(response, dict) else {'status': 'error'})
    _append_agent_call_manifest(
        task_context=task_context,
        stage='analysis_evidence_probe_generation',
        response=response,
        agent_response_path=log_path,
        mode='edit',
        extra={
            'analysis_evidence_probe_request_path': str(probe_request_path),
            'analysis_evidence_probe_result_path': str(probe_result_path),
        },
    )

    if response.get('status') == 'error':
        return {
            'status': 'error',
            'error': response.get('error'),
            'analysis_evidence_probe_request_path': str(probe_request_path),
            'analysis_evidence_probe_result_path': str(probe_result_path),
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
        }

    if not probe_request_path.exists():
        return {
            'status': 'error',
            'error': f'Agent finished but did not write analysis_evidence_probe_request.json to {probe_request_path}',
            'analysis_evidence_probe_request_path': str(probe_request_path),
            'analysis_evidence_probe_result_path': str(probe_result_path),
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
        }

    request_payload = _load_probe_json_object(probe_request_path)
    validation = validate_evidence_probe_request(request_payload)
    normalized_request = validation.get('normalized_request')
    if validation.get('status') == 'error' or not isinstance(normalized_request, dict):
        return {
            'status': 'error',
            'error': 'analysis_evidence_probe_request.json validation failed',
            'validation': validation,
            'analysis_evidence_probe_request_path': str(probe_request_path),
            'analysis_evidence_probe_result_path': str(probe_result_path),
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
        }

    if normalized_request.get('status') == 'not_needed':
        return {
            'status': 'not_needed',
            'validation': validation,
            'request': normalized_request,
            'analysis_evidence_probe_request_path': str(probe_request_path),
            'analysis_evidence_probe_result_path': None,
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
        }

    if not probe_script_path.exists():
        return {
            'status': 'error',
            'error': f'Probe was requested but analysis_evidence_probe.py was not written to {probe_script_path}',
            'validation': validation,
            'analysis_evidence_probe_request_path': str(probe_request_path),
            'analysis_evidence_probe_result_path': str(probe_result_path),
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
        }

    code_repo_path = ((task_context.get('inputs') or {}) if isinstance(task_context.get('inputs'), dict) else {}).get('code_repo_path')
    if not isinstance(code_repo_path, str) or not code_repo_path.strip():
        return {
            'status': 'error',
            'error': 'Probe was requested but task_context.inputs.code_repo_path is missing',
            'validation': validation,
            'analysis_evidence_probe_request_path': str(probe_request_path),
            'analysis_evidence_probe_result_path': str(probe_result_path),
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
        }

    execution = execute_evidence_probe(
        script_path=probe_script_path,
        code_repo_path=code_repo_path,
        task_context_path=task_context_file,
        output_path=probe_result_path,
        timeout_sec=min(timeout_sec, 120),
    )
    if execution.get('status') != 'success':
        return {
            'status': 'error',
            'error': execution.get('error'),
            'validation': validation,
            'execution': execution,
            'analysis_evidence_probe_request_path': str(probe_request_path),
            'analysis_evidence_probe_result_path': str(probe_result_path),
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
        }

    return {
        'status': 'success',
        'validation': validation,
        'request': normalized_request,
        'execution': execution,
        'analysis_evidence_probe_request_path': str(probe_request_path),
        'analysis_evidence_probe_result_path': str(probe_result_path),
        'agent_response_path': str(log_path),
        'agent_id': response.get('agent_id'),
        'session_scope': response.get('session_scope'),
        'service_url': response.get('service_url'),
        'service_url_source': response.get('service_url_source'),
    }


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
    repair_plan_path: Path,
    failure_feedback_path: Optional[Path],
    evidence_probe_result_path: Optional[Path],
    attempt_spec: Dict[str, Any],
    failure_feedback: Dict[str, Any],
) -> str:
    attempt_json = json.dumps(attempt_spec, indent=2, ensure_ascii=False)
    feedback_json = json.dumps(failure_feedback, indent=2, ensure_ascii=False)
    analysis_plan_line = f"- analysis_plan.json: {analysis_plan_path}" if analysis_plan_path else "- analysis_plan.json: (not provided)"
    formula_line = f"- loss_formula.json: {loss_formula_path}" if loss_formula_path else "- loss_formula.json: (read it from task_context if present)"
    failure_feedback_line = (
        f"- failure_feedback.json: {failure_feedback_path}"
        if failure_feedback_path is not None
        else "- failure_feedback.json: (inline payload only)"
    )
    probe_result_line = (
        f"- repair_evidence_probe_result.json: {evidence_probe_result_path}"
        if evidence_probe_result_path is not None
        else "- repair_evidence_probe_result.json: (no extra evidence probe was needed)"
    )
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
{failure_feedback_line}
{probe_result_line}
- editable_files.json: {editable_manifest_path}
- current candidate_loss.py: {current_code_path}
- repair_plan.json: {repair_plan_path}

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
14. 必须先更新 repair_plan.json，再修改代码。repair_plan.json 必须是单个 JSON object，且至少包含这些字段：
    - failure_hypothesis: 字符串，说明你认为失败的主要原因
    - planned_changes: 字符串数组，列出本轮准备做的关键改动
    - target_metric: 字符串，优先填想提升的主要指标，如 val_ssim
    - success_criteria: 字符串，说明本轮希望达到什么验证结果
    - fallback_plan: 字符串，说明如果本轮失败下一步准备怎么退路
    - evidence_refs: 字符串数组，引用 paper/code/validator feedback 的依据
15. 如果 repair_evidence_probe_result.json 存在，repair_plan.evidence_refs 必须显式引用它，不要忽略你自己补采集出来的证据。
16. 不要输出 markdown 代码块，不要只在聊天里贴代码，必须把文件真正写回目标路径。

写完之后，只回复一行简短确认：
code_written:{output_code_path}
"""


def _resolve_latest_repair_plan_path(attempt_result: Dict[str, Any]) -> Optional[Path]:
    metadata = attempt_result.get('metadata', {}) if isinstance(attempt_result.get('metadata'), dict) else {}
    agent_repair = metadata.get('agent_repair', {}) if isinstance(metadata.get('agent_repair'), dict) else {}
    direct_path = agent_repair.get('repair_plan_path')
    if isinstance(direct_path, str) and direct_path.strip():
        candidate = Path(direct_path).expanduser().resolve()
        if candidate.exists():
            return candidate

    repair_rounds = attempt_result.get('repair_rounds')
    if not isinstance(repair_rounds, list):
        return None
    for round_info in reversed(repair_rounds):
        if not isinstance(round_info, dict):
            continue
        artifacts = round_info.get('artifacts', {}) if isinstance(round_info.get('artifacts'), dict) else {}
        artifact_path = artifacts.get('repair_plan_path')
        if isinstance(artifact_path, str) and artifact_path.strip():
            candidate = Path(artifact_path).expanduser().resolve()
            if candidate.exists():
                return candidate
        repair_payload = round_info.get('repair', {}) if isinstance(round_info.get('repair'), dict) else {}
        repair_path = repair_payload.get('repair_plan_path')
        if isinstance(repair_path, str) and repair_path.strip():
            candidate = Path(repair_path).expanduser().resolve()
            if candidate.exists():
                return candidate
    return None


def _build_candidate_loss_repair_evidence_probe_prompt(
    *,
    task_context_path: Path,
    analysis_plan_path: Optional[Path],
    current_code_path: Path,
    repair_plan_path: Path,
    failure_feedback_path: Optional[Path],
    probe_request_path: Path,
    probe_script_path: Path,
    probe_result_path: Path,
) -> str:
    analysis_plan_line = (
        f"- analysis_plan.json: {analysis_plan_path}"
        if analysis_plan_path
        else "- analysis_plan.json: (not provided)"
    )
    failure_feedback_line = (
        f"- failure_feedback.json: {failure_feedback_path}"
        if failure_feedback_path is not None
        else "- failure_feedback.json: (not provided)"
    )

    return f"""你在执行 loss-transfer 的代码修复补证据阶段。

请先读取：
- task_context.json: {task_context_path}
{analysis_plan_line}
- current candidate_loss.py: {current_code_path}
- repair_plan.json: {repair_plan_path}
{failure_feedback_line}

然后判断：仅凭当前失败反馈、当前代码和 repair plan 占位信息，是否已经足够开始修复。

你必须先写出：
- repair_evidence_probe_request.json: {probe_request_path}

如果你认为证据已经足够：
1. 把 request.json 写成单个 JSON object，字段至少包含：
   - status: "not_needed"
   - reason: 为什么现有修复证据已经足够
   - evidence_refs: 你依赖的 result/code/repair_plan/task_context 引用
2. 不要写 probe 脚本。

如果你认为证据还不够：
1. 把 request.json 写成单个 JSON object，字段至少包含：
   - status: "probe_needed"
   - reason: 为什么现有修复证据还不够
   - evidence_refs: 已有但仍不足的依据
   - probe_goal: 你还需要验证什么
   - expected_output_keys: 你希望 probe 产出的关键 JSON 字段名
2. 同时写一个只读 Python 脚本：
   - repair_evidence_probe.py: {probe_script_path}
3. 脚本运行方式固定为：
   `python repair_evidence_probe.py --code_repo <repo> --task_context <task_context> --current_code <py> [--analysis_plan <json>] [--failure_feedback <json>] [--repair_plan <json>] --output <json>`
4. 脚本必须只做“读取仓库/工件 + 解析信息 + 写 JSON 到输出路径 {probe_result_path}”。
5. 禁止修改 code_repo 内任何文件，禁止训练、禁止联网、禁止长时间阻塞。
6. 优先使用标准库做静态分析，例如 `ast/json/re/pathlib`；这是分析工具，不是实现补丁。

写完之后，只回复一行简短确认：
probe_written:{probe_request_path}
"""


def _run_candidate_loss_repair_evidence_probe(
    *,
    context: Dict[str, Any],
    failure_feedback: Dict[str, Any],
    service_url: Optional[str],
    api_key: Optional[str],
    timeout_sec: int,
) -> Dict[str, Any]:
    output_path = context['output_path']
    task_context_file = context['task_context_file']
    current_code_path = context['output_path']
    repair_plan_path = context['repair_plan_path'] or (output_path.parent / 'repair_plan.json')
    probe_request_path = output_path.parent / 'repair_evidence_probe_request.json'
    probe_script_path = output_path.parent / 'repair_evidence_probe.py'
    probe_result_path = output_path.parent / 'repair_evidence_probe_result.json'
    notebook_path = output_path.parent / 'candidate_loss_repair_evidence_probe_agent.ipynb'

    files = [str(task_context_file), str(current_code_path), str(repair_plan_path)]
    resolved_analysis_plan = context.get('resolved_analysis_plan')
    if isinstance(resolved_analysis_plan, Path) and resolved_analysis_plan.exists():
        files.append(str(resolved_analysis_plan))
    failure_feedback_path = context.get('failure_feedback_path')
    if isinstance(failure_feedback_path, Path) and failure_feedback_path.exists():
        files.append(str(failure_feedback_path))

    response = run_agent_chat(
        message=_build_candidate_loss_repair_evidence_probe_prompt(
            task_context_path=task_context_file,
            analysis_plan_path=resolved_analysis_plan if isinstance(resolved_analysis_plan, Path) and resolved_analysis_plan.exists() else None,
            current_code_path=current_code_path,
            repair_plan_path=repair_plan_path,
            failure_feedback_path=failure_feedback_path if isinstance(failure_feedback_path, Path) and failure_feedback_path.exists() else None,
            probe_request_path=probe_request_path,
            probe_script_path=probe_script_path,
            probe_result_path=probe_result_path,
        ),
        mode='edit',
        working_dir=str(context['working_dir']),
        outputs_path=str(context['experiment_dir']),
        notebook_path=str(notebook_path),
        files=files,
        service_url=service_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
    )

    log_path = output_path.parent / 'candidate_loss_repair_evidence_probe_agent_response.json'
    _write_json(log_path, response if isinstance(response, dict) else {'status': 'error'})
    _append_agent_call_manifest(
        task_context=context['task_context'],
        stage='candidate_loss_repair_evidence_probe_generation',
        response=response,
        agent_response_path=log_path,
        mode='edit',
        extra={
            'attempt_name': context.get('attempt_spec', {}).get('name') if isinstance(context.get('attempt_spec'), dict) else None,
            'attempt_kind': context.get('attempt_spec', {}).get('kind') if isinstance(context.get('attempt_spec'), dict) else None,
            'code_path': str(output_path),
            'repair_evidence_probe_request_path': str(probe_request_path),
            'repair_evidence_probe_result_path': str(probe_result_path),
        },
    )

    if response.get('status') == 'error':
        return {
            'status': 'error',
            'error': response.get('error'),
            'repair_evidence_probe_request_path': str(probe_request_path),
            'repair_evidence_probe_result_path': str(probe_result_path),
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
        }

    if not probe_request_path.exists():
        return {
            'status': 'error',
            'error': f'Agent finished but did not write repair_evidence_probe_request.json to {probe_request_path}',
            'repair_evidence_probe_request_path': str(probe_request_path),
            'repair_evidence_probe_result_path': str(probe_result_path),
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
        }

    request_payload = _load_probe_json_object(probe_request_path)
    validation = validate_evidence_probe_request(request_payload)
    normalized_request = validation.get('normalized_request')
    if validation.get('status') == 'error' or not isinstance(normalized_request, dict):
        return {
            'status': 'error',
            'error': 'repair_evidence_probe_request.json validation failed',
            'validation': validation,
            'repair_evidence_probe_request_path': str(probe_request_path),
            'repair_evidence_probe_result_path': str(probe_result_path),
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
        }

    if normalized_request.get('status') == 'not_needed':
        return {
            'status': 'not_needed',
            'validation': validation,
            'request': normalized_request,
            'repair_evidence_probe_request_path': str(probe_request_path),
            'repair_evidence_probe_result_path': None,
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
        }

    if not probe_script_path.exists():
        return {
            'status': 'error',
            'error': f'Probe was requested but repair_evidence_probe.py was not written to {probe_script_path}',
            'validation': validation,
            'repair_evidence_probe_request_path': str(probe_request_path),
            'repair_evidence_probe_result_path': str(probe_result_path),
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
        }

    code_repo_path = ((context['task_context'].get('inputs') or {}) if isinstance(context['task_context'].get('inputs'), dict) else {}).get('code_repo_path')
    if not isinstance(code_repo_path, str) or not code_repo_path.strip():
        return {
            'status': 'error',
            'error': 'Probe was requested but task_context.inputs.code_repo_path is missing',
            'validation': validation,
            'repair_evidence_probe_request_path': str(probe_request_path),
            'repair_evidence_probe_result_path': str(probe_result_path),
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
        }

    extra_args = ['--current_code', str(current_code_path)]
    if isinstance(resolved_analysis_plan, Path) and resolved_analysis_plan.exists():
        extra_args.extend(['--analysis_plan', str(resolved_analysis_plan)])
    if isinstance(failure_feedback_path, Path) and failure_feedback_path.exists():
        extra_args.extend(['--failure_feedback', str(failure_feedback_path)])
    if isinstance(repair_plan_path, Path) and repair_plan_path.exists():
        extra_args.extend(['--repair_plan', str(repair_plan_path)])

    execution = execute_evidence_probe(
        script_path=probe_script_path,
        code_repo_path=code_repo_path,
        task_context_path=task_context_file,
        output_path=probe_result_path,
        timeout_sec=min(timeout_sec, 120),
        extra_args=extra_args,
    )
    if execution.get('status') != 'success':
        return {
            'status': 'error',
            'error': execution.get('error'),
            'validation': validation,
            'execution': execution,
            'repair_evidence_probe_request_path': str(probe_request_path),
            'repair_evidence_probe_result_path': str(probe_result_path),
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
        }

    return {
        'status': 'success',
        'validation': validation,
        'request': normalized_request,
        'execution': execution,
        'repair_evidence_probe_request_path': str(probe_request_path),
        'repair_evidence_probe_result_path': str(probe_result_path),
        'agent_response_path': str(log_path),
        'agent_id': response.get('agent_id'),
        'session_scope': response.get('session_scope'),
        'service_url': response.get('service_url'),
        'service_url_source': response.get('service_url_source'),
    }


def _build_followup_attempt_prompt(
    *,
    task_context_path: Path,
    analysis_plan_path: Optional[Path],
    latest_attempt_result_path: Path,
    latest_repair_plan_path: Optional[Path],
    trajectory_path: Optional[Path],
    evidence_probe_result_path: Optional[Path],
    output_attempt_path: Path,
    max_attempts: int,
    next_attempt_id: int,
) -> str:
    analysis_plan_line = f"- analysis_plan.json: {analysis_plan_path}" if analysis_plan_path else "- analysis_plan.json: (not provided)"
    repair_plan_line = (
        f"- latest repair_plan.json: {latest_repair_plan_path}"
        if latest_repair_plan_path
        else "- latest repair_plan.json: (not available)"
    )
    trajectory_line = f"- trajectory.jsonl: {trajectory_path}" if trajectory_path else "- trajectory.jsonl: (not available)"
    probe_result_line = (
        f"- followup_evidence_probe_result.json: {evidence_probe_result_path}"
        if evidence_probe_result_path
        else "- followup_evidence_probe_result.json: (no extra evidence probe was needed)"
    )

    return f"""你在执行 loss-transfer 的逐轮重规划阶段。

请先读取：
- task_context.json: {task_context_path}
{analysis_plan_line}
- latest attempt result.json: {latest_attempt_result_path}
{repair_plan_line}
{trajectory_line}
{probe_result_line}

然后写出：
- next_attempt.json: {output_attempt_path}

目标：
基于最新失败 attempt 的验证结果、repair_plan、trajectory 历史，生成“下一条最值得尝试”的 attempt JSON object。

要求：
1. 输出必须是单个 JSON object，字段必须兼容 task_context.analysis_plan_schema.attempts[]。
2. 这不是重写完整 analysis_plan，只写一个新的 attempt。
3. 必须明确吸收上一轮 repair_plan 的 failure_hypothesis / fallback_plan，而不是机械重复上一轮。
4. evidence_refs 必须非空，并引用 paper/code/result/repair_plan 中的依据。
5. 如果上一轮已经证明某种修法失败，本轮 attempt 必须在 objective 或 notes 中明确体现“变更了什么策略”。
6. 优先最小改动；但如果最新失败证据表明 loss-only 路线不够，需要诚实升级到 adapter/model copy 路线。
7. 如果 latest attempt 已经在 layer4/性能层失败，本轮重点是提高主要指标，而不是只做语法修补。
8. attempts 总预算最多 {max_attempts}，当前将生成第 {next_attempt_id} 个 attempt，请避免无信息增益的重复尝试。
9. 必须填写 strategy_delta 对象，至少包含：
    - previous_attempt_id
    - why_previous_failed
    - what_changes_now
    - why_not_repeat_previous
    - expected_signal
10. 如果 followup_evidence_probe_result.json 存在，必须吸收里面的发现；不要忽略你自己补采集出来的证据。

写完之后，只回复一行简短确认：
attempt_written:{output_attempt_path}
"""


def _build_followup_evidence_probe_prompt(
    *,
    task_context_path: Path,
    latest_attempt_result_path: Path,
    latest_repair_plan_path: Optional[Path],
    analysis_plan_path: Optional[Path],
    trajectory_path: Optional[Path],
    probe_request_path: Path,
    probe_script_path: Path,
    probe_result_path: Path,
) -> str:
    repair_plan_line = (
        f"- latest repair_plan.json: {latest_repair_plan_path}"
        if latest_repair_plan_path
        else "- latest repair_plan.json: (not available)"
    )
    analysis_plan_line = (
        f"- analysis_plan.json: {analysis_plan_path}"
        if analysis_plan_path
        else "- analysis_plan.json: (not provided)"
    )
    trajectory_line = (
        f"- trajectory.jsonl: {trajectory_path}"
        if trajectory_path
        else "- trajectory.jsonl: (not available)"
    )

    return f"""你在执行 loss-transfer 的逐轮重规划补证据阶段。

请先读取：
- task_context.json: {task_context_path}
- latest attempt result.json: {latest_attempt_result_path}
{repair_plan_line}
{analysis_plan_line}
{trajectory_line}

然后判断：仅凭当前失败证据，是否已经足够生成下一条 follow-up attempt。

你必须先写出：
- followup_evidence_probe_request.json: {probe_request_path}

如果你认为证据已经足够：
1. 把 request.json 写成单个 JSON object，字段至少包含：
   - status: "not_needed"
   - reason: 为什么现有失败证据已经足够
   - evidence_refs: 你依赖的 result/repair_plan/task_context 引用
2. 不要写 probe 脚本。

如果你认为证据还不够：
1. 把 request.json 写成单个 JSON object，字段至少包含：
   - status: "probe_needed"
   - reason: 为什么现有失败证据还不够
   - evidence_refs: 已有但仍不足的依据
   - probe_goal: 你还需要验证什么
   - expected_output_keys: 你希望 probe 产出的关键 JSON 字段名
2. 同时写一个只读 Python 脚本：
   - followup_evidence_probe.py: {probe_script_path}
3. 脚本运行方式固定为：
   `python followup_evidence_probe.py --code_repo <repo> --task_context <task_context> --latest_result <json> [--latest_repair_plan <json>] [--analysis_plan <json>] [--trajectory <jsonl>] --output <json>`
4. 脚本必须只做“读取仓库/工件 + 解析信息 + 写 JSON 到输出路径 {probe_result_path}”。
5. 禁止修改 code_repo 内任何文件，禁止训练、禁止联网、禁止长时间阻塞。
6. 优先使用标准库做静态分析，例如 `ast/json/re/pathlib`；这是分析工具，不是实现补丁。

写完之后，只回复一行简短确认：
probe_written:{probe_request_path}
"""


def _run_followup_evidence_probe(
    *,
    task_context: Dict[str, Any],
    task_context_file: Path,
    latest_result_file: Path,
    latest_repair_plan: Optional[Path],
    resolved_analysis_plan: Optional[Path],
    trajectory_path: Optional[Path],
    output_path: Path,
    resolved_next_attempt_id: int,
    working_dir: Path,
    service_url: Optional[str],
    api_key: Optional[str],
    timeout_sec: int,
) -> Dict[str, Any]:
    probe_request_path = output_path.parent / f'followup_attempt_{resolved_next_attempt_id}_evidence_probe_request.json'
    probe_script_path = output_path.parent / f'followup_attempt_{resolved_next_attempt_id}_evidence_probe.py'
    probe_result_path = output_path.parent / f'followup_attempt_{resolved_next_attempt_id}_evidence_probe_result.json'
    notebook_path = output_path.parent / f'followup_attempt_{resolved_next_attempt_id}_evidence_probe_agent.ipynb'

    files = [str(task_context_file), str(latest_result_file)]
    if latest_repair_plan is not None and latest_repair_plan.exists():
        files.append(str(latest_repair_plan))
    if resolved_analysis_plan is not None and resolved_analysis_plan.exists():
        files.append(str(resolved_analysis_plan))
    if trajectory_path is not None and trajectory_path.exists():
        files.append(str(trajectory_path))

    response = run_agent_chat(
        message=_build_followup_evidence_probe_prompt(
            task_context_path=task_context_file,
            latest_attempt_result_path=latest_result_file,
            latest_repair_plan_path=latest_repair_plan if latest_repair_plan and latest_repair_plan.exists() else None,
            analysis_plan_path=resolved_analysis_plan if resolved_analysis_plan and resolved_analysis_plan.exists() else None,
            trajectory_path=trajectory_path if trajectory_path and trajectory_path.exists() else None,
            probe_request_path=probe_request_path,
            probe_script_path=probe_script_path,
            probe_result_path=probe_result_path,
        ),
        mode='edit',
        working_dir=str(working_dir),
        outputs_path=str(output_path.parent),
        notebook_path=str(notebook_path),
        files=files,
        service_url=service_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
    )

    log_path = output_path.parent / f'followup_attempt_{resolved_next_attempt_id}_evidence_probe_agent_response.json'
    _write_json(log_path, response if isinstance(response, dict) else {'status': 'error'})
    _append_agent_call_manifest(
        task_context=task_context,
        stage='followup_attempt_evidence_probe_generation',
        response=response,
        agent_response_path=log_path,
        mode='edit',
        extra={
            'attempt_path': str(output_path),
            'next_attempt_id': resolved_next_attempt_id,
            'followup_evidence_probe_request_path': str(probe_request_path),
            'followup_evidence_probe_result_path': str(probe_result_path),
        },
    )

    if response.get('status') == 'error':
        return {
            'status': 'error',
            'error': response.get('error'),
            'followup_evidence_probe_request_path': str(probe_request_path),
            'followup_evidence_probe_result_path': str(probe_result_path),
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
        }

    if not probe_request_path.exists():
        return {
            'status': 'error',
            'error': f'Agent finished but did not write followup_evidence_probe_request.json to {probe_request_path}',
            'followup_evidence_probe_request_path': str(probe_request_path),
            'followup_evidence_probe_result_path': str(probe_result_path),
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
        }

    request_payload = _load_probe_json_object(probe_request_path)
    validation = validate_evidence_probe_request(request_payload)
    normalized_request = validation.get('normalized_request')
    if validation.get('status') == 'error' or not isinstance(normalized_request, dict):
        return {
            'status': 'error',
            'error': 'followup_evidence_probe_request.json validation failed',
            'validation': validation,
            'followup_evidence_probe_request_path': str(probe_request_path),
            'followup_evidence_probe_result_path': str(probe_result_path),
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
        }

    if normalized_request.get('status') == 'not_needed':
        return {
            'status': 'not_needed',
            'validation': validation,
            'request': normalized_request,
            'followup_evidence_probe_request_path': str(probe_request_path),
            'followup_evidence_probe_result_path': None,
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
        }

    if not probe_script_path.exists():
        return {
            'status': 'error',
            'error': f'Probe was requested but followup_evidence_probe.py was not written to {probe_script_path}',
            'validation': validation,
            'followup_evidence_probe_request_path': str(probe_request_path),
            'followup_evidence_probe_result_path': str(probe_result_path),
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
        }

    code_repo_path = ((task_context.get('inputs') or {}) if isinstance(task_context.get('inputs'), dict) else {}).get('code_repo_path')
    if not isinstance(code_repo_path, str) or not code_repo_path.strip():
        return {
            'status': 'error',
            'error': 'Probe was requested but task_context.inputs.code_repo_path is missing',
            'validation': validation,
            'followup_evidence_probe_request_path': str(probe_request_path),
            'followup_evidence_probe_result_path': str(probe_result_path),
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
        }

    extra_args = ['--latest_result', str(latest_result_file)]
    if latest_repair_plan is not None and latest_repair_plan.exists():
        extra_args.extend(['--latest_repair_plan', str(latest_repair_plan)])
    if resolved_analysis_plan is not None and resolved_analysis_plan.exists():
        extra_args.extend(['--analysis_plan', str(resolved_analysis_plan)])
    if trajectory_path is not None and trajectory_path.exists():
        extra_args.extend(['--trajectory', str(trajectory_path)])

    execution = execute_evidence_probe(
        script_path=probe_script_path,
        code_repo_path=code_repo_path,
        task_context_path=task_context_file,
        output_path=probe_result_path,
        timeout_sec=min(timeout_sec, 120),
        extra_args=extra_args,
    )
    if execution.get('status') != 'success':
        return {
            'status': 'error',
            'error': execution.get('error'),
            'validation': validation,
            'execution': execution,
            'followup_evidence_probe_request_path': str(probe_request_path),
            'followup_evidence_probe_result_path': str(probe_result_path),
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
        }

    return {
        'status': 'success',
        'validation': validation,
        'request': normalized_request,
        'execution': execution,
        'followup_evidence_probe_request_path': str(probe_request_path),
        'followup_evidence_probe_result_path': str(probe_result_path),
        'agent_response_path': str(log_path),
        'agent_id': response.get('agent_id'),
        'session_scope': response.get('session_scope'),
        'service_url': response.get('service_url'),
        'service_url_source': response.get('service_url_source'),
    }


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
    write_run_manifest(
        experiment_dir=experiment_dir,
        paper_slug=str(task_context.get('paper_slug', 'paper')),
        task_context=task_context,
        mode='agent_loop',
        max_attempts=max_attempts,
        service_url=service_url,
        analysis_plan_path=str(analysis_plan_path),
    )
    working_dir = _resolve_working_dir(task_context)
    notebook_path = experiment_dir / 'analysis_plan_agent.ipynb'
    evidence_probe = _run_analysis_evidence_probe(
        task_context=task_context,
        task_context_file=task_context_file,
        experiment_dir=experiment_dir,
        working_dir=working_dir,
        service_url=service_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
    )
    if evidence_probe.get('status') == 'error':
        result = {
            'status': 'error',
            'error': evidence_probe.get('error') or 'analysis evidence probe failed',
            'analysis_plan_path': str(analysis_plan_path),
            'analysis_evidence_probe_status': evidence_probe.get('status'),
            'analysis_evidence_probe_request_path': evidence_probe.get('analysis_evidence_probe_request_path'),
            'analysis_evidence_probe_result_path': evidence_probe.get('analysis_evidence_probe_result_path'),
            'agent_response_path': evidence_probe.get('agent_response_path'),
            'agent_id': evidence_probe.get('agent_id'),
            'session_scope': evidence_probe.get('session_scope'),
            'service_url': evidence_probe.get('service_url'),
            'service_url_source': evidence_probe.get('service_url_source'),
            'evidence_probe': evidence_probe,
        }
        result.update(
            write_run_manifest(
                experiment_dir=experiment_dir,
                paper_slug=str(task_context.get('paper_slug', 'paper')),
                task_context=task_context,
                mode='agent_loop',
                service_url=service_url,
                analysis_plan_path=str(analysis_plan_path),
                plan_generation=result,
            )
        )
        return result

    analysis_files = [str(task_context_file)]
    evidence_graph_path = paths.get('evidence_graph_path')
    if isinstance(evidence_graph_path, str) and evidence_graph_path.strip():
        evidence_graph_file = Path(evidence_graph_path).expanduser().resolve()
        if evidence_graph_file.exists():
            analysis_files.append(str(evidence_graph_file))
    probe_result_path: Optional[Path] = None
    if isinstance(evidence_probe.get('analysis_evidence_probe_result_path'), str):
        candidate_probe_result = Path(evidence_probe['analysis_evidence_probe_result_path']).expanduser().resolve()
        if candidate_probe_result.exists():
            probe_result_path = candidate_probe_result
            analysis_files.append(str(candidate_probe_result))

    response = run_agent_chat(
        message=_build_analysis_plan_prompt(
            task_context,
            task_context_path=task_context_file,
            analysis_plan_path=analysis_plan_path,
            max_attempts=max_attempts,
            evidence_probe_result_path=probe_result_path,
        ),
        mode='edit',
        working_dir=str(working_dir),
        outputs_path=str(experiment_dir),
        notebook_path=str(notebook_path),
        files=analysis_files,
        service_url=service_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
    )

    log_path = experiment_dir / 'analysis_plan_agent_response.json'
    _write_json(log_path, response if isinstance(response, dict) else {'status': 'error'})
    _append_agent_call_manifest(
        task_context=task_context,
        stage='analysis_plan_generation',
        response=response,
        agent_response_path=log_path,
        mode='edit',
        extra={
            'analysis_plan_path': str(analysis_plan_path),
        },
    )

    if response.get('status') == 'error':
        result = {
            'status': 'error',
            'error': response.get('error'),
            'analysis_plan_path': str(analysis_plan_path),
            'analysis_evidence_probe_status': evidence_probe.get('status'),
            'analysis_evidence_probe_request_path': evidence_probe.get('analysis_evidence_probe_request_path'),
            'analysis_evidence_probe_result_path': evidence_probe.get('analysis_evidence_probe_result_path'),
            'agent_response_path': str(log_path),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
            'evidence_probe': evidence_probe,
        }
        result.update(
            write_run_manifest(
                experiment_dir=experiment_dir,
                paper_slug=str(task_context.get('paper_slug', 'paper')),
                task_context=task_context,
                mode='agent_loop',
                service_url=service_url,
                analysis_plan_path=str(analysis_plan_path),
                plan_generation=result,
            )
        )
        return result

    if not analysis_plan_path.exists():
        result = {
            'status': 'error',
            'error': f'Agent finished but did not write analysis_plan.json to {analysis_plan_path}',
            'analysis_plan_path': str(analysis_plan_path),
            'analysis_evidence_probe_status': evidence_probe.get('status'),
            'analysis_evidence_probe_request_path': evidence_probe.get('analysis_evidence_probe_request_path'),
            'analysis_evidence_probe_result_path': evidence_probe.get('analysis_evidence_probe_result_path'),
            'agent_response_path': str(log_path),
            'agent_text': response.get('text', ''),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
            'service_url': response.get('service_url'),
            'service_url_source': response.get('service_url_source'),
            'evidence_probe': evidence_probe,
        }
        result.update(
            write_run_manifest(
                experiment_dir=experiment_dir,
                paper_slug=str(task_context.get('paper_slug', 'paper')),
                task_context=task_context,
                mode='agent_loop',
                service_url=service_url,
                analysis_plan_path=str(analysis_plan_path),
                plan_generation=result,
            )
        )
        return result

    plan = _load_json(analysis_plan_path)
    validation = validate_analysis_plan(plan, task_context=task_context)
    result = {
        'status': 'success' if validation['status'] != 'error' else 'error',
        'analysis_plan_path': str(analysis_plan_path),
        'analysis_evidence_probe_status': evidence_probe.get('status'),
        'analysis_evidence_probe_request_path': evidence_probe.get('analysis_evidence_probe_request_path'),
        'analysis_evidence_probe_result_path': evidence_probe.get('analysis_evidence_probe_result_path'),
        'agent_response_path': str(log_path),
        'agent_id': response.get('agent_id'),
        'validation': validation,
        'agent_text': response.get('text', ''),
        'session_scope': response.get('session_scope'),
        'service_url': response.get('service_url'),
        'service_url_source': response.get('service_url_source'),
        'evidence_probe': evidence_probe,
    }
    normalized_plan = validation.get('normalized_plan')
    if isinstance(normalized_plan, dict):
        result.update(
            write_routing_audit(
                experiment_dir=experiment_dir,
                paper_slug=str(task_context.get('paper_slug', 'paper')),
                task_context=task_context,
                analysis_plan=normalized_plan,
                analysis_plan_path=str(analysis_plan_path),
            )
        )
    result.update(
        write_run_manifest(
            experiment_dir=experiment_dir,
            paper_slug=str(task_context.get('paper_slug', 'paper')),
            task_context=task_context,
            mode='agent_loop',
            service_url=service_url,
            analysis_plan_path=str(analysis_plan_path),
            plan_generation=result,
        )
    )
    _write_json(log_path, {**response, 'validation': validation})
    return result


def generate_followup_attempt(
    latest_attempt_result_path: str,
    *,
    task_context_path: str,
    output_attempt_path: str,
    analysis_plan_path: Optional[str] = None,
    service_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_sec: int = 900,
    max_attempts: int = 4,
    next_attempt_id: Optional[int] = None,
) -> Dict[str, Any]:
    task_context_file = Path(task_context_path).expanduser().resolve()
    latest_result_file = Path(latest_attempt_result_path).expanduser().resolve()
    output_path = Path(output_attempt_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_context = _load_json(task_context_file)
    attempt_result = _load_json(latest_result_file)
    paths = task_context.get('paths', {}) if isinstance(task_context.get('paths'), dict) else {}
    experiment_dir = Path(str(paths.get('experiment_dir') or task_context_file.parent)).expanduser().resolve()
    write_run_manifest(
        experiment_dir=experiment_dir,
        paper_slug=str(task_context.get('paper_slug', 'paper')),
        task_context=task_context,
        mode='agent_loop',
        analysis_plan_path=analysis_plan_path,
        service_url=service_url,
    )
    working_dir = _resolve_working_dir(task_context)
    notebook_path = output_path.parent / 'followup_attempt_agent.ipynb'
    resolved_analysis_plan = (
        Path(analysis_plan_path).expanduser().resolve()
        if analysis_plan_path
        else (
            Path(str(paths.get('analysis_plan_path'))).expanduser().resolve()
            if paths.get('analysis_plan_path')
            else None
        )
    )
    trajectory_path = (
        Path(str(paths.get('trajectory_path'))).expanduser().resolve()
        if paths.get('trajectory_path')
        else None
    )
    latest_repair_plan = _resolve_latest_repair_plan_path(attempt_result)
    latest_repair_plan_payload = _safe_load_json_object(latest_repair_plan)
    resolved_next_attempt_id = (
        int(next_attempt_id)
        if next_attempt_id is not None
        else int(attempt_result.get('attempt_id', 0)) + 1
    )

    files = [str(task_context_file), str(latest_result_file)]
    if resolved_analysis_plan is not None and resolved_analysis_plan.exists():
        files.append(str(resolved_analysis_plan))
    if trajectory_path is not None and trajectory_path.exists():
        files.append(str(trajectory_path))
    if latest_repair_plan is not None and latest_repair_plan.exists():
        files.append(str(latest_repair_plan))

    evidence_probe = _run_followup_evidence_probe(
        task_context=task_context,
        task_context_file=task_context_file,
        latest_result_file=latest_result_file,
        latest_repair_plan=latest_repair_plan,
        resolved_analysis_plan=resolved_analysis_plan,
        trajectory_path=trajectory_path,
        output_path=output_path,
        resolved_next_attempt_id=resolved_next_attempt_id,
        working_dir=working_dir,
        service_url=service_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
    )
    if evidence_probe.get('status') == 'error':
        return {
            'status': 'error',
            'error': evidence_probe.get('error') or 'follow-up evidence probe failed',
            'attempt_path': str(output_path),
            'agent_response_path': evidence_probe.get('agent_response_path'),
            'followup_evidence_probe_status': evidence_probe.get('status'),
            'followup_evidence_probe_request_path': evidence_probe.get('followup_evidence_probe_request_path'),
            'followup_evidence_probe_result_path': evidence_probe.get('followup_evidence_probe_result_path'),
            'agent_id': evidence_probe.get('agent_id'),
            'session_scope': evidence_probe.get('session_scope'),
        }

    probe_result_path: Optional[Path] = None
    if isinstance(evidence_probe.get('followup_evidence_probe_result_path'), str):
        candidate_probe_result = Path(evidence_probe['followup_evidence_probe_result_path']).expanduser().resolve()
        if candidate_probe_result.exists():
            probe_result_path = candidate_probe_result
            files.append(str(candidate_probe_result))

    memory_block = _format_case_memory_block(
        _load_similar_case_memories(
            task_context=task_context,
            latest_attempt_result=attempt_result,
            latest_repair_plan=latest_repair_plan_payload,
            case_memory_path=_CASE_MEMORY_PATH,
        )
    )

    response = run_agent_chat(
        message=_append_memory_block(
            _build_followup_attempt_prompt(
            task_context_path=task_context_file,
            analysis_plan_path=resolved_analysis_plan if resolved_analysis_plan and resolved_analysis_plan.exists() else None,
            latest_attempt_result_path=latest_result_file,
            latest_repair_plan_path=latest_repair_plan,
            trajectory_path=trajectory_path if trajectory_path and trajectory_path.exists() else None,
            evidence_probe_result_path=probe_result_path,
            output_attempt_path=output_path,
            max_attempts=max_attempts,
            next_attempt_id=resolved_next_attempt_id,
            ),
            memory_block,
        ),
        mode='edit',
        working_dir=str(working_dir),
        outputs_path=str(experiment_dir),
        notebook_path=str(notebook_path),
        files=files,
        service_url=service_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
    )

    log_path = output_path.parent / f'followup_attempt_{resolved_next_attempt_id}_agent_response.json'
    _write_json(log_path, response if isinstance(response, dict) else {'status': 'error'})
    _append_agent_call_manifest(
        task_context=task_context,
        stage='followup_attempt_generation',
        response=response,
        agent_response_path=log_path,
        mode='edit',
        extra={
            'attempt_path': str(output_path),
            'next_attempt_id': resolved_next_attempt_id,
            'followup_evidence_probe_status': evidence_probe.get('status'),
            'followup_evidence_probe_request_path': evidence_probe.get('followup_evidence_probe_request_path'),
            'followup_evidence_probe_result_path': evidence_probe.get('followup_evidence_probe_result_path'),
        },
    )

    if response.get('status') == 'error':
        return {
            'status': 'error',
            'error': response.get('error'),
            'attempt_path': str(output_path),
            'agent_response_path': str(log_path),
            'followup_evidence_probe_status': evidence_probe.get('status'),
            'followup_evidence_probe_request_path': evidence_probe.get('followup_evidence_probe_request_path'),
            'followup_evidence_probe_result_path': evidence_probe.get('followup_evidence_probe_result_path'),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
        }

    if not output_path.exists():
        return {
            'status': 'error',
            'error': f'Agent finished but did not write next_attempt.json to {output_path}',
            'attempt_path': str(output_path),
            'agent_response_path': str(log_path),
            'agent_text': response.get('text', ''),
            'followup_evidence_probe_status': evidence_probe.get('status'),
            'followup_evidence_probe_request_path': evidence_probe.get('followup_evidence_probe_request_path'),
            'followup_evidence_probe_result_path': evidence_probe.get('followup_evidence_probe_result_path'),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
        }

    attempt_payload = _load_json(output_path)
    validation = validate_attempt_spec(attempt_payload)
    _write_json(log_path, {**response, 'validation': validation, 'latest_repair_plan_path': str(latest_repair_plan) if latest_repair_plan else None})
    if validation['status'] == 'error' or not isinstance(validation.get('normalized_attempt'), dict):
        return {
            'status': 'error',
            'error': 'follow-up attempt validation failed: ' + '; '.join(validation.get('errors', [])),
            'attempt_path': str(output_path),
            'agent_response_path': str(log_path),
            'agent_text': response.get('text', ''),
            'validation': validation,
            'followup_evidence_probe_status': evidence_probe.get('status'),
            'followup_evidence_probe_request_path': evidence_probe.get('followup_evidence_probe_request_path'),
            'followup_evidence_probe_result_path': evidence_probe.get('followup_evidence_probe_result_path'),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
        }

    normalized_attempt = validation['normalized_attempt']
    if probe_result_path is not None and not _evidence_refs_contain_prefix(
        normalized_attempt.get('evidence_refs'),
        'followup_evidence_probe_result',
    ):
        return {
            'status': 'error',
            'error': (
                'follow-up attempt must reference followup_evidence_probe_result in evidence_refs '
                'when a follow-up evidence probe was requested and executed'
            ),
            'attempt_path': str(output_path),
            'agent_response_path': str(log_path),
            'agent_text': response.get('text', ''),
            'validation': validation,
            'followup_evidence_probe_status': evidence_probe.get('status'),
            'followup_evidence_probe_request_path': evidence_probe.get('followup_evidence_probe_request_path'),
            'followup_evidence_probe_result_path': evidence_probe.get('followup_evidence_probe_result_path'),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
        }

    strategy_delta = normalized_attempt.get('strategy_delta')
    if not isinstance(strategy_delta, dict):
        return {
            'status': 'error',
            'error': 'follow-up attempt must include a non-empty strategy_delta object',
            'attempt_path': str(output_path),
            'agent_response_path': str(log_path),
            'agent_text': response.get('text', ''),
            'validation': validation,
            'followup_evidence_probe_status': evidence_probe.get('status'),
            'followup_evidence_probe_request_path': evidence_probe.get('followup_evidence_probe_request_path'),
            'followup_evidence_probe_result_path': evidence_probe.get('followup_evidence_probe_result_path'),
            'agent_id': response.get('agent_id'),
            'session_scope': response.get('session_scope'),
        }

    return {
        'status': 'success',
        'attempt_path': str(output_path),
        'attempt_spec': normalized_attempt,
        'agent_response_path': str(log_path),
        'followup_evidence_probe_status': evidence_probe.get('status'),
        'followup_evidence_probe_request_path': evidence_probe.get('followup_evidence_probe_request_path'),
        'followup_evidence_probe_result_path': evidence_probe.get('followup_evidence_probe_result_path'),
        'agent_id': response.get('agent_id'),
        'session_scope': response.get('session_scope'),
        'agent_text': response.get('text', ''),
        'validation': validation,
        'latest_repair_plan_path': str(latest_repair_plan) if latest_repair_plan else None,
    }


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
    failure_feedback: Optional[Dict[str, Any]] = None,
    repair_plan_path: Optional[str] = None,
) -> Dict[str, Any]:
    task_context_file = Path(task_context_path).expanduser().resolve()
    output_path = Path(output_code_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_repair_plan_path = (
        Path(repair_plan_path).expanduser().resolve() if repair_plan_path else None
    )
    resolved_failure_feedback_path = (
        output_path.parent / 'failure_feedback.json'
        if failure_feedback is not None
        else None
    )

    task_context = _load_json(task_context_file)
    paths = task_context.get('paths', {}) if isinstance(task_context.get('paths'), dict) else {}
    experiment_dir = Path(str(paths.get('experiment_dir') or task_context_file.parent)).expanduser().resolve()
    write_run_manifest(
        experiment_dir=experiment_dir,
        paper_slug=str(task_context.get('paper_slug', 'paper')),
        task_context=task_context,
        mode='agent_loop',
        analysis_plan_path=analysis_plan_path,
    )
    working_dir = _resolve_working_dir(task_context)
    notebook_path = output_path.parent / notebook_name
    additional_editable_targets: list[Dict[str, Any]] = []
    if resolved_repair_plan_path is not None:
        placeholder = _build_repair_plan_placeholder(failure_feedback or {})
        _write_json(resolved_repair_plan_path, placeholder)
        additional_editable_targets.append(
            {
                'path': str(resolved_repair_plan_path),
                'kind': 'repair_plan',
                'description': 'Structured per-round repair hypothesis and execution plan.',
            }
        )
    if resolved_failure_feedback_path is not None:
        _write_json(resolved_failure_feedback_path, failure_feedback)
    resolved_analysis_plan = (
        Path(analysis_plan_path).expanduser().resolve()
        if analysis_plan_path
        else (
            Path(str(paths.get('analysis_plan_path'))).expanduser().resolve()
            if paths.get('analysis_plan_path')
            else None
        )
    )
    analysis_plan = (
        _load_json(resolved_analysis_plan)
        if resolved_analysis_plan is not None and resolved_analysis_plan.exists()
        else None
    )
    edit_workspace = _prepare_attempt_edit_workspace(
        task_context=task_context,
        attempt_spec=attempt_spec,
        output_code_path=output_path,
        analysis_plan=analysis_plan,
        additional_editable_targets=additional_editable_targets,
    )
    before_snapshot = _snapshot_editable_targets(edit_workspace['editable_targets'])
    support_paths = _resolve_candidate_support_paths(
        paths=paths,
        analysis_plan_path=analysis_plan_path,
    )
    return {
        'task_context_file': task_context_file,
        'task_context': task_context,
        'attempt_spec': attempt_spec,
        'paths': paths,
        'output_path': output_path,
        'experiment_dir': experiment_dir,
        'working_dir': working_dir,
        'notebook_path': notebook_path,
        'edit_workspace': edit_workspace,
        'before_snapshot': before_snapshot,
        'resolved_loss_formula': support_paths['loss_formula_path'],
        'resolved_analysis_plan': support_paths['analysis_plan_path'],
        'analysis_plan': analysis_plan,
        'repair_plan_path': resolved_repair_plan_path,
        'failure_feedback_path': resolved_failure_feedback_path,
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


def _finalize_repair_plan_result(
    *,
    result: Dict[str, Any],
    log_path: Path,
    repair_plan_path: Optional[Path],
) -> Dict[str, Any]:
    if result.get('status') != 'success':
        return result
    if repair_plan_path is None:
        return result

    if not repair_plan_path.exists():
        return {
            **result,
            'status': 'error',
            'error': f'Agent finished but did not write repair_plan.json to {repair_plan_path}',
            'repair_plan_path': str(repair_plan_path),
        }

    repair_plan = _load_json(repair_plan_path)
    validation_error = _validate_repair_plan(repair_plan)
    if validation_error is not None:
        return {
            **result,
            'status': 'error',
            'error': f'Invalid repair_plan.json: {validation_error}',
            'repair_plan_path': str(repair_plan_path),
        }

    summary = {
        'failure_hypothesis': repair_plan['failure_hypothesis'],
        'target_metric': repair_plan['target_metric'],
        'planned_change_count': len(repair_plan['planned_changes']),
    }
    log_payload = _load_json(log_path)
    log_payload['repair_plan_path'] = str(repair_plan_path)
    log_payload['repair_plan'] = repair_plan
    _write_json(log_path, log_payload)

    return {
        **result,
        'repair_plan_path': str(repair_plan_path),
        'repair_plan': repair_plan,
        'repair_plan_summary': summary,
    }


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
    _append_agent_call_manifest(
        task_context=context['task_context'],
        stage='candidate_loss_generation',
        response=response,
        agent_response_path=log_path,
        mode='edit',
        extra={
            'attempt_name': attempt_spec.get('name'),
            'attempt_kind': attempt_spec.get('kind'),
            'code_path': str(context['output_path']),
        },
    )
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
    repair_plan_path: Optional[str] = None,
    analysis_plan_path: Optional[str] = None,
    service_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_sec: int = 900,
) -> Dict[str, Any]:
    output_path = Path(output_code_path).expanduser().resolve()
    if not output_path.exists():
        raise FileNotFoundError(f'Candidate code not found for repair: {output_path}')
    resolved_repair_plan_path = repair_plan_path or str(output_path.parent / 'repair_plan.json')
    context = _prepare_candidate_edit_context(
        task_context_path=task_context_path,
        attempt_spec=attempt_spec,
        output_code_path=output_code_path,
        analysis_plan_path=analysis_plan_path,
        notebook_name='candidate_loss_repair_agent.ipynb',
        failure_feedback=failure_feedback,
        repair_plan_path=resolved_repair_plan_path,
    )
    evidence_probe = _run_candidate_loss_repair_evidence_probe(
        context=context,
        failure_feedback=failure_feedback,
        service_url=service_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
    )
    if evidence_probe.get('status') == 'error':
        return {
            'status': 'error',
            'error': evidence_probe.get('error') or 'candidate loss repair evidence probe failed',
            'code_path': str(context['output_path']),
            'repair_plan_path': str(context['repair_plan_path']) if context.get('repair_plan_path') else None,
            'repair_evidence_probe_status': evidence_probe.get('status'),
            'repair_evidence_probe_request_path': evidence_probe.get('repair_evidence_probe_request_path'),
            'repair_evidence_probe_result_path': evidence_probe.get('repair_evidence_probe_result_path'),
            'agent_response_path': evidence_probe.get('agent_response_path'),
            'agent_id': evidence_probe.get('agent_id'),
            'session_scope': evidence_probe.get('session_scope'),
        }

    repair_probe_result_path: Optional[Path] = None
    if isinstance(evidence_probe.get('repair_evidence_probe_result_path'), str):
        candidate_probe_result = Path(evidence_probe['repair_evidence_probe_result_path']).expanduser().resolve()
        if candidate_probe_result.exists():
            repair_probe_result_path = candidate_probe_result

    repair_input_files = _build_agent_edit_input_files(
        task_context_file=context['task_context_file'],
        editable_manifest_path=Path(context['edit_workspace']['manifest_path']),
        editable_targets=context['edit_workspace']['editable_targets'],
        resolved_loss_formula=context['resolved_loss_formula'],
        resolved_analysis_plan=context['resolved_analysis_plan'],
        current_code_path=context['output_path'],
    )
    if isinstance(context.get('failure_feedback_path'), Path) and context['failure_feedback_path'].exists():
        repair_input_files.append(str(context['failure_feedback_path']))
    if repair_probe_result_path is not None and repair_probe_result_path.exists():
        repair_input_files.append(str(repair_probe_result_path))

    memory_block = _format_case_memory_block(
        _load_similar_case_memories(
            task_context=context['task_context'],
            attempt_spec=attempt_spec,
            failure_feedback=failure_feedback,
            case_memory_path=_CASE_MEMORY_PATH,
        )
    )

    response = run_agent_chat(
        message=_append_memory_block(
            _build_candidate_loss_repair_prompt(
                task_context_path=context['task_context_file'],
                analysis_plan_path=context['resolved_analysis_plan'],
                loss_formula_path=context['resolved_loss_formula'],
                editable_manifest_path=Path(context['edit_workspace']['manifest_path']),
                editable_targets=context['edit_workspace']['editable_targets'],
                current_code_path=context['output_path'],
                output_code_path=context['output_path'],
                repair_plan_path=context['repair_plan_path'] or (context['output_path'].parent / 'repair_plan.json'),
                failure_feedback_path=context.get('failure_feedback_path'),
                evidence_probe_result_path=repair_probe_result_path,
                attempt_spec=attempt_spec,
                failure_feedback=failure_feedback,
            ),
            memory_block,
        ),
        mode='edit',
        working_dir=str(context['working_dir']),
        outputs_path=str(context['experiment_dir']),
        notebook_path=str(context['notebook_path']),
        files=repair_input_files,
        service_url=service_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
    )

    log_path = context['output_path'].parent / 'agent_code_repair_response.json'
    _write_json(log_path, response if isinstance(response, dict) else {'status': 'error'})
    _append_agent_call_manifest(
        task_context=context['task_context'],
        stage='candidate_loss_repair',
        response=response,
        agent_response_path=log_path,
        mode='edit',
        extra={
            'attempt_name': attempt_spec.get('name'),
            'attempt_kind': attempt_spec.get('kind'),
            'code_path': str(context['output_path']),
            'repair_plan_path': str(context['repair_plan_path']) if context.get('repair_plan_path') else None,
            'repair_evidence_probe_status': evidence_probe.get('status'),
            'repair_evidence_probe_request_path': evidence_probe.get('repair_evidence_probe_request_path'),
            'repair_evidence_probe_result_path': evidence_probe.get('repair_evidence_probe_result_path'),
        },
    )
    result = _finalize_candidate_edit_result(
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
    finalized_result = _finalize_repair_plan_result(
        result=result,
        log_path=log_path,
        repair_plan_path=context['repair_plan_path'],
    )
    finalized_result.update(
        {
            'repair_evidence_probe_status': evidence_probe.get('status'),
            'repair_evidence_probe_request_path': evidence_probe.get('repair_evidence_probe_request_path'),
            'repair_evidence_probe_result_path': evidence_probe.get('repair_evidence_probe_result_path'),
        }
    )
    if (
        finalized_result.get('status') == 'success'
        and repair_probe_result_path is not None
        and isinstance(finalized_result.get('repair_plan'), dict)
        and not _evidence_refs_contain_prefix(
            finalized_result['repair_plan'].get('evidence_refs'),
            'repair_evidence_probe_result',
        )
    ):
        return {
            **finalized_result,
            'status': 'error',
            'error': (
                'repair_plan.json must reference repair_evidence_probe_result in evidence_refs '
                'when a repair evidence probe was requested and executed'
            ),
        }
    return finalized_result
