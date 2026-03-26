"""
@file attempt_executor.py
@description Execute one agent-authored or formula-native loss attempt.
@author Leizheng
@date 2026-03-25
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from agent_artifact_generator import generate_candidate_loss, repair_candidate_loss
from attempt_feedback import (
    build_failure_feedback as _assemble_failure_feedback,
    compute_baseline_delta as _compute_baseline_delta,
    extract_primary_metric as _extract_primary_metric,
    validation_error_text as _validation_error_text,
)
from attempt_state import (
    attach_repair_artifact as _attach_repair_artifact,
    build_attempt_result as _build_attempt_result,
    build_code_generation_failure_result as _build_code_generation_failure_result,
    build_initial_repair_record as _build_initial_repair_record,
    mark_repair_reverted as _mark_repair_reverted,
    should_revert_repair as _should_revert_repair,
    snapshot_path as _snapshot_path,
)
from formula_code_generator import generate_formula_loss_code, supports_formula_codegen
from formula_interface_analysis import analyze_formula_interface
from runtime_routing import (
    FULL_RUN_MODEL_CONFIGS,
    build_runtime_routing_feedback,
    probe_model_output_extension_support,
)
from trajectory_logger import (
    append_trajectory_event,
    ensure_experiment_dir,
    write_attempt_artifacts,
)
from validate_formula_alignment import validate_alignment
from validate_loss import (
    _PIPELINE_DIR,
    _PYTHON,
    _SANDBOX_DIR,
    validate_full_run,
    validate_single_model,
    validate_smoke,
    validate_static,
)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BASELINE_FILE = _PROJECT_ROOT / 'workflow' / 'loss_transfer' / 'baseline_thresholds.yaml'
_MAX_AGENT_REPAIR_ROUNDS = 3
_REPAIRABLE_LAYERS = {'layer1', 'layer2', 'formula_alignment', 'layer3', 'layer4'}
_LAYER_ORDER = {
    'formula_interface': 0,
    'layer1': 1,
    'layer2': 2,
    'formula_alignment': 3,
    'layer3': 4,
    'layer4': 5,
    None: 6,
}


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _load_baseline_thresholds() -> Dict[str, Any]:
    if _BASELINE_FILE.exists():
        data = yaml.safe_load(_BASELINE_FILE.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            return data
    return {
        'model': 'swinir',
        'ssim_mean': 0.6645,
        'ssim_std': 0.01,
        'viable_threshold': 0.6545,
        'improvement_threshold': 0.6745,
    }


def _check_formula_runtime_compatibility(
    formula_spec: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not formula_spec:
        return None

    interface_analysis = analyze_formula_interface(formula_spec)
    if not isinstance(interface_analysis, dict):
        return None

    # The sandbox runtime can now forward model-produced loss_inputs through
    # sandbox_adapter / SandboxLossWrapper, so extra auxiliary variables should
    # not be treated as an immediate hard failure here. Let downstream
    # validators synthesize adapter configs and test the candidate end-to-end.
    if interface_analysis.get('auto_experiment_supported', False) and interface_analysis.get(
        'runtime_can_forward_model_loss_inputs',
        False,
    ):
        return None

    return {
        'passed': False,
        'error': 'formula_interface_incompatible',
        'detail': '; '.join(interface_analysis.get('issues', [])) or 'Formula interface incompatible',
        'fix_hint': (
            'Return {"pred": ..., "loss_inputs": {...}} from the model/adapter, '
            'or simplify the loss to pred/target/mask/params only.'
        ),
        'interface_analysis': interface_analysis,
    }


def _resolve_attempt_code(
    attempt_spec: Dict[str, Any],
    formula_spec: Optional[Dict[str, Any]],
    *,
    task_context_path: Optional[str] = None,
    analysis_plan_path: Optional[str] = None,
    output_code_path: Optional[str] = None,
    agent_service_url: Optional[str] = None,
    agent_api_key: Optional[str] = None,
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    kind = str(attempt_spec.get('kind', 'agent_code'))
    if kind == 'agent_code':
        inline_code = attempt_spec.get('code')
        if isinstance(inline_code, str) and inline_code.strip():
            return inline_code, 'agent_code', None

        code_path = attempt_spec.get('code_path')
        if isinstance(code_path, str) and code_path.strip():
            return Path(code_path).read_text(encoding='utf-8'), 'agent_code', None

        objective = attempt_spec.get('objective')
        if isinstance(objective, str) and objective.strip():
            if not task_context_path:
                raise ValueError('agent_code objective requires task_context_path')
            if not output_code_path:
                raise ValueError('agent_code objective requires output_code_path')
            generation = generate_candidate_loss(
                task_context_path=task_context_path,
                attempt_spec=attempt_spec,
                output_code_path=output_code_path,
                analysis_plan_path=analysis_plan_path,
                service_url=agent_service_url,
                api_key=agent_api_key,
            )
            if generation.get('status') != 'success':
                raise ValueError(str(generation.get('error') or 'agent code generation failed'))
            generated_code = Path(output_code_path).read_text(encoding='utf-8')
            return generated_code, 'agent_code', generation

        raise ValueError('agent_code attempt requires `code`, `code_path`, or `objective`')

    if kind == 'formula_variant':
        if not supports_formula_codegen(formula_spec):
            raise ValueError('formula_variant requested but loss_formula.json is not codegen-compatible')
        variant = str(attempt_spec.get('variant', 'faithful'))
        return generate_formula_loss_code(formula_spec, variant=variant), 'formula_variant', None

    raise ValueError(f'Unsupported attempt kind: {kind}')


def _build_failure_feedback(
    *,
    stop_layer: Optional[str],
    validation: Dict[str, Any],
    metrics: Optional[Dict[str, Any]],
    baseline: Dict[str, Any],
    repair_rounds: List[Dict[str, Any]],
    code_path: Path,
    formula_spec: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    override_dir = code_path.parent / 'sandbox_overrides'
    runtime_routing = build_runtime_routing_feedback(
        config_dir=_PROJECT_ROOT / 'sandbox' / 'configs',
        sandbox_override_dir=str(override_dir) if override_dir.is_dir() else None,
        formula_spec=formula_spec,
        model_configs=FULL_RUN_MODEL_CONFIGS,
        support_probe=lambda config_path: probe_model_output_extension_support(
            config_path=config_path,
            sandbox_override_dir=str(override_dir) if override_dir.is_dir() else None,
            formula_spec=formula_spec,
            python_executable=_PYTHON,
            project_root=_PROJECT_ROOT,
            pipeline_dir=_PIPELINE_DIR,
        ),
    )
    return _assemble_failure_feedback(
        stop_layer=stop_layer,
        validation=validation,
        metrics=metrics,
        baseline=baseline,
        repair_rounds=repair_rounds,
        runtime_routing=runtime_routing,
    )


def _run_pretraining_validations(
    code_path: Path,
    formula_spec: Optional[Dict[str, Any]],
    formula_spec_path: Optional[str],
) -> Tuple[Dict[str, Any], Optional[str]]:
    validation: Dict[str, Any] = {}
    stop_layer: Optional[str] = None

    formula_issue = _check_formula_runtime_compatibility(formula_spec)
    if formula_issue is not None:
        validation['formula_interface'] = formula_issue
        return validation, 'formula_interface'

    layer1 = validate_static(str(code_path))
    validation['layer1'] = layer1
    if not layer1.get('passed'):
        return validation, 'layer1'

    layer2 = validate_smoke(str(code_path), formula_spec_path=formula_spec_path)
    validation['layer2'] = layer2
    if not layer2.get('passed'):
        return validation, 'layer2'

    if formula_spec_path:
        alignment = validate_alignment(str(code_path), formula_spec_path)
        validation['formula_alignment'] = alignment
        if not alignment.get('passed'):
            return validation, 'formula_alignment'

    return validation, stop_layer


def _maybe_repair_candidate_code(
    *,
    attempt_spec: Dict[str, Any],
    source_kind: str,
    stop_layer: Optional[str],
    failure_feedback: Dict[str, Any],
    task_context_path: Optional[str],
    analysis_plan_path: Optional[str],
    code_path: Path,
    agent_service_url: Optional[str],
    agent_api_key: Optional[str],
) -> Optional[Dict[str, Any]]:
    if source_kind != 'agent_code':
        return None
    if stop_layer not in _REPAIRABLE_LAYERS:
        return None
    if not task_context_path:
        return None
    if not agent_service_url:
        return None

    return repair_candidate_loss(
        task_context_path=task_context_path,
        attempt_spec=attempt_spec,
        output_code_path=str(code_path),
        failure_feedback=failure_feedback,
        analysis_plan_path=analysis_plan_path,
        service_url=agent_service_url,
        api_key=agent_api_key,
    )


def _run_training_validations(
    code_path: Path,
    *,
    formula_spec_path: Optional[str],
    dataset_root: Optional[str],
    baseline: Dict[str, Any],
    run_training: bool,
) -> Tuple[Dict[str, Any], Optional[str], Dict[str, Any]]:
    validation: Dict[str, Any] = {}
    stop_layer: Optional[str] = None
    metrics: Dict[str, Any] = {}

    if not run_training:
        return validation, stop_layer, metrics

    layer3 = validate_single_model(
        str(code_path),
        formula_spec_path=formula_spec_path,
        dataset_root=dataset_root,
    )
    validation['layer3'] = layer3
    metrics = layer3.get('metrics') or layer3.get('partial_metrics') or {}
    if not layer3.get('passed'):
        return validation, 'layer3', metrics

    layer4 = validate_full_run(
        str(code_path),
        baseline_thresholds=baseline,
        formula_spec_path=formula_spec_path,
        dataset_root=dataset_root,
    )
    validation['layer4'] = layer4
    metrics = layer4.get('metrics') or layer4.get('partial_metrics') or metrics
    if not layer4.get('passed'):
        return validation, 'layer4', metrics

    primary_metric_name, primary_metric_value = _extract_primary_metric(metrics)
    viable_threshold = baseline.get('viable_threshold')
    if (
        primary_metric_name
        and isinstance(primary_metric_value, (int, float))
        and isinstance(viable_threshold, (int, float))
        and float(primary_metric_value) < float(viable_threshold)
    ):
        validation['layer4'] = {
            **layer4,
            'passed': False,
            'error': 'below_viable_threshold',
            'detail': (
                f'{primary_metric_name}={float(primary_metric_value):.6f} is below '
                f'viable_threshold={float(viable_threshold):.6f}'
            ),
            'metrics': metrics,
            'viable_threshold': float(viable_threshold),
            'improvement_threshold': (
                float(baseline['improvement_threshold'])
                if isinstance(baseline.get('improvement_threshold'), (int, float))
                else None
            ),
        }
        return validation, 'layer4', metrics

    return validation, stop_layer, metrics


def _evaluate_candidate(
    code_path: Path,
    *,
    formula_spec: Optional[Dict[str, Any]],
    formula_spec_path: Optional[str],
    dataset_root: Optional[str],
    baseline: Dict[str, Any],
    run_training: bool,
) -> Tuple[Dict[str, Any], Optional[str], Dict[str, Any]]:
    validation, stop_layer = _run_pretraining_validations(
        code_path,
        formula_spec,
        formula_spec_path,
    )
    metrics: Dict[str, Any] = {}

    if stop_layer is not None:
        return validation, stop_layer, metrics

    training_validation, training_stop_layer, training_metrics = _run_training_validations(
        code_path,
        formula_spec_path=formula_spec_path,
        dataset_root=dataset_root,
        baseline=baseline,
        run_training=run_training,
    )
    validation.update(training_validation)
    return validation, training_stop_layer, training_metrics


def execute_attempt(
    paper_slug: str,
    attempt_id: int,
    attempt_spec: Dict[str, Any],
    *,
    dataset_root: Optional[str] = None,
    output_dir: Optional[str] = None,
    task_context_path: Optional[str] = None,
    analysis_plan_path: Optional[str] = None,
    agent_service_url: Optional[str] = None,
    agent_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    experiment_dir = ensure_experiment_dir(paper_slug, output_dir=output_dir)
    baseline = _load_baseline_thresholds()
    formula_path = experiment_dir / 'loss_formula.json'
    formula_spec = _load_json(formula_path)
    formula_spec_path = str(formula_path) if formula_spec else None

    attempt_dir = write_attempt_artifacts(
        experiment_dir,
        attempt_id,
        attempt_spec=attempt_spec,
    )
    code_path = attempt_dir / 'candidate_loss.py'

    append_trajectory_event(
        paper_slug,
        'attempt_started',
        {
            'attempt_id': attempt_id,
            'name': attempt_spec.get('name', f'attempt_{attempt_id}'),
            'kind': str(attempt_spec.get('kind', 'agent_code')),
            'dataset_root': dataset_root,
        },
        output_dir=output_dir,
    )

    try:
        code, source_kind, generation_info = _resolve_attempt_code(
            attempt_spec,
            formula_spec,
            task_context_path=task_context_path,
            analysis_plan_path=analysis_plan_path,
            output_code_path=str(code_path),
            agent_service_url=agent_service_url,
            agent_api_key=agent_api_key,
        )
    except Exception as exc:
        result = _build_code_generation_failure_result(
            attempt_id=attempt_id,
            attempt_spec=attempt_spec,
            attempt_dir=attempt_dir,
            code_path=code_path,
            baseline=baseline,
            max_agent_repair_rounds=_MAX_AGENT_REPAIR_ROUNDS,
            error_text=str(exc),
        )
        write_attempt_artifacts(
            experiment_dir,
            attempt_id,
            result=result,
        )
        append_trajectory_event(
            paper_slug,
            'attempt_finished',
            {
                'attempt_id': attempt_id,
                'status': result['status'],
                'stop_layer': 'code_generation',
                'reward_summary': result['reward_summary'],
            },
            output_dir=output_dir,
        )
        return result

    write_attempt_artifacts(
        experiment_dir,
        attempt_id,
        code=code,
    )

    run_training = bool(attempt_spec.get('run_training', True))
    validation, stop_layer, metrics = _evaluate_candidate(
        code_path,
        formula_spec=formula_spec,
        formula_spec_path=formula_spec_path,
        dataset_root=dataset_root,
        baseline=baseline,
        run_training=run_training,
    )

    repair_rounds: List[Dict[str, Any]] = []
    repair_info: Optional[Dict[str, Any]] = None

    while stop_layer in _REPAIRABLE_LAYERS and len(repair_rounds) < _MAX_AGENT_REPAIR_ROUNDS:
        failure_feedback = _build_failure_feedback(
            stop_layer=stop_layer,
            validation=validation,
            metrics=metrics,
            baseline=baseline,
            repair_rounds=repair_rounds,
            code_path=code_path,
            formula_spec=formula_spec,
        )
        round_number = len(repair_rounds) + 1
        pre_repair_code_path = _snapshot_path(
            attempt_dir,
            'candidate_loss_before_repair',
            round_number,
            '.py',
        )
        shutil.copy2(code_path, pre_repair_code_path)
        append_trajectory_event(
            paper_slug,
            'attempt_repair_started',
            {
                'attempt_id': attempt_id,
                'round': round_number,
                'stop_layer': stop_layer,
                'error': failure_feedback.get('error'),
            },
            output_dir=output_dir,
        )
        repair_info = _maybe_repair_candidate_code(
            attempt_spec=attempt_spec,
            source_kind=source_kind,
            stop_layer=stop_layer,
            failure_feedback=failure_feedback,
            task_context_path=task_context_path,
            analysis_plan_path=analysis_plan_path,
            code_path=code_path,
            agent_service_url=agent_service_url,
            agent_api_key=agent_api_key,
        )
        repair_record = _build_initial_repair_record(
            round_number=round_number,
            trigger_stop_layer=stop_layer,
            failure_feedback=failure_feedback,
            repair_info=repair_info,
            pre_repair_code_path=pre_repair_code_path,
        )

        if not repair_info:
            append_trajectory_event(
                paper_slug,
                'attempt_repair_finished',
                {
                    'attempt_id': attempt_id,
                    'round': round_number,
                    'status': 'skipped',
                    'reason': 'repair_unavailable',
                },
                output_dir=output_dir,
            )
            break

        repair_response_path = repair_info.get('agent_response_path')
        if isinstance(repair_response_path, str) and repair_response_path.strip():
            response_source = Path(repair_response_path)
            if response_source.exists():
                response_snapshot_path = _snapshot_path(
                    attempt_dir,
                    'agent_code_repair_response',
                    round_number,
                    '.json',
                )
                shutil.copy2(response_source, response_snapshot_path)
                _attach_repair_artifact(
                    repair_record,
                    key='repair_response_path',
                    path=response_snapshot_path,
                )

        post_repair_code_path = _snapshot_path(
            attempt_dir,
            'candidate_loss_after_repair',
            round_number,
            '.py',
        )
        shutil.copy2(code_path, post_repair_code_path)
        _attach_repair_artifact(
            repair_record,
            key='post_repair_code_path',
            path=post_repair_code_path,
        )

        if repair_info.get('status') != 'success':
            repair_record['status'] = 'error'
            repair_rounds.append(repair_record)
            append_trajectory_event(
                paper_slug,
                'attempt_repair_finished',
                {
                    'attempt_id': attempt_id,
                    'round': round_number,
                    'status': 'error',
                    'error': repair_info.get('error'),
                },
                output_dir=output_dir,
            )
            break

        validation, stop_layer, metrics = _evaluate_candidate(
            code_path,
            formula_spec=formula_spec,
            formula_spec_path=formula_spec_path,
            dataset_root=dataset_root,
            baseline=baseline,
            run_training=run_training,
        )
        repair_record.update(
            {
                'post_validation': validation,
                'post_stop_layer': stop_layer,
                'post_error': _validation_error_text(stop_layer, validation),
                'post_metrics': metrics,
                'post_baseline_delta': _compute_baseline_delta(metrics, baseline),
            }
        )
        if _should_revert_repair(
            trigger_stop_layer=repair_record['trigger_stop_layer'],
            post_stop_layer=stop_layer,
            layer_order=_LAYER_ORDER,
        ):
            shutil.copy2(pre_repair_code_path, code_path)
            restored_code_path = _snapshot_path(
                attempt_dir,
                'candidate_loss_restored',
                round_number,
                '.py',
            )
            shutil.copy2(code_path, restored_code_path)
            _mark_repair_reverted(
                repair_record,
                restored_code_path=restored_code_path,
            )
            repair_rounds.append(repair_record)
            append_trajectory_event(
                paper_slug,
                'attempt_repair_finished',
                {
                    'attempt_id': attempt_id,
                    'round': round_number,
                    'status': 'reverted_regression',
                    'trigger_stop_layer': repair_record['trigger_stop_layer'],
                    'post_stop_layer': stop_layer,
                },
                output_dir=output_dir,
            )
            validation = repair_record['failure_feedback']['validation']
            metrics = repair_record['failure_feedback'].get('metrics') or metrics
            stop_layer = repair_record['trigger_stop_layer']
            continue

        repair_record['status'] = 'success'
        repair_rounds.append(repair_record)
        append_trajectory_event(
            paper_slug,
            'attempt_repair_finished',
            {
                'attempt_id': attempt_id,
                'round': round_number,
                'status': 'success',
                'post_stop_layer': stop_layer,
                'post_baseline_delta': repair_record.get('post_baseline_delta'),
            },
            output_dir=output_dir,
        )

    result = _build_attempt_result(
        attempt_id=attempt_id,
        attempt_spec=attempt_spec,
        source_kind=source_kind,
        attempt_dir=attempt_dir,
        code_path=code_path,
        validation=validation,
        stop_layer=stop_layer,
        metrics=metrics,
        baseline=baseline,
        repair_rounds=repair_rounds,
        run_training=run_training,
        formula_spec_path=formula_spec_path,
        generation_info=generation_info,
        repair_info=repair_info,
        max_agent_repair_rounds=_MAX_AGENT_REPAIR_ROUNDS,
        validation_error_text_fn=_validation_error_text,
        compute_baseline_delta_fn=_compute_baseline_delta,
        extract_primary_metric_fn=_extract_primary_metric,
    )

    write_attempt_artifacts(
        experiment_dir,
        attempt_id,
        result=result,
    )
    append_trajectory_event(
        paper_slug,
        'attempt_finished',
        {
            'attempt_id': attempt_id,
            'status': result['status'],
            'stop_layer': stop_layer,
            'reward_summary': result['reward_summary'],
        },
        output_dir=output_dir,
    )
    return result
