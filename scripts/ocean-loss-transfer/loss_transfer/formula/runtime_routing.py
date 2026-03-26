"""
@file runtime_routing.py
@description Shared runtime-routing helpers for adapter fallback and model-output extension support.
@author OpenAI Codex
@date 2026-03-25
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml

from loss_transfer.formula.formula_interface_analysis import analyze_formula_interface


FULL_RUN_MODEL_CONFIGS = {
    'SwinIR': 'swinir.yaml',
    'EDSR': 'edsr.yaml',
    'FNO2d': 'fno2d.yaml',
    'UNet2d': 'unet2d.yaml',
}

MODEL_OUTPUT_EXTENSION_POLICY = (
    'Validators prefer copied-model output extension for models whose attempt-scoped '
    'constructors support output_aux_loss_inputs. Models without that support remain on '
    'sandbox_adapter fallback until the copied model path is upgraded.'
)


def get_formula_interface_analysis(formula_spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(formula_spec, dict) or not formula_spec:
        return {}
    analysis = analyze_formula_interface(formula_spec)
    return analysis if isinstance(analysis, dict) else {}


def formula_requires_model_output_extension(formula_spec: Optional[Dict[str, Any]]) -> bool:
    return bool(get_formula_interface_analysis(formula_spec).get('requires_model_output_extension', False))


def formula_requires_sandbox_adapter(formula_spec: Optional[Dict[str, Any]]) -> bool:
    analysis = get_formula_interface_analysis(formula_spec)
    if not analysis:
        return False
    if analysis.get('requires_model_output_extension', False):
        return False
    return bool(analysis.get('extra_required_variables'))


def needs_temporary_runtime_config(
    formula_spec: Optional[Dict[str, Any]],
    dataset_root: Optional[str] = None,
) -> bool:
    return (
        formula_requires_sandbox_adapter(formula_spec)
        or formula_requires_model_output_extension(formula_spec)
        or bool(dataset_root)
    )


def _build_model_probe_env(
    *,
    pipeline_dir: Path,
    sandbox_override_dir: Optional[str],
) -> Dict[str, str]:
    env = os.environ.copy()
    pipeline_path = str(pipeline_dir)
    existing_pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f'{pipeline_path}:{existing_pythonpath}' if existing_pythonpath else pipeline_path
    if sandbox_override_dir:
        env['SANDBOX_OVERRIDE_DIR'] = str(Path(sandbox_override_dir).resolve())
    return env


def probe_model_output_extension_support(
    *,
    config_path: Path,
    sandbox_override_dir: Optional[str],
    formula_spec: Optional[Dict[str, Any]],
    python_executable: str,
    project_root: Path,
    pipeline_dir: Path,
    timeout: int = 60,
    subprocess_run: Callable[..., Any] = subprocess.run,
) -> bool:
    if not formula_requires_model_output_extension(formula_spec):
        return False
    if not sandbox_override_dir:
        return False

    override_path = Path(sandbox_override_dir)
    if not override_path.is_dir():
        return False

    try:
        base_config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    except OSError:
        return False
    if not isinstance(base_config, dict):
        return False

    model_cfg = base_config.get('model', {})
    if not isinstance(model_cfg, dict):
        return False
    model_name = model_cfg.get('name')
    if not isinstance(model_name, str) or not model_name.strip():
        return False

    probe_code = (
        'import inspect, json, os, sys\n'
        'project_root = os.path.abspath(sys.argv[1])\n'
        'override_dir = os.path.abspath(sys.argv[2])\n'
        'model_name = sys.argv[3]\n'
        "pipeline_dir = os.path.join(project_root, 'scripts', 'ocean-SR-training-masked')\n"
        "sandbox_dir = os.path.join(project_root, 'sandbox')\n"
        'if pipeline_dir not in sys.path:\n'
        '    sys.path.insert(0, pipeline_dir)\n'
        'if sandbox_dir not in sys.path:\n'
        '    sys.path.insert(0, sandbox_dir)\n'
        'sys.path.insert(0, override_dir)\n'
        'from models import _model_dict\n'
        'factory = _model_dict.get(model_name)\n'
        'supports = False\n'
        'resolved_file = ""\n'
        'if callable(factory):\n'
        '    module = inspect.getmodule(factory)\n'
        "    resolved_file = getattr(module, '__file__', '') or ''\n"
        '    try:\n'
        '        signature = inspect.signature(factory)\n'
        '        params = signature.parameters.values()\n'
        "        supports = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params) or 'output_aux_loss_inputs' in signature.parameters\n"
        '    except (TypeError, ValueError):\n'
        '        supports = False\n'
        'print(json.dumps({"supports": supports, "resolved_file": resolved_file}))\n'
    )

    env = _build_model_probe_env(
        pipeline_dir=pipeline_dir,
        sandbox_override_dir=str(override_path.resolve()),
    )

    try:
        result = subprocess_run(
            [python_executable, '-c', probe_code, str(project_root), str(override_path.resolve()), model_name],
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False

    if getattr(result, 'returncode', 1) != 0:
        return False

    try:
        payload = json.loads((getattr(result, 'stdout', '') or '').strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return False

    return bool(payload.get('supports'))


def collect_model_output_extension_support(
    *,
    config_dir: Path,
    sandbox_override_dir: Optional[str],
    formula_spec: Optional[Dict[str, Any]],
    model_configs: Optional[Dict[str, str]] = None,
    support_probe: Optional[Callable[[Path], bool]] = None,
) -> Dict[str, bool]:
    if not formula_requires_model_output_extension(formula_spec):
        return {}

    configs = model_configs or FULL_RUN_MODEL_CONFIGS
    if not sandbox_override_dir:
        return {model_name.lower(): False for model_name in configs}

    support_by_model: Dict[str, bool] = {}
    for model_name, config_name in configs.items():
        config_path = config_dir / config_name
        support_by_model[model_name.lower()] = bool(
            support_probe(config_path) if support_probe is not None else False
        )
    return support_by_model


def build_runtime_routing_feedback(
    *,
    config_dir: Path,
    sandbox_override_dir: Optional[str],
    formula_spec: Optional[Dict[str, Any]],
    model_configs: Optional[Dict[str, str]] = None,
    support_probe: Optional[Callable[[Path], bool]] = None,
) -> Optional[Dict[str, Any]]:
    if not formula_requires_model_output_extension(formula_spec):
        return None

    support_by_model = collect_model_output_extension_support(
        config_dir=config_dir,
        sandbox_override_dir=sandbox_override_dir,
        formula_spec=formula_spec,
        model_configs=model_configs,
        support_probe=support_probe,
    )
    return {
        'requires_model_output_extension': True,
        'current_model_output_extension_support': support_by_model,
        'policy': MODEL_OUTPUT_EXTENSION_POLICY,
        'sandbox_override_dir': sandbox_override_dir,
    }
