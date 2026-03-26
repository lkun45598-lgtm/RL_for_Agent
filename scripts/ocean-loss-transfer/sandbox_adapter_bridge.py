"""
@file sandbox_adapter_bridge.py
@description 将 loss_formula.json 中的额外变量需求桥接到 sandbox adapter / validator
@author Leizheng
@date 2026-03-24
@version 1.0.0
"""

from __future__ import annotations

import json
import glob
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from formula_interface_analysis import analyze_formula_interface
from runtime_routing import formula_requires_model_output_extension, formula_requires_sandbox_adapter


def _infer_dataset_metadata(dataset_root: str) -> Dict[str, Any]:
    root = Path(dataset_root)
    train_hr_root = root / "train" / "hr"
    train_lr_root = root / "train" / "lr"

    if not train_hr_root.is_dir():
        raise FileNotFoundError(f"Dataset train/hr directory not found: {train_hr_root}")
    if not train_lr_root.is_dir():
        raise FileNotFoundError(f"Dataset train/lr directory not found: {train_lr_root}")

    dyn_vars = sorted(
        entry.name for entry in train_hr_root.iterdir()
        if entry.is_dir()
    )
    if not dyn_vars:
        raise ValueError(f"No dynamic variable directories found under: {train_hr_root}")

    first_var = dyn_vars[0]
    hr_files = sorted(glob.glob(str(train_hr_root / first_var / "*.npy")))
    lr_files = sorted(glob.glob(str(train_lr_root / first_var / "*.npy")))
    if not hr_files:
        raise FileNotFoundError(f"No HR .npy files found under: {train_hr_root / first_var}")
    if not lr_files:
        raise FileNotFoundError(f"No LR .npy files found under: {train_lr_root / first_var}")

    hr_shape = list(np.load(hr_files[0]).shape[:2])
    lr_shape = list(np.load(lr_files[0]).shape[:2])
    if len(hr_shape) != 2 or len(lr_shape) != 2:
        raise ValueError(f"Expected 2D HR/LR arrays, got HR={hr_shape}, LR={lr_shape}")
    if lr_shape[0] == 0 or lr_shape[1] == 0:
        raise ValueError(f"Invalid LR shape inferred from dataset: {lr_shape}")

    sample_factor_h = hr_shape[0] // lr_shape[0]
    sample_factor_w = hr_shape[1] // lr_shape[1]
    if sample_factor_h != sample_factor_w or sample_factor_h <= 0:
        raise ValueError(f"Inconsistent HR/LR scale inferred from dataset: HR={hr_shape}, LR={lr_shape}")

    return {
        "dyn_vars": dyn_vars,
        "shape": hr_shape,
        "sample_factor": int(sample_factor_h),
        "num_channels": len(dyn_vars),
    }


def load_formula_spec(formula_spec_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not formula_spec_path:
        return None

    spec_path = Path(formula_spec_path)
    if not spec_path.exists():
        return None

    try:
        data = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    return data if isinstance(data, dict) else None


def requires_sandbox_adapter(formula_spec: Optional[Dict[str, Any]]) -> bool:
    return formula_requires_sandbox_adapter(formula_spec)


def requires_model_output_extension(formula_spec: Optional[Dict[str, Any]]) -> bool:
    return formula_requires_model_output_extension(formula_spec)


def infer_adapter_head_template(variable_name: str) -> Dict[str, Any]:
    lowered = variable_name.strip().lower()

    if lowered.startswith("log_"):
        return {
            "activation": "tanh",
            "output_scale": 4.0,
            "bias_init": 0.0,
            "detach_input": True,
            "zero_init": True,
        }
    if lowered == "weight" or lowered.endswith("_weight") or lowered.endswith("_logits"):
        return {
            "activation": "tanh",
            "output_scale": 4.0,
            "bias_init": 0.0,
            "detach_input": True,
            "zero_init": True,
        }
    if lowered in {"sigma", "variance", "var", "uncertainty"} or lowered.endswith("_sigma") or lowered.endswith("_var"):
        return {
            "activation": "softplus",
            "bias_init": 0.0,
            "detach_input": True,
            "zero_init": True,
        }
    if lowered == "confidence" or lowered.endswith("_confidence"):
        return {
            "activation": "sigmoid",
            "bias_init": 0.0,
            "detach_input": True,
            "zero_init": True,
        }
    return {
        "activation": "none",
        "bias_init": 0.0,
        "detach_input": True,
        "zero_init": True,
    }


def draft_adapter_heads(extra_required_variables: list[str]) -> Dict[str, Dict[str, Any]]:
    return {
        variable_name: infer_adapter_head_template(variable_name)
        for variable_name in extra_required_variables
        if isinstance(variable_name, str) and variable_name.strip()
    }


def _normalize_head_config(
    variable_name: str,
    raw_head_cfg: Optional[Dict[str, Any]],
    pred_channels: int,
) -> Dict[str, Any]:
    merged = dict(infer_adapter_head_template(variable_name))
    if isinstance(raw_head_cfg, dict):
        merged.update(raw_head_cfg)

    out_channels = merged.get("out_channels")
    merged["out_channels"] = int(out_channels) if out_channels not in (None, "") else int(pred_channels)

    if "activation" in merged and merged["activation"] is not None:
        merged["activation"] = str(merged["activation"])
    if "bias_init" in merged and merged["bias_init"] is not None:
        merged["bias_init"] = float(merged["bias_init"])
    if "hidden_channels" in merged and merged["hidden_channels"] is not None:
        merged["hidden_channels"] = int(merged["hidden_channels"])
    if "output_scale" in merged and merged["output_scale"] is not None:
        merged["output_scale"] = float(merged["output_scale"])
    if "output_shift" in merged and merged["output_shift"] is not None:
        merged["output_shift"] = float(merged["output_shift"])
    if "detach_input" in merged:
        merged["detach_input"] = bool(merged["detach_input"])
    if "zero_init" in merged:
        merged["zero_init"] = bool(merged["zero_init"])

    return merged


def resolve_adapter_heads(
    formula_spec: Optional[Dict[str, Any]],
    pred_channels: int,
) -> Dict[str, Dict[str, Any]]:
    if not formula_spec:
        return {}

    analysis = analyze_formula_interface(formula_spec)
    extra_required_variables = analysis.get("extra_required_variables", [])
    if not extra_required_variables:
        return {}

    raw_heads = formula_spec.get("adapter_heads", {})
    if not isinstance(raw_heads, dict):
        raw_heads = {}

    resolved: Dict[str, Dict[str, Any]] = {}
    for variable_name in extra_required_variables:
        if not isinstance(variable_name, str) or not variable_name.strip():
            continue
        raw_cfg = raw_heads.get(variable_name)
        resolved[variable_name] = _normalize_head_config(
            variable_name=variable_name,
            raw_head_cfg=raw_cfg if isinstance(raw_cfg, dict) else None,
            pred_channels=pred_channels,
        )
    return resolved


def build_sandbox_adapter_config(
    formula_spec: Optional[Dict[str, Any]],
    pred_channels: int,
) -> Optional[Dict[str, Any]]:
    heads = resolve_adapter_heads(formula_spec, pred_channels=pred_channels)
    if not heads:
        return None

    raw_hidden_channels = 32
    if formula_spec and formula_spec.get("adapter_hidden_channels") is not None:
        raw_hidden_channels = formula_spec["adapter_hidden_channels"]

    try:
        hidden_channels = int(raw_hidden_channels)
    except (TypeError, ValueError):
        hidden_channels = 32

    return {
        "enabled": True,
        "pred_channels": int(pred_channels),
        "hidden_channels": hidden_channels,
        "heads": heads,
    }


def _dummy_fill_value(head_cfg: Dict[str, Any]) -> float:
    activation = str(head_cfg.get("activation", "none")).lower()
    if activation == "sigmoid":
        return 0.5
    if activation == "softplus":
        return 0.5
    if activation == "exp":
        return 1.0
    if activation == "tanh":
        return 0.0

    bias_init = head_cfg.get("bias_init")
    if isinstance(bias_init, (int, float)):
        return float(bias_init)
    return 0.0


def build_smoke_loss_kwargs(formula_spec: Optional[Dict[str, Any]], pred):
    if not formula_spec:
        return {}
    if getattr(pred, "dim", lambda: 0)() != 4:
        return {}

    import torch

    pred_channels = int(pred.shape[-1])
    heads = resolve_adapter_heads(formula_spec, pred_channels=pred_channels)
    kwargs: Dict[str, Any] = {}
    for variable_name, head_cfg in heads.items():
        out_channels = int(head_cfg.get("out_channels", pred_channels))
        fill_value = _dummy_fill_value(head_cfg)
        kwargs[variable_name] = torch.full(
            (pred.shape[0], pred.shape[1], pred.shape[2], out_channels),
            fill_value,
            dtype=pred.dtype,
            device=pred.device,
        )
    return kwargs


def build_config_with_adapter(
    base_config: Dict[str, Any],
    formula_spec: Optional[Dict[str, Any]],
    dataset_root: Optional[str] = None,
    prefer_model_output_extension: bool = False,
) -> Dict[str, Any]:
    config = dict(base_config)
    model_cfg = config.get("model", {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    else:
        model_cfg = dict(model_cfg)

    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        data_cfg = {}
    else:
        data_cfg = dict(data_cfg)

    train_cfg = config.get("train", {})
    if not isinstance(train_cfg, dict):
        train_cfg = {}
    else:
        train_cfg = dict(train_cfg)

    dataset_meta: Dict[str, Any] = {}
    if dataset_root:
        dataset_meta = _infer_dataset_metadata(dataset_root)

    pred_channels = int(model_cfg.get("out_channels", model_cfg.get("out_dim", 1)))
    model_requires_output_extension = (
        requires_model_output_extension(formula_spec) and prefer_model_output_extension
    )
    if dataset_meta.get("num_channels"):
        channel_count = int(dataset_meta["num_channels"])
        pred_channels = channel_count
        if "in_channels" in model_cfg:
            model_cfg["in_channels"] = channel_count
        if "out_channels" in model_cfg:
            model_cfg["out_channels"] = channel_count
        if "out_dim" in model_cfg:
            model_cfg["out_dim"] = channel_count
    if model_requires_output_extension:
        model_cfg["output_aux_loss_inputs"] = True
    adapter_cfg = build_sandbox_adapter_config(formula_spec, pred_channels=pred_channels)
    if adapter_cfg and not model_requires_output_extension:
        config["sandbox_adapter"] = adapter_cfg
    else:
        config.pop("sandbox_adapter", None)
    if dataset_root:
        data_cfg["dataset_root"] = dataset_root
    if dataset_meta:
        data_cfg["dyn_vars"] = dataset_meta["dyn_vars"]
        data_cfg["shape"] = dataset_meta["shape"]
        data_cfg["sample_factor"] = dataset_meta["sample_factor"]
    # Auto-experiment validators run inside restricted sandboxes where
    # multi-process DataLoader workers can fail on FD sharing.
    data_cfg["num_workers"] = 0
    # Default to full precision for transferred-loss experiments because
    # auxiliary adapter heads can overflow under AMP before the loss itself diverges.
    train_cfg["use_amp"] = False
    if data_cfg:
        config["data"] = data_cfg
    if model_cfg:
        config["model"] = model_cfg
    if train_cfg:
        config["train"] = train_cfg
    return config


def write_config_with_adapter(
    base_config_path: str,
    formula_spec: Optional[Dict[str, Any]],
    output_path: str,
    dataset_root: Optional[str] = None,
    prefer_model_output_extension: bool = False,
) -> Dict[str, Any]:
    config_path = Path(base_config_path)
    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(base_config, dict):
        raise ValueError(f"Invalid sandbox config: {base_config_path}")

    final_config = build_config_with_adapter(
        base_config,
        formula_spec=formula_spec,
        dataset_root=dataset_root,
        prefer_model_output_extension=prefer_model_output_extension,
    )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(final_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    return {
        "written_path": str(out_path),
        "sandbox_adapter": final_config.get("sandbox_adapter"),
    }
