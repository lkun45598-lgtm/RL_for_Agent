"""
@file write_loss_formula.py
@description 验证并写入 Loss Formula Spec（LaTeX + params JSON + symbol↔variable 双射映射）
@author Leizheng
@date 2026-03-24
@version 1.0.0
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from formula_interface_analysis import analyze_formula_interface, ensure_formula_interface_analysis


_PY_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED_ADAPTER_ACTIVATIONS = {"none", "sigmoid", "tanh", "softplus", "exp"}


def _is_json_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def _validate_params(params: Any) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if params is None:
        return True, errors
    if not isinstance(params, dict):
        return False, ["params must be an object/dict"]

    for k, v in params.items():
        if not isinstance(k, str) or not k.strip():
            errors.append("params key must be non-empty string")
            continue
        if not _PY_IDENT_RE.match(k):
            errors.append(f'params key "{k}" is not a valid python identifier (recommend snake_case)')
        # allow nested json but keep it simple for now
        if _is_json_scalar(v):
            continue
        if isinstance(v, list) and all(_is_json_scalar(x) for x in v):
            continue
        errors.append(f'params["{k}"] must be a JSON scalar or list of scalars')
    return len(errors) == 0, errors


def _validate_latex(latex: Any) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if isinstance(latex, str):
        if not latex.strip():
            errors.append("latex must be non-empty")
        return len(errors) == 0, errors
    if isinstance(latex, list):
        if not latex:
            errors.append("latex list must be non-empty")
            return False, errors
        for i, s in enumerate(latex):
            if not isinstance(s, str) or not s.strip():
                errors.append(f"latex[{i}] must be a non-empty string")
        return len(errors) == 0, errors
    return False, ["latex must be a string or list of strings"]


def _validate_symbol_map(symbol_map: Any, params: Optional[Dict[str, Any]]) -> Tuple[bool, List[str], List[str]]:
    """
    Returns:
      ok, errors, warnings
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(symbol_map, dict):
        return False, ["symbol_map must be an object/dict"], warnings

    inv: Dict[str, str] = {}
    values: List[str] = []
    for k, v in symbol_map.items():
        if not isinstance(k, str) or not k.strip():
            errors.append("symbol_map key (latex symbol) must be non-empty string")
            continue
        if not isinstance(v, str) or not v.strip():
            errors.append(f'symbol_map["{k}"] value must be non-empty string')
            continue
        if not _PY_IDENT_RE.match(v):
            errors.append(f'symbol_map["{k}"] value "{v}" must be a python identifier')
            continue
        if v in inv:
            errors.append(f'symbol_map is not 1:1: both "{inv[v]}" and "{k}" map to variable "{v}"')
            continue
        inv[v] = k
        values.append(v)

    # Required variables for this repo's target loss interface
    if "pred" not in values:
        errors.append('symbol_map must include a mapping to variable "pred"')
    if "target" not in values:
        errors.append('symbol_map must include a mapping to variable "target"')

    # Optional variable
    # mask is optional, but if latex uses a mask concept it's good to map it.

    # Warnings for unknown variables: not in interface nor params
    allowed_base = {"pred", "target", "mask"}
    allowed_params = set(params.keys()) if isinstance(params, dict) else set()
    for v in values:
        if v in allowed_base:
            continue
        if v in allowed_params:
            continue
        warnings.append(f'symbol_map value "{v}" is not in (pred/target/mask) nor in params keys; target interface may not provide it')

    return len(errors) == 0, errors, warnings


def _validate_adapter_heads(adapter_heads: Any) -> Tuple[bool, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if adapter_heads is None:
        return True, errors, warnings
    if not isinstance(adapter_heads, dict):
        return False, ["adapter_heads must be an object/dict"], warnings

    for variable_name, cfg in adapter_heads.items():
        if not isinstance(variable_name, str) or not variable_name.strip():
            errors.append("adapter_heads key must be a non-empty string")
            continue
        if not _PY_IDENT_RE.match(variable_name):
            errors.append(f'adapter_heads key "{variable_name}" must be a python identifier')
            continue
        if not isinstance(cfg, dict):
            errors.append(f'adapter_heads["{variable_name}"] must be an object/dict')
            continue

        if "out_channels" in cfg:
            out_channels = cfg.get("out_channels")
            if not isinstance(out_channels, int) or out_channels <= 0:
                errors.append(f'adapter_heads["{variable_name}"].out_channels must be a positive integer')

        if "hidden_channels" in cfg:
            hidden_channels = cfg.get("hidden_channels")
            if not isinstance(hidden_channels, int) or hidden_channels <= 0:
                errors.append(f'adapter_heads["{variable_name}"].hidden_channels must be a positive integer')

        if "activation" in cfg:
            activation = cfg.get("activation")
            if not isinstance(activation, str) or activation not in _ALLOWED_ADAPTER_ACTIVATIONS:
                errors.append(
                    f'adapter_heads["{variable_name}"].activation must be one of '
                    + ", ".join(sorted(_ALLOWED_ADAPTER_ACTIVATIONS))
                )

        if "bias_init" in cfg and cfg.get("bias_init") is not None and not isinstance(cfg.get("bias_init"), (int, float)):
            errors.append(f'adapter_heads["{variable_name}"].bias_init must be numeric or null')

    return len(errors) == 0, errors, warnings


def validate_formula_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate formula spec.

    Expected schema (JSON):
      {
        "latex": "..." | ["...", ...],
        "params": { ... },
        "symbol_map": { "<latex_symbol>": "<python_var_name>", ... },
        "notes": "..." (optional),
        "sources": { ... } (optional)
      }
    """
    errors: List[str] = []
    warnings: List[str] = []

    latex = spec.get("latex")
    ok_latex, latex_errs = _validate_latex(latex)
    if not ok_latex:
        errors.extend(latex_errs)

    params = spec.get("params", {})
    ok_params, params_errs = _validate_params(params)
    if not ok_params:
        errors.extend(params_errs)

    symbol_map = spec.get("symbol_map")
    ok_map, map_errs, map_warns = _validate_symbol_map(symbol_map, params if isinstance(params, dict) else None)
    if not ok_map:
        errors.extend(map_errs)
    warnings.extend(map_warns)

    adapter_heads = spec.get("adapter_heads")
    ok_adapter, adapter_errs, adapter_warns = _validate_adapter_heads(adapter_heads)
    if not ok_adapter:
        errors.extend(adapter_errs)
    warnings.extend(adapter_warns)

    # Optional notes
    if "notes" in spec and spec["notes"] is not None and not isinstance(spec["notes"], str):
        errors.append("notes must be a string if provided")
    if "interface_analysis" in spec and not isinstance(spec["interface_analysis"], dict):
        errors.append("interface_analysis must be an object if provided")
    if "adapter_hidden_channels" in spec:
        hidden_channels = spec["adapter_hidden_channels"]
        if not isinstance(hidden_channels, int) or hidden_channels <= 0:
            errors.append("adapter_hidden_channels must be a positive integer if provided")

    interface_analysis = analyze_formula_interface(spec)
    warnings.extend(interface_analysis.get("issues", []))

    status = "ok" if not errors else "error"
    if status == "ok" and warnings:
        status = "warning"

    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "interface_analysis": interface_analysis,
    }


def write_formula_spec(formula_json: str, output_path: str, validate: bool = True) -> Dict[str, Any]:
    """
    Validate and write formula spec JSON to output_path.
    `formula_json` can be JSON string or path to a file containing JSON.
    """
    raw = formula_json
    p = Path(formula_json)
    if p.exists():
        raw = p.read_text(encoding="utf-8")

    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"status": "error", "error": f"Invalid JSON: {e}"}

    if not isinstance(spec, dict):
        return {"status": "error", "error": "Formula spec must be a JSON object"}

    spec = ensure_formula_interface_analysis(spec)

    validation: Dict[str, Any] = {"status": "skipped"}
    if validate:
        validation = validate_formula_spec(spec)
        if validation["status"] == "error":
            return {
                "status": "error",
                "validation": validation,
                "written_path": None,
            }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "status": "success" if validation.get("status") in ("ok", "skipped") else "warning",
        "validation": validation,
        "written_path": str(out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and write Loss Formula Spec JSON")
    parser.add_argument("--formula_json", required=True, help="JSON string or path to JSON file")
    parser.add_argument("--output_path", required=True, help="Output path for formula spec JSON")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    args = parser.parse_args()

    result = write_formula_spec(args.formula_json, args.output_path, validate=not args.no_validate)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
