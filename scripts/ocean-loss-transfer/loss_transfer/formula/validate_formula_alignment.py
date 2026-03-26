"""
@file validate_formula_alignment.py
@description 校验 loss 代码与 formula spec（LaTeX/params/symbol_map）的一致性，用于“符号↔变量名”强约束把关
@author Leizheng
@date 2026-03-24
@version 1.0.0
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from loss_transfer.formula.formula_interface_analysis import analyze_formula_interface


def _load_formula_spec(path: str) -> Dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("formula spec must be a JSON object")
    return data


def _parse_loss_code(path: str) -> ast.Module:
    code = Path(path).read_text(encoding="utf-8")
    return ast.parse(code)


def _find_sandbox_loss(tree: ast.Module) -> Optional[ast.FunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "sandbox_loss":
            return node
    return None


def _collect_name_ids(tree: ast.AST) -> Set[str]:
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names


def _collect_arg_names(func: ast.FunctionDef) -> List[str]:
    args = [a.arg for a in func.args.args]
    if func.args.vararg:
        args.append(func.args.vararg.arg)
    if func.args.kwarg:
        args.append(func.args.kwarg.arg)
    return args


def validate_alignment(loss_file: str, formula_spec_file: str) -> Dict[str, Any]:
    """
    Validate that:
    - symbol_map values appear in code (as args or names)
    - params keys are represented in code (preferably as sandbox_loss args)
    """
    errors: List[str] = []
    warnings: List[str] = []

    spec = _load_formula_spec(formula_spec_file)
    symbol_map = spec.get("symbol_map", {})
    params = spec.get("params", {})
    interface_analysis = analyze_formula_interface(spec)

    if not isinstance(symbol_map, dict):
        return {"passed": False, "errors": ["symbol_map missing or invalid"], "warnings": []}
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return {"passed": False, "errors": ["params invalid (must be object)"], "warnings": []}

    tree = _parse_loss_code(loss_file)
    func = _find_sandbox_loss(tree)
    if func is None:
        return {"passed": False, "errors": ["sandbox_loss() not found in code"], "warnings": []}

    arg_names = _collect_arg_names(func)
    all_names = _collect_name_ids(tree)

    symbol_values: List[str] = []
    for _, v in symbol_map.items():
        if isinstance(v, str) and v.strip():
            symbol_values.append(v.strip())

    missing_symbol_values: List[str] = []
    for v in sorted(set(symbol_values)):
        if v in arg_names:
            continue
        if v in all_names:
            continue
        missing_symbol_values.append(v)

    if missing_symbol_values:
        errors.append(
            "symbol_map values not present in code: " + ", ".join(missing_symbol_values)
        )

    # Prefer params to be explicit sandbox_loss args for tunability.
    params_missing_in_signature: List[str] = []
    params_unused_in_code: List[str] = []
    for k in sorted(params.keys()):
        if not isinstance(k, str) or not k.strip():
            continue
        if k not in arg_names:
            params_missing_in_signature.append(k)
        if k not in all_names and k not in arg_names:
            params_unused_in_code.append(k)

    if params_missing_in_signature:
        warnings.append(
            "params not exposed as sandbox_loss args (still allowed, but reduces tunability): "
            + ", ".join(params_missing_in_signature)
        )

    if params_unused_in_code:
        warnings.append(
            "params keys not referenced in code (possible dead params): "
            + ", ".join(params_unused_in_code)
        )

    if interface_analysis.get("requires_extra_prediction_heads"):
        warnings.append(
            "formula requires extra model-provided loss inputs beyond pred/target/mask/params: "
            + ", ".join(interface_analysis.get("extra_required_variables", []))
        )

    # Hard requirement: pred/target should exist as args (interface)
    if "pred" not in arg_names or "target" not in arg_names:
        errors.append(f"sandbox_loss args must include pred and target, got: {arg_names}")

    passed = not errors
    return {
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "details": {
            "sandbox_loss_args": arg_names,
            "missing_symbol_values": missing_symbol_values,
            "params_missing_in_signature": params_missing_in_signature,
            "params_unused_in_code": params_unused_in_code,
            "interface_analysis": interface_analysis,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate alignment between loss code and formula spec")
    parser.add_argument("--loss_file", required=True)
    parser.add_argument("--formula_spec", required=True)
    args = parser.parse_args()

    result = validate_alignment(args.loss_file, args.formula_spec)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
