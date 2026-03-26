"""
@file loss_spec_builder.py
@description 根据 loss_formula.json 草稿生成原始超分训练可消费的 loss_spec 草稿
@author Leizheng
@date 2026-03-25
@version 1.0.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from formula_code_generator import detect_formula_codegen_pattern
from formula_interface_analysis import analyze_formula_interface


def _joined_latex(formula_spec: Dict[str, Any]) -> str:
    latex = formula_spec.get("latex", [])
    if isinstance(latex, str):
        return latex
    if isinstance(latex, list):
        return "\n".join(str(item) for item in latex if isinstance(item, str))
    return ""


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    for token in (" ", "\n", "\t", "\\", "{", "}", "_", "^"):
        lowered = lowered.replace(token, "")
    return lowered


def infer_recipe(formula_spec: Dict[str, Any]) -> Dict[str, Any]:
    pattern = detect_formula_codegen_pattern(formula_spec)
    if pattern == "mixture_laplace_nll":
        return {
            "status": "ready",
            "recipe": "masked_mixture_laplace_nll",
            "reason": "Recognized mixture Laplace negative log-likelihood pattern",
        }

    normalized = _normalize_text(_joined_latex(formula_spec))
    if any(token in normalized for token in ("smoothl1", "huber")):
        return {
            "status": "unsupported",
            "recipe": None,
            "reason": "SmoothL1/Huber recipe is not implemented in the original SR loss registry yet",
        }
    if any(
        token in normalized
        for token in (
            "mse",
            "squareerror",
            "(pred-target)2",
            "(target-pred)2",
            "(x-y)2",
            "(y-x)2",
            "l2",
        )
    ):
        return {
            "status": "heuristic",
            "recipe": "masked_abs_mse",
            "reason": "Heuristic match on squared-error/L2 style formula text",
        }
    if any(
        token in normalized
        for token in (
            "l1",
            "abs",
            "|r|",
            "|pred-target|",
            "|target-pred|",
            "|x-y|",
            "|y-x|",
        )
    ):
        return {
            "status": "heuristic",
            "recipe": "masked_abs_l1",
            "reason": "Heuristic match on absolute-error/L1 style formula text",
        }
    if "relative" in normalized or "relloss" in normalized:
        return {
            "status": "heuristic",
            "recipe": "masked_rel_l2",
            "reason": "Heuristic match on relative error wording",
        }

    return {
        "status": "unsupported",
        "recipe": None,
        "reason": "Could not map the current formula draft to a registered loss recipe",
    }


def _build_bindings(formula_spec: Dict[str, Any], interface_analysis: Dict[str, Any]) -> Dict[str, str]:
    symbol_map = formula_spec.get("symbol_map", {})
    bindings: Dict[str, str] = {}
    if not isinstance(symbol_map, dict):
        symbol_map = {}

    mapped_values = {
        value.strip()
        for value in symbol_map.values()
        if isinstance(value, str) and value.strip()
    }
    for variable_name in ("pred", "target", "mask"):
        if variable_name in mapped_values:
            bindings[variable_name] = variable_name

    for variable_name in interface_analysis.get("extra_required_variables", []):
        if isinstance(variable_name, str) and variable_name.strip():
            bindings[variable_name] = f"loss_inputs.{variable_name}"

    return bindings


def build_loss_spec_draft(formula_spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not formula_spec:
        return {
            "status": "missing_formula_spec",
            "loss": None,
            "change_level": None,
            "recommended_path": None,
            "issues": ["formula_spec is missing"],
        }

    interface_analysis = analyze_formula_interface(formula_spec)
    recipe_info = infer_recipe(formula_spec)
    bindings = _build_bindings(formula_spec, interface_analysis)
    params = formula_spec.get("params", {})
    if not isinstance(params, dict):
        params = {}

    loss_block = None
    issues = list(interface_analysis.get("issues", []))
    if recipe_info.get("status") != "unsupported" and recipe_info.get("recipe"):
        loss_block = {
            "mode": "spec_driven",
            "recipe": recipe_info["recipe"],
            "params": params,
            "bindings": bindings,
        }
    else:
        issues.append(str(recipe_info.get("reason", "Unsupported recipe")))

    status = "ready"
    if recipe_info.get("status") == "heuristic":
        status = "review_required"
    elif recipe_info.get("status") == "unsupported":
        status = "unsupported"
    elif formula_spec.get("review_required", False):
        status = "review_required"

    if interface_analysis.get("change_level", 2) >= 4 and status == "ready":
        status = "review_required"

    return {
        "status": status,
        "recipe_status": recipe_info.get("status"),
        "recipe_reason": recipe_info.get("reason"),
        "loss": loss_block,
        "change_level": interface_analysis.get("change_level"),
        "change_level_label": interface_analysis.get("change_level_label"),
        "change_level_reasons": interface_analysis.get("change_level_reasons"),
        "recommended_path": interface_analysis.get("recommended_integration_path"),
        "recommended_integration_path": interface_analysis.get("recommended_integration_path"),
        "requires_model_changes": interface_analysis.get("requires_model_changes"),
        "minimal_change_possible": interface_analysis.get("minimal_change_possible"),
        "can_use_original_loss_factory_directly": interface_analysis.get("can_use_original_loss_factory_directly"),
        "issues": issues,
        "review_required": bool(formula_spec.get("review_required", False) or status != "ready"),
    }


def write_loss_spec_yaml(loss_spec_draft: Dict[str, Any], output_path: str) -> Dict[str, Any]:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(loss_spec_draft, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return {
        "written_path": str(out_path),
        "status": loss_spec_draft.get("status", "unknown"),
    }
