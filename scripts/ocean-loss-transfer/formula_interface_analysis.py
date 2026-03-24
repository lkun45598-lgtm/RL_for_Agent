"""
@file formula_interface_analysis.py
@description 分析 loss_formula.json 是否超出当前 sandbox loss 运行时接口能力
@author Leizheng
@date 2026-03-24
@version 1.0.0
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


BASE_INTERFACE_VARIABLES = ("pred", "target", "mask")


def _collect_symbol_pairs(spec: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    symbol_map = spec.get("symbol_map", {})
    if not isinstance(symbol_map, dict):
        return pairs

    for symbol, variable in symbol_map.items():
        if not isinstance(symbol, str) or not symbol.strip():
            continue
        if not isinstance(variable, str) or not variable.strip():
            continue
        pairs.append((symbol.strip(), variable.strip()))
    return pairs


def analyze_formula_interface(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    判断 formula spec 是否只依赖当前 runtime 保证提供的变量。

    当前 sandbox runtime 仅保证向 sandbox_loss 提供:
    - pred
    - target
    - mask

    其余变量如果出现在 symbol_map 中，且又不是 params，可视为
    “需要模型额外预测头 / 额外运行时输入”。
    """
    params = spec.get("params", {})
    param_variables = sorted(
        k for k in params.keys()
        if isinstance(params, dict) and isinstance(k, str) and k.strip()
    )
    param_set = set(param_variables)

    symbol_pairs = _collect_symbol_pairs(spec)
    mapped_variables = sorted({variable for _, variable in symbol_pairs})

    extra_required_variables = sorted(
        variable for variable in mapped_variables
        if variable not in BASE_INTERFACE_VARIABLES and variable not in param_set
    )
    extra_required_symbols = sorted(
        symbol for symbol, variable in symbol_pairs
        if variable in extra_required_variables
    )

    raw_adapter_heads = spec.get("adapter_heads", {})
    declared_adapter_heads = sorted(
        key for key in raw_adapter_heads.keys()
        if isinstance(key, str) and key.strip()
    ) if isinstance(raw_adapter_heads, dict) else []
    missing_adapter_heads = sorted(
        variable for variable in extra_required_variables
        if variable not in declared_adapter_heads
    )

    issues: List[str] = []
    status = "fully_compatible"
    if extra_required_variables:
        status = "requires_adapter"
        issues.append(
            "formula requires additional model-provided loss inputs; auto experiment must enable sandbox_adapter heads for: "
            + ", ".join(extra_required_variables)
        )

    return {
        "status": status,
        "base_interface_variables": list(BASE_INTERFACE_VARIABLES),
        "param_variables": param_variables,
        "mapped_variables": mapped_variables,
        "runtime_can_forward_model_loss_inputs": True,
        "requires_model_output_extension": bool(extra_required_variables),
        "requires_extra_prediction_heads": bool(extra_required_variables),
        "loss_only_pipeline_compatible": not extra_required_variables,
        "auto_experiment_supported": True,
        "extra_required_variables": extra_required_variables,
        "extra_required_symbols": extra_required_symbols,
        "declared_adapter_heads": declared_adapter_heads,
        "missing_adapter_heads": missing_adapter_heads,
        "adapter_config_source": (
            "not_needed" if not extra_required_variables
            else "declared" if not missing_adapter_heads
            else "auto_inferred"
        ),
        "issues": issues,
        "notes": (
            "The sandbox runtime can forward extra kwargs to sandbox_loss when using a "
            "sandbox model adapter that exposes extra loss inputs. The auto experiment "
            "workflow can now synthesize sandbox_adapter configs from loss_formula.json; "
            "explicit adapter_heads remain recommended when channel counts or activations "
            "must be controlled precisely."
        ),
    }


def ensure_formula_interface_analysis(spec: Dict[str, Any]) -> Dict[str, Any]:
    """返回带最新 interface_analysis 的 spec 副本。"""
    updated = dict(spec)
    updated["interface_analysis"] = analyze_formula_interface(spec)
    return updated
