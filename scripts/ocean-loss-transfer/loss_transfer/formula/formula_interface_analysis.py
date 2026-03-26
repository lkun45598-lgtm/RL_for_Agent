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
_HEAD_FRIENDLY_HINTS = (
    "weight",
    "log_b",
    "sigma",
    "variance",
    "var",
    "uncertainty",
    "confidence",
    "scale",
    "logits",
)
_FEATURE_HINTS = ("feat", "feature", "hidden", "latent", "embed", "embedding", "context")
_SIMPLE_FORMULA_HINTS = (
    "l1",
    "l2",
    "mse",
    "mae",
    "abs",
    "relative",
    "charbonnier",
    "|x-y|",
    "|y-x|",
    "|pred-target|",
    "|target-pred|",
    "(x-y)2",
    "(y-x)2",
    "(pred-target)2",
    "(target-pred)2",
)
_COMPLEX_FORMULA_HINTS = (
    "logsumexp",
    "mixture",
    "laplace",
    "gaussian",
    "nll",
    "kl",
    "crossentropy",
    "softmax",
    "huber",
    "smoothl1",
)


def _joined_latex(spec: Dict[str, Any]) -> str:
    latex = spec.get("latex", [])
    if isinstance(latex, str):
        return latex
    if isinstance(latex, list):
        return "\n".join(str(item) for item in latex if isinstance(item, str))
    return ""


def _normalize_text(text: str) -> str:
    normalized = text.lower()
    for token in (" ", "\n", "\t", "\\", "{", "}", "_", "^"):
        normalized = normalized.replace(token, "")
    return normalized


def _collect_structure_hints(spec: Dict[str, Any]) -> Dict[str, Any]:
    hints: Dict[str, Any] = {}

    raw_sources = spec.get("sources", {})
    if isinstance(raw_sources, dict):
        source_hints = raw_sources.get("structure_hints", {})
        if isinstance(source_hints, dict):
            hints.update(source_hints)

    raw_hints = spec.get("structure_hints", {})
    if isinstance(raw_hints, dict):
        hints.update(raw_hints)

    return hints


def _looks_head_friendly(variable_name: str) -> bool:
    lowered = variable_name.strip().lower()
    if not lowered:
        return False
    return any(token in lowered for token in _HEAD_FRIENDLY_HINTS)


def _looks_feature_like(variable_name: str) -> bool:
    lowered = variable_name.strip().lower()
    if not lowered:
        return False
    return any(token in lowered for token in _FEATURE_HINTS)


def _is_simple_formula(spec: Dict[str, Any], extra_required_variables: List[str]) -> bool:
    if extra_required_variables:
        return False

    normalized = _normalize_text(_joined_latex(spec))
    if not normalized:
        return False

    if any(token in normalized for token in _COMPLEX_FORMULA_HINTS):
        return False
    return any(token in normalized for token in _SIMPLE_FORMULA_HINTS)


def _bool_hint(hints: Dict[str, Any], key: str) -> bool:
    return bool(hints.get(key))


def _build_change_level_summary(
    *,
    extra_required_variables: List[str],
    head_friendly_variables: List[str],
    unclear_extra_variables: List[str],
    feature_like_variables: List[str],
    structure_hints: Dict[str, Any],
    simple_formula: bool,
) -> Tuple[int, str, str, List[str]]:
    reasons: List[str] = []

    uses_feature_terms = _bool_hint(structure_hints, "uses_feature_or_hidden_state_terms")
    uses_external_terms = _bool_hint(structure_hints, "uses_pretrained_or_adversarial_terms")
    uses_multi_stage_terms = _bool_hint(structure_hints, "uses_multi_stage_or_multi_scale_terms")
    uses_distributional_heads = _bool_hint(structure_hints, "uses_distributional_aux_heads")
    loss_inside_forward = _bool_hint(structure_hints, "loss_defined_inside_model_forward")

    if uses_external_terms:
        reasons.append("Paper/code hints indicate perceptual/adversarial/external-network dependencies")
        return (
            5,
            "model_surgery",
            "model_surgery",
            reasons,
        )

    if feature_like_variables:
        if feature_like_variables:
            reasons.append("Extra variables look like internal features or hidden states")
        return (
            5,
            "model_surgery",
            "model_surgery",
            reasons,
        )

    if uses_feature_terms and not head_friendly_variables and not uses_distributional_heads:
        if uses_feature_terms:
            reasons.append("Paper/code hints mention feature-level or hidden-state loss terms")
        return (
            5,
            "model_surgery",
            "model_surgery",
            reasons,
        )

    if extra_required_variables:
        if head_friendly_variables:
            reasons.append(
                "Formula needs extra loss inputs that look like lightweight prediction heads: "
                + ", ".join(head_friendly_variables)
            )
        if unclear_extra_variables:
            reasons.append(
                "Some extra variables are not clearly adapter-friendly and likely need custom model outputs: "
                + ", ".join(unclear_extra_variables)
            )
        if uses_multi_stage_terms:
            reasons.append("Paper/code hints indicate multi-stage or multi-scale supervision")
        if loss_inside_forward:
            reasons.append("Paper code appears to compute or couple loss logic inside model.forward")

        if unclear_extra_variables or uses_multi_stage_terms or loss_inside_forward:
            return (
                4,
                "model_output_extension",
                "extend_model_outputs",
                reasons,
            )

        return (
            3,
            "loss_inputs_adapter",
            "add_loss_inputs_adapter",
            reasons,
        )

    if uses_multi_stage_terms or loss_inside_forward:
        if uses_multi_stage_terms:
            reasons.append("Paper/code hints indicate multi-stage or multi-scale supervision")
        if loss_inside_forward:
            reasons.append("Paper code appears to compute or couple loss logic inside model.forward")
        return (
            4,
            "model_output_extension",
            "extend_model_outputs",
            reasons,
        )

    if simple_formula:
        reasons.append("Formula only depends on pred/target/mask/params and looks like a simple pointwise loss")
        return (
            1,
            "params_only",
            "reuse_existing_loss_config",
            reasons,
        )

    reasons.append("Formula stays within the base loss interface but likely needs a spec-driven recipe")
    return (
        2,
        "spec_driven_loss",
        "add_spec_driven_recipe",
        reasons,
    )


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
    head_friendly_variables = sorted(
        variable for variable in extra_required_variables
        if _looks_head_friendly(variable)
    )
    unclear_extra_variables = sorted(
        variable for variable in extra_required_variables
        if variable not in head_friendly_variables
    )
    feature_like_variables = sorted(
        variable for variable in extra_required_variables
        if _looks_feature_like(variable)
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

    structure_hints = _collect_structure_hints(spec)
    simple_formula = _is_simple_formula(spec, extra_required_variables)
    change_level, change_level_label, recommended_path, change_level_reasons = _build_change_level_summary(
        extra_required_variables=extra_required_variables,
        head_friendly_variables=head_friendly_variables,
        unclear_extra_variables=unclear_extra_variables,
        feature_like_variables=feature_like_variables,
        structure_hints=structure_hints,
        simple_formula=simple_formula,
    )

    issues: List[str] = []
    if change_level <= 2:
        status = "fully_compatible"
    elif change_level == 3:
        status = "requires_adapter"
    else:
        status = "requires_model_changes"

    if extra_required_variables:
        issues.append(
            "formula requires additional model-provided loss inputs; auto experiment must enable sandbox_adapter heads for: "
            + ", ".join(extra_required_variables)
        )
    if change_level >= 4:
        issues.append(
            "formula likely needs deeper integration than loss-only migration; recommended path: "
            + recommended_path
        )
    if change_level == 5:
        issues.append(
            "formula appears to depend on feature-level, external-network, or architecture-coupled signals"
        )

    return {
        "status": status,
        "base_interface_variables": list(BASE_INTERFACE_VARIABLES),
        "param_variables": param_variables,
        "mapped_variables": mapped_variables,
        "runtime_can_forward_model_loss_inputs": True,
        "requires_model_output_extension": change_level >= 4,
        "requires_extra_prediction_heads": bool(extra_required_variables),
        "loss_only_pipeline_compatible": change_level <= 2,
        "auto_experiment_supported": True,
        "extra_required_variables": extra_required_variables,
        "extra_required_symbols": extra_required_symbols,
        "head_friendly_variables": head_friendly_variables,
        "unclear_extra_variables": unclear_extra_variables,
        "feature_like_variables": feature_like_variables,
        "declared_adapter_heads": declared_adapter_heads,
        "missing_adapter_heads": missing_adapter_heads,
        "adapter_config_source": (
            "not_needed" if not extra_required_variables
            else "declared" if not missing_adapter_heads
            else "auto_inferred"
        ),
        "structure_hints": structure_hints,
        "simple_formula": simple_formula,
        "change_level": change_level,
        "change_level_label": change_level_label,
        "change_level_reasons": change_level_reasons,
        "recommended_integration_path": recommended_path,
        "minimal_change_possible": change_level <= 3,
        "requires_model_changes": change_level >= 4,
        "can_use_original_loss_factory_directly": change_level <= 3,
        "issues": issues,
        "notes": (
            "Change levels: L1=params only, L2=spec-driven loss recipe, "
            "L3=loss_inputs adapter/wrapper, L4=model output extension, "
            "L5=model surgery. The sandbox runtime can forward extra kwargs to "
            "sandbox_loss when using a sandbox model adapter that exposes extra loss "
            "inputs, but change_level>=4 means the original SR code likely needs deeper "
            "integration than a loss-only patch."
        ),
    }


def ensure_formula_interface_analysis(spec: Dict[str, Any]) -> Dict[str, Any]:
    """返回带最新 interface_analysis 的 spec 副本。"""
    updated = dict(spec)
    updated["interface_analysis"] = analyze_formula_interface(spec)
    return updated
