"""
@file formula_code_generator.py
@description 基于 loss_formula.json 的确定性公式代码生成器
@author Leizheng
@date 2026-03-24
@version 1.0.0
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional


FormulaGenerationVariant = Literal["faithful", "stabilized"]


def _joined_latex_text(formula_spec: Dict[str, Any]) -> str:
    latex = formula_spec.get("latex", [])
    if isinstance(latex, str):
        return latex
    if isinstance(latex, list):
        return "\n".join(str(item) for item in latex if isinstance(item, str))
    return ""


def _normalize_latex_text(text: str) -> str:
    normalized = text.lower()
    for token in (" ", "\n", "\t", "\\", "{", "}", "_", "^"):
        normalized = normalized.replace(token, "")
    return normalized


def detect_formula_codegen_pattern(formula_spec: Optional[Dict[str, Any]]) -> Optional[str]:
    if not formula_spec:
        return None

    symbol_map = formula_spec.get("symbol_map", {})
    if not isinstance(symbol_map, dict):
        return None

    mapped_vars = {
        value.strip()
        for value in symbol_map.values()
        if isinstance(value, str) and value.strip()
    }
    latex_text = _joined_latex_text(formula_spec).lower()
    normalized_latex = _normalize_latex_text(latex_text)

    if {"pred", "target", "weight", "log_b"}.issubset(mapped_vars):
        if "nll" in normalized_latex and "logsum" in normalized_latex:
            return "mixture_laplace_nll"
        if (
            "nll" in normalized_latex
            and "logb" in normalized_latex
            and ("|r|/b" in normalized_latex or "laplace" in normalized_latex)
        ):
            return "mixture_laplace_nll"

    return None


def supports_formula_codegen(formula_spec: Optional[Dict[str, Any]]) -> bool:
    return detect_formula_codegen_pattern(formula_spec) is not None


def _serialize_formula_metadata(formula_spec: Dict[str, Any]) -> str:
    summary = {
        "params": formula_spec.get("params", {}),
        "symbol_map": formula_spec.get("symbol_map", {}),
        "sources": formula_spec.get("sources", {}),
    }
    return json.dumps(summary, ensure_ascii=False)


def _generate_mixture_laplace_nll_code(
    formula_spec: Dict[str, Any],
    variant: FormulaGenerationVariant,
) -> str:
    params = formula_spec.get("params", {})
    gamma = float(params.get("gamma", 0.85))
    use_var = bool(params.get("use_var", True))
    var_min = float(params.get("var_min", 0.0))
    var_max = float(params.get("var_max", 10.0))

    if variant == "faithful":
        weight_clip = 20.0
        log_b_clip = max(8.0, abs(var_max))
        aux_reg = 0.0
        mixture_nll_block = """\
    log_norm = torch.logsumexp(weight_logits.squeeze(-2), dim=-1).unsqueeze(-1)
    nll = log_norm - torch.logsumexp(component_logits, dim=-1)
"""
        description = "Formula-native faithful Mixture Laplace NLL loss"
    else:
        weight_clip = 12.0
        log_b_clip = min(max(4.0, abs(var_max)), 6.0)
        aux_reg = 1e-4
        mixture_nll_block = """\
    log_weight = F.log_softmax(weight_logits.squeeze(-2), dim=-1).unsqueeze(-2)
    nll = -torch.logsumexp(
        log_weight - math.log(2.0) - stabilized_log_b - residual_abs * inv_b,
        dim=-1,
    )
    nll = torch.nan_to_num(nll, nan=1e4, posinf=1e4, neginf=0.0)
"""
        description = "Formula-native stabilized Mixture Laplace NLL loss"

    metadata_blob = _serialize_formula_metadata(formula_spec)

    return f'''"""
@file sandbox_loss.py
@description {description}
@version 1.0.0

Formula metadata:
{metadata_blob}
"""

import math
import torch
import torch.nn.functional as F


def _align_mask(mask, pred):
    if mask is None:
        return None
    if mask.shape[1] == pred.shape[1] and mask.shape[2] == pred.shape[2]:
        return mask
    m = mask.permute(0, 3, 1, 2).float()
    m = F.interpolate(m, size=(pred.shape[1], pred.shape[2]), mode='nearest')
    return m.permute(0, 2, 3, 1).bool()


def _align_aux_tensor(value, pred, name):
    if value is None:
        raise ValueError(f"Missing required auxiliary loss input: {{name}}")
    if not torch.is_tensor(value):
        raise TypeError(f"Auxiliary loss input {{name}} must be a tensor, got {{type(value)}}")
    if value.dim() != 4:
        raise ValueError(f"Auxiliary loss input {{name}} must be BHWC 4D tensor, got shape {{tuple(value.shape)}}")
    if value.shape[0] != pred.shape[0]:
        raise ValueError(
            f"Auxiliary loss input {{name}} batch mismatch: {{tuple(value.shape)}} vs {{tuple(pred.shape)}}"
        )
    if value.shape[1] == pred.shape[1] and value.shape[2] == pred.shape[2]:
        return value.to(device=pred.device, dtype=pred.dtype)
    t = value.permute(0, 3, 1, 2).to(device=pred.device, dtype=pred.dtype)
    t = F.interpolate(t, size=(pred.shape[1], pred.shape[2]), mode='bilinear', align_corners=False)
    return t.permute(0, 2, 3, 1).contiguous()


def _stabilize_weight_logits(weight, clip_value):
    return weight.float().clamp(min=-float(clip_value), max=float(clip_value))


def _stabilize_log_b(log_b, var_min, var_max, clip_value):
    log_b = log_b.float()
    positive_cap = max(float(var_max), 1e-3)
    negative_floor = float(var_min)
    if negative_floor >= 0.0:
        negative_floor = -positive_cap

    if log_b.shape[-1] == 1:
        stabilized = log_b.clamp(min=negative_floor, max=positive_cap)
    else:
        positive_branch = log_b[..., :1].clamp(min=0.0, max=positive_cap)
        negative_branch = log_b[..., 1:].clamp(min=negative_floor, max=0.0)
        stabilized = torch.cat([positive_branch, negative_branch], dim=-1)

    if clip_value is not None:
        clip_value = abs(float(clip_value))
        stabilized = stabilized.clamp(min=-clip_value, max=clip_value)
    return stabilized


def _masked_channel_sum_mean(value, mask=None):
    per_pixel = value.sum(dim=-1, keepdim=True)
    if mask is not None:
        valid = _align_mask(mask, per_pixel).float()
        return (per_pixel * valid).sum() / valid.sum().clamp(min=1.0)
    return per_pixel.mean()


def sandbox_loss(
    pred,
    target,
    mask=None,
    gamma={gamma},
    use_var={str(use_var)},
    var_min={var_min},
    var_max={var_max},
    weight_clip={weight_clip},
    log_b_clip={log_b_clip},
    aux_reg={aux_reg},
    eps=1e-6,
    **kwargs,
):
    pred_f = pred.float()
    target_f = target.float()

    if not use_var:
        fallback = (pred_f - target_f).abs()
        return _masked_channel_sum_mean(fallback, mask=mask)

    weight = _align_aux_tensor(kwargs.get("weight"), pred, "weight")
    log_b = _align_aux_tensor(kwargs.get("log_b"), pred, "log_b")

    weight_logits = _stabilize_weight_logits(weight, clip_value=weight_clip)
    stabilized_log_b = _stabilize_log_b(log_b, var_min=var_min, var_max=var_max, clip_value=log_b_clip)

    residual_abs = (pred_f - target_f).abs().unsqueeze(-1)
    weight_logits = weight_logits.unsqueeze(-2)
    stabilized_log_b = stabilized_log_b.unsqueeze(-2)

    inv_b = torch.exp((-stabilized_log_b).clamp(max=12.0)).clamp(max=1.0 / max(float(eps), 1e-6))
    component_logits = (
        weight_logits
        - math.log(2.0)
        - stabilized_log_b
        - residual_abs * inv_b
    )

{mixture_nll_block}

    sequence_weight = pred_f.new_tensor(float(gamma)) * 0.0 + 1.0
    loss = _masked_channel_sum_mean(nll, mask=mask) * sequence_weight

    if aux_reg > 0.0:
        loss = loss + float(aux_reg) * (
            weight_logits.square().mean() + stabilized_log_b.square().mean()
        )

    return loss
'''


def generate_formula_loss_code(
    formula_spec: Dict[str, Any],
    variant: FormulaGenerationVariant = "faithful",
) -> str:
    pattern = detect_formula_codegen_pattern(formula_spec)
    if pattern == "mixture_laplace_nll":
        return _generate_mixture_laplace_nll_code(formula_spec, variant=variant)
    raise ValueError(f"Unsupported formula codegen pattern: {pattern}")


def load_formula_spec_for_paper(project_root: str, paper_slug: str) -> Optional[Dict[str, Any]]:
    formula_path = (
        Path(project_root)
        / "sandbox"
        / "loss_transfer_experiments"
        / paper_slug
        / "loss_formula.json"
    )
    if not formula_path.exists():
        return None
    return json.loads(formula_path.read_text(encoding="utf-8"))
