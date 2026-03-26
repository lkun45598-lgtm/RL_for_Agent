"""
@file extract_loss_formula.py
@description 从论文 PDF 和代码仓库中自动起草 Loss Formula Spec（draft），供 Agent 进一步校对
@author Leizheng
@date 2026-03-24
@version 1.0.0

@changelog
  - 2026-03-24 Leizheng: v1.0.0 initial version
    - 提取论文中的 loss/equation 相关片段作为 latex 草稿
    - 从代码/配置中提取高置信参数
    - 生成 review_required 的 loss_formula.json draft
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from formula_interface_analysis import ensure_formula_interface_analysis
from loss_spec_builder import build_loss_spec_draft, write_loss_spec_yaml
from prepare_context import prepare_context
from sandbox_adapter_bridge import draft_adapter_heads
from write_loss_formula import validate_formula_spec


_PARAM_NAME_RE = re.compile(
    r"^(gamma|alpha|beta|lambda|eps|epsilon|var_min|var_max|max_flow|scale_weights|scales|weight.*|use_var)$",
    re.IGNORECASE,
)
_EQUATION_HINT_RE = re.compile(r"(=|MixLap|NLL|log|loss|Laplace|gamma|alpha|beta)", re.IGNORECASE)
_STRONG_FORMULA_RE = re.compile(r"(MixLap|LMoL|Lall|NLL|\\log|log|Laplace|gamma|alpha|beta|=)", re.IGNORECASE)
_AUX_VAR_CANDIDATES = ("weight", "log_b", "sigma", "variance", "var", "uncertainty", "confidence")


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _cleanup_pdf_formula_candidate(text: str) -> str:
    cleaned = text.replace('-\n', '')
    cleaned = re.sub(r'\n+', '\n', cleaned)
    cleaned = re.sub(r'\\\s+([A-Za-z])', r'\\\1', cleaned)
    cleaned = re.sub(
        r'\\([A-Za-z])(?:\s+([A-Za-z]))+',
        lambda m: '\\' + re.sub(r'\s+', '', m.group(0)[1:]),
        cleaned,
    )
    cleaned = re.sub(r'(?<=[A-Za-z])\s+_(?=\w)', '_', cleaned)
    cleaned = re.sub(r'(?<=[A-Za-z])\s+\^(?=\w)', '^', cleaned)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n\s+', '\n', cleaned)
    return cleaned.strip()


def _extract_equation_like_snippets(paper: Optional[Dict[str, Any]]) -> List[str]:
    """
    从 paper.loss_snippets 和 sections 中提取适合作为 latex 草稿的片段。
    注意：PDF 抽取后的文本不一定是真 LaTeX，这里只是高相关公式文本草稿。
    """
    if not paper or not paper.get("success"):
        return []

    def score_text(text: str, heading: str = "") -> int:
        score = 0
        if "=" in text:
            score += 6
        if re.search(r"(MixLap|LMoL|Lall|NLL)", text, re.IGNORECASE):
            score += 8
        if re.search(r"(Laplace|log-likelihood|log term)", text, re.IGNORECASE):
            score += 4
        if re.search(r"\bgamma\b", text, re.IGNORECASE):
            score += 3
        if "loss" in heading.lower() or "objective" in heading.lower():
            score += 4
        if "3.2" in heading:
            score += 4
        return score

    def add_windows(text: str, score_bias: int = 0) -> List[Tuple[int, str]]:
        windows: List[Tuple[int, str]] = []
        for m in _STRONG_FORMULA_RE.finditer(text):
            start = max(0, m.start() - 350)
            end = min(len(text), m.end() + 700)
            snippet = text[start:end].strip()
            if len(snippet) < 80:
                continue
            windows.append((score_text(snippet) + score_bias, snippet[:1200]))
        return windows

    scored: List[Tuple[int, str]] = []

    for entry in paper.get("loss_snippets", [])[:20]:
        snippet = entry.get("snippet", "").strip()
        if snippet and _EQUATION_HINT_RE.search(snippet):
            scored.extend(add_windows(snippet, score_bias=1))

    for sec in paper.get("sections", [])[:20]:
        heading = sec.get("heading", "")
        text = sec.get("text", "").strip()
        if not text:
            continue
        if "loss" in heading.lower() or "objective" in heading.lower() or "3.2" in heading:
            scored.extend(add_windows(text, score_bias=3))

    # Fallback to the full text for local windows around high-signal markers
    full_text_path = paper.get("full_text_path")
    if isinstance(full_text_path, str) and full_text_path:
        try:
            full_text = Path(full_text_path).read_text(encoding="utf-8", errors="ignore")
            scored.extend(add_windows(full_text, score_bias=2))
        except OSError:
            pass

    scored.sort(key=lambda x: x[0], reverse=True)
    return _dedupe_keep_order([snippet for _, snippet in scored])[:8]


def _infer_symbol_map_from_params(
    params: Dict[str, Any],
    paper: Optional[Dict[str, Any]],
    aux_vars: List[str],
) -> Dict[str, str]:
    """
    生成保守的 symbol_map：
    - pred / target 是强约束，必须存在
    - mask 只有论文/代码里出现 mask/valid 概念时才补
    - 参数只对常见希腊字母做直接映射，其余不强行猜
    """
    symbol_map: Dict[str, str] = {
        "PRED_SYMBOL_TODO": "pred",
        "TARGET_SYMBOL_TODO": "target",
    }

    paper_text = ""
    if paper and paper.get("success"):
        paper_text = " ".join(
            [paper.get("abstract", "")] +
            [s.get("snippet", "") for s in paper.get("loss_snippets", [])[:10]]
        ).lower()

    if "mask" in paper_text or "valid" in paper_text:
        symbol_map["MASK_OR_VALID_SYMBOL_TODO"] = "mask"

    greek_candidates = {
        "gamma": "\\gamma",
        "alpha": "\\alpha",
        "beta": "\\beta",
        "epsilon": "\\epsilon",
        "eps": "\\epsilon",
        "lambda": "\\lambda",
    }

    for k in params.keys():
        key_l = k.lower()
        if key_l in greek_candidates and k not in symbol_map.values():
            symbol_map[greek_candidates[key_l]] = k

    for aux_var in aux_vars:
        if aux_var not in symbol_map.values():
            symbol_map[f"{aux_var.upper()}_SYMBOL_TODO"] = aux_var

    return symbol_map


def _extract_literal(node: ast.AST) -> Optional[Any]:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        out = []
        for elt in node.elts:
            val = _extract_literal(elt)
            if val is None:
                return None
            out.append(val)
        return out
    if isinstance(node, ast.Tuple):
        out = []
        for elt in node.elts:
            val = _extract_literal(elt)
            if val is None:
                return None
            out.append(val)
        return out
    return None


def _extract_params_from_python_file(py_file: Path) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    try:
        content = py_file.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(content)
    except Exception:
        return params

    # Module-level constants and likely hyperparameters
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    if _PARAM_NAME_RE.match(name):
                        val = _extract_literal(node.value)
                        if val is not None:
                            params.setdefault(name, val)

    # Function args defaults for loss-like functions
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and "loss" in node.name.lower():
            pos_args = node.args.args
            defaults = node.args.defaults
            start = len(pos_args) - len(defaults)
            for idx, default in enumerate(defaults):
                arg_name = pos_args[start + idx].arg
                if _PARAM_NAME_RE.match(arg_name):
                    val = _extract_literal(default)
                    if val is not None:
                        params.setdefault(arg_name, val)

    return params


def _extract_params_from_json_file(json_file: Path) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
    except Exception:
        return params

    if not isinstance(data, dict):
        return params

    for k, v in data.items():
        if _PARAM_NAME_RE.match(k) and isinstance(v, (bool, int, float, str, list)):
            params.setdefault(k, v)
    return params


def _collect_params(code_repo_path: str, primary_files: List[Dict[str, Any]]) -> Dict[str, Any]:
    repo = Path(code_repo_path)
    params: Dict[str, Any] = {}

    # First pass: high-priority files from prepare_context
    for file_info in primary_files[:10]:
        py_file = repo / file_info["path"]
        params.update({k: v for k, v in _extract_params_from_python_file(py_file).items() if k not in params})

    # Second pass: config json files override code defaults because they are closer to the
    # actual training run used by the paper/repo.
    for json_file in repo.rglob("*.json"):
        if ".git" in json_file.parts:
            continue
        for k, v in _extract_params_from_json_file(json_file).items():
            params[k] = v

    return params


def _collect_code_evidence(primary_files: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    evidence: List[Dict[str, str]] = []
    for file_info in primary_files[:5]:
        evidence.append({
            "path": file_info["path"],
            "functions": ", ".join(file_info.get("functions", [])[:10]),
            "priority": file_info.get("priority", "unknown"),
        })
    return evidence


def _collect_aux_vars(primary_files: List[Dict[str, Any]]) -> List[str]:
    """
    检测 loss 可能依赖的模型额外输出变量。
    这类变量如果不在 pred/target/mask/params 中，通常意味着“当前 loss 接口不够表达”。
    """
    found: List[str] = []
    for file_info in primary_files[:10]:
        content = file_info.get("content", "")
        lowered = content.lower()
        for candidate in _AUX_VAR_CANDIDATES:
            if re.search(rf"\b{re.escape(candidate.lower())}\b", lowered):
                found.append(candidate)
    return _dedupe_keep_order(found)


def _join_paper_text(paper: Optional[Dict[str, Any]]) -> str:
    if not paper or not paper.get("success"):
        return ""

    parts: List[str] = []
    abstract = paper.get("abstract")
    if isinstance(abstract, str):
        parts.append(abstract)

    for entry in paper.get("loss_snippets", [])[:10]:
        snippet = entry.get("snippet")
        if isinstance(snippet, str):
            parts.append(snippet)

    for section in paper.get("sections", [])[:10]:
        heading = section.get("heading")
        text = section.get("text")
        if isinstance(heading, str):
            parts.append(heading)
        if isinstance(text, str):
            parts.append(text[:1200])

    return "\n".join(parts)


def _infer_structure_hints(
    primary_files: List[Dict[str, Any]],
    paper: Optional[Dict[str, Any]],
    aux_vars: List[str],
) -> Dict[str, Any]:
    feature_pattern = re.compile(r"\b(feat|feature|hidden|latent|embedding|context)\b", re.IGNORECASE)
    external_pattern = re.compile(r"\b(vgg|lpips|perceptual|discriminator|gan|critic|adversarial|teacher)\b", re.IGNORECASE)
    multi_stage_pattern = re.compile(
        r"\b(multi[-_ ]scale|multiscale|multi[-_ ]stage|pyramid|deep[_ ]supervision|scale_weights|auxiliary loss|aux loss)\b",
        re.IGNORECASE,
    )
    structured_return_pattern = re.compile(r"return\s*(\{|\w+\s*,)", re.IGNORECASE)

    forward_loss_paths: List[str] = []
    structured_output_paths: List[str] = []
    feature_term_paths: List[str] = []
    external_term_paths: List[str] = []
    multi_stage_paths: List[str] = []
    explicit_loss_inputs_paths: List[str] = []

    for file_info in primary_files[:10]:
        path = str(file_info.get("path", ""))
        content = str(file_info.get("content", ""))
        lowered = content.lower()

        if "def forward" in lowered and "loss" in lowered:
            forward_loss_paths.append(path)
        if "loss_inputs" in lowered:
            explicit_loss_inputs_paths.append(path)
        if structured_return_pattern.search(content):
            structured_output_paths.append(path)
        if feature_pattern.search(content):
            feature_term_paths.append(path)
        if external_pattern.search(content):
            external_term_paths.append(path)
        if multi_stage_pattern.search(content):
            multi_stage_paths.append(path)

    paper_text = _join_paper_text(paper)
    lowered_aux = [item.lower() for item in aux_vars]
    distributional_aux = any(
        candidate in lowered_aux
        for candidate in ("weight", "log_b", "sigma", "variance", "var", "uncertainty", "confidence")
    ) or bool(
        re.search(r"\b(mixture|laplace|gaussian|uncertainty|variance|sigma|log_b|weight logits?)\b", paper_text, re.IGNORECASE)
    )

    return {
        "loss_defined_inside_model_forward": bool(forward_loss_paths),
        "model_returns_structured_outputs": bool(structured_output_paths),
        "uses_feature_or_hidden_state_terms": bool(feature_term_paths) or bool(feature_pattern.search(paper_text)),
        "uses_pretrained_or_adversarial_terms": bool(external_term_paths) or bool(external_pattern.search(paper_text)),
        "uses_multi_stage_or_multi_scale_terms": bool(multi_stage_paths) or bool(multi_stage_pattern.search(paper_text)),
        "uses_distributional_aux_heads": distributional_aux,
        "explicit_loss_inputs_container": bool(explicit_loss_inputs_paths),
        "detected_aux_variables": aux_vars,
        "evidence_paths": {
            "forward_loss": forward_loss_paths[:5],
            "structured_outputs": structured_output_paths[:5],
            "feature_terms": feature_term_paths[:5],
            "external_terms": external_term_paths[:5],
            "multi_stage_terms": multi_stage_paths[:5],
            "explicit_loss_inputs": explicit_loss_inputs_paths[:5],
        },
    }


def extract_loss_formula_draft(
    code_repo_path: str,
    paper_slug: str,
    *,
    paper_pdf_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a conservative, review-required loss formula draft.
    """
    context = prepare_context(
        code_repo_path=code_repo_path,
        paper_slug=paper_slug,
        paper_pdf_path=paper_pdf_path,
    )

    paper = context.get("paper")
    primary_files = context.get("primary_files", [])
    params = _collect_params(code_repo_path, primary_files)
    raw_formula_candidates = _extract_equation_like_snippets(paper)
    latex_candidates = [_cleanup_pdf_formula_candidate(item) for item in raw_formula_candidates]
    latex_candidates = _dedupe_keep_order(latex_candidates)
    if not latex_candidates:
        latex_candidates = ["LOSS_FORMULA_TODO"]

    aux_vars = _collect_aux_vars(primary_files)
    symbol_map = _infer_symbol_map_from_params(params, paper, aux_vars)
    adapter_heads = draft_adapter_heads(aux_vars) if aux_vars else {}
    structure_hints = _infer_structure_hints(primary_files, paper, aux_vars)

    spec: Dict[str, Any] = {
        "latex": latex_candidates,
        "raw_formula_candidates": raw_formula_candidates,
        "latex_extraction_status": (
            "heuristic_pdf_text_windows"
            if raw_formula_candidates
            else "missing_formula_candidates"
        ),
        "params": params,
        "symbol_map": symbol_map,
        "adapter_heads": adapter_heads,
        "notes": (
            "AUTO-DRAFT: generated heuristically from paper snippets + code/config evidence. "
            "The `latex` field is currently a cleaned candidate reconstructed from PDF text windows, not guaranteed "
            "to be exact author LaTeX. Check `raw_formula_candidates` when formula formatting matters. "
            "You must manually review symbol names, equation formatting, and whether all predicted variables "
            "required by the paper are represented. adapter_heads is only a sandbox-side draft and may need "
            "manual correction for channel counts or activations."
        ),
        "review_required": True,
        "sources": {
            "paper_pdf_path": paper_pdf_path,
            "code_repo_path": code_repo_path,
            "paper_title": ((paper or {}).get("metadata") or {}).get("title", paper_slug) if isinstance(paper, dict) else paper_slug,
            "primary_files": _collect_code_evidence(primary_files),
            "detected_aux_variables": aux_vars,
            "structure_hints": structure_hints,
        },
    }

    spec = ensure_formula_interface_analysis(spec)
    validation = validate_formula_spec(spec)
    loss_spec_draft = build_loss_spec_draft(spec)
    result: Dict[str, Any] = {
        "status": "success" if validation["status"] != "error" else "error",
        "validation": validation,
        "formula_spec": spec,
        "loss_spec_draft": loss_spec_draft,
    }

    final_output_path = output_path or context.get("formula_output_path")
    if final_output_path and validation["status"] != "error":
        out = Path(final_output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
        result["written_path"] = str(out)
        loss_spec_result = write_loss_spec_yaml(loss_spec_draft, str(out.parent / "loss_spec.yaml"))
        result["loss_spec_written_path"] = loss_spec_result["written_path"]
        result["loss_spec_write"] = loss_spec_result
    else:
        result["written_path"] = None
        result["loss_spec_written_path"] = None

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft Loss Formula Spec from paper PDF + code repo")
    parser.add_argument("--code_repo", required=True, help="Path to paper code repository")
    parser.add_argument("--paper_slug", required=True, help="Paper slug")
    parser.add_argument("--paper_pdf", help="Optional paper PDF path")
    parser.add_argument("--output_path", help="Optional output JSON path")
    args = parser.parse_args()

    result = extract_loss_formula_draft(
        code_repo_path=args.code_repo,
        paper_slug=args.paper_slug,
        paper_pdf_path=args.paper_pdf,
        output_path=args.output_path,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
