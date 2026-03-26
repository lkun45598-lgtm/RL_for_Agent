"""
@file extract_paper_text.py
@description 从论文 PDF 提取文本上下文（abstract/sections/loss 相关片段），并将全文落盘供 Agent 分析
@author Leizheng
@date 2026-03-24
@version 1.0.0
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except Exception:  # pragma: no cover
    fitz = None  # type: ignore[assignment]
    _HAS_FITZ = False


def _extract_abstract(text: str) -> str:
    """
    从全文中粗略提取 Abstract 段落（基于常见 heading 规则的启发式）。
    PDF 文本质量差异很大，这里宁可保守返回空字符串，也不返回错误内容。
    """
    # "Abstract\n ... \n Introduction"
    m = re.search(
        r"(?:^|\n)\s*Abstract\s*\n(.*?)(?=\n\s*(?:\d+\.?\s+)?(?:Introduction|1\s))",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    # "Abstract: ..."
    m = re.search(
        r"(?:^|\n)\s*Abstract[:\s]*\n?(.*?)(?:\n\n|\n\s*\n)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    return ""


def _detect_sections(text: str, max_body_chars: int = 5000) -> List[Dict[str, str]]:
    """
    简单的 section detection：匹配类似 "1 Introduction" / "2.3 Loss Function" 的 heading。
    """
    sections: List[Dict[str, str]] = []
    pattern = re.compile(r"(?:^|\n)\s*(\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{2,80})\s*\n", re.MULTILINE)
    matches = list(pattern.finditer(text))
    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append({"heading": heading, "text": body[:max_body_chars]})
    return sections


def _normalize_snippet(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _extract_loss_snippets(text: str, *, max_snippets: int = 25, window: int = 450) -> List[Dict[str, str]]:
    """
    从全文中抓取 loss/objective 相关上下文窗口，供 Agent 快速定位公式位置。
    不追求 100% 精准，追求覆盖率和可读性。
    """
    patterns: List[tuple[str, re.Pattern[str]]] = [
        ("loss_keyword", re.compile(r"\b(loss|objective|criterion|optimi[sz]e|minimi[sz]e)\b", re.IGNORECASE)),
        ("loss_symbol", re.compile(r"(?:^|\n)\s*(?:L|ℒ|𝓛)\s*=\s*", re.MULTILINE)),
        ("equation_ref", re.compile(r"\b(Eq\.|Equation)\s*\(?\d+\)?", re.IGNORECASE)),
        ("regularization", re.compile(r"\b(regulari[sz]ation|penalty|term)\b", re.IGNORECASE)),
    ]

    hits: List[Dict[str, str]] = []
    seen: set[str] = set()

    for tag, pat in patterns:
        for m in pat.finditer(text):
            if len(hits) >= max_snippets:
                break
            start = max(0, m.start() - window)
            end = min(len(text), m.end() + window)
            snippet = _normalize_snippet(text[start:end])
            if not snippet:
                continue
            # 去重：不同 pattern 可能抓到相同片段
            key = f"{tag}:{snippet[:200]}"
            if key in seen:
                continue
            seen.add(key)
            hits.append({"tag": tag, "snippet": snippet})
        if len(hits) >= max_snippets:
            break

    return hits


def extract_paper_text(
    paper_pdf_path: str,
    *,
    output_dir: Path,
    max_pages: int = 0,
    include_full_text_excerpt_chars: int = 2000,
) -> Dict[str, Any]:
    """
    提取 PDF 文本，并写入 output_dir/paper_full_text.txt。

    Returns:
        dict（可 JSON 序列化），即使失败也会返回 error 字段，避免上游崩溃。
    """
    pdf_path = Path(paper_pdf_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not _HAS_FITZ:
        return {
            "success": False,
            "paper_pdf_path": str(pdf_path),
            "error": "PyMuPDF (fitz) not installed. Please install pymupdf/fitz in your python env.",
            "backend": "pymupdf",
        }

    if not pdf_path.exists():
        return {
            "success": False,
            "paper_pdf_path": str(pdf_path),
            "error": f"File not found: {pdf_path}",
            "backend": "pymupdf",
        }

    try:
        with fitz.open(str(pdf_path)) as doc:  # type: ignore[union-attr]
            page_count = int(doc.page_count)
            pages_to_read = page_count
            if max_pages and max_pages > 0:
                pages_to_read = min(pages_to_read, int(max_pages))

            parts: List[str] = []
            for i in range(pages_to_read):
                page = doc[i]
                parts.append(page.get_text())
            full_text = "\n".join(parts)

            meta = doc.metadata or {}
            title = (meta.get("title") or "").strip()

        # 落盘全文（给后续 Agent/工具再利用）
        full_text_path = output_dir / "paper_full_text.txt"
        full_text_path.write_text(full_text, encoding="utf-8", errors="ignore")

        abstract = _extract_abstract(full_text)
        sections = _detect_sections(full_text)
        loss_snippets = _extract_loss_snippets(full_text)

        excerpt = full_text[: max(0, include_full_text_excerpt_chars)]
        excerpt = _normalize_snippet(excerpt)

        return {
            "success": True,
            "paper_pdf_path": str(pdf_path),
            "backend": "pymupdf",
            "page_count": page_count,
            "metadata": {
                "title": title or pdf_path.stem,
                "author": (meta.get("author") or "").strip() if isinstance(meta, dict) else "",
                "subject": (meta.get("subject") or "").strip() if isinstance(meta, dict) else "",
                "creator": (meta.get("creator") or "").strip() if isinstance(meta, dict) else "",
            },
            "abstract": abstract,
            "sections": sections[:20],  # 防止 JSON 过大
            "loss_snippets": loss_snippets,
            "full_text_path": str(full_text_path),
            "full_text_chars": len(full_text),
            "full_text_excerpt": excerpt,
        }
    except Exception as e:  # pragma: no cover
        return {
            "success": False,
            "paper_pdf_path": str(pdf_path),
            "error": str(e),
            "backend": "pymupdf",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract paper text context from a PDF")
    parser.add_argument("--paper_pdf", required=True, help="Path to paper PDF")
    parser.add_argument("--output_dir", required=True, help="Output directory to write extracted artifacts")
    parser.add_argument("--max_pages", type=int, default=0, help="Max pages to extract (0=all)")
    args = parser.parse_args()

    result = extract_paper_text(
        args.paper_pdf,
        output_dir=Path(args.output_dir),
        max_pages=args.max_pages,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

