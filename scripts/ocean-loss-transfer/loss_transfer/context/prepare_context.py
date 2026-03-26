"""
@file prepare_context.py
@description 扫描代码仓库，准备 loss transfer 闭环所需的论文/代码上下文材料
@author Leizheng
@date 2026-03-23
@version 1.2.0

@changelog
  - 2026-03-23 Leizheng: v1.0.0 初始版本
    - 智能文件发现（文件名 + 内容关键词）
    - 代码预处理（提取函数签名、移除注释）
    - 依赖分析（import 语句提取）
    - 返回结构化 JSON
  - 2026-03-24 Leizheng: v1.1.0 支持论文 PDF 上下文提取（abstract/sections/loss_snippets）
  - 2026-03-26 OpenAI Codex: v1.2.0 补充 analysis_plan 输出路径，供 Agent 主流程使用
"""

import os
import re
import json
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

from loss_transfer.common.paths import LOSS_TRANSFER_EXPERIMENTS_DIR


def find_loss_files(repo_path: str) -> List[Dict[str, Any]]:
    """
    智能发现 loss 相关文件

    优先级：
    1. 文件名包含 loss/criterion/objective
    2. 文件内容包含 def.*loss / class.*Loss
    """
    repo_path = Path(repo_path)
    candidates = []

    # 排除目录
    exclude_dirs = {'test', 'tests', 'docs', 'examples', '__pycache__', '.git', 'build', 'dist'}

    for py_file in repo_path.rglob('*.py'):
        # 跳过排除目录
        if any(ex in py_file.parts for ex in exclude_dirs):
            continue

        rel_path = py_file.relative_to(repo_path)
        filename_lower = py_file.name.lower()

        # 优先级评分
        priority = 0

        # 文件名匹配
        if any(kw in filename_lower for kw in ['loss', 'criterion', 'objective']):
            priority += 10

        # 读取内容检查
        try:
            content = py_file.read_text(encoding='utf-8', errors='ignore')

            # 关键词匹配
            if re.search(r'def\s+\w*loss\w*\s*\(', content, re.IGNORECASE):
                priority += 5
            if re.search(r'class\s+\w*Loss\w*\s*[\(:]', content, re.IGNORECASE):
                priority += 5
            if 'criterion' in content.lower():
                priority += 2
            # 很多论文会把 loss 计算塞进 model.forward 里（比如 nf_loss / aux_loss），
            # 这种情况文件名不含 loss、也没有 def loss()，但依然是关键信息。
            if 'loss' in content.lower():
                priority += 1

            if priority > 0:
                candidates.append({
                    'path': str(rel_path),
                    'abs_path': str(py_file),
                    'priority': priority,
                    'size': len(content)
                })
        except Exception:
            continue

    # 按优先级排序
    candidates.sort(key=lambda x: (-x['priority'], x['size']))

    return candidates[:10]  # 最多返回 10 个文件


def extract_functions(content: str) -> List[str]:
    """提取函数签名"""
    functions = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
    except:
        pass
    return functions


def extract_imports(content: str) -> List[str]:
    """提取 import 语句"""
    imports = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
    except:
        pass
    return list(set(imports))


def preprocess_code(content: str, max_lines: int = 5000) -> str:
    """
    预处理代码：
    - 移除注释
    - 限制行数
    - 保留关键结构
    """
    lines = content.split('\n')

    # 移除单行注释
    processed_lines = []
    for line in lines[:max_lines]:
        # 保留非注释行
        stripped = line.strip()
        if not stripped.startswith('#'):
            processed_lines.append(line)

    return '\n'.join(processed_lines)


def prepare_context(
    code_repo_path: str,
    paper_slug: str,
    output_dir: str = None,
    paper_pdf_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    准备 loss transfer 闭环的上下文

    Args:
        code_repo_path: 代码仓库路径
        paper_slug: 论文标识符
        output_dir: 输出目录（可选）
        paper_pdf_path: 论文 PDF 路径（可选）

    Returns:
        包含文件列表、内容、schema 的字典
    """
    repo_path = Path(code_repo_path)
    if not repo_path.exists():
        raise FileNotFoundError(f"代码仓库不存在: {code_repo_path}")

    # 默认输出目录
    if output_dir is None:
        output_dir = LOSS_TRANSFER_EXPERIMENTS_DIR / paper_slug
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_yaml_path = output_dir / 'loss_ir.yaml'
    output_formula_path = output_dir / 'loss_formula.json'
    output_analysis_plan_path = output_dir / 'analysis_plan.json'

    # 0. 提取论文 PDF 上下文（可选，不影响代码扫描）
    paper_context: Optional[Dict[str, Any]] = None
    if paper_pdf_path:
        try:
            from loss_transfer.context.extract_paper_text import extract_paper_text

            paper_context = extract_paper_text(
                paper_pdf_path,
                output_dir=output_dir,
                # 0 = all pages；若你希望进一步控大小，可改成 30/40
                max_pages=0,
            )
        except Exception as e:
            paper_context = {
                "success": False,
                "paper_pdf_path": paper_pdf_path,
                "error": f"extract_paper_text failed: {e}",
            }

    # 1. 发现 loss 文件
    loss_files = find_loss_files(code_repo_path)

    # 2. 读取文件内容并预处理
    primary_files = []
    for file_info in loss_files:
        try:
            content = Path(file_info['abs_path']).read_text(encoding='utf-8', errors='ignore')

            primary_files.append({
                'path': file_info['path'],
                'content': preprocess_code(content),
                'functions': extract_functions(content),
                'imports': extract_imports(content),
                'priority': 'high' if file_info['priority'] >= 10 else 'medium'
            })
        except Exception as e:
            print(f"Warning: 无法读取 {file_info['path']}: {e}")
            continue

    # 3. 读取 Loss IR schema
    schema_doc = '见 loss_transfer/ir/loss_ir_schema.py 中的 LossIR 数据类定义'

    # 4. 构建返回结果
    result = {
        'paper': paper_context,
        'primary_files': primary_files,
        'schema': schema_doc,
        'output_path': str(output_yaml_path),
        'formula_output_path': str(output_formula_path),
        'analysis_plan_output_path': str(output_analysis_plan_path),
        'code_repo': str(repo_path),
        'paper_slug': paper_slug
    }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='准备 Loss IR 提取上下文')
    parser.add_argument('--code_repo', required=True, help='代码仓库路径')
    parser.add_argument('--paper_slug', required=True, help='论文标识符')
    parser.add_argument('--output_dir', help='输出目录（可选）')
    parser.add_argument('--paper_pdf', help='论文 PDF 路径（可选）')

    args = parser.parse_args()

    result = prepare_context(
        args.code_repo,
        args.paper_slug,
        args.output_dir,
        paper_pdf_path=args.paper_pdf,
    )

    # 输出 JSON
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
