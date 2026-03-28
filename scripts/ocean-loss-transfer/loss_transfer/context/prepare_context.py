"""
@file prepare_context.py
@description 扫描代码仓库，准备 loss transfer 闭环所需的论文/代码上下文材料
@author Leizheng
@date 2026-03-23
@version 1.3.0

@changelog
  - 2026-03-23 Leizheng: v1.0.0 初始版本
    - 智能文件发现（文件名 + 内容关键词）
    - 代码预处理（提取函数签名、移除注释）
    - 依赖分析（import 语句提取）
    - 返回结构化 JSON
  - 2026-03-24 Leizheng: v1.1.0 支持论文 PDF 上下文提取（abstract/sections/loss_snippets）
  - 2026-03-26 OpenAI Codex: v1.2.0 补充 analysis_plan 输出路径，供 Agent 主流程使用
  - 2026-03-27 OpenAI Codex: v1.3.0 扩展 trainer/model/config 扫描，并输出 evidence_graph.json
"""

import re
import json
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

from loss_transfer.common.paths import LOSS_TRANSFER_EXPERIMENTS_DIR


_EXCLUDE_DIRS = {'test', 'tests', 'docs', 'examples', '__pycache__', '.git', 'build', 'dist'}
_CONFIG_SUFFIXES = {'.yaml', '.yml', '.json', '.toml'}
_CATEGORY_LIMIT = 8
_PRIMARY_FILE_LIMIT = 10


def _is_excluded(path: Path) -> bool:
    return any(part in _EXCLUDE_DIRS for part in path.parts)


def _read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='ignore')


def _safe_rel_path(repo_path: Path, path: Path) -> str:
    return str(path.relative_to(repo_path))


def _priority_label(score: int) -> str:
    if score >= 12:
        return 'high'
    if score >= 6:
        return 'medium'
    return 'low'


def _score_loss_candidate(rel_path: str, content: str) -> Dict[str, Any]:
    lowered_path = rel_path.lower()
    lowered_content = content.lower()
    score = 0
    signals: List[str] = []
    if any(keyword in lowered_path for keyword in ('loss', 'criterion', 'objective')):
        score += 10
        signals.append('loss_filename_keyword')
    if re.search(r'def\s+\w*loss\w*\s*\(', content, re.IGNORECASE):
        score += 5
        signals.append('loss_function_definition')
    if re.search(r'class\s+\w*Loss\w*\s*[\(:]', content, re.IGNORECASE):
        score += 5
        signals.append('loss_class_definition')
    if 'criterion' in lowered_content:
        score += 2
        signals.append('criterion_reference')
    if 'loss' in lowered_content:
        score += 1
        signals.append('loss_token_present')
    return {'score': score, 'signals': signals}


def _score_trainer_candidate(rel_path: str, content: str) -> Dict[str, Any]:
    lowered_path = rel_path.lower()
    lowered_content = content.lower()
    score = 0
    signals: List[str] = []
    if any(keyword in lowered_path for keyword in ('train', 'trainer', 'engine', 'runner', 'loop', 'fit')):
        score += 6
        signals.append('trainer_path_keyword')
    if re.search(r'def\s+(train|fit|training_step|run_epoch)\s*\(', content):
        score += 4
        signals.append('trainer_function_definition')
    if 'optimizer' in lowered_content:
        score += 2
        signals.append('optimizer_reference')
    if 'backward(' in lowered_content:
        score += 3
        signals.append('backward_call')
    if 'criterion' in lowered_content or 'loss' in lowered_content:
        score += 2
        signals.append('loss_flow_reference')
    return {'score': score, 'signals': signals}


def _score_model_candidate(rel_path: str, content: str) -> Dict[str, Any]:
    lowered_path = rel_path.lower()
    lowered_content = content.lower()
    score = 0
    signals: List[str] = []
    if any(keyword in lowered_path for keyword in ('model', 'models', 'network', 'net', 'backbone', 'encoder', 'decoder', 'head')):
        score += 4
        signals.append('model_path_keyword')
    if re.search(r'class\s+\w+\s*\([^)]*(nn\.module|module)', content, re.IGNORECASE):
        score += 4
        signals.append('nn_module_class')
    if re.search(r'def\s+forward\s*\(', content):
        score += 4
        signals.append('forward_definition')
    if 'loss_inputs' in lowered_content:
        score += 5
        signals.append('loss_inputs_reference')
    if 'output_aux_loss_inputs' in lowered_content:
        score += 5
        signals.append('output_aux_loss_inputs_reference')
    if any(token in lowered_content for token in ('aux_loss', 'nf_loss', 'return {', 'return dict(')):
        score += 2
        signals.append('structured_output_or_aux_reference')
    return {'score': score, 'signals': signals}


def _score_config_candidate(rel_path: str, content: str) -> Dict[str, Any]:
    lowered_path = rel_path.lower()
    lowered_content = content.lower()
    score = 0
    signals: List[str] = []
    if Path(rel_path).suffix.lower() in _CONFIG_SUFFIXES:
        score += 2
        signals.append('config_extension')
    if any(keyword in lowered_path for keyword in ('config', 'configs', 'train', 'experiment')):
        score += 2
        signals.append('config_path_keyword')
    if any(keyword in lowered_content for keyword in ('loss:', 'criterion', '"loss"', "'loss'", 'optimizer', 'scheduler', 'model', 'train')):
        score += 4
        signals.append('training_control_reference')
    return {'score': score, 'signals': signals}


def _build_inventory_record(
    repo_path: Path,
    file_path: Path,
    *,
    category: str,
    score: int,
    signals: List[str],
    content: str,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        'path': _safe_rel_path(repo_path, file_path),
        'abs_path': str(file_path),
        'category': category,
        'score': score,
        'priority': _priority_label(score),
        'signals': signals,
        'size': len(content),
        'content_preview': preprocess_code(content, max_lines=120)[:4000],
    }
    if file_path.suffix.lower() == '.py':
        record['functions'] = extract_functions(content)
        record['imports'] = extract_imports(content)
    else:
        record['functions'] = []
        record['imports'] = []
    return record


def _collect_code_inventory(repo_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    inventory: Dict[str, List[Dict[str, Any]]] = {
        'loss_files': [],
        'trainer_files': [],
        'model_files': [],
        'config_files': [],
    }

    for file_path in repo_path.rglob('*'):
        if not file_path.is_file() or _is_excluded(file_path):
            continue

        suffix = file_path.suffix.lower()
        if suffix != '.py' and suffix not in _CONFIG_SUFFIXES:
            continue

        try:
            content = _read_text(file_path)
        except Exception:
            continue

        rel_path = _safe_rel_path(repo_path, file_path)
        if suffix == '.py':
            loss_score = _score_loss_candidate(rel_path, content)
            trainer_score = _score_trainer_candidate(rel_path, content)
            model_score = _score_model_candidate(rel_path, content)
            if loss_score['score'] > 0:
                inventory['loss_files'].append(
                    _build_inventory_record(
                        repo_path,
                        file_path,
                        category='loss_files',
                        score=loss_score['score'],
                        signals=loss_score['signals'],
                        content=content,
                    )
                )
            if trainer_score['score'] > 0:
                inventory['trainer_files'].append(
                    _build_inventory_record(
                        repo_path,
                        file_path,
                        category='trainer_files',
                        score=trainer_score['score'],
                        signals=trainer_score['signals'],
                        content=content,
                    )
                )
            if model_score['score'] > 0:
                inventory['model_files'].append(
                    _build_inventory_record(
                        repo_path,
                        file_path,
                        category='model_files',
                        score=model_score['score'],
                        signals=model_score['signals'],
                        content=content,
                    )
                )
            continue

        config_score = _score_config_candidate(rel_path, content)
        if config_score['score'] > 0:
            inventory['config_files'].append(
                _build_inventory_record(
                    repo_path,
                    file_path,
                    category='config_files',
                    score=config_score['score'],
                    signals=config_score['signals'],
                    content=content,
                )
            )

    for key, records in inventory.items():
        records.sort(key=lambda item: (-int(item['score']), int(item['size']), str(item['path'])))
        inventory[key] = records[:_CATEGORY_LIMIT]
    return inventory


def _claim_paths(records: List[Dict[str, Any]], prefix: str, limit: int = 3) -> List[str]:
    refs: List[str] = []
    for idx, _ in enumerate(records[:limit]):
        refs.append(f'prepared_context.code_inventory.{prefix}[{idx}]')
    return refs


def _build_evidence_graph(code_inventory: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    loss_files = code_inventory.get('loss_files', [])
    trainer_files = code_inventory.get('trainer_files', [])
    model_files = code_inventory.get('model_files', [])
    config_files = code_inventory.get('config_files', [])
    claims: List[Dict[str, Any]] = []

    if loss_files:
        claims.append(
            {
                'claim_id': 'loss_files_present',
                'label': 'Standalone loss-related files are present in the repository.',
                'strength': 'high',
                'evidence_refs': _claim_paths(loss_files, 'loss_files'),
                'why_it_matters': 'Inspect these files first before deciding the paper requires deeper integration.',
            }
        )
    if trainer_files:
        claims.append(
            {
                'claim_id': 'trainer_loss_flow_present',
                'label': 'Trainer or engine files explicitly wire optimization/loss flow.',
                'strength': 'medium',
                'evidence_refs': _claim_paths(trainer_files, 'trainer_files'),
                'why_it_matters': 'Useful for locating where candidate loss kwargs and validation metrics are passed.',
            }
        )
    aux_model_refs = [
        item for item in model_files
        if any(signal in item.get('signals', []) for signal in ('loss_inputs_reference', 'output_aux_loss_inputs_reference', 'structured_output_or_aux_reference'))
    ]
    if aux_model_refs:
        claims.append(
            {
                'claim_id': 'model_aux_loss_inputs_present',
                'label': 'Model files may already emit structured outputs or auxiliary loss inputs.',
                'strength': 'medium',
                'evidence_refs': _claim_paths(aux_model_refs, 'model_files'),
                'why_it_matters': 'Check these files before forcing a loss-only migration path.',
            }
        )
    elif model_files:
        claims.append(
            {
                'claim_id': 'model_forward_present',
                'label': 'Model forward definitions are available for inspection.',
                'strength': 'low',
                'evidence_refs': _claim_paths(model_files, 'model_files'),
                'why_it_matters': 'Use these files to verify whether the paper loss depends on extra outputs from the model.',
            }
        )
    if config_files:
        claims.append(
            {
                'claim_id': 'config_training_controls_present',
                'label': 'Configuration files mention training or loss controls.',
                'strength': 'medium',
                'evidence_refs': _claim_paths(config_files, 'config_files'),
                'why_it_matters': 'Config files often expose loss weights, trainer flags, and model names that affect migration.',
            }
        )

    recommended_read_order: List[str] = []
    for key in ('loss_files', 'trainer_files', 'model_files', 'config_files'):
        recommended_read_order.extend(_claim_paths(code_inventory.get(key, []), key, limit=1))

    return {
        'schema_version': 'evidence_graph.v1',
        'summary': {
            'loss_files_count': len(loss_files),
            'trainer_files_count': len(trainer_files),
            'model_files_count': len(model_files),
            'config_files_count': len(config_files),
        },
        'claims': claims,
        'recommended_read_order': recommended_read_order,
    }


def find_loss_files(repo_path: str) -> List[Dict[str, Any]]:
    """
    智能发现 loss 相关文件

    优先级：
    1. 文件名包含 loss/criterion/objective
    2. 文件内容包含 def.*loss / class.*Loss
    """
    repo_root = Path(repo_path)
    inventory = _collect_code_inventory(repo_root)
    return [
        {
            'path': record['path'],
            'abs_path': record['abs_path'],
            'priority': record['score'],
            'size': record['size'],
            'signals': record.get('signals', []),
        }
        for record in inventory.get('loss_files', [])[:_PRIMARY_FILE_LIMIT]
    ]


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
    evidence_graph_path = output_dir / 'evidence_graph.json'

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

    # 1. 扫描代码仓库，保留旧 primary_files 接口，并补充 trainer/model/config inventory
    code_inventory = _collect_code_inventory(repo_path)
    loss_files = [
        {
            'path': record['path'],
            'abs_path': record['abs_path'],
            'priority': int(record['score']),
            'size': int(record['size']),
            'signals': list(record.get('signals', [])),
        }
        for record in code_inventory.get('loss_files', [])[:_PRIMARY_FILE_LIMIT]
    ]

    # 2. 读取文件内容并预处理
    primary_files = []
    for file_info in loss_files:
        try:
            content = _read_text(Path(file_info['abs_path']))

            primary_files.append({
                'path': file_info['path'],
                'content': preprocess_code(content),
                'functions': extract_functions(content),
                'imports': extract_imports(content),
                'priority': _priority_label(int(file_info['priority'])),
                'signals': file_info.get('signals', []),
            })
        except Exception as e:
            print(f"Warning: 无法读取 {file_info['path']}: {e}")
            continue

    evidence_graph = _build_evidence_graph(code_inventory)
    evidence_graph_path.write_text(
        json.dumps(evidence_graph, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )

    # 3. 读取 Loss IR schema
    schema_doc = '见 loss_transfer/ir/loss_ir_schema.py 中的 LossIR 数据类定义'

    # 4. 构建返回结果
    result = {
        'paper': paper_context,
        'primary_files': primary_files,
        'code_inventory': code_inventory,
        'evidence_graph': evidence_graph,
        'evidence_graph_path': str(evidence_graph_path),
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
