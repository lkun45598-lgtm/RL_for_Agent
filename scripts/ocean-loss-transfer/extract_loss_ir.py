"""
@file extract_loss_ir.py
@description 从论文 PDF + 代码提取 Loss IR
@author Leizheng
@date 2026-03-22
@version 1.0.0
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from .loss_ir_schema import LossIR, validate_loss_ir
    from .llm_extractor import extract_with_llm
except ImportError:
    from loss_ir_schema import LossIR, validate_loss_ir
    from llm_extractor import extract_with_llm


def scan_code_for_loss(code_repo_path: str) -> Dict[str, Any]:
    """扫描代码仓库,找到 loss 相关文件"""
    repo = Path(code_repo_path)
    loss_files = []
    
    # 搜索 loss 相关文件
    for pattern in ['*loss*.py', '*criterion*.py', '*objective*.py']:
        loss_files.extend(repo.rglob(pattern))
    
    code_snippets = []
    for f in loss_files[:5]:  # 最多 5 个文件
        try:
            content = f.read_text()
            if len(content) < 10000:  # 只读小文件
                code_snippets.append({
                    'file': str(f.relative_to(repo)),
                    'content': content[:2000]  # 前 2000 字符
                })
        except:
            pass
    
    return {'files': [str(f.relative_to(repo)) for f in loss_files], 'snippets': code_snippets}


def generate_template_yaml(output_path: str) -> str:
    """生成模板 Loss IR YAML 供手动填写"""
    template = {
        'metadata': {
            'paper_title': 'TODO: 论文标题',
            'paper_url': 'TODO: 论文链接',
            'code_repo': 'TODO: 代码仓库'
        },
        'interface': {
            'input_tensors': [
                {'name': 'pred', 'shape': '[B,H,W,C]', 'required': True},
                {'name': 'target', 'shape': '[B,H,W,C]', 'required': True}
            ]
        },
        'components': [
            {
                'name': 'pixel_loss',
                'type': 'pixel_loss',
                'weight': 1.0,
                'implementation': {
                    'reduction': 'mean',
                    'operates_on': 'pixel_space'
                }
            }
        ],
        'multi_scale': {'enabled': False},
        'combination': {'method': 'weighted_sum'},
        'incompatibility_flags': {
            'requires_model_features': False,
            'requires_pretrained_network': False,
            'requires_adversarial': False,
            'requires_multiple_forward_passes': False
        }
    }
    
    Path(output_path).write_text(yaml.dump(template, allow_unicode=True))
    return output_path


def extract_loss_ir(
    paper_pdf_path: Optional[str] = None,
    code_repo_path: Optional[str] = None,
    output_yaml_path: str = 'loss_ir.yaml',
    manual_mode: bool = False
) -> str:
    """
    提取 Loss IR
    
    Args:
        paper_pdf_path: 论文 PDF 路径
        code_repo_path: 代码仓库路径
        output_yaml_path: 输出 YAML 路径
        manual_mode: 手动模式 (生成模板)
    
    Returns:
        输出文件路径
    """
    
    # 手动模式: 生成模板
    if manual_mode or (not paper_pdf_path and not code_repo_path):
        print('Generating template YAML for manual editing...')
        return generate_template_yaml(output_yaml_path)
    
    # 扫描代码
    code_info = {}
    if code_repo_path and Path(code_repo_path).exists():
        print(f'Scanning code repo: {code_repo_path}')
        code_info = scan_code_for_loss(code_repo_path)
        print(f'Found {len(code_info["files"])} loss-related files')

    # LLM 自动提取
    loss_ir = {
        'metadata': {
            'paper_title': Path(paper_pdf_path).stem if paper_pdf_path else 'Unknown',
            'code_repo': code_repo_path or 'N/A',
            'loss_files': code_info.get('files', [])
        },
        'interface': {'input_tensors': []},
        'components': [],
        'multi_scale': {'enabled': False},
        'combination': {'method': 'weighted_sum'},
        'incompatibility_flags': {}
    }

    # 如果有代码片段,用 LLM 提取
    if code_info.get('snippets'):
        print('Extracting with LLM...')
        try:
            extracted = extract_with_llm(code_info['snippets'])
            if extracted:
                loss_ir.update(extracted)
                print('✓ LLM extraction completed')
        except Exception as e:
            print(f'⚠️  LLM extraction failed: {e}, using template')

    output_path = Path(output_yaml_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.dump(loss_ir, allow_unicode=True))
    print(f'Loss IR saved to: {output_yaml_path}')
    print('NOTE: This is a basic extraction. Please review and complete manually.')

    return output_yaml_path
