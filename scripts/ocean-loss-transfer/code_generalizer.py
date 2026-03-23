"""
@file code_generalizer.py
@description 将测试通过的代码泛化为可复用 torch 模块
@author Leizheng
@contributors kongzhiquan
@date 2026-03-22
@version 1.1.0

@changelog
  - 2026-03-22 Leizheng: v1.0.0 initial version
  - 2026-03-23 kongzhiquan: v1.1.0 refine type annotations
"""

import ast
from pathlib import Path
from typing import Dict, Optional
from _types import Innovation

def extract_component_function(loss_file: str, func_name: str) -> Optional[str]:
    """从 sandbox_loss.py 中提取单个组件函数"""
    code = Path(loss_file).read_text()
    tree = ast.parse(code)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            # 提取函数代码
            lines = code.split('\n')
            start = node.lineno - 1
            end = node.end_lineno
            if end is not None:
                return '\n'.join(lines[start:end])

    return None


def generalize_to_module(
    component_name: str,
    func_code: str,
    innovation: Innovation,
    output_path: str
) -> str:
    """将组件函数泛化为 torch 模块"""

    module_code = f'''"""
{component_name} - from {innovation['paper']}
Improvement: +{innovation['improvement']:.4f} SSIM
{innovation['why_works']}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class {component_name}(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask=None):
        # 原始实现
{_indent_code(func_code, 8)}
        return loss
'''

    Path(output_path).write_text(module_code)
    return output_path


def _indent_code(code: str, spaces: int) -> str:
    """缩进代码"""
    lines = code.split('\n')
    return '\n'.join(' ' * spaces + line if line.strip() else '' for line in lines)


def generalize_best_trial(
    paper_slug: str,
    trial_dir: str,
    innovation: Innovation
) -> Optional[str]:
    """泛化最佳 trial 的代码"""
    loss_file = Path(trial_dir) / 'sandbox_loss.py'
    if not loss_file.exists():
        return None

    # 提取 FFT loss 函数作为示例
    func_code = extract_component_function(str(loss_file), '_fft_loss')
    if not func_code:
        return None

    # 生成模块名
    module_name = f"{paper_slug.replace('-', '_')}_loss"
    output_path = Path(__file__).parent.parent.parent / f'workflow/loss_transfer/knowledge_base/modules/{module_name}.py'

    return generalize_to_module(module_name, func_code, innovation, str(output_path))
