"""
@file validate_loss.py
@description 4层渐进式验证器
@author Leizheng
@date 2026-03-22
@version 1.0.0
"""

import ast
import sys
import json
import torch
import importlib.util
import subprocess
import re
import traceback
from pathlib import Path


def validate_static(loss_file_path: str) -> dict:
    """
    Layer 1: 静态检查 (<1s)
    - 语法检查
    - Import 白名单
    - 函数签名
    - 禁止模式
    """
    code = Path(loss_file_path).read_text()

    # 1. 语法检查
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            'passed': False,
            'error': 'syntax_error',
            'detail': str(e),
            'fix_hint': 'Python 语法错误，检查括号、冒号、缩进'
        }

    # 2. Import 白名单
    allowed = {'torch', 'torch.nn.functional', 'math'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in allowed:
                    return {
                        'passed': False,
                        'error': 'forbidden_import',
                        'detail': f'不允许 import {alias.name}',
                        'fix_hint': f'移除 import {alias.name}'
                    }
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module not in allowed:
                return {
                    'passed': False,
                    'error': 'forbidden_import',
                    'detail': f'不允许 from {node.module} import'
                }

    # 3. 函数签名检查
    sandbox_loss_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'sandbox_loss':
            sandbox_loss_func = node
            break

    if not sandbox_loss_func:
        return {
            'passed': False,
            'error': 'missing_function',
            'detail': '缺少 sandbox_loss 函数'
        }

    args = sandbox_loss_func.args
    arg_names = [a.arg for a in args.args]
    if 'pred' not in arg_names or 'target' not in arg_names:
        return {
            'passed': False,
            'error': 'invalid_signature',
            'detail': f'函数参数错误: {arg_names}'
        }

    # 4. 禁止模式
    forbidden = [
        ('open(', 'file_io'),
        ('subprocess', 'subprocess'),
        ('__import__', 'dynamic_import'),
    ]

    for pattern, error_type in forbidden:
        if pattern in code:
            return {
                'passed': False,
                'error': error_type,
                'detail': f'不能使用 {pattern}'
            }

    # 5. 已知失败模式
    code_lower = code.lower()
    if 'ssim' in code_lower:
        return {
            'passed': False,
            'error': 'blocked_pattern',
            'detail': 'SSIM loss 已知会崩溃 (exp#11)',
            'fix_hint': '移除 SSIM，使用 L1/L2/gradient'
        }

    if 'laplacian' in code_lower:
        return {
            'passed': False,
            'error': 'blocked_pattern',
            'detail': 'Laplacian 已知会崩溃 (exp#20,#40,#66)',
            'fix_hint': '移除 Laplacian，使用 Sobel/Scharr'
        }

    return {'passed': True}


def validate_single_model(loss_file_path: str, gpu_id: int = 4) -> dict:
    """
    Layer 3: Single Model (2-5min)
    - 真实数据，完整 15 epochs
    - 检查 SSIM 是否合理
    """
    import shutil
    import os

    # 复制 loss 文件到 sandbox/
    sandbox_dir = Path(__file__).parent.parent.parent / 'sandbox'
    target_loss = sandbox_dir / 'sandbox_loss.py'

    # 只在文件不同时复制
    if Path(loss_file_path).resolve() != target_loss.resolve():
        shutil.copy(loss_file_path, target_loss)

    # 运行训练
    project_root = Path(__file__).parent.parent.parent
    cmd = f'cd {sandbox_dir} && PYTHONPATH={project_root}/scripts/ocean-SR-training-masked:$PYTHONPATH CUDA_VISIBLE_DEVICES={gpu_id} /home/lz/miniconda3/envs/pytorch/bin/python _run_once.py --config configs/swinir.yaml'

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )
    except subprocess.TimeoutExpired:
        return {
            'passed': False,
            'error': 'timeout',
            'detail': '训练超时（>10分钟）'
        }

    if result.returncode != 0:
        # 训练崩溃
        stderr = result.stderr

        if 'CUDA out of memory' in stderr:
            return {
                'passed': False,
                'error': 'oom',
                'detail': 'GPU 显存不足'
            }
        elif 'nan' in stderr.lower():
            return {
                'passed': False,
                'error': 'nan_during_training',
                'detail': '训练过程中出现 NaN',
                'fix_hint': 'Loss 数值不稳定'
            }
        else:
            return {
                'passed': False,
                'error': 'crash',
                'detail': stderr[-500:]
            }

    # 解析指标
    stdout = result.stdout
    try:
        val_ssim = float(re.search(r'val_ssim:\s+([\d.]+)', stdout).group(1))
        val_psnr = float(re.search(r'val_psnr:\s+([\d.]+)', stdout).group(1))
    except:
        return {
            'passed': False,
            'error': 'parse_failed',
            'detail': '无法解析输出指标'
        }

    # 检查 SSIM
    if val_ssim < 0.3:
        return {
            'passed': False,
            'error': 'ssim_collapse',
            'detail': f'SSIM={val_ssim:.4f} 太低',
            'metrics': {'val_ssim': val_ssim, 'val_psnr': val_psnr},
            'fix_hint': '检查是否使用了已知失败组件'
        }

    return {
        'passed': True,
        'metrics': {
            'val_ssim': val_ssim,
            'val_psnr': val_psnr
        }
    }


def validate_smoke(loss_file_path: str) -> dict:
    """
    Layer 2: Smoke Test (<10s)
    - 动态导入
    - Dummy forward/backward
    - NaN/Inf 检查
    """
    # 动态导入
    try:
        spec = importlib.util.spec_from_file_location("test_loss", loss_file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sandbox_loss = mod.sandbox_loss
    except Exception as e:
        return {
            'passed': False,
            'error': 'import_failed',
            'detail': str(e),
            'traceback': traceback.format_exc()
        }

    # 创建 dummy 数据
    torch.manual_seed(42)
    pred = torch.randn(2, 128, 128, 2, requires_grad=True)
    target = torch.randn(2, 128, 128, 2)
    mask = torch.ones(1, 128, 128, 1, dtype=torch.bool)

    try:
        # Forward
        loss = sandbox_loss(pred, target, mask=mask)

        if not isinstance(loss, torch.Tensor):
            return {
                'passed': False,
                'error': 'invalid_output_type',
                'detail': f'返回类型是 {type(loss)}'
            }

        if loss.dim() != 0:
            return {
                'passed': False,
                'error': 'not_scalar',
                'detail': f'返回 shape 是 {loss.shape}'
            }

        if torch.isnan(loss):
            return {
                'passed': False,
                'error': 'nan_in_forward',
                'detail': 'Forward 返回 NaN',
                'fix_hint': '检查除法，在分母加 .clamp(min=1e-8)'
            }

        if torch.isinf(loss):
            return {
                'passed': False,
                'error': 'inf_in_forward',
                'detail': 'Forward 返回 Inf'
            }

        # Backward
        loss.backward()

        if pred.grad is None:
            return {
                'passed': False,
                'error': 'no_gradient',
                'detail': 'Backward 没有产生梯度'
            }

        if torch.isnan(pred.grad).any():
            return {
                'passed': False,
                'error': 'nan_in_gradient',
                'detail': 'Gradient 包含 NaN'
            }

        # 测试 mask=None
        pred.grad = None
        loss_no_mask = sandbox_loss(pred, target, mask=None)
        loss_no_mask.backward()

        if torch.isnan(loss_no_mask):
            return {
                'passed': False,
                'error': 'mask_none_failed',
                'detail': 'mask=None 时失败'
            }

        return {
            'passed': True,
            'loss_value': float(loss),
            'grad_norm': float(pred.grad.norm())
        }

    except Exception as e:
        return {
            'passed': False,
            'error': 'runtime_error',
            'detail': str(e),
            'traceback': traceback.format_exc()
        }


def validate_full_run(loss_file_path: str, baseline_thresholds: dict = None) -> dict:
    """
    Layer 4: Full Run (5-10min)
    - 4 模型并行训练
    - 比对基线阈值
    """
    import shutil

    sandbox_dir = Path(__file__).parent.parent.parent / 'sandbox'
    target_loss = sandbox_dir / 'sandbox_loss.py'
    shutil.copy(loss_file_path, target_loss)

    cmd = f'cd {sandbox_dir} && bash run_all_models.sh'

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=900
        )
    except subprocess.TimeoutExpired:
        return {'passed': False, 'error': 'timeout', 'detail': '训练超时（>15分钟）'}

    if result.returncode != 0:
        return {'passed': False, 'error': 'crash', 'detail': result.stderr[-500:]}

    # 解析 4 个模型的结果 (从日志文件读取)
    metrics = {}
    for model in ['SwinIR', 'EDSR', 'FNO2d', 'UNet2d']:
        log_file = sandbox_dir / f'run_{model}.log'
        if log_file.exists():
            log_content = log_file.read_text()
            ssim_match = re.search(r'val_ssim:\s+([\d.]+)', log_content)
            if ssim_match:
                metrics[model.lower()] = float(ssim_match.group(1))

    if len(metrics) < 4:
        return {'passed': False, 'error': 'parse_failed', 'detail': f'只解析到 {len(metrics)} 个模型', 'metrics': metrics}

    # 检查是否有模型崩溃
    if any(v < 0.3 for v in metrics.values()):
        return {'passed': False, 'error': 'model_collapse', 'metrics': metrics}

    # 如果有基线阈值,比对
    if baseline_thresholds:
        swinir_baseline = baseline_thresholds.get('swinir_mean', 0.6645)
        if metrics['swinir'] < swinir_baseline - 0.01:
            return {'passed': False, 'error': 'below_baseline', 'metrics': metrics}

    return {'passed': True, 'metrics': metrics}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_file', required=True)
    parser.add_argument('--mode', default='static')
    args = parser.parse_args()

    if args.mode == 'static':
        result = validate_static(args.loss_file)
    elif args.mode == 'smoke':
        result = validate_smoke(args.loss_file)
    elif args.mode == 'single':
        result = validate_single_model(args.loss_file)
    elif args.mode == 'full':
        result = validate_full_run(args.loss_file)
    else:
        result = {'error': f'Unknown mode: {args.mode}'}

    print(json.dumps(result))
