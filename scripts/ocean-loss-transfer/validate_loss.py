"""
@file validate_loss.py
@description 4层渐进式验证器
@author Leizheng
@contributors kongzhiquan
@date 2026-03-22
@version 1.3.0

@changelog
  - 2026-03-22 Leizheng: v1.0.0 initial version
  - 2026-03-23 kongzhiquan: v1.1.0 refine type annotations, use ValidationResult TypedDict, fix bare except
  - 2026-03-23 kongzhiquan: v1.2.0 use find_first_python_path instead of hardcoded path
  - 2026-03-24 Leizheng: v1.3.0 enhanced validation (borrowing from AutoResearchClaw):
    - Layer 1: undefined function detection, variable scoping check, device consistency warning
    - Layer 2: gradient magnitude analysis, multi-shape test, boundary condition test
    - Layer 3/4: timeout recovery with partial metrics parsing
"""

import ast
import math
import os
import sys
import json
import tempfile
import torch
import importlib.util
import subprocess
import re
import traceback
from pathlib import Path
from typing import Dict, List, Optional
sys.path.append(str(Path(__file__).parent.parent))  # 添加上层目录（scripts）到路径，以便导入 python_manager
from python_manager import find_first_python_path
from sandbox_adapter_bridge import (
    build_smoke_loss_kwargs,
    load_formula_spec,
    requires_sandbox_adapter,
    write_config_with_adapter,
)
from _types import (
    ValidationResult, TrainingMetrics,
    StaticWarning, GradientAnalysis, SmokeTestDetail,
)
from _utils import parse_training_events

_PYTHON = find_first_python_path() or 'python3'

# Python 内置函数名集合
_BUILTINS = set(__builtins__.keys()) if isinstance(__builtins__, dict) else set(dir(__builtins__))

_SANDBOX_DIR = Path(__file__).parent.parent.parent / 'sandbox'
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PIPELINE_DIR = _PROJECT_ROOT / 'scripts' / 'ocean-SR-training-masked'
_FULL_RUN_MODEL_CONFIGS = {
    'SwinIR': 'swinir.yaml',
    'EDSR': 'edsr.yaml',
    'FNO2d': 'fno2d.yaml',
    'UNet2d': 'unet2d.yaml',
}


def _copy_loss_to_sandbox(loss_file_path: str) -> None:
    import shutil

    target_loss = _SANDBOX_DIR / 'sandbox_loss.py'
    if Path(loss_file_path).resolve() != target_loss.resolve():
        shutil.copy(loss_file_path, target_loss)


def _build_run_env(gpu_id: Optional[int] = None, extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = os.environ.copy()
    pipeline_path = str(_PIPELINE_DIR)
    existing_pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f'{pipeline_path}:{existing_pythonpath}' if existing_pythonpath else pipeline_path
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    if extra_env:
        env.update(extra_env)
    return env


def _prepare_config_path(
    config_name: str,
    formula_spec: Optional[Dict[str, object]],
    temp_dir: Optional[str],
    dataset_root: Optional[str] = None,
) -> str:
    base_config_path = _SANDBOX_DIR / 'configs' / config_name
    if not requires_sandbox_adapter(formula_spec) and not dataset_root:
        return str(base_config_path)
    if not temp_dir:
        raise ValueError('temp_dir is required when a temporary sandbox config must be synthesized')

    output_path = Path(temp_dir) / config_name
    write_config_with_adapter(
        str(base_config_path),
        formula_spec,
        str(output_path),
        dataset_root=dataset_root,
    )
    return str(output_path)


def _collect_valid_epochs(training_curve: Dict[str, object]) -> List[Dict[str, object]]:
    epochs = training_curve.get('epochs', [])
    if not isinstance(epochs, list):
        return []
    return [
        ep for ep in epochs
        if isinstance(ep, dict) and ep.get('ssim') is not None and ep.get('psnr') is not None
    ]


def _collect_nan_metrics(training_curve: Dict[str, object]) -> List[str]:
    epochs = training_curve.get('epochs', [])
    if not isinstance(epochs, list):
        return []

    nan_points: List[str] = []
    for ep in epochs:
        if not isinstance(ep, dict):
            continue
        epoch_id = ep.get('epoch', -1)
        for key in ('train_loss', 'valid_loss', 'ssim', 'psnr', 'rmse'):
            value = ep.get(key)
            if isinstance(value, float) and math.isnan(value):
                nan_points.append(f'epoch={epoch_id}:{key}')
    return nan_points


def _check_undefined_functions(tree: ast.Module) -> Optional[ValidationResult]:
    """
    检查裸函数调用是否都有定义。
    只检查 Name 节点的调用（如 my_func()），不检查属性调用（如 torch.abs()）。
    """
    # 收集所有函数/类定义名
    defined_names: set = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined_names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            defined_names.add(node.name)

    # 检查所有裸函数调用
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in defined_names and func_name not in _BUILTINS:
                return {
                    'passed': False,
                    'error': 'undefined_function',
                    'detail': f'函数 `{func_name}` 被调用但未在文件内定义',
                    'fix_hint': f'定义 `{func_name}` 或从 torch/F/math 中调用'
                }

    return None


def _collect_assigned_names(stmts: List[ast.stmt]) -> set:
    """收集语句列表中直接赋值的变量名（不递归进子块）"""
    names: set = set()
    for stmt in stmts:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else ([stmt.target] if stmt.target else [])
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            names.add(elt.id)
        elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            names.add(stmt.target.id)
    return names


def _collect_referenced_names(stmts: List[ast.stmt]) -> set:
    """收集语句列表中所有被引用（Load）的变量名"""
    names: set = set()
    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                names.add(node.id)
    return names


def _check_variable_scoping(tree: ast.Module) -> Optional[ValidationResult]:
    """
    检查 sandbox_loss 函数中的变量作用域问题。
    目标: 发现只在条件块内赋值但在外部使用的变量。
    """
    # 找到 sandbox_loss 函数
    sandbox_func: Optional[ast.FunctionDef] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'sandbox_loss':
            sandbox_func = node
            break

    if not sandbox_func:
        return None  # validate_static 已经检查了

    body = sandbox_func.body

    # 收集函数参数名
    param_names = {a.arg for a in sandbox_func.args.args}
    if sandbox_func.args.kwarg:
        param_names.add(sandbox_func.args.kwarg.arg)
    if sandbox_func.args.vararg:
        param_names.add(sandbox_func.args.vararg.arg)

    # 收集顶层无条件赋值
    unconditional = _collect_assigned_names(body)

    # 对每个 if 条件块，检查是否有仅在内部赋值的变量被后续引用
    # 注意：不检查 for/try/with，因为它们的 body 总是执行
    for i, stmt in enumerate(body):
        if not isinstance(stmt, ast.If):
            continue

        # 收集该 if 块内部赋值的名字（包括 else 分支）
        inner_stmts: List[ast.stmt] = stmt.body + stmt.orelse

        conditional_assigns = _collect_assigned_names(inner_stmts)

        # 只关心「仅在条件块内赋值」的变量（不在顶层无条件赋值中）
        risky_names = conditional_assigns - unconditional - param_names

        if not risky_names:
            continue

        # 检查后续语句是否引用这些变量
        subsequent = body[i + 1:]
        if not subsequent:
            continue

        referenced_after = _collect_referenced_names(subsequent)
        problematic = risky_names & referenced_after

        if problematic:
            var_name = sorted(problematic)[0]
            return {
                'passed': False,
                'error': 'variable_scope_error',
                'detail': f'变量 `{var_name}` 仅在条件块内赋值，但在外部被引用（可能 NameError）',
                'fix_hint': f'在条件块之前初始化 `{var_name}`，如 `{var_name} = None`'
            }

    return None


def _check_device_consistency(code: str) -> List[StaticWarning]:
    """
    检查硬编码设备调用（警告，不阻断）。
    """
    warnings: List[StaticWarning] = []
    lines = code.split('\n')

    patterns = [
        (r'\.cuda\(\)', 'device_hardcoded', '使用 .cuda() 硬编码设备，建议用 tensor.device'),
        (r'\.to\([\'"]cuda[\'"]', 'device_hardcoded', '使用 .to("cuda") 硬编码设备'),
        (r'\.to\([\'"]cpu[\'"]', 'device_hardcoded', '使用 .to("cpu") 硬编码设备'),
        (r'\.to\([\'"]cuda:\d+[\'"]', 'device_hardcoded', '使用 .to("cuda:N") 硬编码设备'),
    ]

    for line_num, line in enumerate(lines, 1):
        for pattern, warn_type, detail in patterns:
            if re.search(pattern, line):
                warnings.append({
                    'type': warn_type,
                    'detail': detail,
                    'line': line_num,
                })

    return warnings


def validate_static(loss_file_path: str) -> ValidationResult:
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
    sandbox_loss_func: Optional[ast.FunctionDef] = None
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
    arg_names: List[str] = [a.arg for a in args.args]
    if 'pred' not in arg_names or 'target' not in arg_names:
        return {
            'passed': False,
            'error': 'invalid_signature',
            'detail': f'函数参数错误: {arg_names}'
        }

    # 4. 禁止模式
    forbidden: List[tuple[str, str]] = [
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

    # 6. 未定义函数检测
    undef_result = _check_undefined_functions(tree)
    if undef_result:
        return undef_result

    # 7. 变量作用域检查
    scope_result = _check_variable_scoping(tree)
    if scope_result:
        return scope_result

    # 8. 设备一致性警告（不阻断）
    warnings = _check_device_consistency(code)
    if warnings:
        return {'passed': True, 'warnings': warnings}

    return {'passed': True}


def validate_single_model(
    loss_file_path: str,
    gpu_id: int = 4,
    formula_spec_path: Optional[str] = None,
    dataset_root: Optional[str] = None,
) -> ValidationResult:
    """
    Layer 3: Single Model (2-5min)
    - 真实数据，完整 15 epochs
    - 检查 SSIM 是否合理
    """
    formula_spec = load_formula_spec(formula_spec_path)
    _copy_loss_to_sandbox(loss_file_path)

    with tempfile.TemporaryDirectory(prefix='sandbox_single_') as temp_dir:
        config_path = _prepare_config_path(
            'swinir.yaml',
            formula_spec,
            temp_dir,
            dataset_root=dataset_root,
        )
        cmd = [_PYTHON, '_run_once.py', '--config', config_path]

        try:
            result = subprocess.run(
                cmd,
                cwd=_SANDBOX_DIR,
                env=_build_run_env(gpu_id=gpu_id),
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
        except subprocess.TimeoutExpired as e:
            # 改进6: 超时恢复 - 解析已完成的部分结果
            partial_stdout = ''
            if e.stdout:
                partial_stdout = e.stdout if isinstance(e.stdout, str) else e.stdout.decode('utf-8', errors='replace')

            training_curve = parse_training_events(partial_stdout)
            partial_metrics: TrainingMetrics = {}
            valid_epochs = [ep for ep in training_curve.get('epochs', []) if ep.get('ssim') is not None]
            if valid_epochs:
                last = valid_epochs[-1]
                partial_metrics = {'val_ssim': last.get('ssim', 0), 'val_psnr': last.get('psnr', 0)}

            detail = f'训练超时（>10分钟）'
            if training_curve.get('last_epoch', -1) >= 0:
                detail += f'，已完成 {training_curve["last_epoch"]} epoch'
                if valid_epochs:
                    detail += f'，最后 SSIM={valid_epochs[-1].get("ssim", "?"):.4f}' if isinstance(valid_epochs[-1].get("ssim"), float) else ''
                detail += f'，趋势: {training_curve.get("trend", "unknown")}'

            return {
                'passed': False,
                'error': 'timeout',
                'detail': detail,
                'partial_metrics': partial_metrics,
                'training_curve': training_curve,
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
        training_curve = parse_training_events(stdout)
        valid_epochs = _collect_valid_epochs(training_curve)
        nan_metrics = _collect_nan_metrics(training_curve)

        if nan_metrics:
            partial_metrics: TrainingMetrics = {}
            if valid_epochs:
                last_valid = valid_epochs[-1]
                partial_metrics = {
                    'val_ssim': last_valid.get('ssim', 0),
                    'val_psnr': last_valid.get('psnr', 0),
                }

            detail = '训练过程中出现 NaN'
            if valid_epochs:
                last_valid = valid_epochs[-1]
                detail += (
                    f'，最后有效 epoch={last_valid.get("epoch", "?")}'
                    f'，SSIM={last_valid.get("ssim", 0):.4f}'
                    f'，PSNR={last_valid.get("psnr", 0):.4f}'
                )

            return {
                'passed': False,
                'error': 'nan_during_training',
                'detail': detail,
                'partial_metrics': partial_metrics,
                'training_curve': training_curve,
                'fix_hint': 'Loss 数值不稳定',
            }

        try:
            if valid_epochs:
                last_valid = valid_epochs[-1]
                val_ssim = float(last_valid['ssim'])  # type: ignore[arg-type]
                val_psnr = float(last_valid['psnr'])  # type: ignore[arg-type]
            else:
                ssim_match = re.search(r'val_ssim:\s+([\d.]+)', stdout)
                psnr_match = re.search(r'val_psnr:\s+([\d.]+)', stdout)
                if not ssim_match or not psnr_match:
                    raise ValueError("Missing val_ssim or val_psnr in output")
                val_ssim = float(ssim_match.group(1))
                val_psnr = float(psnr_match.group(1))
        except (ValueError, AttributeError) as e:
            return {
                'passed': False,
                'error': 'parse_failed',
                'detail': f'无法解析输出指标: {e}'
            }

        metrics: TrainingMetrics = {'val_ssim': val_ssim, 'val_psnr': val_psnr}

        # 检查 SSIM
        if val_ssim < 0.3:
            return {
                'passed': False,
                'error': 'ssim_collapse',
                'detail': f'SSIM={val_ssim:.4f} 太低',
                'metrics': metrics,
                'fix_hint': '检查是否使用了已知失败组件'
            }

        return {
            'passed': True,
            'metrics': metrics
        }


def validate_smoke(loss_file_path: str, formula_spec_path: Optional[str] = None) -> ValidationResult:
    """
    Layer 2: Smoke Test (<10s)
    - 动态导入
    - Dummy forward/backward
    - NaN/Inf 检查
    """
    # 动态导入
    try:
        spec = importlib.util.spec_from_file_location("test_loss", loss_file_path)
        if spec is None or spec.loader is None:
            return {
                'passed': False,
                'error': 'import_failed',
                'detail': f'Cannot create module spec from {loss_file_path}'
            }
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        sandbox_loss = mod.sandbox_loss
    except Exception as e:
        return {
            'passed': False,
            'error': 'import_failed',
            'detail': str(e),
            'traceback': traceback.format_exc()
        }

    # 创建 dummy 数据
    formula_spec = load_formula_spec(formula_spec_path)
    torch.manual_seed(42)
    pred = torch.randn(2, 128, 128, 2, requires_grad=True)
    target = torch.randn(2, 128, 128, 2)
    mask = torch.ones(2, 128, 128, 1, dtype=torch.bool)
    loss_kwargs = build_smoke_loss_kwargs(formula_spec, pred)

    try:
        # Forward
        loss = sandbox_loss(pred, target, mask=mask, **loss_kwargs)

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
        loss_no_mask = sandbox_loss(pred, target, mask=None, **loss_kwargs)
        loss_no_mask.backward()

        if torch.isnan(loss_no_mask):
            return {
                'passed': False,
                'error': 'mask_none_failed',
                'detail': 'mask=None 时失败'
            }

        # === 改进2: 梯度幅度分析 ===
        grad_norm_val = float(pred.grad.norm())
        grad_analysis: GradientAnalysis = {
            'grad_norm': grad_norm_val,
            'grad_min': float(pred.grad.min()),
            'grad_max': float(pred.grad.max()),
        }

        if grad_norm_val < 1e-7:
            grad_analysis['warning'] = 'vanishing'
            return {
                'passed': False,
                'error': 'gradient_vanish',
                'detail': f'梯度范数 = {grad_norm_val:.2e}，可能梯度消失',
                'fix_hint': 'Loss 函数可能无法有效传播梯度',
                'smoke_detail': {'gradient_analysis': grad_analysis}
            }

        if grad_norm_val > 1e5:
            grad_analysis['warning'] = 'exploding'
            return {
                'passed': False,
                'error': 'gradient_explode',
                'detail': f'梯度范数 = {grad_norm_val:.2e}，可能梯度爆炸',
                'fix_hint': '添加梯度裁剪或数值稳定化 (clamp, eps)',
                'smoke_detail': {'gradient_analysis': grad_analysis}
            }

        # === 改进2: 多尺寸输入测试 ===
        shapes_tested = ['(2, 128, 128, 2)']
        additional_shapes = [(1, 64, 64, 2), (4, 256, 256, 2)]
        for shape in additional_shapes:
            try:
                p = torch.randn(*shape, requires_grad=True)
                t = torch.randn(*shape)
                m = torch.ones(shape[0], shape[1], shape[2], 1, dtype=torch.bool)
                shape_kwargs = build_smoke_loss_kwargs(formula_spec, p)
                l = sandbox_loss(p, t, mask=m, **shape_kwargs)
                if torch.isnan(l) or torch.isinf(l):
                    return {
                        'passed': False,
                        'error': 'shape_dependent_nan',
                        'detail': f'输入 shape {shape} 时出现 NaN/Inf'
                    }
                l.backward()
                shapes_tested.append(str(shape))
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    # OOM 跳过该 shape，不视为失败
                    shapes_tested.append(f'{shape} (skipped: OOM)')
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                else:
                    return {
                        'passed': False,
                        'error': 'shape_dependent_error',
                        'detail': f'输入 shape {shape} 时出错: {e}',
                        'traceback': traceback.format_exc()
                    }
            except Exception as e:
                return {
                    'passed': False,
                    'error': 'shape_dependent_error',
                    'detail': f'输入 shape {shape} 时出错: {e}',
                    'traceback': traceback.format_exc()
                }

        # === 改进2: 边界条件测试 (pred ≈ target) ===
        pred_boundary = torch.randn(2, 128, 128, 2)
        target_boundary = pred_boundary.clone() + torch.randn_like(pred_boundary) * 1e-4
        pred_boundary.requires_grad_(True)
        try:
            boundary_kwargs = build_smoke_loss_kwargs(formula_spec, pred_boundary)
            loss_boundary = sandbox_loss(pred_boundary, target_boundary, mask=None, **boundary_kwargs)
            if torch.isnan(loss_boundary) or torch.isinf(loss_boundary):
                return {
                    'passed': False,
                    'error': 'boundary_instability',
                    'detail': f'pred ≈ target 时出现 NaN/Inf (差值 ~1e-4)',
                    'fix_hint': '分母除以近零范数，添加 .clamp(min=eps)'
                }
            loss_boundary.backward()
            if pred_boundary.grad is not None and torch.isnan(pred_boundary.grad).any():
                return {
                    'passed': False,
                    'error': 'boundary_gradient_nan',
                    'detail': 'pred ≈ target 时梯度包含 NaN',
                    'fix_hint': '近零残差时梯度不稳定'
                }
        except Exception as e:
            return {
                'passed': False,
                'error': 'boundary_instability',
                'detail': f'边界条件测试失败: {e}',
                'traceback': traceback.format_exc()
            }

        smoke_detail: SmokeTestDetail = {
            'shapes_tested': shapes_tested,
            'boundary_test_passed': True,
            'gradient_analysis': grad_analysis,
        }

        return {
            'passed': True,
            'loss_value': float(loss),
            'grad_norm': grad_norm_val,
            'smoke_detail': smoke_detail,
        }

    except Exception as e:
        return {
            'passed': False,
            'error': 'runtime_error',
            'detail': str(e),
            'traceback': traceback.format_exc()
        }


def validate_full_run(
    loss_file_path: str,
    baseline_thresholds: Optional[Dict[str, float]] = None,
    formula_spec_path: Optional[str] = None,
    dataset_root: Optional[str] = None,
) -> ValidationResult:
    """
    Layer 4: Full Run (5-10min)
    - 4 模型并行训练
    - 比对基线阈值
    """
    formula_spec = load_formula_spec(formula_spec_path)
    _copy_loss_to_sandbox(loss_file_path)

    for model_name in _FULL_RUN_MODEL_CONFIGS:
        log_file = _SANDBOX_DIR / f'run_{model_name}.log'
        if log_file.exists():
            log_file.unlink()

    extra_env: Dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix='sandbox_full_') as temp_dir:
        if requires_sandbox_adapter(formula_spec) or dataset_root:
            for config_name in _FULL_RUN_MODEL_CONFIGS.values():
                _prepare_config_path(
                    config_name,
                    formula_spec,
                    temp_dir,
                    dataset_root=dataset_root,
                )
            extra_env['SANDBOX_CONFIG_DIR'] = temp_dir

        try:
            result = subprocess.run(
                ['bash', 'run_all_models.sh'],
                cwd=_SANDBOX_DIR,
                env=_build_run_env(extra_env=extra_env),
                capture_output=True,
                text=True,
                timeout=900
            )
        except subprocess.TimeoutExpired:
        # 改进6: 超时恢复 - 从已写入的日志文件解析部分结果
            partial_metrics: TrainingMetrics = {}
            for model in _FULL_RUN_MODEL_CONFIGS:
                log_file = _SANDBOX_DIR / f'run_{model}.log'
                if log_file.exists():
                    log_content = log_file.read_text()
                    training_curve = parse_training_events(log_content)
                    valid_epochs = _collect_valid_epochs(training_curve)
                    if valid_epochs:
                        partial_metrics[model.lower()] = float(valid_epochs[-1]['ssim'])  # type: ignore[literal-required]
                        continue
                    ssim_match = re.search(r'val_ssim:\s+([\d.]+)', log_content)
                    if ssim_match:
                        partial_metrics[model.lower()] = float(ssim_match.group(1))  # type: ignore[literal-required]

            detail = f'训练超时（>15分钟），已解析 {len(partial_metrics)}/4 个模型结果'
            return {
                'passed': False,
                'error': 'timeout',
                'detail': detail,
                'partial_metrics': partial_metrics,
            }

        if result.returncode != 0:
            return {'passed': False, 'error': 'crash', 'detail': result.stderr[-500:]}

        # 解析 4 个模型的结果 (从日志文件读取)
        metrics: TrainingMetrics = {}
        nan_models: List[str] = []
        for model in _FULL_RUN_MODEL_CONFIGS:
            log_file = _SANDBOX_DIR / f'run_{model}.log'
            if log_file.exists():
                log_content = log_file.read_text()
                training_curve = parse_training_events(log_content)
                if _collect_nan_metrics(training_curve):
                    nan_models.append(model)
                valid_epochs = _collect_valid_epochs(training_curve)
                if valid_epochs:
                    metrics[model.lower()] = float(valid_epochs[-1]['ssim'])  # type: ignore[literal-required]
                    continue
                ssim_match = re.search(r'val_ssim:\s+([\d.]+)', log_content)
                if ssim_match:
                    metrics[model.lower()] = float(ssim_match.group(1))  # type: ignore[literal-required]

        if nan_models:
            return {
                'passed': False,
                'error': 'nan_during_training',
                'detail': '以下模型训练过程中出现 NaN: ' + ', '.join(sorted(nan_models)),
                'metrics': metrics,
                'fix_hint': 'Loss 数值不稳定',
            }

        if len(metrics) < 4:
            return {'passed': False, 'error': 'parse_failed', 'detail': f'只解析到 {len(metrics)} 个模型', 'metrics': metrics}

        # 检查是否有模型崩溃
        if any(v < 0.3 for v in metrics.values()):
            return {'passed': False, 'error': 'model_collapse', 'metrics': metrics}

        # 如果有基线阈值,比对
        if baseline_thresholds:
            swinir_baseline = baseline_thresholds.get('swinir_mean', 0.6645)
            swinir_ssim = metrics.get('swinir', 0)  # type: ignore[arg-type]
            if swinir_ssim < swinir_baseline - 0.01:
                return {'passed': False, 'error': 'below_baseline', 'metrics': metrics}

        return {'passed': True, 'metrics': metrics}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_file', required=True)
    parser.add_argument('--mode', default='static')
    parser.add_argument('--formula_spec', default=None)
    parser.add_argument('--dataset_root', default=None)
    args = parser.parse_args()

    result: ValidationResult
    if args.mode == 'static':
        result = validate_static(args.loss_file)
    elif args.mode == 'smoke':
        result = validate_smoke(args.loss_file, formula_spec_path=args.formula_spec)
    elif args.mode == 'single':
        result = validate_single_model(
            args.loss_file,
            formula_spec_path=args.formula_spec,
            dataset_root=args.dataset_root,
        )
    elif args.mode == 'full':
        result = validate_full_run(
            args.loss_file,
            formula_spec_path=args.formula_spec,
            dataset_root=args.dataset_root,
        )
    else:
        result = {'passed': False, 'error': f'Unknown mode: {args.mode}'}

    print(json.dumps(result))
