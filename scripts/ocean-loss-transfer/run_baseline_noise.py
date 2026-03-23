"""
@file run_baseline_noise.py
@description 测量当前最优 loss 的性能噪声范围
@author Leizheng
@contributors kongzhiquan
@date 2026-03-22
@version 1.2.0

@changelog
  - 2026-03-22 Leizheng: v1.0.0 initial version
  - 2026-03-23 kongzhiquan: v1.1.0 refine type annotations
  - 2026-03-23 kongzhiquan: v1.2.0 use find_first_python_path instead of hardcoded path
"""

import subprocess
import re
import sys
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Union
sys.path.append(str(Path(__file__).parent.parent)) # 添加上层目录（scripts）到路径，以便导入 python_manager
from python_manager import find_first_python_path

from _types import BaselineThresholds

_PYTHON = find_first_python_path() or 'python3'


# 失败时返回的错误结果
BaselineResult = Union[BaselineThresholds, Dict[str, str]]


def run_baseline_noise(n_runs: int = 3, model: str = 'swinir', gpu_id: int = 4) -> BaselineResult:
    """
    运行 n 次当前 loss,测量噪声

    Returns:
        BaselineThresholds on success, or error dict on failure
    """
    sandbox_dir = Path(__file__).parent.parent.parent / 'sandbox'
    config_file = f'configs/{model}.yaml'

    ssim_values: List[float] = []
    psnr_values: List[float] = []

    for i in range(n_runs):
        print(f'Running baseline trial {i+1}/{n_runs}...')

        cmd = f'cd {sandbox_dir} && CUDA_VISIBLE_DEVICES={gpu_id} {_PYTHON} _run_once.py --config {config_file}'

        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            print(f'Trial {i+1} timeout')
            continue

        if result.returncode != 0:
            print(f'Trial {i+1} failed')
            continue

        # 解析指标
        stdout = result.stdout
        ssim_match = re.search(r'val_ssim:\s+([\d.]+)', stdout)
        psnr_match = re.search(r'val_psnr:\s+([\d.]+)', stdout)

        if ssim_match:
            ssim_values.append(float(ssim_match.group(1)))
        if psnr_match:
            psnr_values.append(float(psnr_match.group(1)))

    if len(ssim_values) < 2:
        return {'error': 'Not enough successful runs'}

    # 计算统计量
    ssim_mean = float(np.mean(ssim_values))
    ssim_std = float(np.std(ssim_values))
    psnr_mean = float(np.mean(psnr_values))
    psnr_std = float(np.std(psnr_values))

    thresholds: BaselineThresholds = {
        'model': model,
        'n_runs': len(ssim_values),
        'ssim_mean': ssim_mean,
        'ssim_std': ssim_std,
        'psnr_mean': psnr_mean,
        'psnr_std': psnr_std,
        'viable_threshold': ssim_mean - ssim_std,
        'improvement_threshold': ssim_mean + ssim_std
    }

    # 保存
    output_file = Path(__file__).parent.parent.parent / 'workflow/loss_transfer/baseline_thresholds.yaml'
    output_file.write_text(yaml.dump(thresholds))

    return thresholds
