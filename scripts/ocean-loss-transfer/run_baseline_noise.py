"""
@file run_baseline_noise.py
@description 测量当前最优 loss 的性能噪声范围
@author Leizheng
@date 2026-03-22
@version 1.0.0
"""

import subprocess
import re
import yaml
import numpy as np
from pathlib import Path


def run_baseline_noise(n_runs: int = 3, model: str = 'swinir', gpu_id: int = 4) -> dict:
    """
    运行 n 次当前 loss,测量噪声
    
    Returns:
        {swinir_mean, swinir_std, psnr_mean, psnr_std, ...}
    """
    sandbox_dir = Path(__file__).parent.parent.parent / 'sandbox'
    config_file = f'configs/{model}.yaml'
    
    ssim_values = []
    psnr_values = []
    
    for i in range(n_runs):
        print(f'Running baseline trial {i+1}/{n_runs}...')
        
        cmd = f'cd {sandbox_dir} && CUDA_VISIBLE_DEVICES={gpu_id} /home/lz/miniconda3/envs/pytorch/bin/python _run_once.py --config {config_file}'
        
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
    
    thresholds = {
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
