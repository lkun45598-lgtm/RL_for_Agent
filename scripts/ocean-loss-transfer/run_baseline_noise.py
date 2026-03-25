"""
@file run_baseline_noise.py
@description 测量原始训练默认 loss 的性能噪声范围
@author Leizheng
@contributors kongzhiquan
@date 2026-03-22
@version 1.3.0

@changelog
  - 2026-03-22 Leizheng: v1.0.0 initial version
  - 2026-03-23 kongzhiquan: v1.1.0 refine type annotations
  - 2026-03-23 kongzhiquan: v1.2.0 use find_first_python_path instead of hardcoded path
  - 2026-03-25 OpenAI Codex: v1.3.0 run baseline through the original
    training entrypoint instead of sandbox_loss.py; synthesize temp configs with
    dataset-aware metadata and AMP disabled for fair comparison to transferred losses
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

sys.path.append(str(Path(__file__).parent.parent))  # 添加上层目录（scripts）到路径，以便导入 python_manager
from python_manager import find_first_python_path

from _types import BaselineThresholds
from _utils import parse_training_events
from sandbox_adapter_bridge import write_config_with_adapter

_PYTHON = sys.executable or find_first_python_path() or 'python3'

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PIPELINE_DIR = _PROJECT_ROOT / 'scripts' / 'ocean-SR-training-masked'
_SANDBOX_CONFIG_DIR = _PROJECT_ROOT / 'sandbox' / 'configs'
_MODEL_CONFIG_MAP = {
    'swinir': 'swinir.yaml',
    'edsr': 'edsr.yaml',
    'fno2d': 'fno2d.yaml',
    'unet2d': 'unet2d.yaml',
}


# 失败时返回的错误结果
BaselineResult = Union[BaselineThresholds, Dict[str, str]]


def _baseline_log_dir(model: str, run_idx: int) -> str:
    return str(_PROJECT_ROOT / 'sandbox' / 'runs' / f'baseline_{model}_{run_idx}')


def _build_baseline_config(
    model: str,
    dataset_root: Optional[str],
    run_idx: int,
    num_workers: int,
) -> Tuple[Path, str]:
    config_name = _MODEL_CONFIG_MAP.get(model.lower())
    if not config_name:
        raise ValueError(f'Unknown baseline model: {model}')

    base_config_path = _SANDBOX_CONFIG_DIR / config_name
    temp_dir = Path(tempfile.mkdtemp(prefix=f'baseline_{model.lower()}_'))
    output_path = temp_dir / config_name

    write_config_with_adapter(
        str(base_config_path),
        formula_spec=None,
        output_path=str(output_path),
        dataset_root=dataset_root,
    )

    config = yaml.safe_load(output_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError(f'Invalid generated baseline config: {output_path}')

    data_cfg = config.get('data', {})
    if not isinstance(data_cfg, dict):
        data_cfg = {}
    data_cfg['num_workers'] = int(num_workers)
    config['data'] = data_cfg

    log_cfg = config.get('log', {})
    if not isinstance(log_cfg, dict):
        log_cfg = {}
    log_dir = _baseline_log_dir(model.lower(), run_idx)
    log_cfg['log_dir'] = log_dir
    config['log'] = log_cfg

    output_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding='utf-8',
    )
    return output_path, log_dir


def _run_single_baseline(
    config_path: Path,
    gpu_id: int,
    timeout_s: int,
) -> Tuple[Optional[float], Optional[float], str]:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cmd = [_PYTHON, 'main.py', '--config', str(config_path)]
    try:
        result = subprocess.run(
            cmd,
            cwd=_PIPELINE_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ''
        return None, None, f'timeout: {stdout[-300:]}'

    stdout = result.stdout or ''
    stderr = result.stderr or ''
    if result.returncode != 0:
        detail = stderr[-500:] if stderr else stdout[-500:]
        return None, None, f'crash: {detail}'

    curve = parse_training_events(stdout)
    valid_epochs = [
        ep for ep in curve.get('epochs', [])
        if isinstance(ep, dict) and ep.get('ssim') is not None and ep.get('psnr') is not None
    ]
    if not valid_epochs:
        return None, None, 'parse_failed: missing validation metrics'

    last_valid = valid_epochs[-1]
    return float(last_valid['ssim']), float(last_valid['psnr']), ''


def run_baseline_noise(
    n_runs: int = 3,
    model: str = 'swinir',
    gpu_id: int = 4,
    dataset_root: Optional[str] = None,
    num_workers: int = 0,
    timeout_s: int = 900,
) -> BaselineResult:
    """
    运行 n 次原始训练默认 loss，测量 baseline 噪声。

    Returns:
        BaselineThresholds on success, or error dict on failure
    """
    ssim_values: List[float] = []
    psnr_values: List[float] = []

    for i in range(n_runs):
        print(f'Running baseline trial {i + 1}/{n_runs}...')
        try:
            config_path, log_dir = _build_baseline_config(
                model=model,
                dataset_root=dataset_root,
                run_idx=i + 1,
                num_workers=num_workers,
            )
        except Exception as exc:
            return {'error': f'baseline_config_failed: {exc}'}

        ssim, psnr, error = _run_single_baseline(
            config_path=config_path,
            gpu_id=gpu_id,
            timeout_s=timeout_s,
        )
        if ssim is None or psnr is None:
            print(f'Trial {i + 1} failed: {error}')
            continue

        print(f'  -> val_ssim={ssim:.6f}, val_psnr={psnr:.6f}, log_dir={log_dir}')
        ssim_values.append(ssim)
        psnr_values.append(psnr)

    if not ssim_values:
        return {'error': 'Not enough successful runs'}

    # 计算统计量；单次 run 时 std 记为 0，便于快速 baseline 对比
    ssim_mean = float(np.mean(ssim_values))
    ssim_std = float(np.std(ssim_values)) if len(ssim_values) > 1 else 0.0
    psnr_mean = float(np.mean(psnr_values))
    psnr_std = float(np.std(psnr_values)) if len(psnr_values) > 1 else 0.0

    thresholds: BaselineThresholds = {
        'model': model.lower(),
        'n_runs': len(ssim_values),
        'ssim_mean': ssim_mean,
        'ssim_std': ssim_std,
        'psnr_mean': psnr_mean,
        'psnr_std': psnr_std,
        'viable_threshold': ssim_mean - max(ssim_std, 0.01),
        'improvement_threshold': ssim_mean + max(ssim_std, 0.01),
    }

    output_file = _PROJECT_ROOT / 'workflow' / 'loss_transfer' / 'baseline_thresholds.yaml'
    output_file.write_text(yaml.safe_dump(thresholds, sort_keys=False), encoding='utf-8')
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description='Run baseline training noise measurement')
    parser.add_argument('--n_runs', type=int, default=3)
    parser.add_argument('--model', type=str, default='swinir')
    parser.add_argument('--gpu_id', type=int, default=4)
    parser.add_argument('--dataset_root', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--timeout_s', type=int, default=900)
    args = parser.parse_args()

    result = run_baseline_noise(
        n_runs=args.n_runs,
        model=args.model,
        gpu_id=args.gpu_id,
        dataset_root=args.dataset_root,
        num_workers=args.num_workers,
        timeout_s=args.timeout_s,
    )
    print(yaml.safe_dump(result, sort_keys=False), end='')


if __name__ == '__main__':
    main()
