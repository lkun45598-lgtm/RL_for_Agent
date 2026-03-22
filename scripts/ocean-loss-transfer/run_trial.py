"""
@file run_trial.py
@description 执行单次 trial: 生成 → 验证 → 记录
@author Leizheng
@date 2026-03-22
@version 1.0.0
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Any

try:
    from .loss_ir_schema import LossIR
    from .generate_patch import generate_patch_from_ir
    from .experiment_recorder import record_trial
except ImportError:
    from loss_ir_schema import LossIR
    from generate_patch import generate_patch_from_ir
    from experiment_recorder import record_trial


def run_single_trial(
    loss_ir: LossIR,
    patch_spec: Dict[str, Any],
    trial_id: int,
    paper_slug: str
) -> Dict[str, Any]:
    """
    执行单次 trial
    
    Returns:
        {passed, layer_stopped, metrics, trial_dir}
    """
    
    # 1. 生成 patch
    temp_loss_file = Path('/tmp') / f'trial_{trial_id}_loss.py'
    patch_result = generate_patch_from_ir(loss_ir, patch_spec, str(temp_loss_file))
    
    validation_results = {}
    layer_stopped = None

    # 获取绝对路径
    validator_script = Path(__file__).parent / 'validate_loss.py'

    # 2. Layer 1: Static
    cmd = f'/home/lz/miniconda3/envs/pytorch/bin/python {validator_script} --loss_file {temp_loss_file} --mode static'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if not result.stdout.strip():
        return {'passed': False, 'layer_stopped': 'layer1', 'error': 'No output from validator'}

    try:
        layer1 = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {'passed': False, 'layer_stopped': 'layer1', 'error': f'Invalid JSON: {result.stdout[:200]}'}

    validation_results['layer1'] = layer1
    
    if not layer1.get('passed'):
        layer_stopped = 'layer1'
        record_trial(paper_slug, trial_id, patch_spec, validation_results, str(temp_loss_file))
        return {'passed': False, 'layer_stopped': layer_stopped, 'validation': validation_results}
    
    # 3. Layer 2: Smoke
    cmd = f'/home/lz/miniconda3/envs/pytorch/bin/python {validator_script} --loss_file {temp_loss_file} --mode smoke'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    layer2 = json.loads(result.stdout)
    validation_results['layer2'] = layer2
    
    if not layer2.get('passed'):
        layer_stopped = 'layer2'
        record_trial(paper_slug, trial_id, patch_spec, validation_results, str(temp_loss_file))
        return {'passed': False, 'layer_stopped': layer_stopped, 'validation': validation_results}
    
    # 4. Layer 3: Single Model
    cmd = f'/home/lz/miniconda3/envs/pytorch/bin/python {validator_script} --loss_file {temp_loss_file} --mode single'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    layer3 = json.loads(result.stdout)
    validation_results['layer3'] = layer3
    
    if not layer3.get('passed') or layer3.get('metrics', {}).get('val_ssim', 0) < 0.3:
        layer_stopped = 'layer3'
        record_trial(paper_slug, trial_id, patch_spec, validation_results, str(temp_loss_file))
        return {'passed': False, 'layer_stopped': layer_stopped, 'validation': validation_results}
    
    # 5. Layer 4: Full Run
    cmd = f'/home/lz/miniconda3/envs/pytorch/bin/python scripts/ocean-loss-transfer/validate_loss.py --loss_file {temp_loss_file} --mode full'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
    layer4 = json.loads(result.stdout)
    validation_results['layer4'] = layer4
    
    # 记录
    trial_dir = record_trial(paper_slug, trial_id, patch_spec, validation_results, str(temp_loss_file))
    
    return {
        'passed': layer4.get('passed', False),
        'layer_stopped': None if layer4.get('passed') else 'layer4',
        'validation': validation_results,
        'metrics': layer4.get('metrics', {}),
        'trial_dir': trial_dir
    }
