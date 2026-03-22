"""
@file orchestrate_trials.py
@description 编排 5-trial 固定策略
@author Leizheng
@date 2026-03-22
@version 1.0.0
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List

try:
    from .loss_ir_schema import LossIR
    from .run_trial import run_single_trial
    from .run_baseline_noise import run_baseline_noise
except ImportError:
    from loss_ir_schema import LossIR
    from run_trial import run_single_trial
    from run_baseline_noise import run_baseline_noise


def generate_trial_specs(loss_ir: LossIR) -> List[Dict[str, Any]]:
    """生成 5 个 trial 的 patch 规格"""
    
    # Trial 1: Faithful Core (忠实移植)
    trial_1 = {
        'name': 'Faithful Core',
        'pixel_variant': 'rel_l2',
        'gradient_variant': 'sobel_3x3',
        'fft_variant': 'residual_rfft2_abs',
        'scales': [1, 2, 4],
        'scale_weights': [0.5, 0.3, 0.2],
        'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2
    }
    
    # Trial 2: Normalization Aligned
    trial_2 = trial_1.copy()
    trial_2['name'] = 'Normalization Aligned'
    
    # Trial 3: Weight Aligned
    trial_3 = trial_2.copy()
    trial_3['name'] = 'Weight Aligned'
    trial_3['alpha'] = 0.4
    trial_3['beta'] = 0.4
    trial_3['gamma'] = 0.2
    
    # Trial 4: Numerical Stabilized
    trial_4 = trial_3.copy()
    trial_4['name'] = 'Numerical Stabilized'
    
    # Trial 5: Fallback Hybrid
    trial_5 = trial_1.copy()
    trial_5['name'] = 'Fallback Hybrid'
    trial_5['gradient_variant'] = 'scharr_3x3'
    
    return [trial_1, trial_2, trial_3, trial_4, trial_5]


def orchestrate_trials(loss_ir: LossIR, paper_slug: str) -> Dict[str, Any]:
    """
    编排 5-trial 搜索
    
    Returns:
        summary dict
    """
    
    # 1. 检查基线阈值
    baseline_file = Path(__file__).parent.parent.parent / 'workflow/loss_transfer/baseline_thresholds.yaml'
    if not baseline_file.exists():
        print('Baseline thresholds not found, using default from exp#71...')
        baseline = {
            'model': 'swinir',
            'ssim_mean': 0.6645,
            'ssim_std': 0.01,
            'viable_threshold': 0.6545,
            'improvement_threshold': 0.6745
        }
    else:
        baseline = yaml.safe_load(baseline_file.read_text())
    
    # 2. 生成 5 个 trial 规格
    trial_specs = generate_trial_specs(loss_ir)
    
    # 3. 执行 trials
    results = []
    for i, spec in enumerate(trial_specs, 1):
        print(f'\n=== Trial {i}/5: {spec["name"]} ===')
        result = run_single_trial(loss_ir, spec, i, paper_slug)
        results.append({
            'trial_id': i,
            'name': spec['name'],
            'passed': result['passed'],
            'layer_stopped': result.get('layer_stopped'),
            'metrics': result.get('metrics', {})
        })
        print(f'Result: {"PASSED" if result["passed"] else "FAILED at " + str(result.get("layer_stopped"))}')
    
    # 4. 找出最佳 trial
    best_trial = None
    best_ssim = 0
    
    for r in results:
        if r['passed'] and r['metrics'].get('swinir', 0) > best_ssim:
            best_ssim = r['metrics']['swinir']
            best_trial = r['trial_id']
    
    # 5. 生成 summary
    summary = {
        'paper_slug': paper_slug,
        'baseline': baseline,
        'trials': results,
        'best_trial': best_trial,
        'best_ssim': best_ssim,
        'improvement': best_ssim - baseline['ssim_mean'] if best_trial else None
    }
    
    # 6. 保存
    summary_dir = Path(__file__).parent.parent.parent / 'sandbox/loss_transfer_experiments' / paper_slug
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / 'summary.yaml').write_text(yaml.dump(summary, allow_unicode=True))

    # 6.5. 提取创新点并泛化代码
    if best_trial:
        try:
            from innovation_extractor import extract_and_save_innovations
            from code_generalizer import generalize_best_trial

            # 提取创新点
            innovations = extract_and_save_innovations(paper_slug, summary)
            print(f'\n✓ Extracted {len(innovations)} innovations to knowledge base')

            # 泛化代码
            if innovations:
                trial_dir = summary_dir / f'trial_{best_trial}'
                module_path = generalize_best_trial(paper_slug, str(trial_dir), {'paper': paper_slug, 'improvement': summary['improvement'], 'why_works': 'Best trial'})
                if module_path:
                    print(f'✓ Generalized code to: {module_path}')
        except Exception as e:
            print(f'⚠️  Knowledge extraction failed: {e}')

    # 7. 自动 git commit & push
    if best_trial:
        import subprocess
        repo_root = Path(__file__).parent.parent.parent
        commit_msg = f"Loss Transfer: {paper_slug} - Best Trial {best_trial} (SSIM={best_ssim:.4f})"

        try:
            subprocess.run(['git', 'add', f'sandbox/loss_transfer_experiments/{paper_slug}'], cwd=repo_root, check=True)
            subprocess.run(['git', 'commit', '-m', commit_msg], cwd=repo_root, check=True)
            subprocess.run(['git', 'push'], cwd=repo_root, check=True)
            print(f'\n✓ Results pushed to GitHub: {commit_msg}')
        except Exception as e:
            print(f'\n⚠️  Git push failed: {e}')

    return summary
