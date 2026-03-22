"""
@file run_auto_experiment.py
@description 全自动实验: 论文+代码 → Loss IR → 5-trial → 结果
@author Leizheng
@date 2026-03-22
@version 1.0.0
"""

import sys
import yaml
from pathlib import Path

try:
    from .extract_loss_ir import extract_loss_ir
    from .check_compatibility import check_compatibility
    from .orchestrate_trials import orchestrate_trials
    from .loss_ir_schema import LossIR
except ImportError:
    from extract_loss_ir import extract_loss_ir
    from check_compatibility import check_compatibility
    from orchestrate_trials import orchestrate_trials
    from loss_ir_schema import LossIR


def run_auto_experiment(
    paper_slug: str,
    paper_pdf_path: str = None,
    code_repo_path: str = None,
    loss_ir_yaml: str = None
):
    """
    全自动实验流程
    
    Args:
        paper_slug: 论文标识符
        paper_pdf_path: 论文 PDF (可选)
        code_repo_path: 代码仓库 (可选)
        loss_ir_yaml: 已有的 Loss IR (可选,跳过提取)
    """
    
    print(f"\n{'='*60}")
    print(f"  Auto Experiment: {paper_slug}")
    print(f"{'='*60}\n")
    
    # Step 1: 提取或加载 Loss IR
    if loss_ir_yaml and Path(loss_ir_yaml).exists():
        print(f"[1/4] Loading existing Loss IR: {loss_ir_yaml}")
        loss_ir_path = loss_ir_yaml
    else:
        print(f"[1/4] Extracting Loss IR...")
        loss_ir_path = f'sandbox/loss_transfer_experiments/{paper_slug}/loss_ir.yaml'
        extract_loss_ir(
            paper_pdf_path=paper_pdf_path,
            code_repo_path=code_repo_path,
            output_yaml_path=loss_ir_path,
            manual_mode=(not paper_pdf_path and not code_repo_path)
        )
        print(f"  → Generated: {loss_ir_path}")
        
        if not paper_pdf_path and not code_repo_path:
            print("\n⚠️  Manual mode: Please edit the Loss IR YAML and re-run with --loss_ir_yaml")
            return {'status': 'manual_edit_required', 'loss_ir_path': loss_ir_path}
    
    # Step 2: 检查兼容性
    print(f"\n[2/4] Checking compatibility...")
    data = yaml.safe_load(Path(loss_ir_path).read_text())
    loss_ir = LossIR(**data)
    compat = check_compatibility(loss_ir)
    print(f"  → Status: {compat['status']}")
    
    if compat['status'] == 'incompatible':
        print(f"  ✗ Incompatible: {compat['issues']}")
        return {'status': 'incompatible', 'issues': compat['issues']}
    
    if compat.get('warnings'):
        print(f"  ⚠️  Warnings: {compat['warnings']}")
    
    # Step 3: 运行 5-trial 搜索
    print(f"\n[3/4] Running 5-trial search...")
    print("  This will take ~10-30 minutes depending on failures")
    summary = orchestrate_trials(loss_ir, paper_slug)
    
    # Step 4: 输出结果
    print(f"\n[4/4] Results")
    print(f"{'='*60}")
    for trial in summary['trials']:
        status = '✓' if trial['passed'] else '✗'
        print(f"  {status} Trial {trial['trial_id']}: {trial['name']}")
        if trial['passed']:
            print(f"      SSIM: {trial['metrics'].get('swinir', 'N/A')}")
    
    if summary['best_trial']:
        print(f"\n  🏆 Best: Trial {summary['best_trial']} (SSIM={summary['best_ssim']:.4f})")
        print(f"  📈 Improvement: {summary['improvement']:+.4f}")
    else:
        print(f"\n  ✗ No trial passed all validations")
    
    print(f"\n  📁 Results: sandbox/loss_transfer_experiments/{paper_slug}/")
    print(f"{'='*60}\n")
    
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper_slug', required=True)
    parser.add_argument('--paper_pdf', default=None)
    parser.add_argument('--code_repo', default=None)
    parser.add_argument('--loss_ir_yaml', default=None)
    args = parser.parse_args()
    
    run_auto_experiment(
        paper_slug=args.paper_slug,
        paper_pdf_path=args.paper_pdf,
        code_repo_path=args.code_repo,
        loss_ir_yaml=args.loss_ir_yaml
    )
