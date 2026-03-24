"""
@file run_auto_experiment.py
@description 全自动实验: 论文+代码 → Loss IR → 5-trial → 结果
@author Leizheng
@contributors kongzhiquan
@date 2026-03-22
@version 1.1.0

@changelog
  - 2026-03-22 Leizheng: v1.0.0 initial version
  - 2026-03-23 kongzhiquan: v1.1.0 refine type annotations
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union
from extract_loss_ir import extract_loss_ir
from check_compatibility import check_compatibility
from formula_interface_analysis import analyze_formula_interface
from orchestrate_trials import orchestrate_trials
from loss_ir_schema import LossIR
from _types import ExperimentSummary, CompatibilityResult


# 手动编辑提示结果
class ManualEditResult(Dict[str, str]):
    pass


AutoExperimentResult = Union[ExperimentSummary, Dict[str, str]]


def run_auto_experiment(
    paper_slug: str,
    paper_pdf_path: Optional[str] = None,
    code_repo_path: Optional[str] = None,
    loss_ir_yaml: Optional[str] = None,
    dataset_root: Optional[str] = None,
) -> AutoExperimentResult:
    """
    全自动实验流程

    Args:
        paper_slug: 论文标识符
        paper_pdf_path: 论文 PDF (可选)
        code_repo_path: 代码仓库 (可选)
        loss_ir_yaml: 已有的 Loss IR (可选,跳过提取)
        dataset_root: 训练验证使用的数据集根目录（可选）

    Returns:
        ExperimentSummary on success, or status dict on early exit
    """

    print(f"\n{'='*60}")
    print(f"  Auto Experiment: {paper_slug}")
    print(f"{'='*60}\n")

    # Step 1: 提取或加载 Loss IR
    loss_ir_path: str
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
    loss_ir = LossIR.from_yaml(loss_ir_path)
    compat: CompatibilityResult = check_compatibility(loss_ir)
    print(f"  → Status: {compat['status']}")

    if compat['status'] == 'incompatible':
        issues = compat.get('issues', [])
        print(f"  ✗ Incompatible: {issues}")
        return {'status': 'incompatible', 'issues': str(issues)}

    if  compat.get('warnings'):
        print(f"  ⚠️  Warnings: {compat['warnings']}")

    formula_path = Path(f'sandbox/loss_transfer_experiments/{paper_slug}/loss_formula.json')
    if formula_path.exists():
        try:
            formula_spec = json.loads(formula_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            formula_spec = None

        if isinstance(formula_spec, dict):
            interface_analysis = analyze_formula_interface(formula_spec)
            print(f"  → Formula interface: {interface_analysis['status']}")
            if interface_analysis.get('status') == 'incompatible':
                issues = interface_analysis.get('issues', [])
                print(f"  ✗ Formula interface incompatible: {issues}")
                return {'status': 'incompatible', 'issues': str(issues)}
            if interface_analysis.get('status') == 'requires_adapter':
                extra_vars = interface_analysis.get('extra_required_variables', [])
                adapter_source = interface_analysis.get('adapter_config_source', 'auto_inferred')
                print(f"  → Sandbox adapter required for: {extra_vars} ({adapter_source})")

    # Step 3: 运行 5-trial 搜索
    print(f"\n[3/4] Running 5-trial search...")
    print("  This will take ~10-30 minutes depending on failures")
    if dataset_root:
        print(f"  Using dataset_root: {dataset_root}")
    summary: ExperimentSummary = orchestrate_trials(loss_ir, paper_slug, dataset_root=dataset_root)

    # Step 4: 输出结果
    print(f"\n[4/4] Results")
    print(f"{'='*60}")
    for trial in summary['trials']:
        status = '✓' if trial['passed'] else '✗'
        print(f"  {status} Trial {trial['trial_id']}: {trial['name']}")
        if trial['passed']:
            print(f"      SSIM: {trial.get('metrics', {}).get('swinir', 'N/A')}")

    best = summary.get('best_trial')
    if best:
        print(f"\n  🏆 Best: Trial {best} (SSIM={summary['best_ssim']:.4f})")
        improvement = summary.get('improvement')
        if improvement is not None:
            print(f"  📈 Improvement: {improvement:+.4f}")
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
    parser.add_argument('--dataset_root', default=None)
    args = parser.parse_args()

    run_auto_experiment(
        paper_slug=args.paper_slug,
        paper_pdf_path=args.paper_pdf,
        code_repo_path=args.code_repo,
        loss_ir_yaml=args.loss_ir_yaml,
        dataset_root=args.dataset_root,
    )
