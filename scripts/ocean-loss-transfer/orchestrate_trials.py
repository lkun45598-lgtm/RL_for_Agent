"""
@file orchestrate_trials.py
@description 编排优先公式原生的固定 trial 策略
@author Leizheng
@contributors kongzhiquan
@date 2026-03-22
@version 1.4.0

@changelog
  - 2026-03-22 Leizheng: v1.0.0 initial version
  - 2026-03-23 kongzhiquan: v1.1.0 fix Trial 2/4 duplicate params; IR-driven template selection for Trial 1-3; LLM generate mode for Trial 4-5
  - 2026-03-23 kongzhiquan: v1.2.0 refine type annotations
  - 2026-03-24 Leizheng: v1.3.0 add trial early stopping on consecutive early-layer
    failures with same error pattern; add failure_summary to ExperimentSummary
  - 2026-03-24 Leizheng: v1.4.0 prepend deterministic formula-native trials
    when loss_formula.json is supported by formula_code_generator
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple


from formula_code_generator import (
    generate_formula_loss_code,
    load_formula_spec_for_paper,
    supports_formula_codegen,
)
from loss_ir_schema import LossIR, LossIRLike
from run_trial import run_single_trial
from run_baseline_noise import run_baseline_noise
from _types import (
    ExperimentSummary, TrialSummaryItem, PatchSpec, TemplatePatchSpec,
    AgentPatchSpec, LLMPatchSpec, BaselineThresholds, ComponentsByType, ComponentDict,
    PixelVariant, GradientVariant, FFTVariant, TrainingMetrics,
    EarlyStopInfo,
)


def _extract_components_by_type(loss_ir: LossIRLike) -> ComponentsByType:
    """从 Loss IR 中按类型提取组件信息"""
    result: ComponentsByType = {'pixel': [], 'gradient': [], 'frequency': []}
    components: list = []

    if hasattr(loss_ir, 'components') and loss_ir.components:  # type: ignore[union-attr]
        components = loss_ir.components  # type: ignore[union-attr]
    elif isinstance(loss_ir, dict) and loss_ir.get('components'):
        components = loss_ir['components']

    for comp in components:
        comp_dict: ComponentDict = comp if isinstance(comp, dict) else vars(comp)
        comp_type = comp_dict.get('type', '')
        if 'pixel' in comp_type:
            result['pixel'].append(comp_dict)
        elif 'gradient' in comp_type:
            result['gradient'].append(comp_dict)
        elif 'frequency' in comp_type or 'fft' in comp_type:
            result['frequency'].append(comp_dict)

    return result


def _select_pixel_variant(pixel_comps: List[ComponentDict]) -> PixelVariant:
    """根据 IR 中的 pixel loss 组件选择模板变体"""
    if not pixel_comps:
        return 'rel_l2'  # default

    comp = pixel_comps[0]
    impl = comp.get('implementation', {})
    name = comp.get('name', '').lower()

    # 根据 reduction 或名称选择
    if impl.get('reduction') == 'mean' or 'l1' in name or 'mae' in name:
        return 'abs_l1'
    if 'smooth' in name or 'huber' in name:
        return 'smooth_l1'
    return 'rel_l2'


def _select_gradient_variant(gradient_comps: List[ComponentDict]) -> GradientVariant:
    """根据 IR 中的 gradient 组件选择模板变体"""
    if not gradient_comps:
        return 'sobel_3x3'  # default

    name = gradient_comps[0].get('name', '').lower()
    if 'scharr' in name:
        return 'scharr_3x3'
    return 'sobel_3x3'


def _select_fft_variant(freq_comps: List[ComponentDict]) -> FFTVariant:
    """根据 IR 中的 frequency 组件选择模板变体"""
    if not freq_comps:
        return 'residual_rfft2_abs'  # default

    name = freq_comps[0].get('name', '').lower()
    if 'amplitude' in name or 'magnitude' in name:
        return 'amplitude_diff'
    return 'residual_rfft2_abs'


def _compute_weights(comp_by_type: ComponentsByType) -> Tuple[float, float, float]:
    """根据 IR 组件权重计算 alpha/beta/gamma"""
    pixel_w = sum(c.get('weight', 1.0) for c in comp_by_type['pixel']) if comp_by_type['pixel'] else 1.0
    grad_w = sum(c.get('weight', 1.0) for c in comp_by_type['gradient']) if comp_by_type['gradient'] else 0.6
    freq_w = sum(c.get('weight', 1.0) for c in comp_by_type['frequency']) if comp_by_type['frequency'] else 0.4
    total = pixel_w + grad_w + freq_w

    if total == 0:
        return 0.5, 0.3, 0.2

    alpha = round(pixel_w / total, 2)
    beta = round(grad_w / total, 2)
    gamma = round(1.0 - alpha - beta, 2)

    # 确保 gamma 不为负（浮点误差）
    if gamma < 0:
        gamma = 0.0
        total_ab = alpha + beta
        alpha = round(alpha / total_ab, 2)
        beta = round(1.0 - alpha, 2)

    return alpha, beta, gamma


def _build_formula_native_specs(paper_slug: Optional[str]) -> List[PatchSpec]:
    """若公式规格可识别，则优先构造确定性的 formula-native trials。"""
    if not paper_slug:
        return []

    project_root = Path(__file__).resolve().parent.parent.parent
    formula_spec = load_formula_spec_for_paper(str(project_root), paper_slug)
    if not supports_formula_codegen(formula_spec):
        return []

    assert formula_spec is not None
    variants = [
        ('faithful', 'Formula Native Faithful'),
        ('stabilized', 'Formula Native Stabilized'),
    ]

    trials: List[PatchSpec] = []
    for variant, name in variants:
        spec: AgentPatchSpec = {
            'name': name,
            'mode': 'agent_generate',
            'strategy': 'faithful',
            'code': generate_formula_loss_code(formula_spec, variant=variant),
        }
        trials.append(spec)
    return trials


def generate_trial_specs(
    loss_ir: LossIRLike,
    paper_slug: Optional[str] = None,
) -> List[PatchSpec]:
    """生成 trial 规格：优先公式原生 trial，再回退到 IR 驱动模板。"""

    # 解析 Loss IR 组件
    comp_by_type = _extract_components_by_type(loss_ir)

    # 从 IR 推断模板选择
    pixel_v = _select_pixel_variant(comp_by_type['pixel'])
    gradient_v = _select_gradient_variant(comp_by_type['gradient'])
    fft_v = _select_fft_variant(comp_by_type['frequency'])
    alpha, beta, gamma = _compute_weights(comp_by_type)

    # 是否有频域组件
    has_freq = bool(comp_by_type['frequency'])

    # Trial 1: IR-Driven Faithful (忠实移植 IR 中描述的组合)
    trial_1: TemplatePatchSpec = {
        'name': 'IR-Driven Faithful',
        'pixel_variant': pixel_v,
        'gradient_variant': gradient_v,
        'fft_variant': fft_v,
        'scales': [1, 2, 4],
        'scale_weights': [0.5, 0.3, 0.2],
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma if has_freq else 0.0
    }
    
    # 如果没有频域，将 gamma 权重分给 pixel 和 gradient
    if not has_freq:
        trial_1['alpha'] = round(alpha + gamma * 0.6, 2)
        trial_1['beta'] = round(beta + gamma * 0.4, 2)
        trial_1['gamma'] = 0.0

    # Trial 2: Normalization Aligned (量纲对齐变体)
    trial_2: TemplatePatchSpec = {
        'name': 'Normalization Aligned',
        'pixel_variant': 'abs_l1',
        'gradient_variant': gradient_v,
        'fft_variant': fft_v,
        'scales': [1, 2, 4],
        'scale_weights': [0.5, 0.3, 0.2],
        'alpha': 0.34, 'beta': 0.33, 'gamma': 0.33
    }

    # Trial 3: Numerical Stabilized (数值稳定变体)
    trial_3: TemplatePatchSpec = {
        'name': 'Numerical Stabilized',
        'pixel_variant': 'smooth_l1',
        'gradient_variant': gradient_v,
        'fft_variant': 'amplitude_diff',
        'scales': [1, 2, 4],
        'scale_weights': [0.5, 0.3, 0.2],
        'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2
    }

    formula_trials = _build_formula_native_specs(paper_slug)
    return formula_trials + [trial_1, trial_2, trial_3]

def orchestrate_trials(
    loss_ir: LossIRLike,
    paper_slug: str,
    dataset_root: Optional[str] = None,
) -> ExperimentSummary:
    """
    编排 5-trial 搜索

    Returns:
        ExperimentSummary
    """

    # 1. 检查基线阈值
    baseline_file = Path(__file__).parent.parent.parent / 'workflow/loss_transfer/baseline_thresholds.yaml'
    baseline: BaselineThresholds
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

    # 2. 生成 trial 规格（优先 formula-native，随后模板回退）
    trial_specs = generate_trial_specs(loss_ir, paper_slug=paper_slug)

    # 3. 执行 trials（含早停逻辑）
    results: List[TrialSummaryItem] = []
    consecutive_early_failures = 0
    last_failure_error: Optional[str] = None
    early_stop_info: Optional[EarlyStopInfo] = None
    failure_counts: Dict[str, int] = {}

    for i, spec in enumerate(trial_specs, 1):
        # 检查早停条件: 连续 2 个 trial 在 Layer 1/2 以相同错误失败
        if consecutive_early_failures >= 2 and last_failure_error:
            early_stop_info = {
                'reason': 'systematic_incompatibility',
                'consecutive_failures': consecutive_early_failures,
                'failure_pattern': last_failure_error,
                'trials_completed': i - 1,
                'trials_skipped': len(trial_specs) - (i - 1),
            }
            print(f'\n=== EARLY STOP: {consecutive_early_failures} consecutive early failures ===')
            print(f'  Error pattern: {last_failure_error}')
            print(f'  Skipping remaining {len(trial_specs) - (i - 1)} trials')
            break

        print(f'\n=== Trial {i}/{len(trial_specs)}: {spec["name"]} ===')
        result = run_single_trial(loss_ir, spec, i, paper_slug, dataset_root=dataset_root)
        item: TrialSummaryItem = {
            'trial_id': i,
            'name': spec['name'],
            'passed': result['passed'],
            'layer_stopped': result.get('layer_stopped'),
            'metrics': result.get('metrics', {})
        }
        results.append(item)

        # 追踪失败统计
        layer_stopped = result.get('layer_stopped')
        if not result['passed'] and layer_stopped in ('layer1', 'layer2'):
            # 从验证结果中提取具体错误
            current_error = 'unknown'
            validation = result.get('validation', {})
            if layer_stopped and layer_stopped in validation:
                current_error = validation[layer_stopped].get('error', 'unknown')

            failure_counts[current_error] = failure_counts.get(current_error, 0) + 1

            if current_error == last_failure_error:
                consecutive_early_failures += 1
            else:
                consecutive_early_failures = 1
                last_failure_error = current_error
        else:
            consecutive_early_failures = 0
            last_failure_error = None

            if not result['passed'] and layer_stopped:
                # Layer 3/4 失败也记录但不触发早停
                validation = result.get('validation', {})
                if layer_stopped in validation:
                    err = validation[layer_stopped].get('error', 'unknown')
                    failure_counts[err] = failure_counts.get(err, 0) + 1

        print(f'Result: {"PASSED" if result["passed"] else "FAILED at " + str(layer_stopped)}')

    # 4. 找出最佳 trial
    best_trial: Optional[int] = None
    best_ssim: float = 0

    for r in results:
        if r['passed'] and r.get('metrics', {}).get('swinir', 0) > best_ssim:
            best_ssim = r['metrics']['swinir']  # type: ignore[typeddict-item]
            best_trial = r['trial_id']

    # 5. 生成 summary
    summary: ExperimentSummary = {
        'paper_slug': paper_slug,
        'baseline': baseline,
        'trials': results,
        'best_trial': best_trial,
        'best_ssim': best_ssim,
        'improvement': best_ssim - baseline['ssim_mean'] if best_trial else None,
        'early_stop': early_stop_info,
        'failure_summary': failure_counts if failure_counts else {},
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

    # 7. 自动 git commit & push（仅实验分支）
    if best_trial:
        import subprocess
        repo_root = Path(__file__).parent.parent.parent
        commit_msg = f"Loss Transfer: {paper_slug} - Best Trial {best_trial} (SSIM={best_ssim:.4f})"
        experiment_branch = 'autoresearch/mar20'

        try:
            branch_result = subprocess.run(
                ['git', 'branch', '--show-current'],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            current_branch = branch_result.stdout.strip()
            if current_branch != experiment_branch:
                print(
                    f'\n⚠️  Skip auto git push: current branch is {current_branch!r}, '
                    f'experiment artifacts must be committed on {experiment_branch!r}'
                )
                return summary

            subprocess.run(['git', 'add', f'sandbox/loss_transfer_experiments/{paper_slug}'], cwd=repo_root, check=True)
            subprocess.run(['git', 'commit', '-m', commit_msg], cwd=repo_root, check=True)
            subprocess.run(['git', 'push'], cwd=repo_root, check=True)
            print(f'\n✓ Results pushed to GitHub: {commit_msg}')
        except Exception as e:
            print(f'\n⚠️  Git push failed: {e}')

    return summary
