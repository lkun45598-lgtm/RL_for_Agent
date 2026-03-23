"""
@file innovation_extractor.py
@description 从实验结果中提取关键创新点
@author Leizheng
@contributors kongzhiquan
@date 2026-03-22
@version 1.1.0

@changelog
  - 2026-03-22 Leizheng: v1.0.0 initial version
  - 2026-03-23 kongzhiquan: v1.1.0 refine type annotations, fix bare except
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional

from knowledge_db import KnowledgeDB
from llm_extractor import call_llm
from _types import (
    Innovation, InnovationEvidence, LLMInnovationExtract,
    ExperimentSummary, TrialSummaryItem, LossIRDict,
)


def extract_innovation_from_trial(
    paper_slug: str,
    trial_result: TrialSummaryItem,
    baseline_ssim: float,
    loss_ir: LossIRDict
) -> Optional[Innovation]:
    """
    从单次 trial 结果中提取创新点

    Args:
        paper_slug: 论文标识
        trial_result: trial 结果
        baseline_ssim: 基线 SSIM
        loss_ir: Loss IR dict

    Returns:
        Innovation or None
    """

    if not trial_result.get('passed'):
        return None

    new_ssim: float = trial_result.get('metrics', {}).get('swinir', 0)
    improvement = new_ssim - baseline_ssim

    # 只记录有提升的创新
    if improvement <= 0.001:
        return None

    # 构建 prompt 让 LLM 分析
    prompt = f"""分析这个 loss 实验的关键创新:

论文: {paper_slug}
基线 SSIM: {baseline_ssim:.4f}
新 SSIM: {new_ssim:.4f}
提升: +{improvement:.4f}

Loss 组件:
{yaml.dump(loss_ir.get('components', []))}

请提取:
1. component_type (pixel_loss/gradient_loss/frequency_loss)
2. key_idea (一句话描述创新点)
3. why_works (为什么有效)
4. tags (3-5个关键词)

只输出 YAML:
```yaml
component_type:
key_idea:
why_works:
tags: []
```"""

    try:
        response = call_llm(prompt)
        # 提取 YAML
        if '```yaml' in response:
            yaml_text = response.split('```yaml')[1].split('```')[0]
        else:
            yaml_text = response

        extracted: LLMInnovationExtract = yaml.safe_load(yaml_text)
    except (yaml.YAMLError, ValueError, IndexError, Exception) as e:
        # LLM 失败或解析失败,使用简化版
        extracted = {
            'component_type': 'unknown',
            'key_idea': f'Improvement from {paper_slug}',
            'why_works': 'Unknown',
            'tags': [paper_slug]
        }

    # 构建完整的 innovation
    evidence: InnovationEvidence = {
        'baseline_ssim': baseline_ssim,
        'new_ssim': new_ssim
    }

    innovation: Innovation = {
        'paper': paper_slug,
        'component_type': extracted.get('component_type', 'unknown'),
        'key_idea': extracted.get('key_idea', ''),
        'why_works': extracted.get('why_works', ''),
        'improvement': improvement,
        'confidence': min(improvement / 0.01, 1.0),  # 提升越大置信度越高
        'evidence': evidence,
        'tags': extracted.get('tags', [])
    }

    return innovation


def extract_and_save_innovations(paper_slug: str, summary: ExperimentSummary) -> List[str]:
    """从实验 summary 中提取并保存所有创新点"""
    db = KnowledgeDB()
    baseline_ssim: float = summary['baseline']['ssim_mean']

    innovations_added: List[str] = []
    for trial in summary['trials']:
        if trial['passed']:
            # 这里需要读取 trial 的 Loss IR
            # 简化版: 使用 summary 中的信息
            innovation = extract_innovation_from_trial(
                paper_slug, trial, baseline_ssim, {}  # type: ignore[arg-type]
            )
            if innovation:
                inn_id = db.add_innovation(innovation)
                innovations_added.append(inn_id)

    return innovations_added
