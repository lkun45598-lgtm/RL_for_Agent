"""
@file innovation_extractor.py
@description 从实验结果中提取关键创新点
@author Leizheng
@date 2026-03-22
@version 1.0.0
"""

import yaml
from pathlib import Path
from typing import Dict, Any

try:
    from .knowledge_db import KnowledgeDB
    from .llm_extractor import call_llm
except ImportError:
    from knowledge_db import KnowledgeDB
    from llm_extractor import call_llm


def extract_innovation_from_trial(
    paper_slug: str,
    trial_result: Dict[str, Any],
    baseline_ssim: float,
    loss_ir: Dict[str, Any]
) -> Dict[str, Any]:
    """
    从单次 trial 结果中提取创新点
    
    Args:
        paper_slug: 论文标识
        trial_result: trial 结果
        baseline_ssim: 基线 SSIM
        loss_ir: Loss IR
    
    Returns:
        innovation dict
    """
    
    if not trial_result.get('passed'):
        return None
    
    new_ssim = trial_result.get('metrics', {}).get('swinir', 0)
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
        
        extracted = yaml.safe_load(yaml_text)
    except:
        # LLM 失败,使用简化版
        extracted = {
            'component_type': 'unknown',
            'key_idea': f'Improvement from {paper_slug}',
            'why_works': 'Unknown',
            'tags': [paper_slug]
        }
    
    # 构建完整的 innovation
    innovation = {
        'paper': paper_slug,
        'component_type': extracted.get('component_type', 'unknown'),
        'key_idea': extracted.get('key_idea', ''),
        'why_works': extracted.get('why_works', ''),
        'improvement': improvement,
        'confidence': min(improvement / 0.01, 1.0),  # 提升越大置信度越高
        'evidence': {
            'baseline_ssim': baseline_ssim,
            'new_ssim': new_ssim
        },
        'tags': extracted.get('tags', [])
    }
    
    return innovation


def extract_and_save_innovations(paper_slug: str, summary: Dict[str, Any]):
    """从实验 summary 中提取并保存所有创新点"""
    db = KnowledgeDB()
    baseline_ssim = summary['baseline']['ssim_mean']
    
    innovations_added = []
    for trial in summary['trials']:
        if trial['passed']:
            # 这里需要读取 trial 的 Loss IR
            # 简化版: 使用 summary 中的信息
            innovation = extract_innovation_from_trial(
                paper_slug, trial, baseline_ssim, {}
            )
            if innovation:
                inn_id = db.add_innovation(innovation)
                innovations_added.append(inn_id)
    
    return innovations_added
