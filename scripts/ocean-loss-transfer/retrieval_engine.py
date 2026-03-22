"""
@file retrieval_engine.py
@description 检索相关创新点
@author Leizheng
@date 2026-03-22
@version 1.0.0
"""

from typing import List, Dict, Any

try:
    from .knowledge_db import KnowledgeDB
except ImportError:
    from knowledge_db import KnowledgeDB


def retrieve_innovations(
    query: str,
    component_type: str = None,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    检索相关创新点
    
    Args:
        query: 查询描述
        component_type: 组件类型过滤
        top_k: 返回前 K 个
    
    Returns:
        相关创新点列表
    """
    db = KnowledgeDB()
    
    # 提取查询关键词
    keywords = query.lower().split()
    
    # 获取所有创新点
    all_innovations = db.get_all_innovations()
    
    # 过滤和评分
    scored = []
    for inn in all_innovations:
        score = 0
        
        # 类型匹配
        if component_type and inn.get('component_type') == component_type:
            score += 10
        
        # 关键词匹配
        inn_text = f"{inn.get('key_idea', '')} {' '.join(inn.get('tags', []))}".lower()
        for kw in keywords:
            if kw in inn_text:
                score += 5
        
        # 性能加权
        score += inn.get('improvement', 0) * 100
        
        if score > 0:
            scored.append((score, inn))
    
    # 排序并返回 Top-K
    scored.sort(reverse=True, key=lambda x: x[0])
    return [inn for _, inn in scored[:top_k]]
