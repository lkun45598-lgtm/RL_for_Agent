"""
@file knowledge_db.py
@description 知识库管理 - 存储和检索创新点
@author Leizheng
@contributors kongzhiquan
@date 2026-03-22
@version 1.1.0

@changelog
  - 2026-03-22 Leizheng: v1.0.0 initial version
  - 2026-03-23 kongzhiquan: v1.1.0 refine type annotations
"""

import yaml
import json
from pathlib import Path
from typing import List, Optional, Set, Union
from datetime import datetime

from loss_transfer.common._types import Innovation, KnowledgeIndex, KnowledgeData
from loss_transfer.common.paths import KNOWLEDGE_BASE_DIR


class KnowledgeDB:
    """知识库管理器"""

    def __init__(self, base_path: Optional[Union[str, Path]] = None) -> None:
        if base_path is None:
            base_path = KNOWLEDGE_BASE_DIR
        self.base_path = Path(base_path)
        self.innovations_file = self.base_path / 'innovations.yaml'
        self.modules_dir = self.base_path / 'modules'
        self.index_file = self.base_path / 'index.json'

        self._ensure_structure()

    def _ensure_structure(self) -> None:
        """确保目录结构存在"""
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.modules_dir.mkdir(exist_ok=True)

        if not self.innovations_file.exists():
            data: KnowledgeData = {'innovations': []}
            self.innovations_file.write_text(yaml.dump(data))

        if not self.index_file.exists():
            index: KnowledgeIndex = {'next_id': 1, 'tags_index': {}}
            self.index_file.write_text(json.dumps(index))

    def add_innovation(self, innovation: Innovation) -> str:
        """添加创新点"""
        # 读取现有数据
        data: KnowledgeData = yaml.safe_load(self.innovations_file.read_text())
        index: KnowledgeIndex = json.loads(self.index_file.read_text())

        # 生成 ID
        innovation_id = f"inn_{index['next_id']:03d}"
        innovation['id'] = innovation_id
        innovation['date'] = datetime.now().isoformat()

        # 添加到列表
        data['innovations'].append(innovation)

        # 更新索引
        index['next_id'] += 1
        for tag in innovation.get('tags', []):
            if tag not in index['tags_index']:
                index['tags_index'][tag] = []
            index['tags_index'][tag].append(innovation_id)

        # 保存
        self.innovations_file.write_text(yaml.dump(data, allow_unicode=True))
        self.index_file.write_text(json.dumps(index, indent=2))

        return innovation_id

    def get_all_innovations(self) -> List[Innovation]:
        """获取所有创新点"""
        data: KnowledgeData = yaml.safe_load(self.innovations_file.read_text())
        return data.get('innovations', [])

    def search_by_tags(self, tags: List[str]) -> List[Innovation]:
        """按标签搜索"""
        index: KnowledgeIndex = json.loads(self.index_file.read_text())
        innovation_ids: Set[str] = set()

        for tag in tags:
            if tag in index['tags_index']:
                innovation_ids.update(index['tags_index'][tag])

        all_innovations = self.get_all_innovations()
        return [inn for inn in all_innovations if inn.get('id') in innovation_ids]

    def get_top_innovations(self, n: int = 5) -> List[Innovation]:
        """获取 Top-N 创新点 (按 improvement 排序)"""
        innovations = self.get_all_innovations()
        innovations.sort(key=lambda x: x.get('improvement', 0), reverse=True)
        return innovations[:n]
