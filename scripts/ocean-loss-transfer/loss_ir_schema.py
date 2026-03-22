"""
@file loss_ir_schema.py
@description Loss IR (Intermediate Representation) 数据结构定义
@author Leizheng
@date 2026-03-22
@version 1.0.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import yaml


@dataclass
class LossComponent:
    """单个 loss 组件"""
    name: str
    type: str  # pixel_loss | gradient_loss | frequency_loss | ...
    weight: float
    formula: str
    implementation: Dict[str, Any]
    code_evidence: Dict[str, Any]
    required_tensors: List[str]
    required_imports: List[str]


@dataclass
class LossIR:
    """Loss 中间表示"""
    metadata: Dict[str, str]
    interface: Dict[str, Any]
    components: List[LossComponent]
    multi_scale: Dict[str, Any]
    combination: Dict[str, Any]
    incompatibility_flags: Dict[str, bool]

    def to_yaml(self, file_path: str):
        """保存为 YAML"""
        data = {
            'metadata': self.metadata,
            'interface': self.interface,
            'components': [vars(c) for c in self.components],
            'multi_scale': self.multi_scale,
            'combination': self.combination,
            'incompatibility_flags': self.incompatibility_flags
        }
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, file_path: str):
        """从 YAML 加载"""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        components = [LossComponent(**c) for c in data['components']]

        return cls(
            metadata=data['metadata'],
            interface=data['interface'],
            components=components,
            multi_scale=data['multi_scale'],
            combination=data['combination'],
            incompatibility_flags=data['incompatibility_flags']
        )


def validate_loss_ir(loss_ir: LossIR, repo_path: str = None) -> Dict[str, Any]:
    """
    校验 Loss IR 的完整性
    """
    errors = []

    # 检查必填字段
    if not loss_ir.metadata.get('paper_title'):
        errors.append("缺少 metadata.paper_title")

    if not loss_ir.components:
        errors.append("缺少 components")

    # 检查每个组件
    for comp in loss_ir.components:
        if not comp.required_tensors:
            errors.append(f"组件 {comp.name} 缺少 required_tensors")

        if not comp.required_imports:
            errors.append(f"组件 {comp.name} 缺少 required_imports")

    return {
        'valid': len(errors) == 0,
        'errors': errors
    }
