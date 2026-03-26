"""
@file write_loss_ir.py
@description 验证并写入 Loss IR YAML
@author Leizheng
@date 2026-03-23
@version 1.0.0

@changelog
  - 2026-03-23 Leizheng: v1.0.0 初始版本
    - YAML 语法验证
    - Schema 结构验证
    - 语义一致性检查
    - 已知失败模式检测
    - 智能修复建议
"""

import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from loss_transfer.ir.loss_ir_schema import LossIR


# 已知失败模式（来自 71 次实验）
BLOCKED_PATTERNS = {
    'ssim_loss': {
        'keywords': ['ssim', 'structural_similarity'],
        'reason': '导致所有模型崩溃 (exp#11, SSIM=0.109)',
        'severity': 'critical'
    },
    'laplacian': {
        'keywords': ['laplacian'],
        'reason': '严重性能下降 (exp#20, SSIM=0.439)',
        'severity': 'critical'
    },
    'sobel_5x5': {
        'keywords': ['sobel', '5x5', 'kernel_size=5'],
        'reason': '梯度过度平滑 (exp#38, SSIM=0.6395)',
        'severity': 'critical'
    },
    'relative_fft': {
        'keywords': ['fft', 'relative', 'division'],
        'reason': '除法不稳定 (exp#36, SSIM=0.088)',
        'severity': 'critical'
    },
    'scale_8': {
        'keywords': ['scale', '8'],
        'reason': '信息损失过大 (exp#37, SSIM=0.6156)',
        'severity': 'warning'
    }
}


def validate_yaml_syntax(yaml_content: str) -> Dict[str, Any]:
    """验证 YAML 语法"""
    try:
        data = yaml.safe_load(yaml_content)
        return {'status': 'ok', 'data': data}
    except yaml.YAMLError as e:
        return {'status': 'error', 'message': f'YAML 语法错误: {e}'}


def validate_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """验证 Loss IR schema"""
    try:
        from loss_transfer.ir.loss_ir_schema import LossComponent
        components = [LossComponent(**c) for c in data['components']]
        loss_ir = LossIR(
            metadata=data['metadata'],
            interface=data['interface'],
            components=components,
            multi_scale=data['multi_scale'],
            combination=data['combination'],
            incompatibility_flags=data['incompatibility_flags']
        )
        return {'status': 'ok', 'loss_ir': loss_ir}
    except Exception as e:
        return {'status': 'error', 'message': f'Schema 验证失败: {e}'}


def check_semantic_consistency(loss_ir: LossIR) -> List[str]:
    """检查语义一致性"""
    warnings = []

    # 检查组件权重和是否与 combination 一致
    if loss_ir.components:
        component_weights = {comp.name: comp.weight for comp in loss_ir.components}

        if loss_ir.combination.get('method') == 'weighted_sum':
            combo_weights = loss_ir.combination.get('weights', {})

            for name, weight in component_weights.items():
                if name in combo_weights and abs(combo_weights[name] - weight) > 1e-6:
                    warnings.append(f"组件 {name} 的权重不一致: component.weight={weight}, combination.weights={combo_weights[name]}")

    # 检查是否缺少 epsilon/clamp
    for comp in loss_ir.components:
        impl = comp.implementation
        if impl.get('normalization') == 'relative':
            if not impl.get('clamp_or_eps'):
                warnings.append(f"组件 {comp.name} 使用 relative normalization 但缺少 clamp_or_eps，可能导致除零")

    return warnings


def check_blocked_patterns(loss_ir: LossIR) -> List[Dict[str, Any]]:
    """检查已知失败模式"""
    issues = []

    for comp in loss_ir.components:
        comp_name = comp.name.lower()
        comp_type = comp.type.lower()

        for pattern_name, pattern_info in BLOCKED_PATTERNS.items():
            for keyword in pattern_info['keywords']:
                if keyword in comp_name or keyword in comp_type:
                    issues.append({
                        'component': comp.name,
                        'pattern': pattern_name,
                        'reason': pattern_info['reason'],
                        'severity': pattern_info['severity']
                    })
                    break

    return issues


def generate_suggestions(warnings: List[str], blocked: List[Dict[str, Any]]) -> List[str]:
    """生成修复建议"""
    suggestions = []

    # 基于警告的建议
    for warning in warnings:
        if 'clamp_or_eps' in warning:
            suggestions.append("建议添加 clamp_or_eps: [{location: 'denominator', method: 'clamp_min', value: 1e-8}]")
        if '权重不一致' in warning:
            suggestions.append("建议统一 component.weight 和 combination.weights 中的权重值")

    # 基于已知失败的建议
    for issue in blocked:
        if issue['severity'] == 'critical':
            suggestions.append(f"强烈建议移除 {issue['component']}，{issue['reason']}")
        else:
            suggestions.append(f"警告：{issue['component']} 可能有风险，{issue['reason']}")

    return suggestions


def write_loss_ir(yaml_content: str, output_path: str, validate: bool = True) -> Dict[str, Any]:
    """
    验证并写入 Loss IR

    Args:
        yaml_content: YAML 内容字符串
        output_path: 输出文件路径
        validate: 是否验证（默认 True）

    Returns:
        验证结果和建议
    """
    result = {
        'status': 'success',
        'written_path': output_path,
        'validation_results': {},
        'suggestions': []
    }

    if not validate:
        # 直接写入
        Path(output_path).write_text(yaml_content, encoding='utf-8')
        return result

    # 1. YAML 语法验证
    syntax_result = validate_yaml_syntax(yaml_content)
    result['validation_results']['syntax'] = syntax_result['status']

    if syntax_result['status'] == 'error':
        result['status'] = 'error'
        result['validation_results']['error'] = syntax_result['message']
        return result

    # 2. Schema 验证
    schema_result = validate_schema(syntax_result['data'])
    result['validation_results']['schema'] = schema_result['status']

    if schema_result['status'] == 'error':
        result['status'] = 'error'
        result['validation_results']['error'] = schema_result['message']
        return result

    loss_ir = schema_result['loss_ir']

    # 3. 语义一致性检查
    warnings = check_semantic_consistency(loss_ir)
    if warnings:
        result['validation_results']['semantic'] = f"warning: {'; '.join(warnings)}"
        result['status'] = 'warning'

    # 4. 已知失败模式检测
    blocked = check_blocked_patterns(loss_ir)
    if blocked:
        critical_blocked = [b for b in blocked if b['severity'] == 'critical']
        if critical_blocked:
            result['status'] = 'blocked'
            result['validation_results']['blocked_patterns'] = critical_blocked
        else:
            result['validation_results']['risk_patterns'] = blocked

    # 5. 生成建议
    result['suggestions'] = generate_suggestions(warnings, blocked)

    # 6. 写入文件
    if result['status'] != 'blocked':
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(yaml_content, encoding='utf-8')
    else:
        result['written_path'] = None
        result['suggestions'].insert(0, "由于检测到严重失败模式，已阻止写入文件")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='验证并写入 Loss IR')
    parser.add_argument('--yaml_content', required=True, help='YAML 内容（字符串或文件路径）')
    parser.add_argument('--output_path', required=True, help='输出文件路径')
    parser.add_argument('--no-validate', action='store_true', help='跳过验证')

    args = parser.parse_args()

    # 判断是文件路径还是内容
    yaml_content = args.yaml_content
    if Path(yaml_content).exists():
        yaml_content = Path(yaml_content).read_text(encoding='utf-8')

    result = write_loss_ir(yaml_content, args.output_path, validate=not args.no_validate)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
