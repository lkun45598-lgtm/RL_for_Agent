"""
@file llm_code_generator.py
@description LLM 直接生成 sandbox_loss.py 代码，带多轮修复循环
@author kongzhiquan
@date 2026-03-23
@version 1.1.0

@changelog
  - 2026-03-23 kongzhiquan: v1.0.0 initial version
  - 2026-03-23 kongzhiquan: v1.1.0 refine type annotations, type loss_ir parameter
"""

import yaml
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union
from llm_extractor import call_llm
from validate_loss import validate_static, validate_smoke
from _types import (
    CodeSnippet, GenerateResult, ValidationResult,
    GenerationStrategy, LossIRDict, BlockedPatternsConfig
)

# 延迟导入避免循环依赖
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from loss_ir_schema import LossIR, LossIRLike


def _load_target_interface_spec() -> str:
    """加载 target interface spec 作为 prompt 上下文"""
    spec_file = Path(__file__).parent.parent.parent / 'workflow/loss_transfer/target_interface_spec.yaml'
    if spec_file.exists():
        return spec_file.read_text()
    return ""


def _load_blocked_patterns() -> str:
    """加载已知失败模式作为 prompt 上下文"""
    blocked_file = Path(__file__).parent.parent.parent / 'workflow/loss_transfer/blocked_patterns.yaml'
    if blocked_file.exists():
        data: BlockedPatternsConfig = yaml.safe_load(blocked_file.read_text())
        lines: List[str] = []
        for comp in data.get('blocked_components', []):
            if comp.get('action') == 'REJECT':
                lines.append(f"- FORBIDDEN: {comp['name']} ({comp['reason']}). {comp.get('fix_hint', '')}")
            elif comp.get('action') == 'WARN':
                lines.append(f"- WARNING: {comp['name']} ({comp['reason']}). {comp.get('fix_hint', '')}")
        for scale in data.get('blocked_scales', []):
            if scale.get('action') == 'REJECT':
                lines.append(f"- FORBIDDEN: scales={scale['scales']} ({scale['reason']})")
        return '\n'.join(lines)
    return ""


def _format_code_snippets(code_snippets: List[CodeSnippet]) -> str:
    """格式化论文代码片段"""
    if not code_snippets:
        return "(No code snippets available)"
    parts: List[str] = []
    for s in code_snippets[:5]:
        parts.append(f"File: {s['file']}\n```python\n{s['content']}\n```")
    return '\n\n'.join(parts)


def _format_loss_ir_components(loss_ir: 'LossIRLike') -> str:
    """格式化 Loss IR 组件描述"""
    components: list = []
    if hasattr(loss_ir, 'components'):
        components = loss_ir.components  # type: ignore[union-attr]
    elif isinstance(loss_ir, dict):
        components = loss_ir.get('components', [])

    if not components:
        return "(No components extracted)"

    return yaml.dump(components, allow_unicode=True)


def _build_generation_prompt(
    code_snippets: List[CodeSnippet],
    loss_ir: 'LossIRLike',
    strategy: GenerationStrategy,
    interface_spec: str,
    blocked_patterns: str
) -> str:
    """构建代码生成 prompt"""

    code_text = _format_code_snippets(code_snippets)
    components_text = _format_loss_ir_components(loss_ir)

    strategy_instruction = ""
    if strategy == 'faithful':
        strategy_instruction = """策略: FAITHFUL (忠实迁移)
- 尽可能忠实地迁移论文中的 loss 函数核心思想
- 适配 BHWC 张量布局和 sandbox_loss 接口
- 保留论文 loss 的数学本质，只做必要的接口适配"""
    elif strategy == 'creative':
        strategy_instruction = """策略: CREATIVE (创新融合)
- 在论文 loss 核心思想基础上，融合多尺度和梯度特征
- 可以添加 multi-scale wrapper（scales=[1,2,4]，用 avg_pool2d 降采样）
- 可以添加 gradient loss（Sobel 3x3）作为辅助项
- 目标是超越 baseline SSIM=0.6645"""

    prompt = f"""你是一个 PyTorch loss 函数专家。请根据论文代码生成一个 `sandbox_loss` 函数。

## 目标接口规范

函数签名: `sandbox_loss(pred, target, mask=None, **kwargs)`
- pred: shape [B, H, W, 2], layout BHWC, dtype float32/float16
- target: shape [B, H, W, 2], layout BHWC
- mask: shape [B, H, W, 1], layout BHWC, dtype torch.bool, 可选
- 返回: 标量 torch.Tensor (0维)

只允许 import: torch, torch.nn.functional, math
不允许: open(), subprocess, __import__

## 已知失败模式（绝对不要使用）

{blocked_patterns}

## 论文原始代码

{code_text}

## 论文 Loss 组件分析

{components_text}

## {strategy_instruction}

## 输出要求

输出完整的 Python 文件，包含所有辅助函数和 `sandbox_loss` 主函数。
注意事项：
1. 张量 layout 是 BHWC（不是 NCHW），需要 permute 后再做卷积
2. mask 可能为 None，需要正确处理
3. 分母必须 clamp(min=1e-8) 避免除零
4. 确保 loss 可以正常 backward

只输出 Python 代码，用 ```python 包裹，不要其他解释。"""

    return prompt


def _build_repair_prompt(code: str, error_info: ValidationResult, round_num: int) -> str:
    """构建修复 prompt"""
    error_type = error_info.get('error', 'unknown')
    detail = error_info.get('detail', '')
    fix_hint = error_info.get('fix_hint', '')
    traceback_info = error_info.get('traceback', '')

    prompt = f"""你之前生成的 sandbox_loss 代码有错误，请修复。

## 错误信息 (第 {round_num} 轮)

错误类型: {error_type}
详情: {detail}
修复提示: {fix_hint}
"""
    if traceback_info:
        prompt += f"\nTraceback:\n```\n{traceback_info[:1000]}\n```\n"

    prompt += f"""
## 当前代码

```python
{code}
```

## 修复要求

1. 修复上述错误
2. 确保函数签名为 sandbox_loss(pred, target, mask=None, **kwargs)
3. 只允许 import: torch, torch.nn.functional, math
4. 返回标量 tensor
5. 张量 layout 是 BHWC

只输出修复后的完整 Python 文件，用 ```python 包裹。"""

    return prompt


def _extract_code_from_response(response: str) -> str:
    """从 LLM 响应中提取 Python 代码"""
    if '```python' in response:
        code = response.split('```python')[1].split('```')[0]
    elif '```' in response:
        code = response.split('```')[1].split('```')[0]
    else:
        code = response

    return code.strip()


def generate_loss_code(
    loss_ir: 'LossIRLike',
    code_snippets: List[CodeSnippet],
    strategy: GenerationStrategy = 'faithful',
    max_repair_rounds: int = 3
) -> GenerateResult:
    """
    LLM 直接生成 sandbox_loss.py 代码，带多轮修复循环。

    Args:
        loss_ir: LossIR 对象或 dict
        code_snippets: 论文原始代码片段
        strategy: 'faithful' | 'creative'
        max_repair_rounds: 最大修复轮数

    Returns:
        GenerateResult
    """
    interface_spec = _load_target_interface_spec()
    blocked_patterns = _load_blocked_patterns()

    # Round 1: 初始生成
    prompt = _build_generation_prompt(
        code_snippets=code_snippets,
        loss_ir=loss_ir,
        strategy=strategy,
        interface_spec=interface_spec,
        blocked_patterns=blocked_patterns,
    )

    try:
        response = call_llm(prompt)
    except Exception as e:
        return {
            'code': '',
            'passed_static': False,
            'passed_smoke': False,
            'repair_rounds': 0,
            'error': f'LLM call failed: {e}'
        }

    code = _extract_code_from_response(response)

    for round_num in range(1, max_repair_rounds + 1):
        print(f'  LLM generate round {round_num}: validating...')

        # 写入临时文件进行验证
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            tmp_path = f.name

        # Layer 1: Static check
        static_result = validate_static(tmp_path)
        if not static_result.get('passed'):
            print(f'  Static check failed: {static_result.get("error")}')
            if round_num < max_repair_rounds:
                repair_prompt = _build_repair_prompt(code, static_result, round_num)
                try:
                    response = call_llm(repair_prompt)
                    code = _extract_code_from_response(response)
                except Exception as e:
                    Path(tmp_path).unlink(missing_ok=True)
                    return {
                        'code': code,
                        'passed_static': False,
                        'passed_smoke': False,
                        'repair_rounds': round_num,
                        'error': f'LLM repair call failed: {e}'
                    }
                Path(tmp_path).unlink(missing_ok=True)
                continue
            else:
                Path(tmp_path).unlink(missing_ok=True)
                return {
                    'code': code,
                    'passed_static': False,
                    'passed_smoke': False,
                    'repair_rounds': round_num,
                    'error': f'Static check failed after {round_num} rounds: {static_result.get("detail")}'
                }

        # Layer 2: Smoke test
        smoke_result = validate_smoke(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)

        if not smoke_result.get('passed'):
            print(f'  Smoke test failed: {smoke_result.get("error")}')
            if round_num < max_repair_rounds:
                repair_prompt = _build_repair_prompt(code, smoke_result, round_num)
                try:
                    response = call_llm(repair_prompt)
                    code = _extract_code_from_response(response)
                except Exception as e:
                    return {
                        'code': code,
                        'passed_static': True,
                        'passed_smoke': False,
                        'repair_rounds': round_num,
                        'error': f'LLM repair call failed: {e}'
                    }
                continue
            else:
                return {
                    'code': code,
                    'passed_static': True,
                    'passed_smoke': False,
                    'repair_rounds': round_num,
                    'error': f'Smoke test failed after {round_num} rounds: {smoke_result.get("detail")}'
                }

        # Both passed
        print(f'  LLM code passed static + smoke (round {round_num})')
        return {
            'code': code,
            'passed_static': True,
            'passed_smoke': True,
            'repair_rounds': round_num,
            'error': None
        }

    # Should not reach here, but just in case
    return {
        'code': code,
        'passed_static': False,
        'passed_smoke': False,
        'repair_rounds': max_repair_rounds,
        'error': 'Exhausted repair rounds'
    }
