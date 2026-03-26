"""
@file llm_extractor.py
@description LLM 自动提取 Loss IR
@author Leizheng
@contributors kongzhiquan
@date 2026-03-22
@version 1.1.0

@changelog
  - 2026-03-22 Leizheng: v1.0.0 initial version
  - 2026-03-23 kongzhiquan: v1.1.0 refine type annotations
"""

import os
import json
import requests
from typing import Any, Dict, List, Literal, Optional
from pathlib import Path
from loss_transfer.common._types import CodeSnippet, LossIRDict
from loss_transfer.common.paths import PROJECT_ENV_FILE

# 加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ENV_FILE)
except ImportError:
    pass  # dotenv 不是必需的


ProviderKind = Literal['anthropic', 'openai']
OpenAIApiMode = Literal['chat', 'responses']


def _get_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _resolve_provider() -> ProviderKind:
    explicit = (_get_env('LLM_PROVIDER', 'MODEL_PROVIDER') or '').lower()
    if explicit in {'anthropic', 'openai'}:
        return explicit  # type: ignore[return-value]
    if _get_env('LLM_API_KEY', 'OPENAI_API_KEY'):
        return 'openai'
    return 'anthropic'


def _resolve_openai_api_mode() -> OpenAIApiMode:
    mode = (_get_env('LLM_API_MODE', 'OPENAI_API_MODE') or 'chat').lower()
    if mode == 'responses':
        return 'responses'
    return 'chat'


def _normalize_openai_base_url(base_url: str) -> str:
    normalized = base_url.rstrip('/')
    if not normalized.endswith('/v1'):
        normalized += '/v1'
    return normalized


def _normalize_anthropic_base_url(base_url: str) -> str:
    normalized = base_url.rstrip('/')
    if normalized.endswith('/v1'):
        normalized = normalized[:-3]
    return normalized


def _extract_openai_text(payload: Dict[str, Any], api_mode: OpenAIApiMode) -> str:
    if api_mode == 'responses':
        parts = []
        for output in payload.get('output', []):
            for content in output.get('content', []):
                text = content.get('text')
                if isinstance(text, str) and text.strip():
                    parts.append(text)
        if parts:
            return '\n'.join(parts)
        output_text = payload.get('output_text')
        if isinstance(output_text, str) and output_text.strip():
            return output_text
        raise ValueError('OpenAI responses API 返回中未找到文本内容')

    message = ((payload.get('choices') or [{}])[0]).get('message', {})
    content = message.get('content')
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = [item.get('text', '') for item in content if isinstance(item, dict)]
        text = '\n'.join(part for part in parts if part)
        if text.strip():
            return text
    raise ValueError('OpenAI chat API 返回中未找到文本内容')


def call_llm(prompt: str) -> str:
    """调用 LLM API"""
    provider = _resolve_provider()

    if provider == 'openai':
        api_key = _get_env('LLM_API_KEY', 'OPENAI_API_KEY')
        if not api_key:
            raise ValueError('缺少 LLM_API_KEY 或 OPENAI_API_KEY')
        base_url = _normalize_openai_base_url(_get_env('LLM_BASE_URL', 'OPENAI_BASE_URL') or 'https://api.openai.com/v1')
        model = _get_env('LLM_MODEL_ID', 'LLM_MODEL', 'OPENAI_MODEL_ID', 'OPENAI_MODEL') or 'gpt-4o'
        api_mode = _resolve_openai_api_mode()
        url = f"{base_url}/responses" if api_mode == 'responses' else f"{base_url}/chat/completions"
        headers: Dict[str, str] = {
            'Authorization': f'Bearer {api_key}',
            'content-type': 'application/json',
        }
        data: Dict[str, Any]
        if api_mode == 'responses':
            data = {
                'model': model,
                'input': prompt,
                'max_output_tokens': 4096,
            }
        else:
            data = {
                'model': model,
                'max_tokens': 4096,
                'messages': [{'role': 'user', 'content': prompt}],
            }
    else:
        api_key = _get_env('LLM_API_KEY', 'ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError('缺少 LLM_API_KEY 或 ANTHROPIC_API_KEY')
        base_url = _normalize_anthropic_base_url(_get_env('LLM_BASE_URL', 'ANTHROPIC_BASE_URL') or 'https://api.anthropic.com')
        model = _get_env('LLM_MODEL_ID', 'LLM_MODEL', 'ANTHROPIC_MODEL_ID') or 'claude-sonnet-4-5-20250929'
        url = f"{base_url}/v1/messages"
        headers = {
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        }
        data = {
            'model': model,
            'max_tokens': 4096,
            'messages': [{'role': 'user', 'content': prompt}],
        }

    response = requests.post(url, headers=headers, json=data, timeout=60)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"API Error: {e}")
        print(f"Response: {response.text[:500]}")
        raise

    payload = response.json()
    if provider == 'openai':
        return _extract_openai_text(payload, _resolve_openai_api_mode())
    return payload['content'][0]['text']


def build_extraction_prompt(code_snippets: List[CodeSnippet], loss_ir_schema: str) -> str:
    """构建提取 prompt"""

    code_text = "\n\n".join([
        f"File: {s['file']}\n```python\n{s['content'][:1500]}\n```"
        for s in code_snippets[:3]
    ])

    prompt = f"""分析以下代码中的 loss 函数,提取其结构化信息。

代码:
{code_text}

请按照以下 YAML schema 输出 Loss IR:

```yaml
components:
  - name: "loss组件名称"
    type: "pixel_loss|gradient_loss|frequency_loss"
    weight: 1.0
    implementation:
      reduction: "mean|sum"
      operates_on: "pixel_space|frequency_space"

multi_scale:
  enabled: true/false
  scales: [1, 2, 4]

incompatibility_flags:
  requires_model_features: false
  requires_pretrained_network: false
```

只输出 YAML,不要其他解释。"""

    return prompt


def extract_with_llm(code_snippets: List[CodeSnippet]) -> LossIRDict:
    """使用 LLM 提取 Loss IR"""

    prompt = build_extraction_prompt(code_snippets, "")

    try:
        response = call_llm(prompt)

        # 提取 YAML 部分
        import yaml
        if '```yaml' in response:
            yaml_text = response.split('```yaml')[1].split('```')[0]
        elif '```' in response:
            yaml_text = response.split('```')[1].split('```')[0]
        else:
            yaml_text = response

        extracted: LossIRDict = yaml.safe_load(yaml_text)
        return extracted

    except Exception as e:
        print(f"LLM extraction failed: {e}")
        return {}  # type: ignore[return-value]
