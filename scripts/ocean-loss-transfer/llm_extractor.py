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
from typing import Dict, List, Optional
from pathlib import Path
from _types import CodeSnippet, LossIRDict

# 加载 .env 文件
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass  # dotenv 不是必需的


def call_llm(prompt: str) -> str:
    """调用 LLM API"""
    api_key: Optional[str] = os.getenv('ANTHROPIC_API_KEY')
    base_url: str = os.getenv('ANTHROPIC_BASE_URL', 'https://node-hk.sssaicode.com/api')
    model: str = os.getenv('ANTHROPIC_MODEL_ID', 'claude-sonnet-4-5-20250929')

    url = f"{base_url}/v1/messages"
    headers: Dict[str, Optional[str]] = {
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json'
    }

    data = {
        'model': model,
        'max_tokens': 4096,
        'messages': [{'role': 'user', 'content': prompt}]
    }

    response = requests.post(url, headers=headers, json=data, timeout=60)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"API Error: {e}")
        print(f"Response: {response.text[:500]}")
        raise
    return response.json()['content'][0]['text']


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
