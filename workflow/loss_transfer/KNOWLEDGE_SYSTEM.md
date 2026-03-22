# 知识积累系统 (Knowledge Accumulation System)

自动从实验中学习、积累和复用创新点的系统。

## 概述

知识积累系统让 Agent 能够:
1. **学习**: 从每次实验中提取"什么有效"
2. **积累**: 将创新点和代码存入知识库
3. **复用**: 检索历史知识并融合设计新 loss

## 核心理念

**传统方式**: 每次实验都从零开始,没有记忆  
**知识积累**: 每次实验都增强知识库,越用越聪明

## 系统架构

```
实验完成
  ↓
innovation_extractor.py (提取创新点)
  ├─ 分析: 为什么有效?
  ├─ 证据: SSIM 提升多少?
  └─ 存储: innovations.yaml
  ↓
code_generalizer.py (泛化代码)
  ├─ 提取: 组件函数
  ├─ 泛化: torch.nn.Module
  └─ 存储: modules/*.py
  ↓
知识库更新
  ↓
下次新任务
  ↓
retrieval_engine.py (检索知识)
  ├─ 关键词匹配
  ├─ 类型过滤
  └─ 性能排序
  ↓
fusion_designer.py (融合设计)
  ├─ 历史最优
  ├─ 新论文创新
  └─ 生成融合策略
  ↓
新实验 → 循环积累
```

## 核心模块

### 1. innovation_extractor.py - 创新点提取

**功能**: 从实验结果中提取关键创新

**提取内容**:
```yaml
innovation:
  paper: "exp#41"
  component_type: "frequency_loss"
  key_idea: "residual FFT"
  why_works: "保留相位信息"
  improvement: 0.0021
  confidence: 0.85
  tags: ["frequency", "fft", "residual"]
```

**工作流程**:
1. 检查 trial 是否通过
2. 计算 SSIM 提升
3. LLM 分析"为什么有效"
4. 存入 innovations.yaml

**示例**:
```python
from innovation_extractor import extract_and_save_innovations

innovations = extract_and_save_innovations('my_paper', summary)
# 返回: ['inn_001', 'inn_002']
```

### 2. code_generalizer.py - 代码泛化

**功能**: 将测试通过的代码转为可复用 torch 模块

**泛化策略**:
- 提取组件函数 (如 `_fft_loss`)
- 转为 `nn.Module` 类
- 添加文档和性能证据
- 支持 BHWC/BCHW layout

**输出示例**:
```python
class ResidualFFTLoss(nn.Module):
    """
    from exp#41
    Improvement: +0.0021 SSIM
    """
    def forward(self, pred, target):
        residual = pred - target
        fft = torch.fft.rfft2(residual, norm='ortho')
        return fft.abs().mean()
```

### 3. knowledge_db.py - 知识库

**存储结构**:
```
knowledge_base/
├── innovations.yaml    # 创新点数据库
├── modules/           # 泛化模块
│   ├── exp41_loss.py
│   └── sea_raft_loss.py
└── index.json        # 检索索引
```

**API**:
```python
from knowledge_db import KnowledgeDB

db = KnowledgeDB()
db.add_innovation(innovation)
results = db.search_by_tags(['fft'])
top = db.get_top_innovations(n=5)
```

### 4. retrieval_engine.py - 检索引擎

**功能**: 根据任务检索相关创新点

**检索策略**:
1. 关键词匹配 (tags, key_idea)
2. 类型过滤 (component_type)
3. 性能排序 (improvement)

**示例**:
```python
from retrieval_engine import retrieve_innovations

results = retrieve_innovations(
    query="improve frequency loss",
    component_type="frequency_loss",
    top_k=3
)
# 返回 Top-3 相关创新
```

### 5. fusion_designer.py - 融合设计

**功能**: 融合历史创新和新论文设计新 loss

**融合策略**:
- Strategy 1: 使用最佳历史创新
- Strategy 2: 融合 Top-2 创新
- Strategy 3: 新论文 + 历史最优

**示例**:
```python
from fusion_designer import design_fusion_loss

specs = design_fusion_loss(
    new_loss_ir={},
    task_description="improve ocean SR"
)
# 返回融合后的 patch 规格列表
```

## 使用示例

### 完整工作流

```bash
# 1. 运行实验 (自动提取创新点)
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug my_paper \
  --code_repo path/to/code

# 实验完成后自动:
# - 提取创新点 → innovations.yaml
# - 泛化代码 → modules/my_paper_loss.py
# - Git push
```

### 查看知识库

```python
from knowledge_db import KnowledgeDB

db = KnowledgeDB()

# 查看所有创新
innovations = db.get_all_innovations()
print(f'Total: {len(innovations)} innovations')

# 查看 Top-5
top5 = db.get_top_innovations(n=5)
for inn in top5:
    print(f"{inn['paper']}: {inn['key_idea']} (+{inn['improvement']:.4f})")
```

## 知识库格式

### innovations.yaml

```yaml
innovations:
  - id: "inn_001"
    paper: "exp#41"
    date: "2026-03-22"
    component_type: "frequency_loss"
    key_idea: "residual FFT"
    why_works: "保留相位信息"
    improvement: 0.0021
    confidence: 0.85
    tags: ["frequency", "fft", "residual"]
    
  - id: "inn_002"
    paper: "sea_raft"
    component_type: "gradient_loss"
    key_idea: "Scharr gradient"
    improvement: 0.0015
    tags: ["gradient", "scharr"]
```

### index.json

```json
{
  "next_id": 3,
  "tags_index": {
    "fft": ["inn_001"],
    "gradient": ["inn_002"],
    "frequency": ["inn_001"]
  }
}
```

## 关键特性

1. **自动化**: 实验完成后自动提取和存储
2. **智能检索**: 关键词+类型+性能多维度匹配
3. **代码复用**: 泛化为标准 torch 模块
4. **持续学习**: 每次实验都增强知识库
5. **版本控制**: 自动 git push 保存历史

## 注意事项

1. **提升阈值**: 只记录 improvement > 0.001 的创新
2. **置信度**: 基于提升幅度计算 (improvement / 0.01)
3. **LLM 依赖**: 创新点分析需要 LLM,失败时使用简化版
4. **去重**: 未来可添加相似度检查避免重复

## 与 Loss Transfer 的集成

知识积累系统已集成到 `orchestrate_trials.py`:
- 实验完成 → 自动提取创新点
- 最佳 trial → 自动泛化代码
- 全部推送到 GitHub

## 版本历史

- v1.0.0 (2026-03-22): 初始版本,完整功能实现
