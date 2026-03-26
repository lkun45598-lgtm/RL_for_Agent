# Ocean Loss Transfer — 全流程指南

将论文中的 loss 函数自动迁移到海洋超分辨率训练管线的端到端系统。

---

## 目录

- [概览](#概览)
- [快速开始](#快速开始)
- [全流程详解](#全流程详解)
  - [Step 1 — 提取 Loss IR](#step-1--提取-loss-ir)
  - [Step 2 — 兼容性检查](#step-2--兼容性检查)
  - [Step 3 — 生成 Trial 规格](#step-3--生成-trial-规格)
  - [Step 4 — 执行 Trial](#step-4--执行-trial)
  - [Step 5 — 4 层验证](#step-5--4-层验证)
  - [Step 6 — 结果优选与记录](#step-6--结果优选与记录)
  - [Step 7 — 知识沉淀](#step-7--知识沉淀)
- [核心数据类型](#核心数据类型)
- [目录结构](#目录结构)
- [配置文件](#配置文件)

---

## 概览

```
论文 PDF / 代码仓库
        ↓
   [Step 1] 提取 Loss IR (LossIR)
        ↓
   [Step 2] 兼容性检查 (CompatibilityResult)
        ↓
   [Step 3] 构建 Task Context (task_context.json)
        ↓
   [Step 4] Agent 生成 Analysis Plan (analysis_plan.json)
        ↓
   [Step 5] 逐一执行 Attempt
     ├─ formula_variant: formula_code_generator → candidate_loss.py
     └─ agent_code: Agent 直接提交代码
        ↓
   [Step 6] 4 层渐进式验证 (ValidationResult)
     Layer 1: 静态检查 (<1s)
     Layer 2: Smoke Test (<10s)
     Layer 3: Single Model (2-5min)
     Layer 4: Full Run, 4 模型 (5-10min)
        ↓
   [Step 7] 轨迹记录与优选最佳 Attempt (ExperimentSummary)
        ↓
   [Step 7] 提取创新点 (Innovation) → 泛化代码 → git push
```

整个流程通过 `run_auto_experiment.py` 一键触发，也可以分步调用各子模块。

---

## 快速开始

```bash
# 全自动模式 (论文代码仓库 → 结果)
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug my_paper \
  --code_repo /path/to/paper/code

# 手动模式 (先生成模板，手工填写后再运行)
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug my_paper
# → 编辑 sandbox/loss_transfer_experiments/my_paper/loss_ir.yaml
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug my_paper \
  --loss_ir_yaml sandbox/loss_transfer_experiments/my_paper/loss_ir.yaml

# 从已有 Loss IR 直接跑实验
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug my_paper \
  --loss_ir_yaml /path/to/loss_ir.yaml
```

---

## 全流程详解

### Step 1 — 提取 Loss IR

**模块**: `extract_loss_ir.py` + `llm_extractor.py`

**输入**

| 参数 | 类型 | 说明 |
|------|------|------|
| `paper_pdf_path` | `Optional[str]` | 论文 PDF 路径 (可选) |
| `code_repo_path` | `Optional[str]` | 代码仓库根目录 (可选) |
| `output_yaml_path` | `str` | 输出的 YAML 文件路径 |
| `manual_mode` | `bool` | 强制手动模式，生成空模板 |

**处理过程**

1. 扫描代码仓库中匹配 `*loss*.py`、`*criterion*.py`、`*objective*.py` 的文件，读取前 2000 字符作为代码片段。
2. 调用 LLM，将代码片段送入提取 prompt，解析返回的 YAML。
3. 将原始代码片段保存到 `_raw_code_snippets` 字段，供后续 LLM 代码生成使用。

**输出**: 写入 YAML 文件，返回文件路径 `str`

YAML 内容的结构对应 `LossIRDict`：

```python
class LossIRDict(TypedDict, total=False):
    metadata:              Dict[str, Union[str, List[str]]]
    interface:             LossInterfaceDict       # 含 input_tensors
    components:            List[ComponentDict]     # 见下方
    multi_scale:           MultiScaleConfig        # {enabled, scales}
    combination:           CombinationConfig       # {method}
    incompatibility_flags: IncompatibilityFlags    # {requires_*: bool}
    _raw_code_snippets:    List[CodeSnippet]       # 保留给 LLM 生成器

class ComponentDict(TypedDict, total=False):
    name:             str
    type:             str   # 'pixel_loss' | 'gradient_loss' | 'frequency_loss' | ...
    weight:           float
    implementation:   ComponentImplementation   # {reduction, operates_on}
    required_tensors: List[str]
    required_imports: List[str]
    formula:          str
    code_evidence:    Dict[str, str]
```

---

### Step 2 — 兼容性检查

**模块**: `check_compatibility.py`

**输入**: `LossIR` 对象（从 YAML 反序列化）

**处理过程**

1. **硬性不兼容**：检查 `incompatibility_flags` 中的四个标志位（需要模型特征、预训练网络、对抗训练、多次 forward）。任意一个为 `True` 即终止流程。
2. **已知失败模式**：扫描组件名称，SSIM 和 Laplacian 是已知必崩溃模式，触发警告。

**输出**: `CompatibilityResult`

```python
class CompatibilityResult(TypedDict, total=False):
    status:  CompatibilityStatus  # 'incompatible' | 'partially_compatible' | 'fully_compatible'
    reason:  str                  # 不兼容原因 (incompatible 时)
    issues:  List[str]            # 硬性不兼容问题列表
    warnings: List[str]           # 已知失败模式警告
```

`status == 'incompatible'` 时流程终止；`'partially_compatible'` 时打印警告但继续。

---

### Step 3 — 构建 Task Context

**模块**: `context_builder.py`

**输入**: 论文 PDF、论文代码仓库、已有 `loss_ir.yaml`（可选）

**处理过程**

1. 复用 `prepare_context.py` 扫描论文代码中的 loss 相关上下文
2. 若 `loss_formula.json` 不存在，则调用 `extract_loss_formula.py` 生成公式草稿
3. 加载 `loss_ir.yaml`，运行 `check_compatibility.py`
4. 汇总为单一的 `task_context.json`

**输出**

- `task_context.json`: Agent 分析入口
- `loss_formula.json`: LaTeX / params / symbol_map
- `loss_spec.yaml`: loss 规格草稿
- `analysis_plan.json` 路径约定

---

### Step 4 — 执行 Attempt

**模块**: `attempt_executor.py`

**输入**

| 参数 | 类型 |
|------|------|
| `attempt_spec` | `Dict[str, Any]` |
| `attempt_id` | `int` |
| `paper_slug` | `str` |

#### 支持的 Attempt 类型

1. `agent_code`: Agent 直接提交 `candidate_loss.py`
2. `formula_variant`: 对 `loss_formula.json` 使用确定性公式代码生成（如 `faithful` / `stabilized`）

执行层本身不做“枚举修补”，只负责：
- 写入候选代码
- 跑 static / smoke / formula-alignment
- 按需跑 Layer 3 / 4 训练
- 记录结果和奖励摘要

代码结构固定为：
```
_align_mask() + _downsample() + _downsample_mask()
+ _pixel_loss()      ← 由 pixel_variant 选择
+ _gradient_loss()   ← 由 gradient_variant 选择
+ _fft_loss()        ← 由 fft_variant 选择
+ sandbox_loss()     ← multi-scale 主函数
```

#### LLM 生成模式（保留为代码生成器，不再作为固定 Trial 枚举）

**模块**: `llm_code_generator.py`

1. 将 `_raw_code_snippets`（原始论文代码）+ `components`（IR 分析）+ `strategy` 组合成 prompt。
2. 调用 LLM 生成完整 Python 文件，提取 ` ```python ` 代码块。
3. 执行 **Layer 1 (static) + Layer 2 (smoke)** 验证。
4. 若失败，将错误信息和当前代码一并发给 LLM 进行修复，最多 `max_repair_rounds=3` 轮。
5. 双层验证通过后，**跳过 Layer 1-2，直接进入 Layer 3**。
6. 若修复后仍失败，**降级为模板模式**（使用默认 `rel_l2 + sobel_3x3 + residual_rfft2_abs`）走完整 4 层。

LLM 生成结果：

```python
class GenerateResult(TypedDict):
    code:          str
    passed_static: bool
    passed_smoke:  bool
    repair_rounds: int
    error:         Optional[str]
```

---

### Step 5 — 4 层验证

**模块**: `validate_loss.py`

每一层的结果类型相同：

```python
class ValidationResult(TypedDict, total=False):
    passed:     bool         # 必填
    error:      str          # 错误类型标识
    detail:     str          # 详细描述
    fix_hint:   str          # 修复建议 (给 LLM 修复 prompt 用)
    traceback:  str          # Python traceback (import/runtime 失败时)
    metrics:    TrainingMetrics
    loss_value: float        # smoke test 时的 loss 数值
    grad_norm:  float        # smoke test 时的梯度范数

class TrainingMetrics(TypedDict, total=False):
    val_ssim: float
    val_psnr: float
    swinir:   float
    edsr:     float
    fno2d:    float
    unet2d:   float
```

| Layer | 名称 | 耗时 | 检查内容 |
|-------|------|------|---------|
| **Layer 1** | 静态检查 | <1s | 语法、import 白名单 (`torch` / `torch.nn.functional` / `math`)、函数签名 (`pred`, `target`)、禁止模式 (`open`, `subprocess`)、已知崩溃模式 (SSIM, Laplacian) |
| **Layer 2** | Smoke Test | <10s | 动态导入、dummy forward (BHWC `[2,128,128,2]`)、NaN/Inf 检查、backward 梯度检查、`mask=None` 分支 |
| **Layer 3** | Single Model | 2-5min | 真实数据，SwinIR 完整训练，`val_ssim` ≥ 0.3 |
| **Layer 4** | Full Run | 5-10min | 4 个模型 (SwinIR / EDSR / FNO2d / UNet2d) 并行，全部 `val_ssim` ≥ 0.3，可选对比基线阈值 |

任意一层失败即停止，记录 `layer_stopped`，不进入后续层。

---

### Step 6 — 结果优选与记录

**模块**: `agent_repair_loop.py` + `trajectory_logger.py`

每个 Attempt 完成后，系统会写入 `sandbox/loss_transfer_experiments/{paper_slug}/attempt_{N}/`：
- `candidate_loss.py` — 当前候选 loss
- `attempt_spec.json` — Agent 的动作描述
- `result.json` — 分层验证结果 + 指标 + baseline delta

全流程还会持续写入：
- `task_context.json`
- `analysis_plan.json`
- `trajectory.jsonl`
- `agent_loop_summary.json`

最终 summary 结构保持“最优候选 + 全部尝试列表”的形式：

```python
class ExperimentSummary(TypedDict, total=False):
    paper_slug:  str
    baseline:    BaselineThresholds   # 基线 SSIM 统计
    trials:      List[TrialSummaryItem]
    best_trial:  Optional[int]        # 最高 SwinIR SSIM 的 trial_id
    best_ssim:   float
    improvement: Optional[float]      # best_ssim - baseline.ssim_mean

class TrialSummaryItem(TypedDict, total=False):
    trial_id:     int
    name:         str
    passed:       bool
    layer_stopped: Optional[str]      # 'layer1' | 'layer2' | 'layer3' | 'layer4'
    metrics:      TrainingMetrics

class BaselineThresholds(TypedDict, total=False):
    model:                str
    ssim_mean:            float
    ssim_std:             float
    viable_threshold:     float   # ssim_mean - ssim_std
    improvement_threshold: float  # ssim_mean + ssim_std
```

`agent_loop_summary.json` 写入 `sandbox/loss_transfer_experiments/{paper_slug}/`。

---

### Step 7 — 知识沉淀

**模块**: `innovation_extractor.py` + `code_generalizer.py` + `knowledge_db.py`

仅当存在通过全部验证的 Trial 时执行。

**1. 提取创新点**

将 Trial 结果和 Loss IR 发给 LLM，提取结构化的创新描述，存入知识库：

```python
class Innovation(TypedDict, total=False):
    id:             str      # 'inn_001', 'inn_002', ...
    paper:          str
    component_type: str      # 'pixel_loss' | 'gradient_loss' | 'frequency_loss'
    key_idea:       str      # 一句话核心创新
    why_works:      str      # 有效原因分析
    improvement:    float    # SSIM 提升量
    confidence:     float    # min(improvement / 0.01, 1.0)
    evidence:       InnovationEvidence   # {baseline_ssim, new_ssim}
    tags:           List[str]
    date:           str      # ISO 8601
```

知识库存储在 `workflow/loss_transfer/knowledge_base/innovations.yaml`，索引在 `index.json`（按 tag 倒排）。

**2. 泛化代码**

从最佳 Trial 的 `sandbox_loss.py` 中提取 `_fft_loss()` 函数，包装成独立的 `nn.Module`，写入 `workflow/loss_transfer/knowledge_base/modules/{paper_slug}_loss.py`，供后续 Trial 的 `retrieval_engine.py` 检索复用。

**3. 自动 Push**

若有最佳 Trial，自动执行：
```bash
git add sandbox/loss_transfer_experiments/{paper_slug}
git commit -m "Loss Transfer: {paper_slug} - Best Trial {N} (SSIM={best:.4f})"
git push
```

---

## 核心数据类型

所有类型定义集中在 `_types.py`，以下列出模块间传递的主要类型。

### Literal 枚举

```python
PixelVariant      = Literal['rel_l2', 'abs_l1', 'smooth_l1']
GradientVariant   = Literal['sobel_3x3', 'scharr_3x3']
FFTVariant        = Literal['residual_rfft2_abs', 'amplitude_diff']
GenerationStrategy = Literal['faithful', 'creative']
CompatibilityStatus = Literal['incompatible', 'partially_compatible', 'fully_compatible']
ValidationLayer   = Literal['layer1', 'layer2', 'layer3', 'layer4']
```

### 流程主干类型

```
CodeSnippet  ──→  LossIRDict / LossIR  ──→  CompatibilityResult
                        ↓
                   PatchSpec × 5
                        ↓
                  GenerateResult (LLM 模式)
                        ↓
                  ValidationResult × 4层
                        ↓
                  TrialResult  ──→  TrialSummaryItem
                        ↓
                  ExperimentSummary
                        ↓
                  Innovation → KnowledgeDB
```

### 公共导入示例

```python
import sys
sys.path.append('./scripts/ocean-loss-transfer') # 允许跨模块导入，路径根据实际情况调整
from _types import (
    CodeSnippet, LossIRDict, LossIRLike,
    ValidationResult, TrainingMetrics,
    PatchSpec, TemplatePatchSpec, LLMPatchSpec,
    GenerateResult,
    TrialResult, TrialSummaryItem,
    ExperimentSummary, BaselineThresholds,
    Innovation,
    PixelVariant, GradientVariant, FFTVariant, GenerationStrategy,
)
from loss_ir_schema import LossIR, LossIRLike
```

---

## 目录结构

```
scripts/ocean-loss-transfer/
├── _types.py                  # 所有共享类型定义 (TypedDict + Literal)
├── loss_ir_schema.py          # LossIR / LossComponent 数据类，LossIRLike
├── extract_loss_ir.py         # Step 1: 扫描代码 + LLM 提取
├── llm_extractor.py           # LLM API 调用
├── check_compatibility.py     # Step 2: 兼容性检查
├── context_builder.py         # Step 3: 构建 task_context.json
├── attempt_executor.py        # Step 4: 执行单个 Agent attempt
├── agent_repair_loop.py       # Step 6: analysis_plan 驱动闭环
├── llm_code_generator.py      # LLM 生成 + 多轮修复
├── patch_templates.py         # 模板代码片段 + assemble_sandbox_loss()
├── generate_patch.py          # TemplatePatchSpec → sandbox_loss.py
├── validate_loss.py           # Step 5: 4 层验证器
├── trajectory_logger.py       # task / attempt / result 轨迹落盘
├── innovation_extractor.py    # Step 7: 创新点提取
├── code_generalizer.py        # Step 7: 代码泛化为 nn.Module
├── knowledge_db.py            # 知识库 CRUD
├── retrieval_engine.py        # 按 query 检索历史创新
├── fusion_designer.py         # 基于历史创新的融合规格设计
├── run_auto_experiment.py     # 入口: 全流程编排
└── run_baseline_noise.py      # 基线噪声测量工具

sandbox/
├── sandbox_loss.py            # 当前生效的 loss (每次 Trial 覆盖)
├── _run_once.py               # 单模型训练脚本
├── run_all_models.sh          # 4 模型并行训练脚本
└── loss_transfer_experiments/
    └── {paper_slug}/
        ├── loss_ir.yaml             # 提取的 Loss IR
        ├── task_context.json        # Agent 分析入口
        ├── analysis_plan.json       # Agent 的动作计划
        ├── trajectory.jsonl         # 结构化闭环轨迹
        ├── agent_loop_summary.json  # 最终实验汇总
        └── attempt_{N}/
            ├── candidate_loss.py
            └── result.json

workflow/loss_transfer/
├── baseline_thresholds.yaml   # 基线 SSIM 统计 (BaselineThresholds)
├── blocked_patterns.yaml      # 已知崩溃模式 (BlockedPatternsConfig)
├── target_interface_spec.yaml # sandbox_loss 目标接口规范
└── knowledge_base/
    ├── innovations.yaml       # Innovation 列表
    ├── index.json             # tag 倒排索引 (KnowledgeIndex)
    └── modules/               # 泛化后的 nn.Module 代码
```

---

## 配置文件

### `blocked_patterns.yaml`

记录实验中发现的必崩溃模式，静态检查 (Layer 1) 和 LLM prompt 中都会引用：

```yaml
blocked_components:
  - name: ssim_loss
    action: REJECT
    reason: "已知会导致训练崩溃 (exp#11)"
    fix_hint: "使用 L1/L2/gradient loss 替代"
  - name: laplacian_loss
    action: REJECT
    reason: "已知会崩溃 (exp#20,#40,#66)"
    fix_hint: "使用 Sobel/Scharr 替代"
```

### `baseline_thresholds.yaml`

由 `run_baseline_noise.py` 生成，记录当前 loss 的性能基准：

```yaml
model: swinir
ssim_mean: 0.6645
ssim_std: 0.0042
viable_threshold: 0.6603      # ssim_mean - ssim_std
improvement_threshold: 0.6687 # ssim_mean + ssim_std
```
