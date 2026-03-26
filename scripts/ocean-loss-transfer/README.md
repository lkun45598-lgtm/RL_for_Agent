# Ocean Loss Transfer

将论文中的 loss 迁移到海洋超分训练管线的当前版本说明。

这份 README 只描述现在实际在跑的主流程。和早期版本相比，当前系统有两个重要变化：

- 主流程以 `loss_formula.json + task_context.json + analysis_plan.json` 为中心，`loss_ir.yaml` 现在是可选参考材料，不再是唯一入口。
- 执行单位已经从早期文档里的 “trial” 收敛为 `attempt`，并且支持 agent 代码生成、repair round、以及按 integration path 约束可编辑文件范围。

## 主流程

```text
论文 PDF / 论文代码仓库 / 可选 loss_ir.yaml
                ↓
      prepare_context.py 扫描代码和论文上下文
                ↓
 extract_loss_formula.py 生成 loss_formula.json + loss_spec.yaml
                ↓
 context_builder.py 汇总为 task_context.json
                ↓
 analysis_plan.json
   ├─ 手写
   └─ generate_analysis_plan() 通过本地 Agent 服务自动生成
                ↓
 agent_repair_loop.py 执行 attempts
   ├─ formula_variant: 公式原生代码生成
   └─ agent_code: inline code / code_path / objective → Agent 产出代码
                ↓
 execute_attempt()
   ├─ formula_interface 检查
   ├─ layer1 static
   ├─ layer2 smoke
   ├─ formula_alignment
   ├─ layer3 single model
   └─ layer4 full run
                ↓
 repair rounds（最多 3 轮，按 stop_layer 触发）
                ↓
 trajectory.jsonl + attempt_{N}/result.json + agent_loop_summary.json
```

## 快速开始

### 1. 只构建上下文

```bash
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug my_paper \
  --code_repo /path/to/paper/code \
  --paper_pdf /path/to/paper.pdf \
  --mode context_only
```

这会先构建实验目录，并尽量生成：

- `task_context.json`
- `loss_formula.json`
- `loss_spec.yaml`
- 可选的 `loss_ir.yaml` 路径占位

### 2. 手工写 `analysis_plan.json` 再执行

```bash
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug my_paper \
  --code_repo /path/to/paper/code \
  --paper_pdf /path/to/paper.pdf \
  --analysis_plan_json sandbox/loss_transfer_experiments/my_paper/analysis_plan.json
```

### 3. 让本地 Agent 服务自动生成 `analysis_plan.json`

```bash
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug my_paper \
  --code_repo /path/to/paper/code \
  --paper_pdf /path/to/paper.pdf \
  --auto_generate_plan \
  --service_url http://localhost:8787 \
  --service_api_key your-secret
```

### 4. 没有 `analysis_plan.json` 时走 bootstrap

如果不传 `--analysis_plan_json`，且保留默认 `--bootstrap_formula`，系统会按 `task_context.integration_assessment.recommended_path` 自动构造 bootstrap attempts：

- `loss_only` 且公式可代码生成时，优先生成 `formula_variant`
- 否则生成 `agent_code` bootstrap attempts

如果你明确不想自动 bootstrap：

```bash
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug my_paper \
  --code_repo /path/to/paper/code \
  --no_bootstrap_formula
```

这时如果没有 `analysis_plan.json`，结果会返回 `status=analysis_required`。

## 当前入口

### 顶层 CLI wrapper

这些脚本只是薄封装，真正实现都在 `loss_transfer/` 包内：

- `scripts/ocean-loss-transfer/run_auto_experiment.py`
- `scripts/ocean-loss-transfer/prepare_context.py`
- `scripts/ocean-loss-transfer/validate_loss.py`
- `scripts/ocean-loss-transfer/extract_loss_formula.py`
- `scripts/ocean-loss-transfer/write_loss_formula.py`
- `scripts/ocean-loss-transfer/write_loss_ir.py`
- `scripts/ocean-loss-transfer/run_baseline_noise.py`
- `scripts/ocean-loss-transfer/validate_analysis_plan.py`

### 包内主入口

- `loss_transfer/orchestration/run_auto_experiment.py`
- `loss_transfer/context/context_builder.py`
- `loss_transfer/agent/agent_repair_loop.py`
- `loss_transfer/attempts/attempt_executor.py`

## 核心工件

每个实验默认写到 `sandbox/loss_transfer_experiments/{paper_slug}/`。

常见文件：

- `task_context.json`: Agent 分析入口，包含 paper/code/formula/integration assessment
- `loss_formula.json`: 当前主流程的公式真值源
- `loss_spec.yaml`: 从公式 draft 派生出的 loss spec 草稿
- `analysis_plan.json`: 执行计划
- `trajectory.jsonl`: 结构化事件流
- `agent_loop_summary.json`: 全流程结果汇总
- `attempt_{N}/candidate_loss.py`: 当前 attempt 代码
- `attempt_{N}/result.json`: 当前 attempt 的验证与奖励摘要

## 主流程分解

### Step 1. `prepare_context()`

文件: `loss_transfer/context/prepare_context.py`

作用：

- 扫描论文代码仓库中的 loss 相关文件
- 可选解析论文 PDF 文本
- 为下游约定输出路径

函数签名：

```python
prepare_context(
    code_repo_path: str,
    paper_slug: str,
    output_dir: str | None = None,
    paper_pdf_path: str | None = None,
) -> dict[str, Any]
```

返回的关键字段：

```python
{
  "paper": {...},                    # 可选 PDF 解析结果
  "primary_files": [...],            # 代码侧重点文件
  "output_path": ".../loss_ir.yaml",
  "formula_output_path": ".../loss_formula.json",
  "analysis_plan_output_path": ".../analysis_plan.json",
  "code_repo": "...",
  "paper_slug": "..."
}
```

### Step 2. `extract_loss_formula_draft()`

文件: `loss_transfer/formula/extract_loss_formula.py`

这是当前主流程里最关键的前置步骤。它会从论文和代码联合生成 `loss_formula.json` draft，而不是直接把 `loss_ir.yaml` 当成主输入。

输出的最小结构：

```json
{
  "latex": ["..."],
  "params": {
    "gamma": 0.85
  },
  "symbol_map": {
    "\\hat{y}": "pred",
    "y": "target",
    "\\gamma": "gamma"
  }
}
```

当前实现里通常还会带这些扩展字段：

- `raw_formula_candidates`
- `adapter_heads`
- `notes`
- `review_required`
- `sources`
- `interface_analysis`

注意：

- `symbol_map` 必须是 1:1 映射
- 至少要能映射到 `pred` 和 `target`
- 额外变量如果不在 `pred/target/mask/params` 中，会被视为需要额外 runtime/model 支持

### Step 3. `build_task_context()`

文件: `loss_transfer/context/context_builder.py`

它会把前面的材料收敛成一个给 Agent 用的单一上下文文件。

函数签名：

```python
build_task_context(
    paper_slug: str,
    *,
    paper_pdf_path: str | None = None,
    code_repo_path: str | None = None,
    loss_ir_yaml: str | None = None,
    dataset_root: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]
```

`task_context.json` 当前最重要的字段：

- `paper_analysis`
- `code_analysis`
- `integration_assessment`
- `formula_spec`
- `formula_interface`
- `loss_spec`
- `compatibility`
- `analysis_plan_schema`
- `agent_guidance`
- `paths`

其中 `integration_assessment` 会给出当前推荐的 integration path：

- `loss_only`
- `adapter_wrapper`
- `extend_model_outputs`
- `model_surgery`

同时还会明确写出：

- `loss_ir_role`: `optional_reference` 或 `not_required_for_agent_analysis`
- `legacy_loss_ir_status.auto_extraction_enabled = False`

也就是说，旧的 `extract_loss_ir -> check_compatibility -> generate_patch` 主链现在已经不是默认 agentic path。

### Step 4. `analysis_plan.json`

文件: `loss_transfer/agent/validate_analysis_plan.py`

当前支持的最小 schema：

```json
{
  "summary": "why this plan is correct",
  "stop_on_first_pass": false,
  "integration_decision": {
    "path": "loss_only | adapter_wrapper | extend_model_outputs | model_surgery",
    "rationale": "why this path is required",
    "evidence_refs": ["paper_analysis.loss_snippets[0]"]
  },
  "attempts": [
    {
      "name": "Attempt 1",
      "kind": "agent_code | formula_variant",
      "variant": "faithful | stabilized",
      "code": "optional inline code",
      "code_path": "optional path",
      "objective": "required for agent_code if code/code_path omitted",
      "files_to_edit": ["optional logical edit targets"],
      "required_edit_paths": ["optional paths that must actually change"],
      "evidence_refs": ["task_context references"],
      "run_training": true,
      "notes": "optional notes"
    }
  ]
}
```

约束：

- `kind=formula_variant` 时只支持 `faithful` 和 `stabilized`
- `kind=agent_code` 时必须提供 `code`、`code_path` 或 `objective` 之一
- `integration_decision.path` 只能是四种 integration path 之一

### Step 5. `run_agent_repair_loop()`

文件: `loss_transfer/agent/agent_repair_loop.py`

行为：

- 加载并校验 `analysis_plan.json`
- 如果没有 plan 且允许 bootstrap，则自动构造 attempts
- 逐个执行 attempts
- 根据 `stop_on_first_pass` 决定是否首个成功后立即停止
- 生成 `agent_loop_summary.json`

当前 summary 的关键字段：

```python
{
  "status": "completed | completed_with_failures | analysis_required",
  "paper_slug": "...",
  "task_context_path": "...",
  "analysis_plan_path": "...",
  "trajectory_path": ".../trajectory.jsonl",
  "attempts": [...],
  "best_attempt_id": 1,
  "best_metric_name": "swinir",
  "best_metric_value": 0.72
}
```

### Step 6. `execute_attempt()`

文件: `loss_transfer/attempts/attempt_executor.py`

当前支持两类 attempt：

1. `formula_variant`
2. `agent_code`

`agent_code` 又有三种来源：

- 直接内联 `code`
- 指向现成 `code_path`
- 只给 `objective`，由 `generate_candidate_loss()` 调本地 Agent 服务去写代码

执行顺序是：

1. 解析或生成 `candidate_loss.py`
2. 检查公式接口是否在当前 runtime 能支撑
3. 跑 static
4. 跑 smoke
5. 如果有 `loss_formula.json`，跑 `formula_alignment`
6. 若 `run_training=True`，继续跑 single model 和 full run
7. 若 stop layer 可修复，进入 repair rounds

repair 机制：

- 最多 3 轮
- 可修复层包括 `layer1 / layer2 / formula_alignment / layer3 / layer4`
- 如果修复后反而从更高层退化到更低层，会自动回滚到修复前版本

### Step 7. 验证层

文件: `loss_transfer/validation/validate_loss.py`

当前真实验证链不是只有早期 README 里的 “4 层”，而是：

1. `formula_interface`
2. `layer1`
3. `layer2`
4. `formula_alignment`
5. `layer3`
6. `layer4`

含义：

- `formula_interface`: 先判断公式是否要求 runtime/model 提供额外 loss inputs
- `layer1`: 静态检查
- `layer2`: smoke test
- `formula_alignment`: 公式和代码的一致性检查
- `layer3`: 单模型训练验证
- `layer4`: 四模型 full run

对外 CLI 仍按四种模式暴露：

```bash
python scripts/ocean-loss-transfer/validate_loss.py --loss_file ... --mode static
python scripts/ocean-loss-transfer/validate_loss.py --loss_file ... --mode smoke
python scripts/ocean-loss-transfer/validate_loss.py --loss_file ... --mode single
python scripts/ocean-loss-transfer/validate_loss.py --loss_file ... --mode full
```

## Integration Path 语义

当前系统围绕四种 integration path 约束 attempt 的编辑范围：

- `loss_only`: 只改 `candidate_loss.py`
- `adapter_wrapper`: 可以同时修改 attempt-scoped `sandbox_model_adapter.py` / `sandbox_trainer.py`
- `extend_model_outputs`: 可以改 adapter 和 copied `models/`
- `model_surgery`: 允许对 copied `models/` 做更深修改

这些路径会影响：

- `analysis_plan.json` 的 `integration_decision.path`
- 自动补全的 `files_to_edit`
- 自动补全的 `required_edit_paths`
- Agent codegen/repair 时的白名单

## Optional / Legacy 工具

以下模块还在，但不再是默认 agentic 主链：

- `loss_transfer/ir/extract_loss_ir.py`
- `loss_transfer/ir/write_loss_ir.py`
- `loss_transfer/ir/check_compatibility.py`
- `loss_transfer/generation/generate_patch.py`

当前推荐理解是：

- `loss_formula.json` 是主流程真值源
- `loss_ir.yaml` 是可选补充参考

## Baseline

文件: `loss_transfer/validation/run_baseline_noise.py`

它会跑原始训练默认 loss，输出：

- `workflow/loss_transfer/baseline_thresholds.yaml`

当前字段：

```yaml
model: swinir
n_runs: 3
ssim_mean: 0.6645
ssim_std: 0.0042
psnr_mean: 38.1
psnr_std: 0.2
viable_threshold: 0.6545
improvement_threshold: 0.6745
```

`execute_attempt()` 会优先读取这个文件；如果没有，则使用内置默认阈值。

## 目录结构

```text
scripts/ocean-loss-transfer/
├── loss_transfer/
│   ├── common/
│   │   ├── _types.py
│   │   ├── _utils.py
│   │   ├── paths.py
│   │   └── trajectory_logger.py
│   ├── context/
│   ├── formula/
│   ├── agent/
│   ├── attempts/
│   ├── validation/
│   ├── ir/
│   ├── generation/
│   └── orchestration/
├── run_auto_experiment.py
├── prepare_context.py
├── validate_loss.py
├── extract_loss_formula.py
├── write_loss_formula.py
├── write_loss_ir.py
├── run_baseline_noise.py
├── validate_analysis_plan.py
└── tests/
```

## 代码导入示例

推荐直接从包导入，不要再依赖旧的平铺模块名：

```python
from loss_transfer.context.context_builder import build_task_context
from loss_transfer.agent.validate_analysis_plan import validate_analysis_plan
from loss_transfer.attempts.attempt_executor import execute_attempt
from loss_transfer.validation.validate_loss import validate_static, validate_smoke
```

如果是在仓库外或 ad-hoc 脚本里调用，至少先把 `scripts/ocean-loss-transfer` 放到 `sys.path`：

```python
import sys
sys.path.insert(0, 'scripts/ocean-loss-transfer')

from loss_transfer.orchestration.run_auto_experiment import run_auto_experiment
```

## 现阶段最重要的判断

如果你只记住一件事，应该是：

- 当前主流程优先让 Agent 读取 `task_context.json`
- 再围绕 `loss_formula.json` 和 `analysis_plan.json` 做集成决策
- 不要再把 `loss_ir.yaml` 当成唯一驱动源
