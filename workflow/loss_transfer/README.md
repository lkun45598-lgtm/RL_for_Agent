# Loss Transfer System

面向 Agent 的论文 loss 迁移工作流说明。

这份文档是中层概览，和 `scripts/ocean-loss-transfer/README.md` 保持同一口径，但更偏向工作流和目录约定，不展开每个模块的实现细节。

## 当前口径

相对于早期版本，当前系统已经切换到下面这条主链：

```text
paper/code evidence
  ↓
prepare_context.py
  ↓
extract_loss_formula.py
  ↓
task_context.json
  ↓
analysis_plan.json
  ↓
agent_repair_loop.py
  ↓
attempt_{N}/candidate_loss.py + result.json
  ↓
agent_loop_summary.json
```

两点要特别注意：

- 主流程以 `loss_formula.json + task_context.json + analysis_plan.json` 为中心。
- `loss_ir.yaml` 还在，但现在是可选参考，不再是默认主入口。

## 快速开始

### 1. 直接运行自动实验

```bash
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug my_paper \
  --code_repo /path/to/paper/code \
  --paper_pdf /path/to/paper.pdf
```

### 2. 只构建上下文，不执行 attempts

```bash
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug my_paper \
  --code_repo /path/to/paper/code \
  --paper_pdf /path/to/paper.pdf \
  --mode context_only
```

### 3. 使用手写 `analysis_plan.json`

```bash
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug my_paper \
  --code_repo /path/to/paper/code \
  --analysis_plan_json sandbox/loss_transfer_experiments/my_paper/analysis_plan.json
```

### 4. 让本地 Agent 服务自动生成 `analysis_plan.json`

```bash
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug my_paper \
  --code_repo /path/to/paper/code \
  --auto_generate_plan \
  --service_url http://localhost:8787 \
  --service_api_key your-secret
```

如果没有传 `analysis_plan.json`，且保留默认 `--bootstrap_formula`，系统会根据 `integration_assessment.recommended_path` 自动构造 bootstrap attempts。

## 核心工件

默认实验目录：

`sandbox/loss_transfer_experiments/{paper_slug}/`

最重要的文件：

- `task_context.json`: 汇总 paper/code/formula/integration assessment 的 Agent 分析入口
- `loss_formula.json`: 当前主流程的公式真值源
- `loss_spec.yaml`: 从公式 draft 派生出的 loss spec 草稿
- `analysis_plan.json`: 执行计划
- `trajectory.jsonl`: 结构化事件轨迹
- `agent_loop_summary.json`: 全流程摘要
- `attempt_{N}/candidate_loss.py`: 第 N 个 attempt 的候选代码
- `attempt_{N}/result.json`: 第 N 个 attempt 的验证结果和奖励摘要

## 执行模型

### 1. `task_context.json`

由 `loss_transfer/context/context_builder.py` 生成，当前最关键的字段包括：

- `paper_analysis`
- `code_analysis`
- `formula_spec`
- `formula_interface`
- `integration_assessment`
- `analysis_plan_schema`
- `agent_guidance`
- `paths`

其中 `integration_assessment.recommended_path` 当前只会落在四种 integration path 之一：

- `loss_only`
- `adapter_wrapper`
- `extend_model_outputs`
- `model_surgery`

### 2. `analysis_plan.json`

当前最小有效结构：

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
      "files_to_edit": ["optional logical targets"],
      "required_edit_paths": ["optional paths that must really change"],
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
- `integration_decision.path` 必须是四种 integration path 之一

### 3. Attempt 执行

当前只支持两类 attempt：

- `formula_variant`
- `agent_code`

`agent_code` 有三种来源：

- 直接给 `code`
- 给现成 `code_path`
- 只给 `objective`，由本地 Agent 服务生成候选代码

每个 attempt 最终会产出：

- `candidate_loss.py`
- `result.json`
- 可选的 repair artifacts

## 验证语义

外部 CLI 仍然是四个模式：

- `static`
- `smoke`
- `single`
- `full`

但当前真实执行链已经扩展成：

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
- `formula_alignment`: 公式与代码的一致性校验
- `layer3`: 单模型训练
- `layer4`: 四模型 full run

repair 逻辑：

- 最多 3 轮
- 可修复 stop layer: `layer1 / layer2 / formula_alignment / layer3 / layer4`
- 如果修复后从更高层退化到更低层，会自动回滚

## Integration Path 语义

integration path 会决定 Agent 允许修改哪些文件：

- `loss_only`: 只改 `candidate_loss.py`
- `adapter_wrapper`: 可改 attempt-scoped `sandbox_model_adapter.py` / `sandbox_trainer.py`
- `extend_model_outputs`: 可改 adapter 和 copied `models/`
- `model_surgery`: 允许对 copied `models/` 做更深修改

这会同时影响：

- `analysis_plan.json` 的 `integration_decision.path`
- 自动补全的 `files_to_edit`
- 自动补全的 `required_edit_paths`
- agent codegen / repair 时的白名单

## 目录结构

```text
scripts/ocean-loss-transfer/
├── loss_transfer/
│   ├── common/
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

workflow/loss_transfer/
├── blocked_patterns.yaml
├── target_interface_spec.yaml
├── baseline_thresholds.yaml
├── knowledge_base/
└── README.md
```

## Optional / Legacy

以下模块依然存在，但不再是默认 agentic 主链：

- `loss_transfer/ir/extract_loss_ir.py`
- `loss_transfer/ir/write_loss_ir.py`
- `loss_transfer/ir/check_compatibility.py`
- `loss_transfer/generation/generate_patch.py`

现在更推荐这样理解：

- `loss_formula.json` 是主流程真值源
- `loss_ir.yaml` 是补充参考材料

## 参考

- 详细版本说明：`scripts/ocean-loss-transfer/README.md`
- 目标接口规范：`workflow/loss_transfer/target_interface_spec.yaml`
- 已知失败模式：`workflow/loss_transfer/blocked_patterns.yaml`
