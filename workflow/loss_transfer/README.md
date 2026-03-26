# Loss Transfer System

自动化论文 Loss 函数迁移系统

## 概述

Loss Transfer System 是一个面向 Agent 的自动化工具链,用于将研究论文中的 loss 函数迁移到 `sandbox/sandbox_loss.py`。系统通过公式抽取、task context、analysis plan、4层渐进式验证和结构化轨迹记录,支持 Agent 自主分析、修改和迭代。

## 核心特性

- **LLM 自动提取**: 分析论文代码,提取 loss 结构
- **4层渐进式验证**: Static → Smoke → Single Model → Full Run
- **已知失败拦截**: 基于 71 次实验的失败模式库
- **Task Context / Analysis Plan**: 把“提取事实”和“决策修改”明确拆开
- **轨迹记录**: 为后续在线 RL 优化保留状态、动作、结果

## 快速开始

### 基本用法

```bash
cd /data1/user/lz/RL_for_Agent

# 自动实验
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug my_paper \
  --code_repo path/to/paper/code
```

### 手动填写 Loss IR

```bash
# 1. 生成模板
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug my_paper

# 2. 编辑 YAML
vim sandbox/loss_transfer_experiments/my_paper/loss_ir.yaml

# 3. 运行实验
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug my_paper \
  --loss_ir_yaml sandbox/loss_transfer_experiments/my_paper/loss_ir.yaml
```

## 系统架构

```
输入: 论文 PDF + 论文代码 + 目标仓库
  ↓
prepare_context.py / extract_loss_formula.py
  ↓
loss_formula.json + loss_spec.yaml + loss_ir.yaml
  ↓
context_builder.py
  ↓
task_context.json
  ↓
analysis_plan.json (由 Agent 生成)
  ↓
agent_repair_loop.py
  ├─ attempt_executor.py (执行单个候选方案)
  ├─ validate_loss.py (4层验证)
  └─ trajectory_logger.py (task/attempt 轨迹)
  ↓
输出: agent_loop_summary.json + trajectory.jsonl
```

## 核心模块

### 1. Loss IR (中间表示)

结构化描述 loss 函数的 YAML 格式:

```yaml
components:
  - name: "pixel_loss"
    type: "pixel_loss"
    weight: 1.0
    implementation:
      reduction: "mean"
      operates_on: "pixel_space"

multi_scale:
  enabled: true
  scales: [1, 2, 4]
```

### 2. 4层渐进式验证

| 层级 | 时间 | 检查内容 |
|------|------|----------|
| Layer 1: Static | <1s | AST解析、import白名单、已知失败模式 |
| Layer 2: Smoke | <10s | 动态导入、forward/backward、NaN检查 |
| Layer 3: Single | ~2min | SwinIR 完整训练15 epochs |
| Layer 4: Full | ~5min | 4模型并行训练 |

### 3. Agentic 闭环

1. **Task Context**: 汇总公式、参数、symbol_map、代码上下文、兼容性
2. **Analysis Plan**: Agent 决定要改 loss 参数化，还是要改模型/adapter 接口
3. **Attempt Execution**: 每个候选方案单独验证和训练
4. **Trajectory Logging**: 全部结果结构化写入，供后续 RL 使用

### 4. 已知失败模式 (71次实验)

- SSIM loss → 崩溃 (exp#11)
- Laplacian → 崩溃 (exp#20, #40, #66)
- 5x5 Sobel → 性能下降 (exp#38)
- Scale=8 → 太粗糙 (exp#37)

## 文件结构

```
scripts/ocean-loss-transfer/
├── extract_loss_ir.py          # LLM 提取
├── llm_extractor.py            # LLM API 调用
├── check_compatibility.py      # 兼容性检查
├── context_builder.py          # task_context.json 构建
├── attempt_executor.py         # 单个候选方案执行
├── agent_repair_loop.py        # analysis_plan 驱动的闭环
├── validate_loss.py            # 4层验证
├── trajectory_logger.py        # 轨迹记录
└── run_auto_experiment.py      # 端到端脚本

workflow/loss_transfer/
├── target_interface_spec.yaml  # 接口规格
├── patch_type_registry.yaml    # Patch 类型
├── blocked_patterns.yaml       # 失败模式
└── README.md                   # 本文档

sandbox/loss_transfer_experiments/
└── {paper_slug}/
    ├── task_context.json
    ├── analysis_plan.json
    ├── loss_ir.yaml
    ├── loss_formula.json
    ├── agent_loop_summary.json
    ├── trajectory.jsonl
    └── attempt_1/
        ├── candidate_loss.py
        └── result.json
```

## 完整示例

### 示例: SEA-RAFT 论文

```bash
# 运行实验
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug sea_raft \
  --code_repo Benchmark/SEA-RAFT-main

# 查看闭环汇总
cat sandbox/loss_transfer_experiments/sea_raft/agent_loop_summary.json

# 查看某次候选代码
cat sandbox/loss_transfer_experiments/sea_raft/attempt_1/candidate_loss.py
```

### 输出示例

```json
{
  "paper_slug": "sea_raft",
  "status": "completed",
  "best_attempt_id": 2,
  "best_metric_name": "swinir",
  "best_metric_value": 0.6680
}
```

## 故障排除

### 常见问题

**Q: LLM 提取失败?**
- 检查 `.env` 中的 `LLM_PROVIDER / LLM_API_KEY / LLM_BASE_URL`
- 如果不用通用变量,则检查对应 provider 的 `OPENAI_*` 或 `ANTHROPIC_*`
- 使用手动模式: `--manual_mode` 生成模板手动填写

**Q: Layer 1 验证失败?**
- 检查是否使用了禁止的 import
- 查看 `workflow/loss_transfer/blocked_patterns.yaml`

**Q: 所有 attempt 都失败?**
- 检查 Loss IR 是否正确填写
- 查看 `attempt_1/result.json` 了解具体失败原因

**Q: Git push 失败?**
- 检查 git 配置和权限
- 手动推送: `git push origin branch_name`

## 注意事项

1. **实验时间**: 完整闭环耗时取决于 attempt 数和是否跑 Layer 4
2. **GPU 占用**: Layer 3-4 会占用 GPU 4-7
3. **轨迹文件**: `task_context.json / analysis_plan.json / trajectory.jsonl` 都应保留
4. **Loss IR 质量**: 手动复核 `loss_formula.json` 和 `symbol_map` 很重要

## 版本历史

- v1.0.0 (2026-03-22): 初始版本,完整功能实现
