---
name: ocean-loss-transfer
description: 论文 Loss 自动迁移闭环 - 论文/代码联合分析、公式提取、analysis_plan、attempt 沙箱验证与修复
version: 3.1.0
author: Leizheng
contributors: Leizheng
last_modified: 2026-03-26
---

# Loss Transfer 技能

## 核心原则

1. **主证据源**：论文文本 + 代码上下文 + `loss_formula.json`，不要只靠单一公式或单一脚本猜测。
2. **先判断接入路径，再写代码**：优先决定 `loss_only` / `adapter_wrapper` / `extend_model_outputs` / `model_surgery`。
3. **模型级修改只在 attempt 沙箱副本里发生**：不要直接改 repo-root 训练或模型源码。
4. **Loss IR 只是可选参考**：主流程是 `loss_formula.json + analysis_plan.json + orchestrate`，不是先写 IR。

---

## 主工具

| 工具 | 用途 | 使用时机 |
|------|------|----------|
| `ocean_loss_transfer_prepare_context` | 准备论文/代码上下文和输出路径 | 开始分析时 |
| `ocean_loss_transfer_extract_formula` | 起草 `loss_formula.json` | 需要快速抽公式时 |
| `ocean_loss_transfer_write_formula` | 校验并写入公式 spec | 公式确认后 |
| `ocean_loss_transfer_generate_plan` | 根据 `task_context.json` 生成 `analysis_plan.json` 初稿 | 公式确认后、正式执行前 |
| `ocean_loss_transfer_orchestrate` | 主入口：执行 task context + analysis plan + attempt repair 闭环 | 公式和计划就绪后 |
| `ocean_loss_transfer_validate` | 单独验证某个 candidate/attempt | 调试 patch 时 |
| `ocean_loss_transfer_submit_code` | 手工提交一版候选 loss 做快速验证 | 人工探针或对照实验时 |

可选旧路径工具：
- `ocean_loss_transfer_write_ir`
- `ocean_loss_transfer_check_compat`
- `ocean_loss_transfer_extract`

---

## 默认工作流

```text
1. prepare_context
2. extract_formula / write_formula
3. generate_plan
4. orchestrate
5. 失败后按 stop_layer 修复
6. 查看 trajectory.jsonl 和 agent_loop_summary.json
```

默认只做这条主流程。只有在你明确需要补充结构化参考时，再去读 Loss IR 相关文档或工具。

---

## 何时读参考文档

| 文档 | 内容 | 何时读取 |
|------|------|----------|
| `references/workflow-detail.md` | 完整闭环和关键产物 | 需要看全流程时 |
| `references/extraction-guide.md` | 如何从论文+代码提公式并收集证据 | 写 `loss_formula.json` 时 |
| `references/analysis-plan.md` | `analysis_plan.json` 结构与样例 | 写计划时 |
| `references/integration-paths.md` | 4 种 integration path 的判断规则 | 不确定该改哪一层时 |
| `references/validation-layers.md` | Layer 1-4 验证解释 | 看 stop_layer 时 |
| `references/known-failures.md` | 已知不稳定 loss 模式 | 做数值稳定性判断时 |
| `references/trial-strategies.md` | attempt 设计与 repair 试探策略 | 设计多轮尝试时 |
| `references/troubleshooting.md` | 常见故障排查 | 出现异常时 |
| `references/loss-ir-schema.md` | Loss IR schema | 只有在明确要写 IR 时 |

---

## 禁止行为

| 类别 | 禁止行为 |
|------|----------|
| **路径判断** | 看到额外张量需求仍强行走 `loss_only` |
| **代码修改** | 直接改 repo-root 的训练或模型源码 |
| **流程执行** | 跳过公式校验，直接写候选代码 |
| **环境** | 在 skill 指令里硬编码 Python 路径 |
