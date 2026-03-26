# Loss Transfer 工作流细节

本文档展开 `ocean-loss-transfer` 的主闭环。只有在你需要看完整产物流向或排查流程断点时再读它。

---

## 输入

- 论文 PDF
- 对应代码仓库
- 可选：已有 `loss_ir.yaml`
- 可选：训练数据根目录

---

## 主闭环

```text
1. prepare_context
   - 输出 paper snippets、primary_files、formula_output_path、analysis_plan_output_path

2. extract_formula / write_formula
   - 产出并校验 loss_formula.json

3. Agent 编写 analysis_plan.json
   - 选择 integration path
   - 设计 attempts

4. orchestrate
   - 构建 task_context.json
   - 执行 attempts
   - 每轮根据 stop_layer 做 repair

5. 查看结果
   - trajectory.jsonl
   - attempt_*/result.json
   - agent_loop_summary.json
```

---

## 关键产物

- `task_context.json`
- `loss_formula.json`
- `analysis_plan.json`
- `attempt_*/candidate_loss.py`
- `attempt_*/sandbox_overrides/`
- `attempt_*/models/`
- `trajectory.jsonl`
- `agent_loop_summary.json`

---

## 什么时候需要额外读取参考文档

- 不确定公式怎么落地：读 `extraction-guide.md`
- 不确定该选哪条接入路径：读 `integration-paths.md`
- 不知道计划该怎么写：读 `analysis-plan.md`
- 不知道 stop_layer 怎么修：读 `validation-layers.md` 和 `troubleshooting.md`

---

## 工作流中的硬约束

- `loss_formula.json` 的 `symbol_map` 必须是一一映射。
- 论文 loss 依赖 model.forward 额外输出时，不能把问题强行压成纯 loss 文件改动。
- 模型级修改只能发生在 attempt-scoped 副本中。
- `loss_ir.yaml` 不是必需产物；只有在需要额外结构化总结时才写。
