# analysis_plan.json 指南

本文档说明如何编写 `analysis_plan.json`。当 Agent 已经拿到论文、代码和 `loss_formula.json`，准备决定怎么改时再读。

---

## 最小结构

```json
{
  "summary": "一句话总结计划",
  "stop_on_first_pass": false,
  "integration_decision": {
    "path": "adapter_wrapper",
    "rationale": "论文 loss 依赖 model.forward 产生的 aux tensors",
    "evidence_refs": ["paper.loss", "code.model_forward"]
  },
  "attempts": []
}
```

---

## integration_decision

必须包含：

- `path`
- `rationale`
- `evidence_refs`

`evidence_refs` 应尽量引用明确来源，例如：

- `paper.loss`
- `paper.implementation_details`
- `code.loss_callsite`
- `code.model_forward`
- `code.adapter_wrapper`

---

## attempts 设计

每个 attempt 至少应写清楚：

- `name`
- `kind`
- `objective`
- `evidence_refs`

如果已经明确知道需要改哪些 attempt 文件，再补：

- `files_to_edit`
- `required_edit_paths`

---

## 推荐 attempt 模式

### 1. faithful

- 目标：先忠实迁移论文 loss
- 适合：你对 path 判断已经很确定，但还没做数值稳健增强

### 2. stabilized

- 目标：在 faithful 基础上补 epsilon、clamp、dtype guard 等稳定性修复
- 适合：论文公式可能数值敏感，或者历史经验提示高风险

### 3. path-corrective

- 目标：当上一轮 stop_layer 暴露 path 选错时，修正接入层次
- 适合：loss 本身没错，但缺少 loss inputs / adapter outputs / model outputs

---

## 不要这样写

- 只写“实现论文 loss”，没有说明为什么选这条 integration path
- `evidence_refs` 为空
- 明明需要 adapter/model 改动，却只允许编辑 `candidate_loss.py`
- 把 repo-root 文件当作直接编辑目标
