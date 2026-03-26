# Integration Path 选择指南

本文档帮助 Agent 决定该走哪一条接入路径。只有在你不确定需要改 loss、adapter 还是 model 时再读。

---

## 1. loss_only

适用条件：

- 论文 loss 只依赖 `pred / target / mask / params`
- 不需要 model.forward 额外输出
- 不需要中间特征或 auxiliary tensors

典型编辑面：

- `candidate_loss.py`

---

## 2. adapter_wrapper

适用条件：

- loss 需要额外 loss inputs
- 这些输入可以通过 adapter 包装现有模型输出得到
- 不需要改原模型结构

典型编辑面：

- `candidate_loss.py`
- `sandbox_overrides/sandbox_model_adapter.py`
- `sandbox_overrides/sandbox_trainer.py`

---

## 3. extend_model_outputs

适用条件：

- 需要 model.forward 返回更多字段
- 但改动仍可以局限在 attempt-scoped copied model tree 中

典型编辑面：

- `candidate_loss.py`
- `sandbox_overrides/`
- `attempt_*/models/`

---

## 4. model_surgery

适用条件：

- 论文 loss 深度依赖模型内部结构
- 不是简单加一个 adapter 就能解决
- 必须修改复制出的模型实现

典型编辑面：

- `candidate_loss.py`
- `attempt_*/models/`
- 必要时配套 trainer override

---

## 快速判断规则

- 只要论文 loss 依赖 `pred` 之外的中间量，就优先排除 `loss_only`
- 如果额外量可以由包装层补出来，优先 `adapter_wrapper`
- 如果必须让模型本身多返回内容，考虑 `extend_model_outputs`
- 如果论文方法和模型结构强耦合，才走 `model_surgery`

---

## 常见误判

- 看到公式里只有一个总损失表达式，就误以为一定能 `loss_only`
- 论文把 feature/uncertainty 写在方法章节而不是公式附近，结果漏读
- 代码里 loss 调用点简单，但真实的辅助量是在 `model.forward` 里计算
