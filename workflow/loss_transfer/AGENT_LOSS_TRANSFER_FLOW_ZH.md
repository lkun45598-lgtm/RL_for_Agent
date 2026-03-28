# Agent Loss Transfer 详细流程

这份文档说明当前仓库里的自动化 loss transfer 闭环，目标是把“给 Agent 一篇论文和对应代码，然后让 Agent 自动提取 loss、迁移到现有训练体系里、验证、修复、再迭代”的真实执行流程讲清楚。

## 1. 总体目标

当前系统不是“给大模型一个 prompt，让它直接改仓库”，而是一条带中间产物、带验证层、带回滚和重规划的受控闭环：

```text
论文 PDF + 对应代码仓库
  -> 结构化上下文与公式草稿
  -> Agent 生成 analysis_plan
  -> 在 attempt 沙箱中生成候选 loss / 模型适配修改
  -> 接入现有训练框架做分层验证
  -> 失败后自动 repair
  -> repair 不够时自动生成 follow-up attempt
  -> 沉淀 decision trace / RL dataset
```

单篇论文入口在 `scripts/ocean-loss-transfer/run_auto_experiment.py`，真实实现位于 `scripts/ocean-loss-transfer/loss_transfer/orchestration/run_auto_experiment.py`。

## 2. 第一阶段：构建任务上下文

系统首先执行 `build_task_context(...)`，实现位于 `loss_transfer/context/context_builder.py`。

这一阶段会做四件事：

1. 扫描代码仓库中的 loss 相关文件  
   由 `loss_transfer/context/prepare_context.py` 完成，优先寻找文件名和内容里带 `loss / criterion / objective` 的文件，也会关注把 loss 写在 `model.forward` 里的实现。

2. 提取论文文本  
   如果传入 PDF，会提取 abstract、sections、loss snippets，形成 `paper_analysis`。

3. 自动起草公式文件  
   由 `loss_transfer/formula/extract_loss_formula.py` 生成 `loss_formula.json` 和 `loss_spec.yaml`。  
   这里会从论文和代码中抽取：
   - latex 草稿
   - 参数
   - `symbol_map`

4. 分析公式接口  
   由 `formula_interface_analysis` 判断这篇论文的 loss 是不是只依赖 `pred/target/mask`，还是还依赖模型输出的辅助量，比如 `weight / log_b / sigma`。

这一阶段结束后，实验目录下会有几个关键文件：

- `task_context.json`
- `loss_formula.json`
- `loss_spec.yaml`

其中 `task_context.json` 是整个 Agent 主流程的分析入口，里面包含：

- `paper_analysis`
- `code_analysis`
- `formula_spec`
- `integration_assessment`
- `analysis_plan_schema`
- `agent_guidance`
- `paths`

## 3. 第二阶段：Agent 先写分析计划，不直接写代码

如果开启 `--auto_generate_plan`，系统会调用 `generate_analysis_plan(...)`，实现位于 `loss_transfer/agent/agent_artifact_generator.py`。

这里 Agent 先读取 `task_context.json`，再输出 `analysis_plan.json`。  
这个计划必须回答：

- 这篇论文应该走哪条迁移路径
- 为什么这么判断
- 依据来自论文哪里、代码哪里
- 准备先试哪些 attempt
- 每个 attempt 预计改哪些文件

`analysis_plan.json` 的核心结构包括：

- `summary`
- `integration_decision`
- `attempts[]`

其中 `integration_decision.path` 只能是四类之一：

- `loss_only`
- `adapter_wrapper`
- `extend_model_outputs`
- `model_surgery`

## 4. 第三阶段：决定是只改 loss，还是连模型输出链一起改

这一步的策略逻辑在 `loss_transfer/attempts/integration_policy.py`。

不同路径含义如下：

- `loss_only`  
  只改 `candidate_loss.py`

- `adapter_wrapper`  
  允许同时修改 attempt 私有的 `sandbox_model_adapter.py`、`sandbox_trainer.py`

- `extend_model_outputs`  
  允许在 attempt 私有目录里修改复制出来的 `models/`

- `model_surgery`  
  允许做更深的模型级改动，但仍然是在 attempt 私有副本里完成

这一步很关键，因为它把“最小改动优先”和“必要时允许更深模型改动”统一到了一个受控策略里。

## 5. 第四阶段：在 attempt 沙箱里生成候选实现

系统不会让 Agent 直接乱改仓库，而是先由 `loss_transfer/agent/agent_edit_workspace.py` 为每个 attempt 创建独立工作区。

这个工作区会生成：

- `candidate_loss.py`
- `editable_files.json`
- 可选的 `sandbox_overrides/`
- 可选的复制版 `models/`

其中 `editable_files.json` 是白名单，明确限制：

- 这次可以改哪些文件
- 哪些路径必须真的被改到
- 当前推荐的 integration path 是什么

也就是说，Agent 的模型层改动能力是有的，但它改的是 attempt 私有副本，不直接动原始模板库。

随后 `generate_candidate_loss(...)` 会让 Agent 在这个受控白名单内生成当前候选实现。

## 6. 第五阶段：把候选实现接到现有训练体系里验证

单个 attempt 的执行入口是 `loss_transfer/attempts/attempt_executor.py`。

它会为每个 attempt 产出：

- `attempt_{N}/candidate_loss.py`
- `attempt_{N}/result.json`
- 可选的 repair 相关文件

然后执行 4 层渐进式验证，具体在 `loss_transfer/validation/validate_loss.py`：

1. `validate_static`  
   静态检查，过滤语法、未定义变量、作用域等问题

2. `validate_smoke`  
   小规模 smoke test，检查 forward/backward、shape、梯度和边界情况

3. `validate_single_model`  
   单模型短训练，看最基本训练链是否可跑

4. `validate_full_run`  
   完整验证，看 SSIM / PSNR 等指标是否达到 baseline 可行阈值

这意味着当前流程不是“直接 full train”，而是先低成本筛掉明显错误，再进入更贵的训练验证。

## 7. 第六阶段：失败后先写 repair plan，再修代码

如果某个 attempt 在 `layer1 / layer2 / formula_alignment / layer3 / layer4` 中失败，系统会整理失败反馈，然后触发 repair。

repair 不是简单“再试一次”，而是要求 Agent 先写 `repair_plan.json`，再改代码。

`repair_plan.json` 至少包含：

- `failure_hypothesis`
- `planned_changes`
- `target_metric`
- `success_criteria`
- `fallback_plan`
- `evidence_refs`

这一步的意义是把 Agent 的修复推理显式结构化下来，后面既方便排错，也方便给 RL/controller 学习。

## 8. 第七阶段：repair 后重新验证，变差就回滚

repair 完成后，系统会重新跑验证。

如果 repair 把情况修得更糟，比如从 `layer3` 退化到 `layer2`，就会自动：

- 恢复修复前的代码
- 标记这轮为 `reverted_regression`

所以这里不是“Agent 改了就算”，而是“改完还要再过验证，变差就撤回”。

## 9. 第八阶段：当前思路不行时，自动生成下一条 attempt

如果当前 attempt 修不动，而且还没有达到最大尝试数，`loss_transfer/agent/agent_repair_loop.py` 会触发 `generate_followup_attempt(...)`。

这一步会读取：

- `task_context.json`
- `analysis_plan.json`
- 最新 `result.json`
- 最新 `repair_plan.json`
- `trajectory.jsonl`

然后让 Agent 生成新的 follow-up attempt。

新 attempt 必须带 `strategy_delta`，明确说明：

- 上一轮为什么失败
- 这轮改了什么
- 为什么不重复旧策略
- 预期会出现什么验证信号

这就是当前系统里“Agent 具有自我分析能力”的主要体现。

## 10. 第九阶段：把整个执行过程沉淀成结构化产物

每次实验目录里，当前最重要的产物包括：

- `task_context.json`
- `loss_formula.json`
- `loss_spec.yaml`
- `analysis_plan.json`
- `trajectory.jsonl`
- `attempt_{N}/candidate_loss.py`
- `attempt_{N}/result.json`
- `attempt_{N}/repair_plan_round_{K}.json`
- `decision_trace.jsonl`
- `rl_decision_dataset.jsonl`

这些文件不只是日志，而是后续继续重规划、离线分析、训练 controller / RL policy 的数据基础。

## 11. 第十阶段：批量 benchmark 模式

如果不是单篇论文，而是多篇论文批量跑，入口在 `scripts/ocean-loss-transfer/run_benchmark_batch.py`。

批量流程是：

1. 用 `build_benchmark_catalog.py` 扫 benchmark 根目录
2. 用 `materialize_benchmark_entry.py` 把 zip / repo 解成 ready-to-run 代码包
3. 对每个 entry 分别构建 `task_context`
4. 可选自动生成 `analysis_plan`
5. 跑 `run_agent_repair_loop`
6. 汇总每篇论文的结果
7. 聚合 batch 级别的：
   - `decision_trace.jsonl`
   - `rl_decision_dataset.jsonl`

所以 batch 只是把“单篇自动闭环”包了一层 catalog / materialize / aggregate。

## 12. 现在这套流程的核心边界

你最关心的一点是：不要直接污染原始模板代码。

当前系统的边界就是：

- 原始仓库是模板经验库
- Agent 的真实改动尽量落在 attempt 私有目录
- 包括：
  - `candidate_loss.py`
  - `sandbox_overrides/`
  - 复制出来的 `models/`

验证器会优先加载这些 attempt-scoped 文件，而不是直接改 repo-root 的训练与数据处理代码。

## 13. 一句话总结

当前真实流程可以浓缩成一句话：

> 先从论文和代码中抽取结构化证据，再让 Agent 先规划、后实现，并把候选 loss 接到现有训练框架里做分层验证；失败后先生成 repair plan 再修复，不行就自动重规划下一条 attempt，最终把整个闭环沉淀成可继续用于 RL/controller 学习的数据。
