# 知识积累系统

当前仓库已经完全移除早期那套独立的 `innovations.yaml + index.json + fusion_designer.py + retrieval_engine.py + knowledge_db.py` 知识库。

主流程现在统一收敛到一套共享经验后端：

- 主事实源: `workflow/loss_transfer/knowledge_base/case_memories.jsonl`
- 过程轨迹: `sandbox/loss_transfer_experiments/<paper_slug>/decision_trace.jsonl`
- RL 数据导出: `sandbox/loss_transfer_experiments/<paper_slug>/rl_decision_dataset.jsonl`

## 当前架构

```text
attempt 执行结果
  ↓
decision_trace.py
  ├─ 写 decision_trace.jsonl
  ├─ 写 rl_decision_dataset.jsonl
  └─ 合并到 case_memories.jsonl
  ↓
memory/case_memory_store.py
  ├─ 统一 schema
  ├─ 统一去重/合并
  └─ 提供兼容视图
  ↓
memory/case_memory_retriever.py
  ├─ 相似案例检索
  ├─ prompt memory block 格式化
  └─ 给 follow-up / repair 使用
  ↓
agent_artifact_generator.py
  ├─ generate_followup_attempt()
  └─ repair_candidate_loss()
```

## 当前模块分工

### `loss_transfer/common/decision_trace.py`

负责把执行过的 `attempt` 转成三类工件：

- `decision_trace.jsonl`
- `rl_decision_dataset.jsonl`
- `case_memories.jsonl`

其中 `case_memories.jsonl` 是唯一共享经验库。

### `loss_transfer/memory/case_memory_store.py`

负责共享经验库的底层读写：

- 统一 `case_memory.v1` / `decision_trace.v1` 的兼容读取
- 统一去重 key
- 合并写回 `case_memories.jsonl`
- 为旧 `KnowledgeDB` 提供兼容的 `Innovation` 视图

### `loss_transfer/memory/case_memory_retriever.py`

负责经验检索和 prompt 回灌：

- 基于 `integration_path`、`kind`、`stop_layer`、关键词重叠做启发式打分
- 返回 top-k 相似历史案例
- 生成可直接拼到 prompt 的 memory block

## 已删除的旧模块

以下模块已经从仓库中移除，因为它们不再接入当前主流程，且会造成“双后端”误解：

- `loss_transfer/ir/innovation_extractor.py`
- `loss_transfer/ir/fusion_designer.py`
- `loss_transfer/ir/knowledge_db.py`
- `loss_transfer/ir/retrieval_engine.py`

删除原因：

- 它们围绕旧的独立知识库概念设计
- 当前主流程并不调用
- 保留会让“case memory 主链”和“旧 innovation 知识库旁路”同时存在，增加维护成本

## 仍然保留的 `knowledge_base/modules/`

`knowledge_base/modules/` 和 `code_generalizer.py` 仍然保留。

原因不是它们还代表独立知识库存储，而是：

- `code_generalizer.py` 仍然依赖 `knowledge_base/modules/` 作为代码泛化输出目录
- 这条线和 `case_memories.jsonl` 不冲突

也就是说：

- 结构化经验后端: `case_memories.jsonl`
- 泛化代码产物目录: `knowledge_base/modules/`

这两者现在是并列关系，不再是早期文档里的“innovations.yaml + index.json + modules/”三件套。

## 使用建议

如果要继续扩展经验系统，优先改这些位置：

- `loss_transfer/memory/case_memory_store.py`
- `loss_transfer/memory/case_memory_retriever.py`
- `loss_transfer/common/decision_trace.py`

不要再新增基于 `innovations.yaml` 或 `index.json` 的独立逻辑。
