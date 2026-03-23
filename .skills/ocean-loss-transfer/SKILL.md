---
name: ocean-loss-transfer
description: 论文 Loss 函数自动迁移 - Agent 分析代码 + 4层验证 + 5-trial 搜索
version: 2.0.0
author: Leizheng
last_modified: 2026-03-23
---

<!--
Changelog:
  - 2026-03-23 Leizheng: v2.0.0 Agent-Native 提取
    - 移除外部 LLM API 依赖
    - Agent 直接分析代码生成 Loss IR
    - 新增 prepare_context 和 write_ir 工具
    - 文档重构：精简 SKILL.md + references/
  - 2026-03-22 Leizheng: v1.0.0 初始版本
-->

# Loss Transfer 技能

## 核心原则

1. **Agent 直接分析**: 不依赖外部 API，Agent 自己读代码生成 Loss IR
2. **渐进式验证**: 4层验证从轻到重，尽早淘汰坏 patch
3. **已知失败拦截**: 基于 71 次实验，自动拦截 SSIM/Laplacian 等
4. **自动化**: 实验完成后自动 git push

---

## 工作流程

```
用户提供代码仓库
         ↓
[准备上下文] ocean_loss_transfer_prepare_context
         ↓
Agent 分析代码
         ↓
Agent 生成 Loss IR YAML
         ↓
[写入验证] ocean_loss_transfer_write_ir
         ↓
[兼容性检查] ocean_loss_transfer_check_compat
         ↓
[5-Trial 搜索] ocean_loss_transfer_orchestrate
         ↓
每个 trial 通过 4 层验证
         ↓
[实验记录] + [Git Push]
```

---

## 可用工具

| 工具 | 用途 | 使用时机 |
|------|------|----------|
| `prepare_context` | 扫描代码，准备分析材料 | 开始提取 Loss IR |
| `write_ir` | 验证并写入 Loss IR | Agent 生成 YAML 后 |
| `check_compat` | 检查兼容性 | Loss IR 写入后 |
| `validate` | 4层渐进式验证 | 测试单个 loss 文件 |
| `orchestrate` | 5-trial 自动实验 | 开始完整实验 |

---

## 典型流程

### 场景 1: 完整自动化实验

```typescript
// Step 1: 准备上下文
const context = await ocean_loss_transfer_prepare_context({
  code_repo_path: "/path/to/paper/code",
  paper_slug: "paper_name"
})

// Step 2: Agent 分析代码并生成 Loss IR YAML
// （参考 context.analysis_guide）

// Step 3: 写入并验证
const result = await ocean_loss_transfer_write_ir({
  yaml_content: "...",  // Agent 生成的 YAML
  output_path: context.output_path
})

// Step 4: 开始实验
await ocean_loss_transfer_orchestrate({
  loss_ir_yaml: context.output_path,
  paper_slug: "paper_name"
})
```

---

## 4 层验证

| 层级 | 时间 | 检查内容 |
|------|------|----------|
| Layer 1 - Static | <1s | AST + 签名 + import 白名单 |
| Layer 2 - Smoke | <10s | dummy forward/backward + NaN 检查 |
| Layer 3 - Single | ~2min | SwinIR 训练 + SSIM > 0.3 |
| Layer 4 - Full | ~5min | 4 模型并行 + 基线对比 |

---

## 禁止行为

| 类别 | 禁止行为 |
|------|----------|
| **Loss IR 生成** | 跳过 prepare_context，直接猜测代码结构 |
| **验证** | 跳过 write_ir 验证，直接写入文件 |
| **实验** | 手动修改 sandbox_loss.py，绕过工具 |

---

## 参考文档索引

详细信息请按需读取以下文档：

| 文档 | 内容 | 何时读取 |
|------|------|----------|
| `references/extraction-guide.md` | Agent 如何分析代码 | 生成 Loss IR 时 |
| `references/loss-ir-schema.md` | Loss IR 完整 schema | 填写 YAML 时 |
| `references/known-failures.md` | 71 次实验失败模式 | 遇到失败时 |
| `references/validation-layers.md` | 4 层验证详解 | 验证失败时 |
| `references/trial-strategies.md` | 5-trial 策略说明 | 理解实验流程时 |
| `references/troubleshooting.md` | 故障排除 | 遇到错误时 |

---

## 性能基线

当前最优 (exp#41):
- **SwinIR**: 0.6645（目标指标）
- **验收标准**: 新 loss 的 SwinIR SSIM >= 0.6545（baseline - 1σ）
