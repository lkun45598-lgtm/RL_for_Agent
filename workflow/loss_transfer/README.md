# Loss Transfer System

自动化论文 Loss 函数迁移系统

## 概述

Loss Transfer System 是一个完全自动化的工具链,用于将研究论文中的 loss 函数迁移到 `sandbox/sandbox_loss.py`。系统通过 LLM 分析代码、4层渐进式验证、5-trial 结构化搜索,自动找到最优的 loss 配置。

## 核心特性

- **LLM 自动提取**: 分析论文代码,提取 loss 结构
- **4层渐进式验证**: Static → Smoke → Single Model → Full Run
- **已知失败拦截**: 基于 71 次实验的失败模式库
- **5-trial 搜索**: 系统化探索 loss 变体
- **自动 Git 推送**: 成功实验自动提交到 GitHub

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
输入: 论文代码仓库
  ↓
extract_loss_ir.py (LLM 分析)
  ↓
Loss IR YAML
  ↓
check_compatibility.py (兼容性检查)
  ↓
orchestrate_trials.py (5-trial 搜索)
  ├─ Trial 1-5: generate_patch.py
  ├─ 每个 trial: validate_loss.py (4层)
  └─ experiment_recorder.py
  ↓
输出: summary.yaml + 自动 git push
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

### 3. 5-Trial 搜索策略

1. **Faithful Core**: 忠实移植论文组件
2. **Normalization Aligned**: 对齐 normalization
3. **Weight Aligned**: 使用论文权重
4. **Numerical Stabilized**: 数值稳定技巧
5. **Fallback Hybrid**: 混合最优组件

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
├── generate_patch.py           # Patch 生成
├── patch_templates.py          # 模板库
├── validate_loss.py            # 4层验证
├── run_trial.py                # 单次 trial
├── orchestrate_trials.py       # 5-trial 编排
├── experiment_recorder.py      # 实验记录
└── run_auto_experiment.py      # 端到端脚本

workflow/loss_transfer/
├── target_interface_spec.yaml  # 接口规格
├── patch_type_registry.yaml    # Patch 类型
├── blocked_patterns.yaml       # 失败模式
└── README.md                   # 本文档

sandbox/loss_transfer_experiments/
└── {paper_slug}/
    ├── loss_ir.yaml
    ├── summary.yaml
    └── trial_1/
        ├── sandbox_loss.py
        └── result.yaml
```

## 完整示例

### 示例: SEA-RAFT 论文

```bash
# 运行实验
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug sea_raft \
  --code_repo Benchmark/SEA-RAFT-main

# 查看结果
cat sandbox/loss_transfer_experiments/sea_raft/summary.yaml

# 查看最佳 trial
cat sandbox/loss_transfer_experiments/sea_raft/trial_X/sandbox_loss.py
```

### 输出示例

```yaml
paper_slug: sea_raft
baseline:
  ssim_mean: 0.6645
trials:
  - trial_id: 1
    name: Faithful Core
    passed: true
    metrics:
      swinir: 0.6680
best_trial: 1
best_ssim: 0.6680
improvement: 0.0035
```

## 故障排除

### 常见问题

**Q: LLM 提取失败?**
- 检查 `.env` 中的 `ANTHROPIC_API_KEY` 和 `ANTHROPIC_BASE_URL`
- 使用手动模式: `--manual_mode` 生成模板手动填写

**Q: Layer 1 验证失败?**
- 检查是否使用了禁止的 import
- 查看 `workflow/loss_transfer/blocked_patterns.yaml`

**Q: 所有 trial 都失败?**
- 检查 Loss IR 是否正确填写
- 查看 `trial_1/result.yaml` 了解具体失败原因

**Q: Git push 失败?**
- 检查 git 配置和权限
- 手动推送: `git push origin branch_name`

## 注意事项

1. **实验时间**: 完整 5-trial 需要 10-30 分钟
2. **GPU 占用**: Layer 3-4 会占用 GPU 4-7
3. **自动推送**: 成功实验会自动 commit & push
4. **Loss IR 质量**: 手动填写比 LLM 自动提取更准确

## 版本历史

- v1.0.0 (2026-03-22): 初始版本,完整功能实现
