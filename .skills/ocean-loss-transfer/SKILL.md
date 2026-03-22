---
name: ocean-loss-transfer
description: 论文 Loss 函数自动迁移技能 - 支持 LLM 提取、4层验证、5-trial 搜索、知识积累（创新点提取+代码泛化+检索融合）
version: 1.0.0
author: Leizheng
last_modified: 2026-03-22
---

<!--
Changelog:
  - 2026-03-22 Leizheng: v1.0.0 初始版本
    - Loss IR 提取 (LLM 自动 + 手动模板)
    - 4层渐进式验证 (static → smoke → single → full)
    - 5-trial 结构化搜索
    - 知识积累系统 (创新点提取、代码泛化、检索融合)
    - 已知失败模式拦截 (基于 71 次实验)
    - 自动 git push
-->

# 论文 Loss 函数自动迁移技能

## 核心原则

1. **渐进式验证**: 4层验证从轻到重,尽早淘汰坏 patch,节省 GPU 时间
2. **已知失败拦截**: 基于 71 次实验的失败模式,自动拦截 SSIM/Laplacian 等
3. **知识积累**: 每次实验自动提取创新点,泛化代码,存入知识库
4. **自动化**: 实验完成后自动 git push,无需手动操作

---

## 工作流程

```
论文 PDF + 代码仓库
         ↓
    [LLM 提取]
         ↓
    Loss IR (YAML)
         ↓
   [兼容性检查] ──→ 不兼容 → 终止
         ↓ 兼容
   [知识检索] ──→ 融合历史创新点
         ↓
  [5-Trial 搜索]
         ↓
    Trial 1: Faithful Core
         ↓
    [4层验证]
      Layer 1: Static (AST + 签名)
      Layer 2: Smoke (dummy forward/backward)
      Layer 3: Single Model (SwinIR 2min)
      Layer 4: Full Run (4 models 5min)
         ↓
    Trial 2-5: 渐进式改进
         ↓
   [实验记录] ──→ 提取创新点 → 知识库
         ↓
   [Git Push] ──→ GitHub 版本控制
```

---

## 可用工具

### 1. ocean_loss_transfer_extract
**功能**: 从论文代码提取 Loss IR

**参数**:
- `code_repo` (string, 必需): 论文代码仓库路径
- `paper_slug` (string, 必需): 论文标识符
- `output_dir` (string, 可选): 输出目录

**输出**: `loss_ir.yaml` 文件

---

### 2. ocean_loss_transfer_check_compat
**功能**: 检查 Loss IR 与目标接口的兼容性

**参数**:
- `loss_ir_path` (string, 必需): Loss IR YAML 文件路径
- `target_spec_path` (string, 可选): 目标接口规格路径

**输出**: 兼容性报告 (fully_compatible / partially_compatible / incompatible)

**拦截规则**:
- 需要模型内部特征 → incompatible
- 需要预训练网络 (VGG/ResNet) → incompatible
- 需要对抗训练 → incompatible
- 包含已知失败模式 (SSIM loss, Laplacian, 5x5 Sobel) → blocked

---

### 3. ocean_loss_transfer_validate
**功能**: 4层渐进式验证 loss 文件

**参数**:
- `loss_file` (string, 必需): 待验证的 loss 文件路径
- `max_layer` (int, 可选): 最大验证层级 (1-4, 默认 4)

**验证层级**:
- **Layer 1 - Static** (<1s): AST 解析 + 签名检查 + import 白名单
- **Layer 2 - Smoke** (<10s): 动态导入 + dummy forward/backward + NaN/Inf 检查
- **Layer 3 - Single** (~2min): SwinIR 单模型训练 + SSIM > 0.3 检查
- **Layer 4 - Full** (~5min): 4 模型并行训练 + 基线对比

**输出**: 验证报告 (passed_layers, errors, metrics)

---

### 4. ocean_loss_transfer_orchestrate
**功能**: 编排 5-trial 结构化搜索

**参数**:
- `paper_slug` (string, 必需): 论文标识符
- `loss_ir_path` (string, 必需): Loss IR 文件路径
- `experiment_dir` (string, 可选): 实验输出目录

**5-Trial 策略**:
1. **Faithful Core**: 忠实移植论文核心组件
2. **+ Normalization**: 对齐 normalization/reduction 方式
3. **+ Weight Alignment**: 使用论文权重比例
4. **+ Numerical Stabilization**: 加入 epsilon/clamp 技巧
5. **Fallback Hybrid**: 混入当前最优 loss 结构

**输出**: `summary.yaml` (最佳 trial, 所有结果, 知识积累)

**自动化**:
- 每个 trial 通过 4 层验证
- 自动提取创新点存入知识库
- 实验完成后自动 git push

---

## 使用场景

### 场景 1: 完整自动化实验
```bash
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug edsr_2017 \
  --code_repo /path/to/EDSR-PyTorch
```

**流程**: 提取 → 兼容性检查 → 5-trial 搜索 → 记录 → git push

---

### 场景 2: 手动分步执行
```typescript
// Step 1: 提取 Loss IR
ocean_loss_transfer_extract({
  code_repo: "/path/to/paper/code",
  paper_slug: "paper_name"
})

// Step 2: 检查兼容性
ocean_loss_transfer_check_compat({
  loss_ir_path: "sandbox/loss_transfer_experiments/paper_name/loss_ir.yaml"
})

// Step 3: 执行实验
ocean_loss_transfer_orchestrate({
  paper_slug: "paper_name",
  loss_ir_path: "sandbox/loss_transfer_experiments/paper_name/loss_ir.yaml"
})
```

---

## 知识积累系统

### 自动化流程
每次实验完成后自动执行：

1. **创新点提取**: 分析 Loss IR，识别新颖组件
2. **代码泛化**: 将实现转换为可复用的 torch.nn.Module
3. **知识存储**: 存入 `workflow/loss_transfer/knowledge_base.yaml`
4. **检索融合**: 下次实验时自动检索相关创新点

### 知识库结构
```yaml
innovations:
  - id: innov_001
    name: "Residual FFT Loss"
    tags: [frequency, fft, residual]
    performance_gain: 0.0021
    source_paper: "exp#41"
    generalized_code: |
      class ResidualFFTLoss(nn.Module):
          def forward(self, pred, target):
              residual = pred - target
              fft = torch.fft.rfft2(residual, norm='ortho')
              return fft.abs().mean()
```

---

## 已知失败模式 (基于 71 次实验)

### 硬性拦截
- **SSIM Loss**: 导致所有模型崩溃 (exp#11, SSIM=0.109)
- **Laplacian**: 严重性能下降 (exp#20, SSIM=0.439)
- **5x5 Sobel**: 梯度过度平滑 (exp#38, SSIM=0.6395)
- **Relative FFT**: 除法不稳定 (exp#36, SSIM=0.088)
- **Scale-8**: 信息损失过大 (exp#37, SSIM=0.6156)

### 风险警告
- **Charbonnier Loss**: FNO2d 敏感，需谨慎
- **Wavelet Transform**: 计算开销大
- **Power Spectrum**: 数值不稳定 (exp#53)

---

## 故障排除

### 问题 1: Layer 1 验证失败 (AST 错误)
**原因**: 生成的代码语法错误或使用了禁止的 import

**解决**:
- 检查 `loss_ir.yaml` 中的 `required_imports` 是否在白名单内
- 确认模板渲染没有语法错误

### 问题 2: Layer 2 验证失败 (NaN/Inf)
**原因**: 数值不稳定，除零或 log(0)

**解决**:
- 在 Loss IR 中添加 `clamp_or_eps` 配置
- 检查 normalization 是否正确

### 问题 3: Layer 3 验证失败 (SSIM < 0.3)
**原因**: Loss 函数导致训练崩溃

**解决**:
- 检查是否触发已知失败模式
- 降低新组件的权重系数
- 使用 Fallback Hybrid 策略

### 问题 4: 兼容性检查失败
**原因**: Loss 需要模型内部特征或预训练网络

**解决**:
- 修改 Loss IR，移除不兼容组件
- 或放弃该论文，寻找其他方案

---

## 输出结构

实验完成后，在 `sandbox/loss_transfer_experiments/{paper_slug}/` 生成：

```
{paper_slug}/
├── loss_ir.yaml              # 提取的 Loss IR
├── compatibility.yaml        # 兼容性报告
├── trial_1/
│   ├── sandbox_loss.py       # 生成的 loss 文件
│   ├── validation.yaml       # 4层验证结果
│   └── result.yaml           # 训练结果
├── trial_2/ ... trial_5/
├── summary.yaml              # 总结 (最佳 trial, 性能对比)
└── innovations.yaml          # 提取的创新点
```

---

## 参考文档

- **完整文档**: `workflow/loss_transfer/README.md`
- **使用指南**: `workflow/loss_transfer/USAGE.md`
- **知识系统**: `workflow/loss_transfer/KNOWLEDGE_SYSTEM.md`
- **Loss IR Schema**: `scripts/ocean-loss-transfer/loss_ir_schema.py`
- **Patch 模板**: `scripts/ocean-loss-transfer/patch_templates.py`

---

## 性能基线

当前最优 (exp#41):
- **SwinIR**: 0.6645 (目标指标)
- **EDSR**: 0.6815
- **FNO2d**: 0.4344
- **UNet2d**: 0.5786

**验收标准**: 新 loss 的 SwinIR SSIM >= 0.6624 (baseline - 1σ)

---
