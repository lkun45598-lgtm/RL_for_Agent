---
name: ocean-loss-transfer
version: 1.0.0
---

# Ocean Loss Transfer

从论文迁移 loss 函数到 sandbox_loss.py 的自动化工具链。

## 概述

该工具链支持:
- 从论文 PDF + 代码提取 Loss IR (中间表示)
- 检查与目标接口的兼容性
- 生成结构化 patch
- 4 层渐进式验证
- 5-trial 自动搜索

## 工作流

### 1. 提取 Loss IR

```
ocean_loss_transfer_extract
  --paper_pdf_path <path>
  --code_repo_path <path>
  --output_yaml loss_ir.yaml
```

或手动模式生成模板:
```
ocean_loss_transfer_extract
  --manual_mode true
  --output_yaml loss_ir.yaml
```

### 2. 检查兼容性

```
ocean_loss_transfer_check_compat
  --loss_ir_yaml loss_ir.yaml
```

返回: fully_compatible / partially_compatible / incompatible

### 3. 编排 5-Trial 搜索

```
ocean_loss_transfer_orchestrate
  --loss_ir_yaml loss_ir.yaml
  --paper_slug paper_name
```

自动执行:
- Trial 1: Faithful Core (忠实移植)
- Trial 2: Normalization Aligned
- Trial 3: Weight Aligned
- Trial 4: Numerical Stabilized
- Trial 5: Fallback Hybrid

### 4. 查看结果

结果保存在:
```
sandbox/loss_transfer_experiments/{paper_slug}/
├── summary.yaml          # 总结
├── trial_1/
│   ├── sandbox_loss.py
│   └── result.yaml
├── trial_2/
...
```

## 工具列表

- `ocean_loss_transfer_extract` - 提取 Loss IR
- `ocean_loss_transfer_check_compat` - 检查兼容性
- `ocean_loss_transfer_validate` - 验证 loss 文件
- `ocean_loss_transfer_orchestrate` - 编排 5-trial

## 注意事项

- 已知失败模式会被自动拦截 (SSIM loss, Laplacian 等)
- 4 层验证从轻到重,尽早淘汰坏 patch
- 基线噪声会自动测量 (首次运行时)
