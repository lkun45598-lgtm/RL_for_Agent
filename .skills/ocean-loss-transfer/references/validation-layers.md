# 4 层验证详解

渐进式验证系统，从轻到重，尽早淘汰坏 patch。

---

## Layer 1 - Static（<1s）

**目的**: 语法和签名检查

**检查项**:
- AST 解析（`ast.parse()`）
- 函数签名匹配（`sandbox_loss(pred, target, mask=None, **kwargs)`）
- Import 白名单（只允许 torch, torch.nn.functional, math）
- 禁止模式检测（eval, exec, subprocess）

**通过条件**: 无语法错误，签名正确，import 合规

---

## Layer 2 - Smoke（<10s）

**目的**: 动态导入和基本功能测试

**检查项**:
- 动态导入 loss 函数
- Dummy forward（随机张量）
- Dummy backward（梯度计算）
- NaN/Inf 检查
- mask=None 兼容性

**通过条件**: 无运行时错误，无 NaN/Inf，梯度正常

---

## Layer 3 - Single Model（~2min）

**目的**: 单模型快速训练测试

**检查项**:
- SwinIR 模型训练 2 分钟
- SSIM > 0.3（崩溃阈值）
- 训练稳定性

**通过条件**: SSIM >= 0.3

---

## Layer 4 - Full Run（~5min）

**目的**: 4 模型并行训练

**检查项**:
- SwinIR, EDSR, FNO2d, UNet2d 并行训练
- 与基线对比
- 所有模型稳定

**通过条件**: SwinIR SSIM >= baseline - 1σ
