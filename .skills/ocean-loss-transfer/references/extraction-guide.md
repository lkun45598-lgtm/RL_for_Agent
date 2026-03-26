# 公式与证据提取指南

本文档指导 Agent 如何从论文和代码联合提取 `loss_formula.json`，并为 `analysis_plan.json` 收集证据。

## 目录

- 分析步骤
- integration 相关证据
- 常见陷阱

---

## 分析步骤

### 1. 先读论文里的 loss 描述

- 优先看摘要、方法、实现细节、loss 相关小节
- 如果有 `loss_snippets`，优先从这里提取公式、权重、符号
- 不要只看一个公式块；很多关键变量定义在正文段落里

### 2. 再定位代码中的真实 loss 入口

优先看：

- `train.py` / `trainer.py` 中的训练循环
- `criterion(...)` / `loss = ...` / `compute_loss(...)`
- `model.forward(...)` 是否顺带计算了辅助量

### 3. 提取公式内容

`loss_formula.json` 重点提三部分：

- `latex`
- `params`
- `symbol_map`

示例：

```json
{
  "latex": ["L = L_{pix} + \\lambda_f L_{freq}"],
  "params": {"lambda_f": 0.1},
  "symbol_map": {
    "L_{pix}": "pixel_loss",
    "L_{freq}": "freq_loss",
    "\\lambda_f": "lambda_f",
    "\\hat{y}": "pred",
    "y": "target"
  }
}
```

### 4. 识别 loss 组件类型

| 代码特征 | Loss 类型 | 示例 |
|---------|----------|------|
| `F.l1_loss`, `torch.abs` | pixel_loss (L1) | `torch.abs(pred - target).mean()` |
| `F.mse_loss`, `**2` | pixel_loss (L2) | `((pred - target) ** 2).mean()` |
| `torch.sqrt(x**2 + eps)` | pixel_loss (Charbonnier) | `torch.sqrt((pred - target)**2 + 1e-3).mean()` |
| `torch.fft.rfft2` | frequency_loss | `torch.fft.rfft2(pred - target, norm='ortho')` |
| `conv2d` with Sobel kernel | gradient_loss | `F.conv2d(x, sobel_kernel)` |
| `vgg(x)` | perceptual_loss | `F.mse_loss(vgg(pred), vgg(target))` |

### 5. 提取实现细节

#### reduction 方式
```python
.mean()           → reduction: mean
.sum()            → reduction: sum
.mean(dim=[2,3])  → reduction: spatial_mean
```

#### mask 处理
```python
loss * mask                    → mask_handling: multiply
loss[mask]                     → mask_handling: index
if mask is not None: loss * mask → mask_handling: multiply
```

#### normalization
```python
(pred - target).norm() / target.norm()  → normalization: relative
(pred - target).norm()                  → normalization: none
```

#### epsilon/clamp
```python
1 / (x + 1e-8)           → clamp_or_eps: [{location: denominator, method: add_eps, value: 1e-8}]
1 / x.clamp(min=1e-8)    → clamp_or_eps: [{location: denominator, method: clamp_min, value: 1e-8}]
```

### 6. 收集 integration 相关证据

**需要模型内部特征**：
```python
features = model.encoder(x)  # requires_model_internals = true
loss = F.mse_loss(features, target_features)
```

**需要预训练网络**：
```python
vgg = VGG19(pretrained=True)  # requires_pretrained_network = true
perceptual_loss = F.mse_loss(vgg(pred), vgg(target))
```

**需要对抗训练**：
```python
discriminator = Discriminator()  # requires_adversarial = true
adv_loss = discriminator(pred)
```

---

这类证据后面要写进 `analysis_plan.integration_decision.evidence_refs`。

建议引用形式：

- `paper.loss`
- `paper.method`
- `paper.implementation_details`
- `code.loss_callsite`
- `code.model_forward`
- `code.adapter_wrapper`

---

## 常见陷阱

1. **只有公式没有语义**：公式里没写清的变量定义，要回正文和代码找
2. **只有代码没有论文**：代码可能是工程简化版，不能直接当论文真相
3. **symbol_map 不是双射**：多个符号映到一个变量，后续很难做公式对齐
4. **忽略 model.forward**：很多额外 loss inputs 根本不在 loss 文件里
5. **把 IR 当主产物**：当前主产物是 `loss_formula.json` 和 `analysis_plan.json`
