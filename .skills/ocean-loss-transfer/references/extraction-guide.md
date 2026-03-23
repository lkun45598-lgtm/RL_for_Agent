# Loss IR 提取指南（Agent 专用）

本文档指导 Agent 如何分析论文代码并生成 Loss IR YAML。

---

## 分析步骤

### 1. 定位主 loss 函数

**查找位置**：
- `train.py` / `trainer.py` 中的训练循环
- 查找 `loss = ...` 或 `criterion(...)`

**识别特征**：
```python
# 典型模式
loss = criterion(pred, target)
loss = compute_loss(output, gt, mask)
total_loss = loss_fn(sr, hr)
```

### 2. 识别 loss 组件类型

| 代码特征 | Loss 类型 | 示例 |
|---------|----------|------|
| `F.l1_loss`, `torch.abs` | pixel_loss (L1) | `torch.abs(pred - target).mean()` |
| `F.mse_loss`, `**2` | pixel_loss (L2) | `((pred - target) ** 2).mean()` |
| `torch.sqrt(x**2 + eps)` | pixel_loss (Charbonnier) | `torch.sqrt((pred - target)**2 + 1e-3).mean()` |
| `torch.fft.rfft2` | frequency_loss | `torch.fft.rfft2(pred - target, norm='ortho')` |
| `conv2d` with Sobel kernel | gradient_loss | `F.conv2d(x, sobel_kernel)` |
| `vgg(x)` | perceptual_loss | `F.mse_loss(vgg(pred), vgg(target))` |

### 3. 提取实现细节

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

### 4. 检查不兼容特征

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

## Loss IR 模板

```yaml
metadata:
  paper_title: "论文标题"
  loss_function_name: "Loss 函数名称"
  loss_function_path: "文件路径:函数名"

interface:
  input_tensors:
    - name: pred
      shape: [B, H, W, C]
      dtype: float32
      required: true
      source: model_output
    - name: target
      shape: [B, H, W, C]
      dtype: float32
      required: true
      source: external
  requires_model_internals: false
  requires_pretrained_network: false

components:
  - name: pixel_loss
    type: pixel_loss
    weight: 0.5
    implementation:
      reduction: mean
      operates_on: pixel_space
      mask_handling: multiply
      normalization: none
    required_tensors: [pred, target]
    required_imports: [torch]

multi_scale:
  enabled: false

combination:
  method: weighted_sum

incompatibility_flags:
  requires_model_features: false
  requires_pretrained_network: false
  requires_adversarial: false
```

---

## 常见陷阱

1. **相对 vs 绝对**：`/ target.norm()` 是 relative，不要遗漏
2. **多尺度**：如果有 `F.avg_pool2d` 循环，提取 scales 列表
3. **权重系数**：`0.5 * loss1 + 0.3 * loss2` → weights: {loss1: 0.5, loss2: 0.3}
4. **epsilon 位置**：注意是加在分子还是分母
