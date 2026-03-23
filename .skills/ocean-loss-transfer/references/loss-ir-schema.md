# Loss IR Schema 说明

完整的 Loss IR 数据结构定义。

---

## 顶层结构

```yaml
metadata:          # 论文元信息
interface:         # 输入输出接口
components:        # Loss 组件列表
multi_scale:       # 多尺度策略
combination:       # 组合方式
incompatibility_flags:  # 不兼容标记
```

---

## metadata（元信息）

```yaml
metadata:
  paper_title: string           # 论文标题
  paper_authors: [string]       # 作者列表（可选）
  loss_function_name: string    # Loss 函数名称
  loss_function_path: string    # 代码路径（格式：file.py:function_name）
  code_repo: string             # 代码仓库路径（可选）
```

---

## interface（接口）

```yaml
interface:
  input_tensors:
    - name: string              # 张量名称（pred/target/mask）
      shape: [B, H, W, C]       # 形状（支持符号维度）
      dtype: float32            # 数据类型
      required: boolean         # 是否必需
      source: string            # 来源（model_output/external/computed）
  requires_model_internals: boolean      # 是否需要模型中间层
  requires_pretrained_network: boolean   # 是否需要预训练网络
```

**source 类型**：
- `model_output` - 模型输出
- `external` - 外部输入（如 ground truth）
- `computed` - 计算得到（如梯度）

---

## components（组件）

```yaml
components:
  - name: string                # 组件名称
    type: string                # 类型（见下表）
    weight: float               # 权重系数
    formula: string             # 数学公式（可选）
    implementation:             # 实现细节
      reduction: string         # 归约方式
      operates_on: string       # 操作空间
      mask_handling: string     # mask 处理
      channel_handling: string  # 通道处理
      normalization: string     # 归一化方式
      clamp_or_eps: [...]       # epsilon/clamp 配置
    required_tensors: [string]  # 所需张量
    required_imports: [string]  # 所需导入
```

**type 类型**：
- `pixel_loss` - 像素级 loss
- `gradient_loss` - 梯度 loss
- `frequency_loss` - 频域 loss
- `perceptual_loss` - 感知 loss
- `structural_loss` - 结构 loss

**reduction**：`mean` / `sum` / `batch_mean` / `masked_mean`

**operates_on**：`pixel_space` / `gradient_space` / `frequency_space`

**mask_handling**：`multiply` / `index` / `ignore` / `none`

**normalization**：`none` / `relative` / `batch_normalized`

---

## multi_scale（多尺度）

```yaml
multi_scale:
  enabled: boolean              # 是否启用
  scales: [int]                 # 尺度列表（如 [1, 2, 4]）
  scale_weights: [float]        # 尺度权重
  downsample_method: string     # 下采样方法（avg_pool2d/bilinear）
```

---

## combination（组合）

```yaml
combination:
  method: string                # 组合方式
  weights: {name: float}        # 权重字典（weighted_sum 时需要）
```

**method**：`weighted_sum` / `geometric_mean` / `curriculum`

---

## incompatibility_flags（不兼容标记）

```yaml
incompatibility_flags:
  requires_model_features: boolean       # 需要模型特征
  requires_pretrained_network: boolean   # 需要预训练网络
  requires_adversarial: boolean          # 需要对抗训练
  requires_multiple_forward_passes: boolean  # 需要多次前向传播
```
