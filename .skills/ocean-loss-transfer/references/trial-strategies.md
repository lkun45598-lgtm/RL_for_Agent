# 5-Trial 策略说明

结构化搜索策略，渐进式改进。

---

## Trial 1: Faithful Core

**目标**: 忠实移植论文核心组件

**策略**:
- 保留论文的核心 loss 组件
- 使用我方的多尺度策略
- 使用我方的 mask 处理

---

## Trial 2: Normalization Aligned

**目标**: 对齐 normalization/reduction

**策略**:
- 在 Trial 1 基础上
- 对齐论文的 normalization 方式
- 对齐论文的 reduction 方式

---

## Trial 3: Weight Aligned

**目标**: 使用论文权重比例

**策略**:
- 在 Trial 2 基础上
- 使用论文的权重系数

---

## Trial 4: Numerical Stabilized

**目标**: 加入数值稳定技巧

**策略**:
- 在 Trial 3 基础上
- 加入 epsilon/clamp
- 加入 dtype cast

---

## Trial 5: Fallback Hybrid

**目标**: 混入当前最优结构

**策略**:
- 取前 4 轮最好的新组件
- 混入 exp#41 的最优结构
