# Attempt 策略说明

本文档描述如何设计 attempts。只在你需要多轮试探或 repair 策略时再读。

---

## Attempt 1: Faithful

**目标**: 忠实迁移论文核心 loss

**策略**:
- 保留论文核心组件和权重关系
- 尽量少做工程化重写
- 重点验证 integration path 是否选对

---

## Attempt 2: Stabilized

**目标**: 在 faithful 基础上补数值稳定性

**策略**:
- 加 epsilon / clamp / dtype guard
- 明确 reduction / normalization / masked mean
- 保留与公式一致的主结构

---

## Attempt 3: Path-Corrective

**目标**: 修正接入路径，而不是继续硬修 loss 表达式

**策略**:
- 如果 stop_layer 暴露缺失 loss inputs，补 adapter 或 model outputs
- 如果只改 `candidate_loss.py` 不够，就切到更深的 path
- 所有模型级改动都放在 attempt-scoped 副本里

---

## Attempt 4: Conservative Fallback

**目标**: 保住训练稳定性，再逐步恢复论文细节

**策略**:
- 降低高风险组件权重
- 暂时关闭最不稳定的附加项
- 先让 Layer 3/4 能稳定跑通
