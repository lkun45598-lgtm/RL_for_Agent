# 故障排除

常见问题和解决方案。

---

## 公式提取不稳定

**原因**: 论文里符号定义和代码变量名对不上

**解决**:
- 回到正文和实现细节小节重新找变量定义
- 收紧 `symbol_map`，确保一一映射
- 必要时让 `params` 显式写出论文常数

---

## Layer 1 验证失败（AST/静态检查）

**原因**: 生成代码语法错误、签名不匹配或 import 不合规

**解决**:
- 先修 `candidate_loss.py`
- 确认 `sandbox_loss(pred, target, mask=None, **kwargs)` 签名正确
- 不要引入 repo 中不存在的依赖

---

## Layer 2 验证失败（NaN/Inf 或 shape 问题）

**原因**: 数值不稳定、tensor shape 不匹配、mask 处理错误

**解决**:
- 加入 epsilon、clamp、dtype cast
- 检查 BHWC/HW/BCHW 维度约定
- 检查是否误把额外 loss inputs 当成了普通 kwargs

---

## Layer 3/4 暴露缺失 loss inputs

**原因**: integration path 选错，只改 loss 文件不够

**解决**:
- 重新判断是否应切换到 `adapter_wrapper` 或 `extend_model_outputs`
- 在 attempt-scoped override/model copy 中补齐所需张量
- 不要直接回退成错误的 `loss_only`
