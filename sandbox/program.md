# Sandbox Loss 优化 — Agent 指令

这是一个自主 loss 函数搜索实验。你是一个完全自主的研究者，目标是通过修改 loss 函数来最大化海洋超分辨率任务的 **masked SSIM**。

## Setup

开始新实验前，与用户确认：

1. **确定 run tag**：基于日期提议一个 tag（如 `mar20`）。分支 `autoresearch/<tag>` 不能已存在。
2. **创建分支**：`git checkout -b autoresearch/<tag>`
3. **读取上下文文件**：
   - `program.md` — 本文件，你的指令
   - `sandbox_loss.py` — 你唯一修改的文件
   - `sandbox_trainer.py` — 了解 loss 如何被加载
   - `sandbox_config.yaml` — 当前训练配置
4. **检查 GPU**：运行 `nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader` 找到空闲 GPU
5. **初始化 results.tsv**：如果不存在，创建只含 header 的文件
6. **确认并开始**

## 硬约束

**你只能做的事：**
- 修改 `sandbox/sandbox_loss.py` — 这是唯一你编辑的文件
- 在 `sandbox/` 目录下运行命令
- 读取 `sandbox/` 和 `scripts/ocean-SR-training-masked/` 下的文件（只读参考）

**你绝对不能做的事：**
- 修改 `scripts/` 下的任何文件
- 修改 `src/` 下的任何文件
- 修改 `sandbox_trainer.py`、`_run_once.py`、`generate_sandbox_config.py`
- 修改或删除数据文件
- 安装新的包或依赖
- 访问 `sandbox/` 以外的目录进行写操作

**sandbox_loss.py 的约束：**
- 只允许 `import torch, torch.nn.functional, math`
- 函数签名固定：`sandbox_loss(pred, target, mask=None, **kwargs)` → 返回标量 tensor
- `pred, target`: `[B, H, W, C]`，`mask`: `[1, H, W, 1]` bool（True=海洋），可能为 None

## GPU 分配

本机有 8 张 RTX 4090（GPU 0-7）。运行前先检查哪些 GPU 空闲：

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
```

选择 `memory.used` 最小的 GPU，通过 `CUDA_VISIBLE_DEVICES` 指定：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python _run_once.py --config sandbox_config.yaml > run.log 2>&1
```

注意：`sandbox_config.yaml` 中 `device: 0` 不需要改，因为 `CUDA_VISIBLE_DEVICES` 会重映射。

## 切换模型

默认使用 SwinIR。如需切换模型：

```bash
python generate_sandbox_config.py --model_name FNO2d   # 或 SwinIR / EDSR / UNet2d
```

这会覆盖 `sandbox_config.yaml`。切换模型后建议先跑一次 baseline。

## 运行实验

每次实验约 1-2 分钟（15 epoch，小数据集）。

```bash
CUDA_VISIBLE_DEVICES=1 python _run_once.py --config sandbox_config.yaml > run.log 2>&1
```

训练完成后脚本会打印摘要：

```
---
val_ssim:         0.611616
val_psnr:         16.523043
val_rmse:         0.608739
val_loss:         0.754253
test_ssim:        0.661485
test_psnr:        18.409143
test_rmse:        0.741943
duration_s:       56.0
model:            SwinIR
```

提取关键指标：

```bash
grep "^val_ssim:\|^val_psnr:\|^test_ssim:" run.log
```

如果 grep 输出为空，说明训练崩溃了。运行 `tail -n 50 run.log` 查看错误。

## 记录结果

`results.tsv` 是 tab 分隔的（不要用逗号），header 和 5 列：

```
commit	val_ssim	val_psnr	status	description
```

1. git commit hash（短，7 字符）
2. val_ssim（如 0.611616）— 崩溃时用 0.000000
3. val_psnr（如 16.52）— 崩溃时用 0.0
4. status：`keep`、`discard` 或 `crash`
5. 简短描述本次实验做了什么

示例：

```
commit	val_ssim	val_psnr	status	description
a1b2c3d	0.611616	16.52	keep	baseline relative L2
b2c3d4e	0.625000	17.10	keep	L1 + SSIM combo
c3d4e5f	0.590000	15.80	discard	pure FFT loss
d4e5f6g	0.000000	0.0	crash	gradient loss OOM
```

注意：不要 commit results.tsv，保持 untracked。

## 实验循环

在专用分支上（如 `autoresearch/mar20`）：

**LOOP FOREVER:**

1. 查看 git 状态：当前分支/commit
2. 查看 `results.tsv` 最近 10 条记录，分析趋势
3. 决定探索方向，修改 `sandbox_loss.py`
4. git commit：`git add sandbox_loss.py && git commit -m "描述"`
5. 检查 GPU：`nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader`
6. 运行实验：`CUDA_VISIBLE_DEVICES=<空闲GPU> python _run_once.py --config sandbox_config.yaml > run.log 2>&1`
7. 读取结果：`grep "^val_ssim:\|^val_psnr:" run.log`
8. 如果 grep 为空 → 崩溃，`tail -n 50 run.log` 看错误，尝试修复
9. 记录到 results.tsv
10. 如果 val_ssim 提升（更高）→ keep，保留 commit
11. 如果 val_ssim 相同或更差 → discard，`git reset --hard HEAD~1`
12. 回到步骤 1

**超时**：如果训练超过 10 分钟没结束，kill 掉当作失败。

**崩溃**：如果是简单 bug（typo、import 错误），修复后重跑。如果想法本身有问题，跳过，记录 crash。

**永不停止**：一旦实验循环开始，不要暂停问用户是否继续。用户可能在睡觉，期望你持续工作直到被手动停止。如果没有想法了，想得更深 — 重读代码、尝试组合之前的近似成功、尝试更激进的变化。

## Loss 函数探索方向

### 基础
- L1 (MAE)：对异常值更鲁棒
- Smooth L1 / Huber
- Charbonnier：`sqrt((pred-target)^2 + eps^2)`

### 感知
- SSIM Loss：`1 - SSIM(pred, target)`，直接优化目标
- 梯度 Loss：`|∇pred - ∇target|`，保持边缘锐度

### 频域
- FFT Loss：`|FFT(pred) - FFT(target)|`
- 高频加权 FFT

### 组合
- L1 + SSIM：`α * L1 + β * (1 - SSIM)`
- L1 + 梯度 + SSIM
- 逐步添加项，观察边际收益

### 海洋特定
- 物理约束：散度/旋度（uo, vo 是速度分量）
- 空间加权：对高梯度区域加权
- 通道交互：利用 uo/vo 物理关系

### 策略建议
- 先跑 baseline（当前的相对 L2）
- 然后试单一替换（纯 L1、纯 SSIM loss）
- 找到最好的单一 loss 后，尝试组合
- 调权重比例时用二分法，不要一次改太多

## 防遗忘

- 每次实验前读 `results.tsv` 最近 10 条
- 每 10 次实验在 commit message 中写一段总结
- 崩溃后检查 `git log --oneline -20` 回顾历史

## 文件说明

| 文件 | 权限 | 用途 |
|------|------|------|
| `sandbox_loss.py` | **读写** | 唯一修改的文件 — 自定义 loss |
| `sandbox_config.yaml` | 只读 | 训练配置（由 generate_sandbox_config.py 生成） |
| `_run_once.py` | 只读 | 训练入口脚本 |
| `sandbox_trainer.py` | 只读 | SandboxTrainer，覆盖 build_loss() |
| `generate_sandbox_config.py` | 只读 | 配置生成器 |
| `program.md` | 只读 | 本文件 |
| `results.tsv` | 追加写 | 实验日志（不要 git commit） |
| `run.log` | 自动生成 | 最近一次训练的完整输出 |
