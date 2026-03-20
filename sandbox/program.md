# Sandbox Loss 优化 — Agent 指令

这是一个自主 loss 函数搜索实验。你是一个完全自主的研究者，目标是通过修改 loss 函数来最大化海洋超分辨率任务的 **masked SSIM**。

## Setup

开始新实验前，与用户确认：

1. **确定 run tag**：基于日期提议一个 tag（如 `mar20`）。分支 `autoresearch/<tag>` 不能已存在。
2. **创建分支**：`git checkout -b autoresearch/<tag>`
3. **读取上下文文件**：
   - `program.md` — 本文件，你的指令
   - `sandbox_loss.py` — 你唯一修改的文件
   - `sandbox_config.yaml` — 当前训练配置
4. **检查 GPU**：`nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader`
5. **确认并开始**

## 硬约束

**你只能做的事：**
- 修改 `sandbox/sandbox_loss.py` — 这是唯一你编辑的文件
- 在 `sandbox/` 目录下运行命令
- 读取 `sandbox/` 和 `scripts/ocean-SR-training-masked/` 下的文件（只读参考）

**你绝对不能做的事：**
- 修改 `scripts/` 下的任何文件
- 修改 `src/` 下的任何文件
- 修改 `sandbox_trainer.py`、`_run_once.py`、`generate_sandbox_config.py`、`run_all_models.sh`、`status.py`
- 修改或删除数据文件
- 安装新的包或依赖

**sandbox_loss.py 的约束：**
- 只允许 `import torch, torch.nn.functional, math`
- 函数签名固定：`sandbox_loss(pred, target, mask=None, **kwargs)` → 返回标量 tensor
- `pred, target`: `[B, H, W, C]`，`mask`: `[1, H, W, 1]` bool（True=海洋），可能为 None

## GPU 分配

本机有 8 张 RTX 4090（GPU 0-7）。GPU 分配如下：

| GPU | 模型   | 配置文件            |
|-----|--------|---------------------|
| 1   | SwinIR | configs/swinir.yaml |
| 2   | FNO2d  | configs/fno2d.yaml  |
| 3   | EDSR   | configs/edsr.yaml   |
| 4   | UNet2d | configs/unet2d.yaml |
| 5-7 | 备用   | —                   |

GPU 0 可能被其他进程占用，默认不用。

## 实验循环（核心流程）

```
1. 读 results.tsv 最近 10 条，分析历史趋势
2. 修改 sandbox_loss.py
3. git add sandbox_loss.py && git commit -m "exp#N: 描述"
4. 在 SwinIR 上快速验证（约 1 分钟）：
   CUDA_VISIBLE_DEVICES=1 python _run_once.py --config configs/swinir.yaml > run_SwinIR.log 2>&1
5. grep 结果：
   grep "^val_ssim:\|^val_psnr:\|^test_ssim:\|^duration_s:" run_SwinIR.log
6. 判断：
   - IMPROVED (> best_ssim)：保留，进入步骤 7
   - WORSE：git reset --hard <best_commit>，回到步骤 1
   - CRASH：检查 log，修复 bug，重试
7. 确认改进后，4 模型并行验证：
   bash run_all_models.sh
8. 查看汇总：python status.py
9. 记录到 results.tsv：
   echo -e "<hash>\t<ssim>\t<psnr>\tkeep\t描述" >> results.tsv
10. git push origin autoresearch/<tag>
11. 回到步骤 1
```

## results.tsv 格式

```
commit  val_ssim  val_psnr  status   description
abc1234 0.6116    16.52     keep     baseline relative L2 (SwinIR)
```

- `status`: `keep` 或 `discard`
- 每次实验追加一行，不要修改历史记录

## 当前最优（SwinIR）

| exp | val_ssim | val_psnr | loss 描述 |
|-----|----------|----------|-----------|
| baseline | 0.6116 | 16.52 | relative L2 |
| exp#7 | **0.6555** | 17.02 | rel L2 + gradient + FFT (α=0.5 β=0.3 γ=0.2) |

## Loss 探索方向

### 已验证
- ✗ 纯 L1：0.6083 (worse)
- ✗ Charbonnier：0.5960 (worse)
- ✓ rel L2 + gradient (α=0.8 β=0.2)：0.6208
- ✓ rel L2 + gradient (α=0.6 β=0.4)：0.6357
- ✓ rel L2 + gradient + FFT (α=0.6 β=0.3 γ=0.1)：0.6452
- ✓ **rel L2 + gradient + FFT (α=0.5 β=0.3 γ=0.2)：0.6555** ← 当前最优
- ✗ rel L2 + gradient + FFT (α=0.4 β=0.3 γ=0.3)：0.6400

### 待探索
- FFT 高频加权（对高频分量加权放大）
- 相位谱 loss（FFT 实部/虚部差异）
- 多尺度梯度（不同 sigma 的 Sobel）
- uo/vo 物理约束（散度/旋度最小化）
- SSIM loss 组合

## 防遗忘

- 每次实验前读 `results.tsv` 最近 10 条
- 崩溃后 `git log --oneline -20` 回顾历史
- 每 10 次实验写一段 commit message 总结

## NEVER STOP

一旦实验循环开始，**不要暂停询问用户是否继续**。用户可能不在电脑旁。持续运行直到被手动中断。如果没有新思路，重新读论文参考、组合历史近优方案、调整权重比例。

## 文件说明

| 文件 | 权限 | 用途 |
|------|------|------|
| `sandbox_loss.py` | **读写** | 唯一修改的文件 |
| `run_all_models.sh` | 只读 | 4 模型并行运行 |
| `status.py` | 只读 | 查看所有模型状态 |
| `configs/` | 只读 | 各模型训练配置 |
| `results.tsv` | 追加写 | 实验日志 |
| `run_<Model>.log` | 自动生成 | 各模型训练输出 |
