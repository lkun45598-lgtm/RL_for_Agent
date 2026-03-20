#!/bin/bash
# run_all_models.sh — 在 4 张 GPU 上并行跑 4 个模型，共享当前 sandbox_loss.py
# 用法: bash run_all_models.sh
# 结果写入各自的 run_<model>.log，完成后打印汇总

PYTHON=/home/lz/miniconda3/envs/pytorch/bin/python
SCRIPT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/_run_once.py"
CONFIG_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/configs"
LOG_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

declare -A MODEL_GPU=(
    [SwinIR]=1
    [FNO2d]=2
    [EDSR]=3
    [UNet2d]=4
)

declare -A MODEL_CONFIG=(
    [SwinIR]="$CONFIG_DIR/swinir.yaml"
    [FNO2d]="$CONFIG_DIR/fno2d.yaml"
    [EDSR]="$CONFIG_DIR/edsr.yaml"
    [UNet2d]="$CONFIG_DIR/unet2d.yaml"
)

echo "=============================="
echo "  Launching 4 models in parallel"
echo "  Loss: sandbox_loss.py"
echo "  $(date)"
echo "=============================="

# 并行启动
for MODEL in SwinIR FNO2d EDSR UNet2d; do
    GPU=${MODEL_GPU[$MODEL]}
    CFG=${MODEL_CONFIG[$MODEL]}
    LOG="$LOG_DIR/run_${MODEL}.log"
    echo "  GPU$GPU → $MODEL"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$SCRIPT" --config "$CFG" > "$LOG" 2>&1 &
done

echo ""
echo "All launched. Waiting for completion..."
wait
echo ""
echo "=============================="
echo "  Results"
echo "=============================="

for MODEL in SwinIR FNO2d EDSR UNet2d; do
    LOG="$LOG_DIR/run_${MODEL}.log"
    if [ -f "$LOG" ]; then
        SSIM=$(grep "^val_ssim:" "$LOG" | awk '{print $2}')
        PSNR=$(grep "^val_psnr:" "$LOG" | awk '{print $2}')
        TEST_SSIM=$(grep "^test_ssim:" "$LOG" | awk '{print $2}')
        DUR=$(grep "^duration_s:" "$LOG" | awk '{print $2}')
        if [ -n "$SSIM" ]; then
            echo "  $MODEL\tSSIM=$SSIM\tPSNR=$PSNR\ttest_SSIM=$TEST_SSIM\t${DUR}s"
        else
            echo "  $MODEL\tCRASH — check run_${MODEL}.log"
        fi
    else
        echo "  $MODEL\tNO LOG"
    fi
done
echo "=============================="
