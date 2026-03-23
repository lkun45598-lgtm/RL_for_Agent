#!/bin/bash
# run_all_models.sh — 在 4 张 GPU 上并行跑 4 个模型，共享当前 sandbox_loss.py
# 用法: bash run_all_models.sh [loss_file_path]
# 结果写入各自的 run_<model>.log，完成后打印汇总

LOSS_FILE=${1:-}
PYTHON=$(python3 "$(dirname "${BASH_SOURCE[0]}")/../scripts/python_manager.py" --module torch 2>/dev/null | grep -oP '(?<=: ).*')
PYTHON=${PYTHON:-python3}
SCRIPT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/_run_once.py"
CONFIG_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/configs"
LOG_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

declare -A MODEL_GPU=(
    [SwinIR]=4
    [FNO2d]=5
    [EDSR]=6
    [UNet2d]=7
)

declare -A MODEL_CONFIG=(
    [SwinIR]="$CONFIG_DIR/swinir.yaml"
    [FNO2d]="$CONFIG_DIR/fno2d.yaml"
    [EDSR]="$CONFIG_DIR/edsr.yaml"
    [UNet2d]="$CONFIG_DIR/unet2d.yaml"
)

echo "=============================="
echo "  Launching 4 models in parallel"
if [ -n "$LOSS_FILE" ]; then
    echo "  Loss: $LOSS_FILE"
else
    echo "  Loss: sandbox_loss.py"
fi
echo "  $(date)"
echo "=============================="

# 构建 loss 参数
if [ -n "$LOSS_FILE" ]; then
    LOSS_ARG="--loss_file $LOSS_FILE"
else
    LOSS_ARG=""
fi

# 并行启动
for MODEL in SwinIR FNO2d EDSR UNet2d; do
    GPU=${MODEL_GPU[$MODEL]}
    CFG=${MODEL_CONFIG[$MODEL]}
    LOG="$LOG_DIR/run_${MODEL}.log"
    echo "  GPU$GPU → $MODEL"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$SCRIPT" --config "$CFG" $LOSS_ARG > "$LOG" 2>&1 &
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
