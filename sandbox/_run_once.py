"""
@file _run_once.py

@description 沙盒训练入口 — 使用 SandboxTrainer 运行一次训练。
    训练完成后打印 grep 友好的指标摘要。
@author Leizheng
@date 2026-03-20
@version 2.0.0

@changelog
  - 2026-03-20 Leizheng: v2.0.0 autoresearch 模式重构
    - 去掉 --log_dir 参数，自动生成带时间戳的 log 目录
    - 训练完成后打印 grep 友好的指标摘要（val_ssim: 等）
    - 解析 __event__ 提取 final_valid/final_test 指标
  - 2026-03-20 Leizheng: v1.0.0 初始版本
"""

import os
import sys
import re
import json
import time
import traceback
import argparse
from datetime import datetime

# 将训练管线加入 sys.path
_PIPELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts', 'ocean-SR-training-masked')
_PIPELINE_DIR = os.path.abspath(_PIPELINE_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from utils.helper import set_up_logger, set_seed, load_config, save_config
from sandbox_trainer import SandboxTrainer

EVENT_RE = re.compile(r'__event__(\{.*?\})__event__')


class EventCapture:
    """捕获 stdout 中的 __event__ 事件，同时正常输出到终端。"""

    def __init__(self, original_stdout):
        self.original = original_stdout
        self.captured = []

    def write(self, text):
        self.original.write(text)
        for m in EVENT_RE.finditer(text):
            try:
                self.captured.append(json.loads(m.group(1)))
            except json.JSONDecodeError:
                pass

    def flush(self):
        self.original.flush()

    def fileno(self):
        return self.original.fileno()


def main():
    trainer = None
    t_start = time.time()

    try:
        parser = argparse.ArgumentParser(description='Sandbox single run')
        parser.add_argument('--config', type=str, required=True, help='Path to sandbox_config.yaml')
        parser.add_argument('--loss_file', type=str, default=None, help='Path to sandbox_loss.py')
        cli_args = parser.parse_args()

        args = {'config': cli_args.config}
        args = load_config(args)

        # 自动生成带时间戳的 log 目录
        sandbox_dir = os.path.dirname(os.path.abspath(__file__))
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = args['model']['name']
        log_dir = os.path.join(sandbox_dir, 'runs', f'{model_name}_{ts}')
        args['log']['log_dir'] = log_dir

        saving_path, saving_name = set_up_logger(args)
        set_seed(args['train'].get('random_seed', 42))
        args['train']['saving_path'] = saving_path
        args['train']['saving_name'] = saving_name
        save_config(args, saving_path)

        loss_file = cli_args.loss_file or os.path.join(sandbox_dir, 'sandbox_loss.py')

        # 安装事件捕获器
        capture = EventCapture(sys.stdout)
        sys.stdout = capture

        trainer = SandboxTrainer(args, loss_file=loss_file)
        trainer.process()

        # 恢复 stdout
        sys.stdout = capture.original

        # 从捕获的事件中提取指标
        final_valid = None
        final_test = None
        for ev in capture.captured:
            if ev.get('event') == 'final_valid':
                final_valid = ev.get('metrics', {})
            elif ev.get('event') == 'final_test':
                final_test = ev.get('metrics', {})

        duration = time.time() - t_start

        # 打印 grep 友好的摘要
        print("\n---")
        if final_valid:
            print(f"val_ssim:         {final_valid.get('ssim', 0.0):.6f}")
            print(f"val_psnr:         {final_valid.get('psnr', 0.0):.6f}")
            print(f"val_rmse:         {final_valid.get('rmse', 0.0):.6f}")
            print(f"val_loss:         {final_valid.get('valid_loss', 0.0):.6f}")
        if final_test:
            print(f"test_ssim:        {final_test.get('ssim', 0.0):.6f}")
            print(f"test_psnr:        {final_test.get('psnr', 0.0):.6f}")
            print(f"test_rmse:        {final_test.get('rmse', 0.0):.6f}")
        print(f"duration_s:       {duration:.1f}")
        print(f"model:            {model_name}")
        print(f"log_dir:          {log_dir}")

    except Exception as e:
        # 恢复 stdout
        if not isinstance(sys.stdout, type(sys.__stdout__)):
            try:
                sys.stdout = sys.stdout.original
            except AttributeError:
                sys.stdout = sys.__stdout__

        duration = time.time() - t_start
        error = {
            "event": "training_error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "phase": "startup" if trainer is None else "train",
        }
        print(f"__event__{json.dumps(error, ensure_ascii=False)}__event__", flush=True)
        print("\n---")
        print("CRASH")
        print(f"error_type:       {type(e).__name__}")
        print(f"error_message:    {e}")
        print(f"duration_s:       {duration:.1f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
