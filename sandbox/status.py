"""
status.py — 一键查看所有 worker 的最新实验状态。
用法: python status.py
"""

import os
import re
import subprocess

SANDBOX = os.path.dirname(os.path.abspath(__file__))

MODELS = [
    ('SwinIR',  1, 'run_SwinIR.log'),
    ('FNO2d',   2, 'run_FNO2d.log'),
    ('EDSR',    3, 'run_EDSR.log'),
    ('UNet2d',  4, 'run_UNet2d.log'),
]


def parse_log(log_path):
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        content = f.read()
    def get(key):
        m = re.search(rf'^{key}:\s+([\S]+)', content, re.MULTILINE)
        return m.group(1) if m else None
    return {
        'val_ssim':  get('val_ssim'),
        'val_psnr':  get('val_psnr'),
        'test_ssim': get('test_ssim'),
        'duration':  get('duration_s'),
        'running':   get('val_ssim') is None,
    }


def gpu_mem():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.free',
             '--format=csv,noheader'], text=True)
        result = {}
        for line in out.strip().splitlines():
            idx, used, free = [x.strip().replace(' MiB','') for x in line.split(',')]
            result[int(idx)] = (int(used), int(free))
        return result
    except Exception:
        return {}


def main():
    mem = gpu_mem()
    print(f"\n{'='*62}")
    print(f"  Sandbox Multi-Model Status")
    print(f"{'='*62}")
    print(f"  {'Model':<10} {'GPU':>4} {'VRAM used':>12} {'val_SSIM':>10} {'val_PSNR':>10} {'test_SSIM':>10} {'Dur':>6}")
    print(f"  {'-'*58}")

    for model, gpu, logfile in MODELS:
        log_path = os.path.join(SANDBOX, logfile)
        info = parse_log(log_path)
        used, free = mem.get(gpu, ('?', '?'))
        vram = f"{used}MiB" if isinstance(used, int) else '?'

        if info is None:
            print(f"  {model:<10} {gpu:>4} {vram:>12} {'no log':>10}")
        elif info['running']:
            print(f"  {model:<10} {gpu:>4} {vram:>12} {'running...':>10}")
        else:
            ssim  = info['val_ssim']  or '-'
            psnr  = info['val_psnr']  or '-'
            tssim = info['test_ssim'] or '-'
            dur   = info['duration']  or '-'
            print(f"  {model:<10} {gpu:>4} {vram:>12} {ssim:>10} {psnr:>10} {tssim:>10} {dur:>6}s")

    print(f"{'='*62}\n")

    # 读 results.tsv 最近 5 条
    tsv = os.path.join(SANDBOX, 'results.tsv')
    if os.path.exists(tsv):
        with open(tsv) as f:
            lines = f.readlines()
        print("  Recent results (last 5):")
        for line in lines[-5:]:
            print(f"  {line.rstrip()}")
        print()


if __name__ == '__main__':
    main()
