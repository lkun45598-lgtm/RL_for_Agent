"""
@file _utils.py
@description 共享工具函数 - 训练事件解析等
@author Leizheng
@date 2026-03-24
@version 1.0.0

@changelog
  - 2026-03-24 Leizheng: v1.0.0 initial version - parse_training_events for timeout recovery
"""

import re
import json
from typing import List

from loss_transfer.common._types import EpochMetric, TrainingCurve, TrainingTrend

EVENT_RE = re.compile(r'__event__(\{.*?\})__event__')


def parse_training_events(stdout: str) -> TrainingCurve:
    """
    解析训练 stdout 中的 __event__ 标记，提取逐 epoch 指标。

    支持的事件类型:
    - epoch_train: {epoch, metrics: {train_loss}}
    - epoch_valid: {epoch, metrics: {valid_loss, ssim, psnr, rmse}}
    - final_valid: {metrics: {ssim, psnr, rmse, valid_loss}}
    - training_start: {total_epochs}
    """
    epochs: List[EpochMetric] = []
    total_expected = 15  # default

    for match in EVENT_RE.finditer(stdout):
        try:
            event = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue

        event_type = event.get('event', '')
        metrics = event.get('metrics', {})

        if event_type == 'training_start':
            total_expected = event.get('total_epochs', total_expected)

        elif event_type == 'epoch_train':
            epoch_num = event.get('epoch', -1)
            existing = next((e for e in epochs if e.get('epoch') == epoch_num), None)
            if existing:
                existing['train_loss'] = metrics.get('train_loss')
            else:
                entry: EpochMetric = {'epoch': epoch_num}
                if 'train_loss' in metrics:
                    entry['train_loss'] = metrics['train_loss']
                epochs.append(entry)

        elif event_type in ('epoch_valid', 'final_valid'):
            epoch_num = event.get('epoch', len(epochs))
            existing = next((e for e in epochs if e.get('epoch') == epoch_num), None)
            if existing:
                for key in ('valid_loss', 'ssim', 'psnr', 'rmse'):
                    if key in metrics:
                        existing[key] = metrics[key]  # type: ignore[literal-required]
            else:
                entry = {'epoch': epoch_num}
                for key in ('valid_loss', 'ssim', 'psnr', 'rmse'):
                    if key in metrics:
                        entry[key] = metrics[key]  # type: ignore[literal-required]
                epochs.append(entry)

    # 计算训练趋势
    trend: TrainingTrend = 'insufficient_data'
    valid_ssims = [e['ssim'] for e in epochs if e.get('ssim') is not None]
    if len(valid_ssims) >= 3:
        diffs = [valid_ssims[i + 1] - valid_ssims[i] for i in range(len(valid_ssims) - 1)]
        pos = sum(1 for d in diffs if d > 0.001)
        neg = sum(1 for d in diffs if d < -0.001)
        if pos > neg:
            trend = 'improving'
        elif neg > pos:
            trend = 'degrading'
        else:
            trend = 'unstable'
    elif len(valid_ssims) == 2:
        trend = 'improving' if valid_ssims[1] > valid_ssims[0] + 0.001 else \
                'degrading' if valid_ssims[1] < valid_ssims[0] - 0.001 else 'unstable'

    last_epoch = epochs[-1].get('epoch', -1) if epochs else -1

    return {
        'epochs': epochs,
        'trend': trend,
        'last_epoch': last_epoch,
        'total_expected_epochs': total_expected,
    }
