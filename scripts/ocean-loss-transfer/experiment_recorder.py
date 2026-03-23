"""
@file experiment_recorder.py
@description 结构化记录实验结果
@author Leizheng
@contributors kongzhiquan
@date 2026-03-22
@version 1.1.0

@changelog
  - 2026-03-22 Leizheng: v1.0.0 initial version
  - 2026-03-23 kongzhiquan: v1.1.0 refine type annotations
"""

import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from _types import PatchSpec, ValidationResult


def record_trial(
    paper_slug: str,
    trial_id: int,
    patch_spec: PatchSpec,
    validation_results: Dict[str, ValidationResult],
    loss_file_path: Optional[str] = None
) -> str:
    """
    记录单次 trial 结果

    Returns:
        trial 目录路径
    """
    base_dir = Path(__file__).parent.parent.parent / 'sandbox/loss_transfer_experiments'
    trial_dir = base_dir / paper_slug / f'trial_{trial_id}'
    trial_dir.mkdir(parents=True, exist_ok=True)

    # 复制生成的 loss 文件
    if loss_file_path and Path(loss_file_path).exists():
        shutil.copy(loss_file_path, trial_dir / 'sandbox_loss.py')

    # 构建结果
    result = {
        'trial_id': trial_id,
        'timestamp': datetime.now().isoformat(),
        'patch_spec': patch_spec,
        'validation': validation_results
    }

    # 保存
    (trial_dir / 'result.yaml').write_text(yaml.dump(result, allow_unicode=True))

    return str(trial_dir)
