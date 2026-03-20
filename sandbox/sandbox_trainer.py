"""
@file sandbox_trainer.py

@description SandboxTrainer — BaseTrainer 子类，仅覆盖 build_loss()。
    通过 importlib 动态加载 sandbox_loss.py，实现零侵入集成。
@author Leizheng
@date 2026-03-20
@version 1.0.0

@changelog
  - 2026-03-20 Leizheng: v1.0.0 初始版本
"""

import os
import sys
import importlib.util

# 将现有训练管线加入 sys.path
_PIPELINE_DIR = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'ocean-SR-training-masked')
_PIPELINE_DIR = os.path.abspath(_PIPELINE_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from trainers.base import BaseTrainer


class SandboxLossWrapper:
    """
    动态加载 sandbox_loss.py 并包装为 loss_fn 接口。
    每次实例化时从文件重新加载，无缓存问题。
    """

    def __init__(self, loss_file: str, size_average: bool = False):
        self.loss_file = os.path.abspath(loss_file)
        self.size_average = size_average
        self._load()

    def _load(self):
        spec = importlib.util.spec_from_file_location("sandbox_loss_module", self.loss_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, 'sandbox_loss'):
            raise AttributeError(f"sandbox_loss.py must define a 'sandbox_loss' function")
        self._fn = mod.sandbox_loss

    def __call__(self, x, y, mask=None, **kwargs):
        return self._fn(x, y, mask=mask, **kwargs)


class SandboxTrainer(BaseTrainer):
    """
    仅覆盖 build_loss()，其余完全继承 BaseTrainer。
    """

    def __init__(self, args, loss_file: str = None):
        self._loss_file = loss_file or os.path.join(
            os.path.dirname(__file__), 'sandbox_loss.py'
        )
        super().__init__(args)

    def build_loss(self, **kwargs):
        return SandboxLossWrapper(self._loss_file, size_average=False)
