"""
@file sandbox_trainer.py

@description SandboxTrainer — BaseTrainer 子类，仅覆盖 build_loss()。
    通过 importlib 动态加载 sandbox_loss.py，实现零侵入集成。
@author Leizheng
@date 2026-03-20
@version 1.0.0

@changelog
  - 2026-03-20 Leizheng: v1.0.0 初始版本
  - 2026-03-24 Leizheng: v1.1.0 支持沙箱模型输出适配器
    - 可选用 sandbox_adapter 包装原始模型
    - 在不改原始 models/ 代码的前提下生成 loss_inputs
"""

import os
import sys
import inspect
import importlib.util
from typing import Any, Dict

# 将现有训练管线加入 sys.path
_PIPELINE_DIR = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'ocean-SR-training-masked')
_PIPELINE_DIR = os.path.abspath(_PIPELINE_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from trainers.base import BaseTrainer
from models import _model_dict
from sandbox_model_adapter import maybe_wrap_model_with_aux_adapter


class SandboxLossWrapper:
    """
    动态加载 sandbox_loss.py 并包装为 loss_fn 接口。
    每次实例化时从文件重新加载，无缓存问题。
    """

    def __init__(self, loss_file: str, size_average: bool = False, extra_kwargs_provider=None):
        self.loss_file = os.path.abspath(loss_file)
        self.size_average = size_average
        self.extra_kwargs_provider = extra_kwargs_provider
        self._load()

    def _load(self):
        spec = importlib.util.spec_from_file_location("sandbox_loss_module", self.loss_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, 'sandbox_loss'):
            raise AttributeError(f"sandbox_loss.py must define a 'sandbox_loss' function")
        self._fn = mod.sandbox_loss

    @staticmethod
    def _align_aux_value(value: Any, pred) -> Any:
        if not hasattr(pred, 'shape') or not hasattr(value, 'shape'):
            return value
        if getattr(pred, 'dim', lambda: 0)() != 4 or getattr(value, 'dim', lambda: 0)() != 4:
            return value

        h, w = pred.shape[1], pred.shape[2]
        if value.shape[1] >= h and value.shape[2] >= w:
            return value[:, :h, :w, :]
        if value.shape[2] >= h and value.shape[3] >= w:
            return value[:, :, :h, :w]
        return value

    def __call__(self, x, y, mask=None, **kwargs):
        merged_kwargs: Dict[str, Any] = {}
        if self.extra_kwargs_provider is not None:
            provided = self.extra_kwargs_provider()
            if provided:
                merged_kwargs.update(provided)
        merged_kwargs.update(kwargs)
        aligned_kwargs = {
            key: self._align_aux_value(value, x)
            for key, value in merged_kwargs.items()
        }
        return self._fn(x, y, mask=mask, **aligned_kwargs)


class SandboxTrainer(BaseTrainer):
    """
    仅覆盖 build_loss()，其余完全继承 BaseTrainer。
    """

    def __init__(self, args, loss_file: str = None):
        self._loss_file = loss_file or os.path.join(
            os.path.dirname(__file__), 'sandbox_loss.py'
        )
        super().__init__(args)

    @staticmethod
    def _filter_supported_model_init_kwargs(model_factory, raw_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if not raw_kwargs:
            return {}
        try:
            signature = inspect.signature(model_factory)
        except (TypeError, ValueError):
            return {}

        parameters = signature.parameters.values()
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
            return dict(raw_kwargs)

        supported_names = {
            param.name
            for param in parameters
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        return {
            key: value
            for key, value in raw_kwargs.items()
            if key in supported_names
        }

    def _resolve_model_init_kwargs(self) -> Dict[str, Any]:
        resolved: Dict[str, Any] = {}

        raw_extra_kwargs = self.args.get('sandbox_model_init_kwargs')
        if isinstance(raw_extra_kwargs, dict):
            resolved.update(raw_extra_kwargs)

        raw_model_extra_kwargs = self.model_args.get('sandbox_model_init_kwargs')
        if isinstance(raw_model_extra_kwargs, dict):
            resolved.update(raw_model_extra_kwargs)

        if 'output_aux_loss_inputs' in self.model_args:
            resolved.setdefault(
                'output_aux_loss_inputs',
                bool(self.model_args.get('output_aux_loss_inputs')),
            )

        return resolved

    def build_model(self, **kwargs):
        if self.model_name not in _model_dict:
            raise NotImplementedError("Model {} not implemented".format(self.model_name))

        model_factory = _model_dict[self.model_name]
        model_init_kwargs = self._resolve_model_init_kwargs()
        supported_kwargs = {}

        if callable(model_factory) and model_init_kwargs:
            supported_kwargs = self._filter_supported_model_init_kwargs(model_factory, model_init_kwargs)
            ignored_keys = sorted(set(model_init_kwargs) - set(supported_kwargs))
            if ignored_keys:
                self.main_log(
                    "Sandbox model init kwargs ignored for {}: {}".format(
                        self.model_name,
                        ", ".join(ignored_keys),
                    )
                )

        if callable(model_factory) and supported_kwargs:
            model = model_factory(self.model_args, **supported_kwargs)
            self.main_log(
                "Sandbox model init kwargs enabled for {}: {}".format(
                    self.model_name,
                    ", ".join(sorted(supported_kwargs.keys())),
                )
            )
        else:
            model = BaseTrainer.build_model(self, **kwargs)

        adapter_cfg = self.args.get('sandbox_adapter', {})
        if adapter_cfg and adapter_cfg.get('enabled', False):
            model = maybe_wrap_model_with_aux_adapter(
                model,
                self.model_args,
                adapter_cfg,
            )
            self.main_log(
                "Sandbox adapter enabled with heads: {}".format(
                    ", ".join(sorted(adapter_cfg.get('heads', {}).keys()))
                )
            )
        return model

    def _get_model_loss_inputs(self):
        model = self._unwrap()
        if hasattr(model, 'consume_latest_loss_inputs'):
            latest = model.consume_latest_loss_inputs()
            if latest:
                return latest
        return self._consume_model_loss_inputs()

    def build_loss(self, **kwargs):
        return SandboxLossWrapper(
            self._loss_file,
            size_average=False,
            extra_kwargs_provider=self._get_model_loss_inputs,
        )
