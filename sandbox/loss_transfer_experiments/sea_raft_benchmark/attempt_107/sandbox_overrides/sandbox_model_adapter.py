"""
@file sandbox_model_adapter.py
@description 沙箱模型输出适配器：在不改原始模型代码的前提下，为 loss 注入额外预测变量
@author Leizheng
@date 2026-03-24
@version 1.0.0
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn


def _split_base_output(model_output: Any) -> Tuple[Any, Dict[str, Any]]:
    loss_inputs: Dict[str, Any] = {}
    pred = model_output

    if isinstance(model_output, dict):
        pred = model_output.get("pred")
        if pred is None:
            pred = model_output.get("prediction")
        if pred is None:
            pred = model_output.get("output")
        if pred is None:
            raise KeyError("Sandbox adapter expects base model dict output to include 'pred'")
        raw_loss_inputs = model_output.get("loss_inputs", {})
        if raw_loss_inputs is None:
            raw_loss_inputs = {}
        if not isinstance(raw_loss_inputs, dict):
            raise TypeError("Base model output field 'loss_inputs' must be a dict or None")
        loss_inputs.update(raw_loss_inputs)
    elif isinstance(model_output, (tuple, list)):
        if not model_output:
            raise ValueError("Base model output tuple/list cannot be empty")
        pred = model_output[0]
        if len(model_output) > 1:
            extra = model_output[1] or {}
            if not isinstance(extra, dict):
                raise TypeError("Base model output tuple/list second element must be a dict or None")
            loss_inputs.update(extra)

    return pred, loss_inputs


def _apply_activation(x: torch.Tensor, name: str) -> torch.Tensor:
    name = (name or "none").lower()
    if name == "none":
        return x
    if name == "sigmoid":
        return torch.sigmoid(x)
    if name == "tanh":
        return torch.tanh(x)
    if name == "softplus":
        return torch.nn.functional.softplus(x)
    if name == "exp":
        return torch.exp(x)
    raise ValueError(f"Unsupported sandbox adapter activation: {name}")


def _apply_affine(x: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    scale = cfg.get("output_scale")
    shift = cfg.get("output_shift")
    if scale is not None:
        x = x * float(scale)
    if shift is not None:
        x = x + float(shift)
    return x


def _ensure_sequence(pred: Any):
    if isinstance(pred, (list, tuple)):
        return list(pred), True
    if not torch.is_tensor(pred):
        raise TypeError(f"Sandbox adapter requires tensor/list prediction, got {type(pred)}")
    return [pred], False


class AuxLossHeadAdapter(nn.Module):
    """
    Wrap a base SR model and cache latest loss_inputs.

    This attempt-scoped adapter performs a minimal but real structured output
    extension: it can expose per-stage `weight` and `log_b` aligned with each
    prediction tensor, instead of only attaching a flat auxiliary dict.
    """

    def __init__(
        self,
        base_model: nn.Module,
        pred_channels: int,
        heads_cfg: Dict[str, Dict[str, Any]],
        hidden_channels: int = 32,
    ):
        super().__init__()
        self.base_model = base_model
        self.pred_channels = pred_channels
        self.heads_cfg = heads_cfg
        self.hidden_channels = hidden_channels
        self.heads = nn.ModuleDict()
        self._latest_loss_inputs: Dict[str, Any] = {}

        for name, cfg in heads_cfg.items():
            out_channels = int(cfg.get("out_channels", 1))
            head_hidden = int(cfg.get("hidden_channels", hidden_channels))
            head = nn.Sequential(
                nn.Conv2d(pred_channels, head_hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(head_hidden, out_channels, kernel_size=1),
            )
            if bool(cfg.get("zero_init", False)) and hasattr(head[-1], "weight"):
                nn.init.zeros_(head[-1].weight)
            bias_init = cfg.get("bias_init")
            if bias_init is not None and hasattr(head[-1], "bias") and head[-1].bias is not None:
                nn.init.constant_(head[-1].bias, float(bias_init))
            self.heads[name] = head

    def _predict_aux_for_tensor(self, pred_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred_nchw = pred_tensor.permute(0, 3, 1, 2).contiguous()
        aux: Dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            cfg = self.heads_cfg.get(name, {})
            head_input = pred_nchw.detach() if bool(cfg.get("detach_input", False)) else pred_nchw
            value = head(head_input).permute(0, 2, 3, 1).contiguous()
            value = _apply_activation(value, str(cfg.get("activation", "none")))
            value = _apply_affine(value, cfg)
            aux[name] = value
        return aux

    def forward(self, x):
        base_output = self.base_model(x)
        pred, loss_inputs = _split_base_output(base_output)
        pred_list, is_sequence = _ensure_sequence(pred)

        staged_aux = {name: [] for name in self.heads.keys()}
        for pred_tensor in pred_list:
            if not torch.is_tensor(pred_tensor):
                raise TypeError("Each prediction stage must be a torch tensor")
            stage_aux = self._predict_aux_for_tensor(pred_tensor)
            for name, value in stage_aux.items():
                staged_aux[name].append(value)

        for name, values in staged_aux.items():
            if not values:
                continue
            loss_inputs[name] = values if is_sequence else values[0]

        if is_sequence:
            loss_inputs["pred_sequence"] = pred_list
            pred_to_return = pred
        else:
            pred_to_return = pred_list[0]

        self._latest_loss_inputs = loss_inputs
        return pred_to_return

    def peek_latest_loss_inputs(self) -> Dict[str, Any]:
        return dict(self._latest_loss_inputs)

    def consume_latest_loss_inputs(self) -> Dict[str, Any]:
        latest = dict(self._latest_loss_inputs)
        self._latest_loss_inputs = {}
        return latest


def maybe_wrap_model_with_aux_adapter(
    base_model: nn.Module,
    model_args: Dict[str, Any],
    adapter_cfg: Dict[str, Any],
) -> nn.Module:
    if not adapter_cfg or not adapter_cfg.get("enabled", False):
        return base_model

    heads_cfg = adapter_cfg.get("heads", {})
    if not isinstance(heads_cfg, dict) or not heads_cfg:
        raise ValueError("sandbox_adapter.enabled=true requires a non-empty heads dict")

    pred_channels = int(
        adapter_cfg.get(
            "pred_channels",
            model_args.get("out_channels", model_args.get("out_dim", 1)),
        )
    )
    hidden_channels = int(adapter_cfg.get("hidden_channels", 32))

    return AuxLossHeadAdapter(
        base_model=base_model,
        pred_channels=pred_channels,
        heads_cfg=heads_cfg,
        hidden_channels=hidden_channels,
    )
