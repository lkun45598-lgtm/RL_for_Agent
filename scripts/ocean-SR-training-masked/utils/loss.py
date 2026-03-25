"""
Loss functions for ocean SR training (masked version).

@author Leizheng
@contributors Leizheng
@date 2026-02-06
@version 2.1.0

@changelog
  - 2026-02-07 Leizheng: v2.1.0 添加结构化日志支持
    - LossRecord 新增 to_json_event() 方法，输出 JSON 格式日志
    - 支持事件类型标记，便于日志解析
  - 2026-02-06 Leizheng: v2.0.0 添加 MaskedLpLoss
    - 支持显式 mask 参数，只在海洋格点上计算 loss
    - 求平均时分母 = 海洋格点数（排除陆地格点）
  - 原始版本: v1.0.0
"""

import inspect
import json
import torch
import torch.nn.functional as F
from time import time
import torch.distributed as dist
from typing import Any, Callable, Dict, Optional

_loss_dict = {

}


def _reduce_batch(values: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def _align_mask(mask: Optional[torch.Tensor], ref: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.dim() != 4 or ref.dim() != 4:
        return mask

    aligned = mask
    if aligned.shape[0] != ref.shape[0]:
        if aligned.shape[0] != 1:
            raise ValueError(
                f"Mask batch size {aligned.shape[0]} does not match reference batch size {ref.shape[0]}"
            )
        aligned = aligned.expand(ref.shape[0], aligned.shape[1], aligned.shape[2], aligned.shape[3])

    if aligned.shape[1] != ref.shape[1] or aligned.shape[2] != ref.shape[2]:
        aligned = aligned.permute(0, 3, 1, 2).float()
        aligned = F.interpolate(aligned, size=(ref.shape[1], ref.shape[2]), mode="nearest")
        aligned = aligned.permute(0, 2, 3, 1)

    if aligned.shape[-1] != ref.shape[-1]:
        if aligned.shape[-1] != 1:
            raise ValueError(
                f"Mask channel size {aligned.shape[-1]} does not match reference channel size {ref.shape[-1]}"
            )
        aligned = aligned.expand(aligned.shape[0], aligned.shape[1], aligned.shape[2], ref.shape[-1])

    return aligned.bool()


def _masked_rel_lp(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    p: int = 2,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    batch = pred.size(0)
    aligned_mask = _align_mask(mask, pred)
    if aligned_mask is None:
        mask_flat = torch.ones_like(pred.reshape(batch, -1))
    else:
        mask_flat = aligned_mask.reshape(batch, -1).float()

    pred_flat = pred.reshape(batch, -1)
    target_flat = target.reshape(batch, -1)
    diff = (pred_flat - target_flat) * mask_flat
    target_masked = target_flat * mask_flat

    diff_norms = torch.norm(diff, p, dim=1)
    target_norms = torch.norm(target_masked, p, dim=1).clamp(min=float(eps))
    rel_errors = diff_norms / target_norms
    return _reduce_batch(rel_errors, reduction=reduction)


def masked_rel_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-8,
    reduction: str = "mean",
    **kwargs,
) -> torch.Tensor:
    return _masked_rel_lp(pred, target, mask=mask, p=1, eps=eps, reduction=reduction)


def masked_rel_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-8,
    reduction: str = "mean",
    **kwargs,
) -> torch.Tensor:
    return _masked_rel_lp(pred, target, mask=mask, p=2, eps=eps, reduction=reduction)


def masked_abs_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    reduction: str = "mean",
    **kwargs,
) -> torch.Tensor:
    diff = (pred - target).abs()
    if mask is not None:
        valid = _align_mask(mask, diff).float()
        per_item = (diff * valid).reshape(diff.size(0), -1).sum(dim=1) / valid.reshape(diff.size(0), -1).sum(dim=1).clamp(min=1.0)
        return _reduce_batch(per_item, reduction=reduction)
    per_item = diff.reshape(diff.size(0), -1).mean(dim=1)
    return _reduce_batch(per_item, reduction=reduction)


def masked_abs_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    reduction: str = "mean",
    **kwargs,
) -> torch.Tensor:
    diff = (pred - target).pow(2)
    if mask is not None:
        valid = _align_mask(mask, diff).float()
        per_item = (diff * valid).reshape(diff.size(0), -1).sum(dim=1) / valid.reshape(diff.size(0), -1).sum(dim=1).clamp(min=1.0)
        return _reduce_batch(per_item, reduction=reduction)
    per_item = diff.reshape(diff.size(0), -1).mean(dim=1)
    return _reduce_batch(per_item, reduction=reduction)


def _align_aux_tensor(value: Optional[torch.Tensor], pred: torch.Tensor, name: str) -> torch.Tensor:
    if value is None:
        raise ValueError(f"Missing required auxiliary loss input: {name}")
    if not torch.is_tensor(value):
        raise TypeError(f"Auxiliary loss input {name} must be a tensor, got {type(value)}")
    if value.dim() != 4:
        raise ValueError(f"Auxiliary loss input {name} must be BHWC 4D tensor, got shape {tuple(value.shape)}")
    if value.shape[0] != pred.shape[0]:
        if value.shape[0] != 1:
            raise ValueError(
                f"Auxiliary loss input {name} batch mismatch: {tuple(value.shape)} vs {tuple(pred.shape)}"
            )
        value = value.expand(pred.shape[0], value.shape[1], value.shape[2], value.shape[3])
    if value.shape[1] == pred.shape[1] and value.shape[2] == pred.shape[2]:
        return value.to(device=pred.device, dtype=pred.dtype)
    tensor = value.permute(0, 3, 1, 2).to(device=pred.device, dtype=pred.dtype)
    tensor = F.interpolate(tensor, size=(pred.shape[1], pred.shape[2]), mode="bilinear", align_corners=False)
    return tensor.permute(0, 2, 3, 1).contiguous()


def _masked_channel_sum_mean(value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    per_pixel = value.sum(dim=-1, keepdim=True)
    if mask is not None:
        valid = _align_mask(mask, per_pixel).float()
        return (per_pixel * valid).sum() / valid.sum().clamp(min=1.0)
    return per_pixel.mean()


def masked_mixture_laplace_nll(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    weight: Optional[torch.Tensor] = None,
    log_b: Optional[torch.Tensor] = None,
    use_var: bool = True,
    var_min: float = 0.0,
    var_max: float = 10.0,
    weight_clip: float = 12.0,
    log_b_clip: float = 6.0,
    aux_reg: float = 0.0,
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    pred_f = pred.float()
    target_f = target.float()

    if not use_var:
        return masked_abs_l1(pred_f, target_f, mask=mask)

    weight_t = _align_aux_tensor(weight, pred, "weight")
    log_b_t = _align_aux_tensor(log_b, pred, "log_b")

    weight_logits = weight_t.float().clamp(min=-abs(float(weight_clip)), max=abs(float(weight_clip)))

    positive_cap = max(float(var_max), 1e-3)
    negative_floor = float(var_min)
    if negative_floor >= 0.0:
        negative_floor = -positive_cap

    if log_b_t.shape[-1] == 1:
        stabilized_log_b = log_b_t.float().clamp(min=negative_floor, max=positive_cap)
    else:
        positive_branch = log_b_t[..., :1].float().clamp(min=0.0, max=positive_cap)
        negative_branch = log_b_t[..., 1:].float().clamp(min=negative_floor, max=0.0)
        stabilized_log_b = torch.cat([positive_branch, negative_branch], dim=-1)
    stabilized_log_b = stabilized_log_b.clamp(min=-abs(float(log_b_clip)), max=abs(float(log_b_clip)))

    residual_abs = (pred_f - target_f).abs().unsqueeze(-1)
    weight_logits = weight_logits.unsqueeze(-2)
    stabilized_log_b = stabilized_log_b.unsqueeze(-2)
    inv_b = torch.exp((-stabilized_log_b).clamp(max=12.0)).clamp(max=1.0 / max(float(eps), 1e-6))
    log_weight = F.log_softmax(weight_logits.squeeze(-2), dim=-1).unsqueeze(-2)
    nll = -torch.logsumexp(
        log_weight - torch.log(pred_f.new_tensor(2.0)) - stabilized_log_b - residual_abs * inv_b,
        dim=-1,
    )
    nll = torch.nan_to_num(nll, nan=1e4, posinf=1e4, neginf=0.0)

    loss = _masked_channel_sum_mean(nll, mask=mask)
    if aux_reg > 0.0:
        loss = loss + float(aux_reg) * (weight_logits.square().mean() + stabilized_log_b.square().mean())
    return loss


_loss_dict.update(
    {
        "l1": masked_abs_l1,
        "l2": masked_abs_mse,
        "masked_abs_l1": masked_abs_l1,
        "masked_abs_mse": masked_abs_mse,
        "masked_rel_l1": masked_rel_l1,
        "masked_rel_l2": masked_rel_l2,
        "masked_mixture_laplace_nll": masked_mixture_laplace_nll,
    }
)


def _resolve_binding(binding: Any, context: Dict[str, Any]) -> Any:
    if not isinstance(binding, str):
        return binding
    if binding in context:
        return context[binding]

    current: Any = context
    for token in binding.split("."):
        if isinstance(current, dict) and token in current:
            current = current[token]
            continue
        raise KeyError(f"Unable to resolve loss binding: {binding}")
    return current


class SpecDrivenLoss:
    """Build a loss from a registry recipe plus params/bindings."""

    def __init__(
        self,
        recipe: str,
        params: Optional[Dict[str, Any]] = None,
        bindings: Optional[Dict[str, Any]] = None,
        extra_kwargs_provider: Optional[Callable[[], Dict[str, Any]]] = None,
    ):
        if recipe not in _loss_dict:
            raise KeyError(f"Unknown loss recipe: {recipe}")
        self.recipe = recipe
        self.recipe_fn = _loss_dict[recipe]
        self.params = dict(params or {})
        self.bindings = dict(bindings or {})
        self.extra_kwargs_provider = extra_kwargs_provider
        self.signature = inspect.signature(self.recipe_fn)
        self.accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in self.signature.parameters.values()
        )

    def _build_context(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
        runtime_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "pred": pred,
            "target": target,
            "mask": mask,
        }
        if self.extra_kwargs_provider is not None:
            extra = self.extra_kwargs_provider() or {}
            if not isinstance(extra, dict):
                raise TypeError("extra_kwargs_provider must return a dict")
            context["loss_inputs"] = extra
            for key, value in extra.items():
                context.setdefault(key, value)
        context.update(runtime_kwargs)
        return context

    def __call__(self, pred, target, mask=None, **kwargs):
        context = self._build_context(pred, target, mask, kwargs)
        call_kwargs: Dict[str, Any] = {}

        for key, value in self.bindings.items():
            call_kwargs[key] = _resolve_binding(value, context)

        for key in ("pred", "target", "mask"):
            if key not in call_kwargs:
                call_kwargs[key] = context[key]

        for key, value in self.params.items():
            call_kwargs.setdefault(key, value)

        if self.accepts_kwargs:
            for key, value in context.items():
                call_kwargs.setdefault(key, value)
        else:
            call_kwargs = {
                key: value
                for key, value in call_kwargs.items()
                if key in self.signature.parameters
            }

        return self.recipe_fn(**call_kwargs)


class CompositeSpecLoss:
    """Weighted sum of multiple spec-driven loss terms."""

    def __init__(self, terms, extra_kwargs_provider: Optional[Callable[[], Dict[str, Any]]] = None):
        self.extra_kwargs_provider = extra_kwargs_provider
        self.terms = []
        for idx, term_cfg in enumerate(terms):
            if not isinstance(term_cfg, dict):
                raise TypeError(f"loss.terms[{idx}] must be a dict")
            recipe = term_cfg.get("recipe")
            if not isinstance(recipe, str) or not recipe.strip():
                raise ValueError(f"loss.terms[{idx}].recipe must be a non-empty string")
            weight = float(term_cfg.get("weight", 1.0))
            term = SpecDrivenLoss(
                recipe=recipe.strip(),
                params=term_cfg.get("params", {}),
                bindings=term_cfg.get("bindings", {}),
                extra_kwargs_provider=None,
            )
            self.terms.append((weight, recipe, term))

    def __call__(self, pred, target, mask=None, **kwargs):
        shared_kwargs = dict(kwargs)
        if self.extra_kwargs_provider is not None:
            extra = self.extra_kwargs_provider() or {}
            if not isinstance(extra, dict):
                raise TypeError("extra_kwargs_provider must return a dict")
            shared_kwargs.setdefault("loss_inputs", extra)
            for key, value in extra.items():
                shared_kwargs.setdefault(key, value)

        total = pred.new_tensor(0.0)
        for weight, _recipe, term in self.terms:
            if weight == 0.0:
                continue
            total = total + float(weight) * term(pred, target, mask=mask, **shared_kwargs)
        return total


def build_loss_from_config(
    loss_cfg: Optional[Dict[str, Any]],
    *,
    extra_kwargs_provider: Optional[Callable[[], Dict[str, Any]]] = None,
):
    if not loss_cfg:
        return MaskedLpLoss(size_average=False)

    if not isinstance(loss_cfg, dict):
        raise TypeError("loss config must be a dict when provided")

    mode = str(loss_cfg.get("mode", loss_cfg.get("type", ""))).strip().lower()
    if mode in {"", "default", "masked_lp"} and "recipe" not in loss_cfg and "terms" not in loss_cfg:
        return MaskedLpLoss(
            p=int(loss_cfg.get("p", 2)),
            reduction=bool(loss_cfg.get("reduction", True)),
            size_average=bool(loss_cfg.get("size_average", False)),
        )

    if "terms" in loss_cfg:
        terms = loss_cfg.get("terms")
        if not isinstance(terms, list) or not terms:
            raise ValueError("loss.terms must be a non-empty list")
        return CompositeSpecLoss(terms, extra_kwargs_provider=extra_kwargs_provider)

    recipe = loss_cfg.get("recipe")
    if not isinstance(recipe, str) or not recipe.strip():
        raise ValueError("loss.recipe must be a non-empty string")
    return SpecDrivenLoss(
        recipe=recipe.strip(),
        params=loss_cfg.get("params", {}),
        bindings=loss_cfg.get("bindings", {}),
        extra_kwargs_provider=extra_kwargs_provider,
    )


class CompositeLoss:
    """
    组合损失：传入 {name: weight}，自动求和并返回总损失与分项日志
    """
    def __init__(self, spec: dict[str, float]):  # e.g. {"l1":1.0,"l2":0.1,"physics":0.5}
        self.spec = spec
        self.loss_list = ["total_loss", "l2", "l1"]
        self.init_record()

    def __call__(self, pred, target, *, batch_size: int | None = None, **batch):
        logs = {}
        total = 0.0
        for name, w in self.spec.items():
            if w == 0:
                continue
            fn = _loss_dict[name]
            val = fn(pred, target, **batch)  # 标量（已mean）
            total = total + w * val
            logs[name] = float(val.detach().item())
        logs["loss_total"] = float(total.detach().item())
        if self.record is not None and batch_size is not None:
            self.record.update(logs, n=batch_size)
        return total  # 用于 backward

    def init_record(self):
        self.record = LossRecord(self.loss_list)

class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    支持 NaN 掩码 - 修改版
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        
        # 创建掩码：标记有效值（非 NaN）
        mask = ~torch.isnan(y)
        
        # 如果全是 NaN，返回 0
        if not mask.any():
            return torch.tensor(0.0, device=x.device)
        
        # 将 NaN 替换为 0（不影响计算，因为会被掩码过滤）
        x_masked = torch.where(mask, x, torch.zeros_like(x))
        y_masked = torch.where(mask, y, torch.zeros_like(y))
        
        # 展平并只保留有效值
        x_flat = x_masked.reshape(num_examples, -1)
        y_flat = y_masked.reshape(num_examples, -1)
        mask_flat = mask.reshape(num_examples, -1)
        
        # 对每个样本计算相对误差
        diff_norms = []
        y_norms = []
        
        for i in range(num_examples):
            valid_mask = mask_flat[i]
            if valid_mask.sum() == 0:
                # 如果该样本全是 NaN，跳过
                continue
            
            x_valid = x_flat[i][valid_mask]
            y_valid = y_flat[i][valid_mask]
            
            diff_norm = torch.norm(x_valid - y_valid, self.p)
            y_norm = torch.norm(y_valid, self.p)
            
            diff_norms.append(diff_norm)
            y_norms.append(y_norm)
        
        if len(diff_norms) == 0:
            return torch.tensor(0.0, device=x.device)
        
        diff_norms = torch.stack(diff_norms)
        y_norms = torch.stack(y_norms)
        
        # 避免除零
        y_norms = torch.clamp(y_norms, min=1e-8)
        
        rel_errors = diff_norms / y_norms
        
        if self.reduction:
            if self.size_average:
                return torch.mean(rel_errors)
            else:
                return torch.sum(rel_errors)
        
        return rel_errors

    def __call__(self, x, y, **kwargs):
        return self.rel(x, y)


class MaskedLpLoss(object):
    """
    带显式 mask 的 Lp Loss。
    求平均时分母只算海洋格点数（不算陆地格点）。

    与 LpLoss 的区别：
    - LpLoss 通过检测 NaN 来推断 mask
    - MaskedLpLoss 接受显式 mask 参数（数据中 NaN 已被填充为 0）
    """
    def __init__(self, p=2, reduction=True, size_average=True):
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def __call__(self, x, y, mask=None, **kwargs):
        reduction = "none"
        if self.reduction:
            reduction = "mean" if self.size_average else "sum"
        return _masked_rel_lp(x, y, mask=mask, p=self.p, eps=1e-8, reduction=reduction)


class AverageRecord(object):
    """Computes and stores the average and current values for multidimensional data"""

    def __init__(self):
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class LossRecord:
    """
    A class for keeping track of loss values during training.

    Attributes:
        start_time (float): The time when the LossRecord was created.
        loss_list (list): A list of loss names to track.
        loss_dict (dict): A dictionary mapping each loss name to an AverageRecord object.
    """

    def __init__(self, loss_list):
        self.start_time = time()
        self.loss_list = loss_list
        self.loss_dict = {loss: AverageRecord() for loss in self.loss_list}
    
    def update(self, update_dict, n=1):
        for key, value in update_dict.items():
            self.loss_dict[key].update(value, n)
    
    def elapsed(self) -> float:
        """返回自创建以来经过的秒数"""
        return time() - self.start_time

    def format_metrics(self):
        parts = ["{}: {:.4f}".format(k, self.loss_dict[k].avg) for k in self.loss_list]
        parts.append("t={:.1f}s".format(self.elapsed()))
        return "  ".join(parts)

    def to_dict(self):
        return {
            loss: self.loss_dict[loss].avg for loss in self.loss_list
        }
        
    def dist_reduce(self, device=None):
        if not (dist.is_available() and dist.is_initialized()):
            return

        device = device if device is not None else (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available() else torch.device("cpu")
        )

        for loss in self.loss_list:
            # 打包 sum 与 count，一次 all_reduce 两次也行，这里演示两次更直观
            t_sum = torch.tensor(self.loss_dict[loss].sum, dtype=torch.float32, device=device)
            t_cnt = torch.tensor(self.loss_dict[loss].count, dtype=torch.float32, device=device)

            dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_cnt, op=dist.ReduceOp.SUM)

            global_sum = t_sum.item()
            global_cnt = t_cnt.item()

            # 防止除零（极端情况：全局没有任何样本）
            if global_cnt > 0:
                self.loss_dict[loss].sum = global_sum
                self.loss_dict[loss].count = global_cnt
                self.loss_dict[loss].avg = global_sum / global_cnt
            else:
                # 保持为 0，或设为 NaN/Inf 按需处理
                self.loss_dict[loss].sum = 0.0
                self.loss_dict[loss].count = 0
                self.loss_dict[loss].avg = 0.0
    
    def to_json_event(self, event_type: str, **extra_fields) -> str:
        """
        生成结构化 JSON 日志事件。

        Args:
            event_type: 事件类型，如 "epoch_train", "epoch_valid", "test_metrics"
            **extra_fields: 额外字段，如 epoch, lr, best_epoch 等

        Returns:
            JSON 格式字符串，包含 __event__ 标记便于解析
        """
        event_data = {
            "event": event_type,
            "metrics": {loss: self.loss_dict[loss].avg for loss in self.loss_list},
            "elapsed_time": time() - self.start_time,
        }
        event_data.update(extra_fields)
        return f"__event__{json.dumps(event_data, ensure_ascii=False)}__event__"

    def __str__(self):
        return self.format_metrics()

    def __repr__(self):
        return self.loss_dict[self.loss_list[0]].avg
