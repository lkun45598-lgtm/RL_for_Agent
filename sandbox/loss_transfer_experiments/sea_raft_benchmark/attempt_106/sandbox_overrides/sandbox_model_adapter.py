import math
import torch
import torch.nn.functional as F


class SandboxModelAdapter(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def _to_bhwc(self, x):
        if x is None:
            return None
        if not torch.is_tensor(x):
            return x
        if x.ndim != 4:
            return x
        if x.shape[1] <= 8:
            return x.permute(0, 2, 3, 1)
        return x

    def _extract_from_info(self, info):
        info = self._to_bhwc(info)
        if info is None or info.ndim != 4:
            return None, None
        channels = info.shape[-1]
        if channels >= 4:
            weight = info[..., :2]
            raw_b = info[..., 2:4]
        elif channels == 3:
            weight = info[..., :2]
            raw_b = info[..., 2:3]
        elif channels == 2:
            weight = info
            raw_b = torch.zeros_like(info[..., :1])
        else:
            weight = torch.cat([info, torch.zeros_like(info)], dim=-1)
            raw_b = torch.zeros_like(info)

        log_b_first = torch.clamp(raw_b[..., :1], min=0.0, max=0.0)
        if raw_b.shape[-1] >= 2:
            log_b_second = torch.clamp(raw_b[..., 1:2], min=0.0, max=10.0)
        else:
            log_b_second = torch.clamp(raw_b[..., :1], min=0.0, max=10.0)
        log_b = torch.cat([log_b_first, log_b_second], dim=-1)
        return weight, log_b

    def _normalize_output(self, output):
        if torch.is_tensor(output):
            pred = self._to_bhwc(output)
            return {
                "pred": pred,
                "loss_inputs": {
                    "weight": torch.zeros_like(pred[..., :2]),
                    "log_b": torch.zeros_like(pred[..., :2]),
                },
            }

        if isinstance(output, (list, tuple)):
            preds = [self._to_bhwc(item) for item in output]
            zero = torch.zeros_like(preds[-1][..., :2])
            return {
                "pred": preds,
                "loss_inputs": {
                    "weight": [zero for _ in preds],
                    "log_b": [zero for _ in preds],
                },
            }

        if isinstance(output, dict):
            pred = output.get("pred", output.get("flow", output.get("final", None)))
            info = output.get("info", output.get("info_predictions", None))
            loss_inputs = output.get("loss_inputs", {})

            pred_seq = pred if isinstance(pred, (list, tuple)) else [pred]
            pred_seq = [self._to_bhwc(item) for item in pred_seq]

            weight = loss_inputs.get("weight", None)
            log_b = loss_inputs.get("log_b", None)

            if (weight is None or log_b is None) and info is not None:
                info_seq = info if isinstance(info, (list, tuple)) else [info]
                weight_seq = []
                log_b_seq = []
                for info_item in info_seq:
                    weight_item, log_b_item = self._extract_from_info(info_item)
                    weight_seq.append(weight_item)
                    log_b_seq.append(log_b_item)
                if weight is None:
                    weight = weight_seq
                if log_b is None:
                    log_b = log_b_seq

            if weight is None:
                zero = torch.zeros_like(pred_seq[-1][..., :2])
                weight = [zero for _ in pred_seq] if len(pred_seq) > 1 else zero
            if log_b is None:
                zero = torch.zeros_like(pred_seq[-1][..., :2])
                log_b = [zero for _ in pred_seq] if len(pred_seq) > 1 else zero

            output["pred"] = pred_seq if isinstance(pred, (list, tuple)) else pred_seq[-1]
            output["loss_inputs"] = {"weight": weight, "log_b": log_b}
            return output

        return output

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        return self._normalize_output(output)
