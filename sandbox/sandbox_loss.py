import torch
import torch.nn.functional as F


def _to_nchw(x: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.dim() != 4:
        raise ValueError(f"{name} must have 4 dims, got {tuple(x.shape)}")

    if x.shape[1] <= 8:
        return x.contiguous()
    if x.shape[-1] <= 8:
        return x.permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"{name} must be NCHW or NHWC, got {tuple(x.shape)}")


def _odd_int(value, default: int) -> int:
    if value is None:
        value = default
    value = int(value)
    if value <= 0:
        raise ValueError(f"kernel size must be positive, got {value}")
    if value % 2 == 0:
        value += 1
    return value


def _normalize_probs(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = torch.clamp(x, min=0.0)
    denom = torch.clamp(x.sum(dim=-1, keepdim=True), min=eps)
    return x / denom


def _ensure_mask_4d(mask: torch.Tensor, pred_nchw: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor")
    if mask.dim() != 4:
        raise ValueError(f"mask must have 4 dims, got {tuple(mask.shape)}")

    batch = pred_nchw.shape[0]
    height = pred_nchw.shape[2]
    width = pred_nchw.shape[3]
    if mask.shape[0] != batch:
        raise ValueError(f"mask batch must match pred, got {tuple(mask.shape)} vs {tuple(pred_nchw.shape)}")

    if mask.shape[1] == 1 and mask.shape[2] == height and mask.shape[3] == width:
        return mask.contiguous()
    if mask.shape[-1] == 1 and mask.shape[1] == height and mask.shape[2] == width:
        return mask.permute(0, 3, 1, 2).contiguous()
    if mask.shape[1] == height and mask.shape[2] == width and mask.shape[3] == 1:
        return mask.permute(0, 3, 1, 2).contiguous()

    raise ValueError(f"mask must be NCHW or NHWC with singleton channel, got {tuple(mask.shape)}")


def _derive_edge_mask(
    target: torch.Tensor,
    threshold: float = 20.0,
    data_range: float = 255.0,
    dilate_kernel: int = 0,
) -> torch.Tensor:
    gray = target.mean(dim=1, keepdim=True) if target.shape[1] > 1 else target
    gray_scaled = gray * float(data_range)

    lap_kernel = target.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 3, 3)
    edge = F.conv2d(gray_scaled, lap_kernel, padding=1).abs()
    mask = (edge > float(threshold)).to(target.dtype)

    if dilate_kernel and dilate_kernel > 1:
        dilate_kernel = _odd_int(dilate_kernel, dilate_kernel)
        pad = dilate_kernel // 2
        mask = F.max_pool2d(mask, kernel_size=dilate_kernel, stride=1, padding=pad)
        mask = (mask > 0).to(target.dtype)
    return mask


def _extract_patch_bank(img: torch.Tensor, kernel_size_window: int) -> torch.Tensor:
    pad = kernel_size_window // 2
    patches = F.unfold(img, kernel_size=kernel_size_window, padding=pad, stride=1)
    return patches.transpose(1, 2).contiguous()


def _masked_ssl_single(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    kernel_size_search: int,
    kernel_size_window: int,
    sigma: float,
    alpha: float,
    sample_stride: int,
    max_masked_pixels: int,
    chunk_size: int,
    eps: float,
) -> torch.Tensor:
    batch_losses = []
    radius = kernel_size_search // 2

    for batch_index in range(pred.shape[0]):
        pred_i = pred[batch_index : batch_index + 1]
        target_i = target[batch_index : batch_index + 1]
        mask_i = mask[batch_index : batch_index + 1, :1]

        valid_coords = torch.nonzero(mask_i[0, 0] > 0.5, as_tuple=False)
        if valid_coords.numel() == 0:
            continue

        if sample_stride > 1:
            valid_coords = valid_coords[::sample_stride]
        if max_masked_pixels > 0 and valid_coords.shape[0] > max_masked_pixels:
            valid_coords = valid_coords[:max_masked_pixels]
        if valid_coords.numel() == 0:
            continue

        patch_bank_pred = _extract_patch_bank(pred_i, kernel_size_window)[0]
        patch_bank_target = _extract_patch_bank(target_i, kernel_size_window)[0]
        height, width = pred_i.shape[-2:]
        sample_losses = []

        for start in range(0, valid_coords.shape[0], chunk_size):
            coord_chunk = valid_coords[start : start + chunk_size]
            ys = coord_chunk[:, 0]
            xs = coord_chunk[:, 1]
            center_indices = ys * width + xs

            offsets_y = torch.arange(-radius, radius + 1, device=pred.device)
            offsets_x = torch.arange(-radius, radius + 1, device=pred.device)
            grid_y, grid_x = torch.meshgrid(offsets_y, offsets_x, indexing="ij")
            neigh_y = ys[:, None, None] + grid_y[None]
            neigh_x = xs[:, None, None] + grid_x[None]
            neigh_y = neigh_y.clamp(0, height - 1)
            neigh_x = neigh_x.clamp(0, width - 1)
            neigh_indices = (neigh_y * width + neigh_x).reshape(coord_chunk.shape[0], -1)

            center_pred = patch_bank_pred[center_indices]
            neigh_pred = patch_bank_pred[neigh_indices]
            center_target = patch_bank_target[center_indices]
            neigh_target = patch_bank_target[neigh_indices]

            dist_pred = ((neigh_pred - center_pred[:, None, :]) ** 2).mean(dim=-1)
            dist_target = ((neigh_target - center_target[:, None, :]) ** 2).mean(dim=-1)

            scale_sigma = max(float(sigma), eps)
            s_pred = torch.exp(-dist_pred / scale_sigma)
            s_target = torch.exp(-dist_target / scale_sigma)
            s_pred = _normalize_probs(s_pred, eps)
            s_target = _normalize_probs(s_target, eps)

            kl = (s_target * ((s_target + eps).log() - (s_pred + eps).log())).sum(dim=-1)
            l1 = (s_pred - s_target).abs().mean(dim=-1)
            sample_losses.append(kl + alpha * l1)

        if sample_losses:
            batch_losses.append(torch.cat(sample_losses, dim=0).mean())

    if not batch_losses:
        return pred.sum() * 0.0
    return torch.stack(batch_losses).mean()


def sandbox_loss(
    pred,
    target,
    mask=None,
    weight=None,
    sigma=1.0,
    var=None,
    alpha=1.0,
    kernel_size_search=25,
    kernel_size_window=9,
    edge_threshold=20.0,
    mask_dilate_kernel=0,
    sample_stride=1,
    max_masked_pixels=4096,
    chunk_size=256,
    eps=1e-12,
    reduction="mean",
    **kwargs,
):
    pred = _to_nchw(pred, "pred")
    target = _to_nchw(target, "target")

    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have identical shape, got {tuple(pred.shape)} vs {tuple(target.shape)}")

    kernel_size_search = _odd_int(kernel_size_search, 25)
    kernel_size_window = _odd_int(kernel_size_window, 9)
    if reduction != "mean":
        raise ValueError(f"Only reduction='mean' is supported, got {reduction}")

    if weight is not None and torch.is_tensor(weight):
        sigma = sigma * (1.0 + 0.0 * weight.mean())
    if var is not None and torch.is_tensor(var):
        sigma = sigma + 0.0 * var.mean()

    sigma_value = float(sigma) if not torch.is_tensor(sigma) else float(torch.as_tensor(sigma).detach().mean().item())

    if mask is None:
        mask = _derive_edge_mask(
            target,
            threshold=edge_threshold,
            data_range=255.0,
            dilate_kernel=mask_dilate_kernel,
        )
    else:
        mask = _ensure_mask_4d(mask, pred)
        mask = (mask > 0.5).to(pred.dtype)

    return _masked_ssl_single(
        pred=pred,
        target=target,
        mask=mask,
        kernel_size_search=kernel_size_search,
        kernel_size_window=kernel_size_window,
        sigma=sigma_value,
        alpha=alpha,
        sample_stride=max(1, int(sample_stride)),
        max_masked_pixels=max(0, int(max_masked_pixels)),
        chunk_size=max(1, int(chunk_size)),
        eps=float(eps),
    )
