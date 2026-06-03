from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1e-6)


def _masked_mean_per_sample(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, values.ndim))
    return (values * mask).sum(dim=dims) / mask.sum(dim=dims).clamp_min(1e-6)


def _weighted_sample_mean(values: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    weights = sample_weights.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def _weighted_reconstruction_metrics(
    prediction: torch.Tensor,
    input_batch: torch.Tensor,
    foreground_mask: torch.Tensor,
    background_mask: torch.Tensor,
    foreground_weight: float,
    background_weight: float,
) -> dict[str, torch.Tensor]:
    squared_error = (prediction - input_batch).pow(2)
    fg_mse = _masked_mean(squared_error, foreground_mask)
    bg_mse = _masked_mean(squared_error, background_mask)
    recon = foreground_weight * fg_mse + background_weight * bg_mse
    return {
        "recon_loss": recon,
        "fg_mse": fg_mse.detach(),
        "bg_mse": bg_mse.detach(),
    }


def _weighted_reconstruction_metrics_for_samples(
    prediction: torch.Tensor,
    input_batch: torch.Tensor,
    foreground_mask: torch.Tensor,
    background_mask: torch.Tensor,
    foreground_weight: float,
    background_weight: float,
    sample_weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    squared_error = (prediction - input_batch).pow(2)
    fg_mse_samples = _masked_mean_per_sample(squared_error, foreground_mask)
    bg_mse_samples = _masked_mean_per_sample(squared_error, background_mask)
    recon_samples = foreground_weight * fg_mse_samples + background_weight * bg_mse_samples
    return {
        "recon_loss": _weighted_sample_mean(recon_samples, sample_weights),
        "fg_mse": fg_mse_samples.mean().detach(),
        "bg_mse": bg_mse_samples.mean().detach(),
    }


def _sobel_magnitude(tensor: torch.Tensor) -> torch.Tensor:
    channels = tensor.shape[1]
    sobel_x = tensor.new_tensor(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ]
    ).view(1, 1, 3, 3)
    sobel_y = tensor.new_tensor(
        [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ]
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(tensor, sobel_x.expand(channels, 1, 3, 3), padding=1, groups=channels)
    grad_y = F.conv2d(tensor, sobel_y.expand(channels, 1, 3, 3), padding=1, groups=channels)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1.0e-12)


def _structure_reconstruction_loss(prediction: torch.Tensor, input_batch: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(_sobel_magnitude(prediction), _sobel_magnitude(input_batch))


def _camera_topology_loss(
    prediction: torch.Tensor,
    input_batch: torch.Tensor,
    foreground_mask: torch.Tensor,
) -> torch.Tensor:
    prediction_soft = prediction.clamp(0.0, 1.0)
    target = input_batch.clamp(0.0, 1.0)
    # 复杂数字主要输在笔画连接和弯折，先在前景邻域内约束边缘与局部形状。
    foreground_context = F.max_pool2d(foreground_mask, kernel_size=5, stride=1, padding=2)
    edge_error = (_sobel_magnitude(prediction_soft) - _sobel_magnitude(target)).abs()
    edge_loss = _masked_mean(edge_error, foreground_context)
    local_prediction = F.avg_pool2d(prediction_soft, kernel_size=3, stride=1, padding=1)
    local_target = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    local_shape_loss = _masked_mean((local_prediction - local_target).abs(), foreground_context)
    return edge_loss + 0.5 * local_shape_loss


def _foreground_shape_loss(prediction: torch.Tensor, input_batch: torch.Tensor, foreground_mask: torch.Tensor) -> torch.Tensor:
    return 1.0 - _foreground_dice_score(prediction, foreground_mask)


def _foreground_shape_loss_for_samples(
    prediction: torch.Tensor,
    foreground_mask: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    prediction_soft = prediction.clamp(0.0, 1.0)
    dims = tuple(range(1, prediction_soft.ndim))
    intersection = (prediction_soft * foreground_mask).sum(dim=dims)
    denominator = prediction_soft.sum(dim=dims) + foreground_mask.sum(dim=dims)
    dice_score = (2.0 * intersection + 1.0e-6) / (denominator + 1.0e-6)
    return _weighted_sample_mean(1.0 - dice_score, sample_weights)


def _foreground_dice_score(prediction: torch.Tensor, foreground_mask: torch.Tensor) -> torch.Tensor:
    prediction_soft = prediction.clamp(0.0, 1.0)
    dims = tuple(range(1, prediction_soft.ndim))
    intersection = (prediction_soft * foreground_mask).sum(dim=dims)
    denominator = prediction_soft.sum(dim=dims) + foreground_mask.sum(dim=dims)
    dice_score = (2.0 * intersection + 1.0e-6) / (denominator + 1.0e-6)
    return dice_score.mean()


def _total_variation(tensor: torch.Tensor) -> torch.Tensor:
    diff_y = (tensor[..., 1:, :] - tensor[..., :-1, :]).abs().mean()
    diff_x = (tensor[..., :, 1:] - tensor[..., :, :-1]).abs().mean()
    return diff_x + diff_y


def _zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def _field_smoothness_loss(outputs: dict[str, torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    if "chip_real" not in outputs or "chip_imag" not in outputs:
        return _zero_like_loss(reference)
    return _total_variation(outputs["chip_real"]) + _total_variation(outputs["chip_imag"])


def _field_energy_loss(outputs: dict[str, torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    if "chip_real" not in outputs or "chip_imag" not in outputs:
        return _zero_like_loss(reference)
    field_energy = outputs["chip_real"].pow(2) + outputs["chip_imag"].pow(2)
    return (field_energy.mean() - 1.0).pow(2)


def _phase_smoothness_loss(outputs: dict[str, torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    chip_phase = outputs.get("chip_phase")
    if chip_phase is None:
        return _zero_like_loss(reference)
    # 用 sin/cos 避免相位在 -pi/pi wrap 边界附近产生虚假的大跳变。
    return _total_variation(torch.sin(chip_phase)) + _total_variation(torch.cos(chip_phase))


def _camera_saturation_ratio(camera_prediction: torch.Tensor) -> torch.Tensor:
    saturated = (camera_prediction >= 0.99) | (camera_prediction <= 0.01)
    return saturated.float().mean()



def _camera_hotspot_loss(camera_prediction: torch.Tensor) -> torch.Tensor:
    return F.relu(camera_prediction - 1.0).pow(2).mean()


def _camera_hotspot_ratio(camera_prediction: torch.Tensor) -> torch.Tensor:
    return (camera_prediction > 1.0).float().mean()



def _camera_saturation_loss(camera_prediction: torch.Tensor) -> torch.Tensor:
    return F.relu(camera_prediction - 0.95).pow(2).mean()


def _camera_artifact_metrics(
    camera_prediction: torch.Tensor,
    input_batch: torch.Tensor,
    foreground_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    prediction = camera_prediction.clamp(0.0, 1.0)
    target = input_batch.clamp(0.0, 1.0)
    local_target = F.avg_pool2d(target, kernel_size=5, stride=1, padding=2)
    foreground_density = F.avg_pool2d(foreground_mask, kernel_size=5, stride=1, padding=2)
    # 只压前景笔画内部，降低对真实数字边缘的过度抹平。
    interior_weight = foreground_mask * foreground_density.pow(2)
    hole_depth = F.relu(local_target * 0.75 - prediction)
    foreground_hole_loss = _masked_mean(hole_depth.pow(2), interior_weight)
    foreground_hole_ratio = _masked_mean((hole_depth > 0.10).float(), interior_weight)
    local_prediction = F.avg_pool2d(prediction, kernel_size=5, stride=1, padding=2)
    foreground_roughness = _masked_mean((prediction - local_prediction).abs(), interior_weight)
    artifact_base = foreground_hole_loss + 0.25 * foreground_roughness
    return {
        "artifact_base": artifact_base,
        "foreground_hole_loss": foreground_hole_loss,
        "foreground_hole_ratio": foreground_hole_ratio,
        "foreground_roughness": foreground_roughness,
    }

