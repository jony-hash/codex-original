from __future__ import annotations

import torch
import torch.nn.functional as F

from .training_loss_common import (
    _camera_artifact_metrics,
    _camera_artifact_metrics_for_samples,
    _camera_hotspot_loss,
    _camera_hotspot_ratio,
    _camera_saturation_loss,
    _camera_saturation_ratio,
    _camera_topology_loss,
    _camera_topology_loss_for_samples,
    _field_energy_loss,
    _field_smoothness_loss,
    _foreground_dice_score,
    _foreground_shape_loss,
    _foreground_shape_loss_for_samples,
    _masked_mean,
    _phase_smoothness_loss,
    _structure_reconstruction_loss,
    _weighted_reconstruction_metrics,
    _weighted_reconstruction_metrics_for_samples,
)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def kl_beta(global_step: int, total_steps: int, target_beta: float, warmup_ratio: float) -> float:
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if global_step >= warmup_steps:
        return target_beta
    return target_beta * float(global_step + 1) / float(warmup_steps)


def select_diffusion_latents(encoded: dict[str, torch.Tensor], latent_source: str) -> torch.Tensor:
    if latent_source == "mu":
        return encoded["mu"]
    if latent_source == "z":
        return encoded["z"]
    raise ValueError(f"Unsupported diffusion latent_source: {latent_source}; expected mu or z.")


def hard_class_sample_weights(
    class_labels: torch.Tensor | None,
    *,
    hard_class_weight_labels: tuple[int, ...] = (),
    hard_class_weight: float = 1.0,
    batch_size: int | None = None,
    device=None,
    dtype=None,
) -> torch.Tensor:
    if hard_class_weight <= 0.0:
        raise ValueError(f"hard_class_weight must be positive, got {hard_class_weight}")
    if class_labels is None:
        if batch_size is None:
            raise ValueError("batch_size is required when class_labels is None.")
        return torch.ones(int(batch_size), device=device, dtype=dtype or torch.float32)
    labels = class_labels.detach().long()
    weights = torch.ones(labels.shape[0], device=labels.device, dtype=dtype or torch.float32)
    if hard_class_weight_labels and hard_class_weight != 1.0:
        hard = torch.zeros_like(labels, dtype=torch.bool)
        for label in hard_class_weight_labels:
            hard = hard | (labels == int(label))
        weights = torch.where(hard, weights.new_full(weights.shape, float(hard_class_weight)), weights)
    return weights.to(device=device, dtype=dtype) if device is not None or dtype is not None else weights


def _gaussian_blur_2d(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return tensor
    radius = max(1, int(round(float(sigma) * 3.0)))
    coords = torch.arange(-radius, radius + 1, device=tensor.device, dtype=tensor.dtype)
    kernel_1d = torch.exp(-0.5 * (coords / float(sigma)).pow(2))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1.0e-8)
    channels = int(tensor.shape[1])
    kernel_y = kernel_1d.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    kernel_x = kernel_1d.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    blurred = F.conv2d(tensor, kernel_y, padding=(radius, 0), groups=channels)
    return F.conv2d(blurred, kernel_x, padding=(0, radius), groups=channels)


def _normalize_samples_for_label_margin(samples: torch.Tensor, smooth_sigma: float) -> torch.Tensor:
    samples = _gaussian_blur_2d(samples.clamp_min(0.0).float(), smooth_sigma)
    flat = samples.flatten(1)
    mn = flat.amin(dim=1, keepdim=True).view(-1, 1, 1, 1)
    mx = flat.amax(dim=1, keepdim=True).view(-1, 1, 1, 1)
    return (samples - mn) / (mx - mn).clamp_min(1.0e-6)


def _prototype_correlation(samples: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    sample_flat = samples.flatten(1)
    prototype_flat = prototypes.detach().float().flatten(1)
    sample_centered = sample_flat - sample_flat.mean(dim=1, keepdim=True)
    prototype_centered = prototype_flat - prototype_flat.mean(dim=1, keepdim=True)
    sample_norm = sample_centered.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-8)
    prototype_norm = prototype_centered.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-8)
    return (sample_centered / sample_norm) @ (prototype_centered / prototype_norm).T / float(sample_flat.shape[1])


def camera_label_margin_metrics(
    camera_prediction: torch.Tensor,
    class_labels: torch.Tensor | None,
    *,
    camera_label_margin_prototypes: torch.Tensor | None = None,
    camera_label_margin_labels: tuple[int, ...] = (),
    camera_label_margin_target: float = 0.08,
    camera_label_margin_weight: float = 0.0,
    smooth_sigma: float = 0.6,
) -> dict[str, torch.Tensor]:
    zero = camera_prediction.sum() * 0.0
    default = {
        "camera_label_margin_loss": zero,
        "camera_label_margin_base": zero.detach(),
        "camera_label_margin_mean": zero.detach(),
        "camera_label_margin_requested_corr": zero.detach(),
        "camera_label_margin_best_other_corr": zero.detach(),
        "camera_label_margin_sample_ratio": zero.detach(),
    }
    if (
        camera_label_margin_weight <= 0.0
        or camera_label_margin_prototypes is None
        or class_labels is None
        or camera_prediction.shape[0] == 0
    ):
        return default

    labels = class_labels.long()
    prototypes = camera_label_margin_prototypes.to(device=camera_prediction.device, dtype=camera_prediction.dtype)
    if prototypes.shape[-2:] != camera_prediction.shape[-2:]:
        prototypes = F.interpolate(prototypes, size=camera_prediction.shape[-2:], mode="bilinear", align_corners=False)
    samples = _normalize_samples_for_label_margin(camera_prediction, smooth_sigma)
    prototypes = _normalize_samples_for_label_margin(prototypes, smooth_sigma)
    class_count = int(prototypes.shape[0])
    valid_mask = (labels >= 0) & (labels < class_count)
    if camera_label_margin_labels:
        hard_mask = torch.zeros_like(valid_mask)
        for label in camera_label_margin_labels:
            hard_mask = hard_mask | (labels == int(label))
        valid_mask = valid_mask & hard_mask
    if not bool(valid_mask.any().item()):
        return default

    selected_labels = labels[valid_mask]
    correlations = _prototype_correlation(samples[valid_mask], prototypes)
    requested = correlations.gather(1, selected_labels.view(-1, 1)).squeeze(1)
    other = correlations.clone()
    other[torch.arange(selected_labels.shape[0], device=selected_labels.device), selected_labels] = -2.0
    best_other = other.max(dim=1).values
    margin = requested - best_other
    base = torch.relu(camera_prediction.new_tensor(float(camera_label_margin_target)) - margin).mean()
    sample_ratio = valid_mask.float().mean()
    return {
        "camera_label_margin_loss": camera_label_margin_weight * base,
        "camera_label_margin_base": base.detach(),
        "camera_label_margin_mean": margin.mean().detach(),
        "camera_label_margin_requested_corr": requested.mean().detach(),
        "camera_label_margin_best_other_corr": best_other.mean().detach(),
        "camera_label_margin_sample_ratio": sample_ratio.detach(),
    }


def first_stage_loss(
    outputs: dict[str, torch.Tensor],
    input_batch: torch.Tensor,
    global_step: int,
    total_steps: int,
    target_beta: float,
    warmup_ratio: float,
    foreground_threshold: float = 0.05,
    foreground_weight: float = 4.0,
    background_weight: float = 1.0,
    image_recon_weight: float = 0.25,
    camera_recon_weight: float = 1.0,
    structure_recon_weight: float = 0.0,
    shape_recon_weight: float = 0.0,
    field_smoothness_weight: float = 0.0,
    phase_smoothness_weight: float = 0.0,
    background_leakage_weight: float = 0.0,
    field_energy_weight: float = 0.0,
    camera_hotspot_weight: float = 0.0,
    camera_topology_weight: float = 0.0,
    camera_saturation_weight: float = 0.0,
    camera_artifact_weight: float = 0.0,
    class_labels: torch.Tensor | None = None,
    hard_class_weight_labels: tuple[int, ...] = (),
    hard_class_weight: float = 1.0,
    camera_label_margin_weight: float = 0.0,
    camera_label_margin_target: float = 0.08,
    camera_label_margin_labels: tuple[int, ...] = (),
    camera_label_margin_prototypes: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    foreground_mask = (input_batch > foreground_threshold).float()
    background_mask = 1.0 - foreground_mask
    camera_prediction = outputs["camera_plane"]
    camera_loss_prediction = outputs.get("camera_plane_loss", camera_prediction)
    sample_weights = hard_class_sample_weights(
        class_labels,
        hard_class_weight_labels=hard_class_weight_labels,
        hard_class_weight=hard_class_weight,
        batch_size=int(input_batch.shape[0]),
        device=input_batch.device,
        dtype=input_batch.dtype,
    )
    use_hard_class_weighting = class_labels is not None and bool(hard_class_weight_labels) and hard_class_weight != 1.0
    hard_class_scale = sample_weights.mean()
    image_metrics = _weighted_reconstruction_metrics(
        outputs["image_recon"],
        input_batch,
        foreground_mask,
        background_mask,
        foreground_weight,
        background_weight,
    )
    camera_metrics = (
        _weighted_reconstruction_metrics_for_samples(
            camera_prediction,
            input_batch,
            foreground_mask,
            background_mask,
            foreground_weight,
            background_weight,
            sample_weights,
        )
        if use_hard_class_weighting
        else _weighted_reconstruction_metrics(
            camera_prediction,
            input_batch,
            foreground_mask,
            background_mask,
            foreground_weight,
            background_weight,
        )
    )
    if use_hard_class_weighting:
        camera_metrics["recon_loss"] = camera_metrics["recon_loss"] * hard_class_scale
    kl = kl_divergence(outputs["mu"], outputs["logvar"])
    beta = kl_beta(global_step, total_steps, target_beta, warmup_ratio)
    recon = image_recon_weight * image_metrics["recon_loss"] + camera_recon_weight * camera_metrics["recon_loss"]
    image_structure = _structure_reconstruction_loss(outputs["image_recon"], input_batch)
    camera_structure = _structure_reconstruction_loss(camera_prediction, input_batch)
    structure_base = image_recon_weight * image_structure + camera_recon_weight * camera_structure
    structure = structure_recon_weight * structure_base
    image_shape = _foreground_shape_loss(outputs["image_recon"], input_batch, foreground_mask)
    camera_shape = (
        _foreground_shape_loss_for_samples(camera_prediction, foreground_mask, sample_weights)
        if use_hard_class_weighting
        else _foreground_shape_loss(camera_prediction, input_batch, foreground_mask)
    )
    if use_hard_class_weighting:
        camera_shape = camera_shape * hard_class_scale
    shape_base = image_recon_weight * image_shape + camera_recon_weight * camera_shape
    shape = shape_recon_weight * shape_base
    field_tv = _field_smoothness_loss(outputs, input_batch)
    field_smoothness = field_smoothness_weight * field_tv
    phase_tv = _phase_smoothness_loss(outputs, input_batch)
    phase_smoothness = phase_smoothness_weight * phase_tv
    field_energy_base = _field_energy_loss(outputs, input_batch)
    field_energy = field_energy_weight * field_energy_base
    camera_background_leakage = _masked_mean(camera_prediction.clamp_min(0.0), background_mask)
    background_leakage = background_leakage_weight * camera_background_leakage
    camera_hotspot_base = _camera_hotspot_loss(camera_loss_prediction)
    camera_hotspot = camera_hotspot_weight * camera_hotspot_base
    camera_topology_base = (
        _camera_topology_loss_for_samples(camera_prediction, input_batch, foreground_mask, sample_weights)
        if use_hard_class_weighting
        else _camera_topology_loss(camera_prediction, input_batch, foreground_mask)
    )
    if use_hard_class_weighting:
        camera_topology_base = camera_topology_base * hard_class_scale
    camera_topology = camera_topology_weight * camera_topology_base
    camera_saturation_base = _camera_saturation_loss(camera_loss_prediction)
    camera_saturation = camera_saturation_weight * camera_saturation_base
    camera_artifact_metrics = (
        _camera_artifact_metrics_for_samples(camera_prediction, input_batch, foreground_mask, sample_weights)
        if use_hard_class_weighting
        else _camera_artifact_metrics(camera_prediction, input_batch, foreground_mask)
    )
    if use_hard_class_weighting:
        camera_artifact_metrics["artifact_base"] = camera_artifact_metrics["artifact_base"] * hard_class_scale
    camera_artifact = camera_artifact_weight * camera_artifact_metrics["artifact_base"]
    label_margin_metrics = camera_label_margin_metrics(
        camera_prediction,
        class_labels,
        camera_label_margin_prototypes=camera_label_margin_prototypes,
        camera_label_margin_labels=camera_label_margin_labels,
        camera_label_margin_target=camera_label_margin_target,
        camera_label_margin_weight=camera_label_margin_weight,
    )
    total = (
        recon
        + beta * kl
        + structure
        + shape
        + field_smoothness
        + phase_smoothness
        + field_energy
        + background_leakage
        + camera_hotspot
        + camera_topology
        + camera_saturation
        + camera_artifact
        + label_margin_metrics["camera_label_margin_loss"]
    )
    x_hat_detached = outputs["x_hat"].detach()
    pred_fg_energy_ratio = (x_hat_detached * foreground_mask).sum() / x_hat_detached.sum().clamp_min(1e-6)
    camera_foreground_dice = _foreground_dice_score(camera_prediction.detach(), foreground_mask)
    return {
        "loss": total,
        "recon_loss": recon.detach(),
        "image_recon_loss": image_metrics["recon_loss"].detach(),
        "image_fg_mse": image_metrics["fg_mse"],
        "image_bg_mse": image_metrics["bg_mse"],
        "camera_recon_loss": camera_metrics["recon_loss"].detach(),
        "camera_fg_mse": camera_metrics["fg_mse"],
        "camera_bg_mse": camera_metrics["bg_mse"],
        "structure_loss": structure.detach(),
        "image_structure_loss": image_structure.detach(),
        "camera_structure_loss": camera_structure.detach(),
        "shape_loss": shape.detach(),
        "image_shape_loss": image_shape.detach(),
        "camera_shape_loss": camera_shape.detach(),
        "camera_foreground_dice_loss": camera_shape.detach(),
        "camera_foreground_dice": camera_foreground_dice,
        "field_smoothness_loss": field_smoothness.detach(),
        "field_tv": field_tv.detach(),
        "phase_smoothness_loss": phase_smoothness.detach(),
        "phase_tv": phase_tv.detach(),
        "field_energy_loss": field_energy.detach(),
        "field_energy": field_energy_base.detach(),
        "background_leakage_loss": background_leakage.detach(),
        "camera_background_leakage": camera_background_leakage.detach(),
        "camera_hotspot_loss": camera_hotspot.detach(),
        "camera_hotspot_ratio": _camera_hotspot_ratio(camera_loss_prediction.detach()),
        "camera_saturation_ratio": _camera_saturation_ratio(camera_prediction.detach()),
        "camera_topology_loss": camera_topology.detach(),
        "camera_topology_base": camera_topology_base.detach(),
        "camera_saturation_loss": camera_saturation.detach(),
        "camera_saturation_penalty": camera_saturation_base.detach(),
        "camera_artifact_loss": camera_artifact.detach(),
        "camera_foreground_hole_loss": camera_artifact_metrics["foreground_hole_loss"].detach(),
        "camera_foreground_hole_ratio": camera_artifact_metrics["foreground_hole_ratio"].detach(),
        "camera_foreground_roughness": camera_artifact_metrics["foreground_roughness"].detach(),
        "camera_label_margin_loss": label_margin_metrics["camera_label_margin_loss"].detach(),
        "camera_label_margin_base": label_margin_metrics["camera_label_margin_base"],
        "camera_label_margin_mean": label_margin_metrics["camera_label_margin_mean"],
        "camera_label_margin_requested_corr": label_margin_metrics["camera_label_margin_requested_corr"],
        "camera_label_margin_best_other_corr": label_margin_metrics["camera_label_margin_best_other_corr"],
        "camera_label_margin_sample_ratio": label_margin_metrics["camera_label_margin_sample_ratio"],
        "kl_loss": kl.detach(),
        "fg_pixel_ratio": foreground_mask.mean(),
        "pred_fg_energy_ratio": pred_fg_energy_ratio,
        "hard_class_weight_mean": sample_weights.mean().detach(),
        "beta": torch.tensor(beta, device=input_batch.device),
    }
