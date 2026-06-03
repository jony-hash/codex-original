from __future__ import annotations

import torch

from .training_loss_common import (
    _camera_artifact_metrics,
    _camera_hotspot_loss,
    _camera_hotspot_ratio,
    _camera_saturation_loss,
    _camera_saturation_ratio,
    _camera_topology_loss,
    _field_energy_loss,
    _field_smoothness_loss,
    _foreground_dice_score,
    _foreground_shape_loss,
    _masked_mean,
    _phase_smoothness_loss,
    _structure_reconstruction_loss,
    _weighted_reconstruction_metrics,
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
) -> dict[str, torch.Tensor]:
    foreground_mask = (input_batch > foreground_threshold).float()
    background_mask = 1.0 - foreground_mask
    camera_prediction = outputs["camera_plane"]
    camera_loss_prediction = outputs.get("camera_plane_loss", camera_prediction)
    image_metrics = _weighted_reconstruction_metrics(
        outputs["image_recon"],
        input_batch,
        foreground_mask,
        background_mask,
        foreground_weight,
        background_weight,
    )
    camera_metrics = _weighted_reconstruction_metrics(
        camera_prediction,
        input_batch,
        foreground_mask,
        background_mask,
        foreground_weight,
        background_weight,
    )
    kl = kl_divergence(outputs["mu"], outputs["logvar"])
    beta = kl_beta(global_step, total_steps, target_beta, warmup_ratio)
    recon = image_recon_weight * image_metrics["recon_loss"] + camera_recon_weight * camera_metrics["recon_loss"]
    image_structure = _structure_reconstruction_loss(outputs["image_recon"], input_batch)
    camera_structure = _structure_reconstruction_loss(camera_prediction, input_batch)
    structure_base = image_recon_weight * image_structure + camera_recon_weight * camera_structure
    structure = structure_recon_weight * structure_base
    image_shape = _foreground_shape_loss(outputs["image_recon"], input_batch, foreground_mask)
    camera_shape = _foreground_shape_loss(camera_prediction, input_batch, foreground_mask)
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
    camera_topology_base = _camera_topology_loss(camera_prediction, input_batch, foreground_mask)
    camera_topology = camera_topology_weight * camera_topology_base
    camera_saturation_base = _camera_saturation_loss(camera_loss_prediction)
    camera_saturation = camera_saturation_weight * camera_saturation_base
    camera_artifact_metrics = _camera_artifact_metrics(camera_prediction, input_batch, foreground_mask)
    camera_artifact = camera_artifact_weight * camera_artifact_metrics["artifact_base"]
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
        "kl_loss": kl.detach(),
        "fg_pixel_ratio": foreground_mask.mean(),
        "pred_fg_energy_ratio": pred_fg_energy_ratio,
        "beta": torch.tensor(beta, device=input_batch.device),
    }

