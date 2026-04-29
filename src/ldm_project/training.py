from __future__ import annotations

import torch
import torch.nn.functional as F

from .diffusion import LatentDiffusionUNet, LinearNoiseScheduler
from .first_stage import OpticalPathVAE


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


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1e-6)


def _weighted_reconstruction_metrics(
    prediction: torch.Tensor,
    input_batch: torch.Tensor,
    foreground_mask: torch.Tensor,
    background_mask: torch.Tensor,
    foreground_weight: float,
    background_weight: float,
) -> dict[str, torch.Tensor]:
    squared_error = (prediction - input_batch).pow(2)
    weights = foreground_mask * foreground_weight + background_mask * background_weight
    recon = (weights * squared_error).sum() / weights.sum().clamp_min(1e-6)
    return {
        "recon_loss": recon,
        "fg_mse": _masked_mean(squared_error.detach(), foreground_mask),
        "bg_mse": _masked_mean(squared_error.detach(), background_mask),
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


def _foreground_shape_loss(prediction: torch.Tensor, input_batch: torch.Tensor, foreground_mask: torch.Tensor) -> torch.Tensor:
    prediction_soft = prediction.clamp(0.0, 1.0)
    dims = tuple(range(1, prediction_soft.ndim))
    intersection = (prediction_soft * foreground_mask).sum(dim=dims)
    denominator = prediction_soft.sum(dim=dims) + foreground_mask.sum(dim=dims)
    dice_score = (2.0 * intersection + 1.0e-6) / (denominator + 1.0e-6)
    return 1.0 - dice_score.mean()


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


def _phase_smoothness_loss(outputs: dict[str, torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    chip_phase = outputs.get("chip_phase")
    if chip_phase is None:
        return _zero_like_loss(reference)
    # 用 sin/cos 避免相位在 -pi/pi wrap 边界附近产生虚假的大跳变。
    return _total_variation(torch.sin(chip_phase)) + _total_variation(torch.cos(chip_phase))


def _camera_saturation_ratio(camera_prediction: torch.Tensor) -> torch.Tensor:
    saturated = (camera_prediction >= 0.99) | (camera_prediction <= 0.01)
    return saturated.float().mean()


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
) -> dict[str, torch.Tensor]:
    foreground_mask = (input_batch > foreground_threshold).float()
    background_mask = 1.0 - foreground_mask
    camera_prediction = outputs.get("camera_plane_loss", outputs["camera_plane"])
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
    background_leakage = background_leakage_weight * _masked_mean(camera_prediction.clamp_min(0.0), background_mask)
    total = recon + beta * kl + structure + shape + field_smoothness + phase_smoothness + background_leakage
    x_hat_detached = outputs["x_hat"].detach()
    pred_fg_energy_ratio = (x_hat_detached * foreground_mask).sum() / x_hat_detached.sum().clamp_min(1e-6)
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
        "field_smoothness_loss": field_smoothness.detach(),
        "field_tv": field_tv.detach(),
        "phase_smoothness_loss": phase_smoothness.detach(),
        "phase_tv": phase_tv.detach(),
        "background_leakage_loss": background_leakage.detach(),
        "camera_saturation_ratio": _camera_saturation_ratio(camera_prediction.detach()),
        "kl_loss": kl.detach(),
        "fg_pixel_ratio": foreground_mask.mean(),
        "pred_fg_energy_ratio": pred_fg_energy_ratio,
        "beta": torch.tensor(beta, device=input_batch.device),
    }


@torch.no_grad()
def estimate_latent_scale(first_stage: OpticalPathVAE, batch: torch.Tensor, latent_source: str = "mu") -> torch.Tensor:
    encoded = first_stage.encode(batch)
    latents = select_diffusion_latents(encoded, latent_source)
    return latents.detach().pow(2).mean().sqrt().clamp_min(1e-6)


def apply_classifier_free_guidance(
    conditioned_prediction: torch.Tensor,
    unconditioned_prediction: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    return unconditioned_prediction + guidance_scale * (conditioned_prediction - unconditioned_prediction)


def diffusion_loss(
    first_stage: OpticalPathVAE,
    diffusion_model: LatentDiffusionUNet,
    scheduler: LinearNoiseScheduler,
    batch: torch.Tensor,
    class_labels: torch.Tensor,
    latent_scale: torch.Tensor,
    latent_source: str = "mu",
) -> torch.Tensor:
    with torch.no_grad():
        encoded = first_stage.encode(batch)
        latents = select_diffusion_latents(encoded, latent_source) / latent_scale

    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, scheduler.timesteps, (latents.shape[0],), device=latents.device)
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    context = diffusion_model.encode_condition(class_labels)
    predicted_noise = diffusion_model(noisy_latents, timesteps, context)
    return F.mse_loss(predicted_noise, noise)


@torch.no_grad()
def sample_latents(
    diffusion_model: LatentDiffusionUNet,
    scheduler: LinearNoiseScheduler,
    shape: tuple[int, int, int, int],
    device: torch.device,
    class_labels: torch.Tensor,
    guidance_scale: float = 3.0,
) -> torch.Tensor:
    if guidance_scale < 0.0:
        raise ValueError(f"guidance_scale must be >= 0, got {guidance_scale}")
    if class_labels.shape[0] != shape[0]:
        raise ValueError(
            f"class_labels batch size ({class_labels.shape[0]}) must match sample batch size ({shape[0]})."
        )

    sample = torch.randn(shape, device=device)
    conditioned_context = diffusion_model.encode_condition(class_labels, dropout_prob=0.0)
    unconditioned_context = diffusion_model.encode_condition(
        class_labels,
        force_unconditional=True,
        dropout_prob=0.0,
    )
    for timestep in reversed(range(scheduler.timesteps)):
        current_t = torch.full((shape[0],), timestep, device=device, dtype=torch.long)
        conditioned_noise = diffusion_model(sample, current_t, conditioned_context)
        if guidance_scale == 1.0:
            predicted_noise = conditioned_noise
        else:
            unconditioned_noise = diffusion_model(sample, current_t, unconditioned_context)
            predicted_noise = apply_classifier_free_guidance(
                conditioned_noise,
                unconditioned_noise,
                guidance_scale,
            )
        sample = scheduler.step(predicted_noise, current_t, sample)
    return sample
