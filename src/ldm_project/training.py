from __future__ import annotations

import torch
import torch.nn.functional as F

from .diffusion import LatentDiffusionUNet, LinearNoiseScheduler
from .first_stage import OpticalPathVAE


DEFAULT_DENOISING_TRACE_TIMESTEPS = (750, 500, 250, 100, 50, 0)
REFERENCE_DENOISING_TIMESTEPS = 1000


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
    fg_mse = _masked_mean(squared_error, foreground_mask)
    bg_mse = _masked_mean(squared_error, background_mask)
    recon = foreground_weight * fg_mse + background_weight * bg_mse
    return {
        "recon_loss": recon,
        "fg_mse": fg_mse.detach(),
        "bg_mse": bg_mse.detach(),
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


def _one_shot_pairwise_diversity_metrics(
    camera_prediction: torch.Tensor,
    target_batch: torch.Tensor,
    class_labels: torch.Tensor | None,
    min_teacher_mse: float,
) -> dict[str, torch.Tensor]:
    zero = _zero_like_loss(camera_prediction)
    if class_labels is None or camera_prediction.shape[0] < 2:
        return {
            "code_diversity_base": zero,
            "teacher_pairwise_diversity_base": zero,
            "teacher_pairwise_mse": zero,
            "student_pairwise_mse": zero,
            "diversity_pair_count": zero,
        }

    labels = class_labels.detach().reshape(-1)
    if labels.shape[0] != camera_prediction.shape[0]:
        raise ValueError("class_labels must match one_shot_loss batch size when diversity weights are enabled.")

    prediction_flat = camera_prediction.flatten(1)
    target_flat = target_batch.detach().flatten(1)
    student_distances = []
    teacher_distances = []
    for label in labels.unique(sorted=True):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        student_class = prediction_flat.index_select(0, indices)
        teacher_class = target_flat.index_select(0, indices)
        student_pairwise = (student_class[:, None, :] - student_class[None, :, :]).pow(2).mean(dim=-1)
        teacher_pairwise = (teacher_class[:, None, :] - teacher_class[None, :, :]).pow(2).mean(dim=-1)
        pair_mask = torch.triu(
            torch.ones(indices.numel(), indices.numel(), dtype=torch.bool, device=camera_prediction.device),
            diagonal=1,
        )
        student_distances.append(student_pairwise[pair_mask])
        teacher_distances.append(teacher_pairwise[pair_mask])

    if not student_distances:
        return {
            "code_diversity_base": zero,
            "teacher_pairwise_diversity_base": zero,
            "teacher_pairwise_mse": zero,
            "student_pairwise_mse": zero,
            "diversity_pair_count": zero,
        }

    student_distance = torch.cat(student_distances)
    teacher_distance = torch.cat(teacher_distances)
    active = teacher_distance >= float(min_teacher_mse)
    if not active.any():
        return {
            "code_diversity_base": zero,
            "teacher_pairwise_diversity_base": zero,
            "teacher_pairwise_mse": teacher_distance.mean().detach(),
            "student_pairwise_mse": student_distance.mean().detach(),
            "diversity_pair_count": zero,
        }

    active_student = student_distance[active]
    active_teacher = teacher_distance[active].clamp_min(1.0e-8)
    # 只在 teacher 自身存在差异的同类样本对上惩罚塌缩，避免鼓励无意义噪声。
    diversity_ratio = active_student / active_teacher
    code_diversity_base = F.relu(1.0 - diversity_ratio).pow(2).mean()
    teacher_pairwise_base = F.smooth_l1_loss(
        torch.log(active_student.clamp_min(1.0e-8)),
        torch.log(active_teacher),
    )
    return {
        "code_diversity_base": code_diversity_base,
        "teacher_pairwise_diversity_base": teacher_pairwise_base,
        "teacher_pairwise_mse": active_teacher.mean().detach(),
        "student_pairwise_mse": active_student.mean().detach(),
        "diversity_pair_count": active.float().sum().detach(),
    }


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


def one_shot_loss(
    outputs: dict[str, torch.Tensor],
    target_batch: torch.Tensor,
    class_labels: torch.Tensor | None = None,
    foreground_threshold: float = 0.05,
    foreground_weight: float = 3.0,
    background_weight: float = 1.0,
    slm_smoothness_weight: float = 0.0,
    camera_saturation_weight: float = 0.0,
    code_diversity_weight: float = 0.0,
    teacher_pairwise_diversity_weight: float = 0.0,
    diversity_min_teacher_mse: float = 1.0e-4,
) -> dict[str, torch.Tensor]:
    foreground_mask = (target_batch > foreground_threshold).float()
    background_mask = 1.0 - foreground_mask
    camera_prediction = outputs["camera_plane"]
    camera_loss_prediction = outputs.get("camera_plane_loss", camera_prediction)
    camera_metrics = _weighted_reconstruction_metrics(
        camera_prediction,
        target_batch,
        foreground_mask,
        background_mask,
        foreground_weight,
        background_weight,
    )
    slm_tv = _total_variation(outputs["slm_amplitude"])
    slm_smoothness = slm_smoothness_weight * slm_tv
    camera_saturation_base = _camera_saturation_loss(camera_loss_prediction)
    camera_saturation = camera_saturation_weight * camera_saturation_base
    diversity_metrics = _one_shot_pairwise_diversity_metrics(
        camera_prediction,
        target_batch,
        class_labels,
        diversity_min_teacher_mse,
    )
    code_diversity = code_diversity_weight * diversity_metrics["code_diversity_base"]
    teacher_pairwise_diversity = (
        teacher_pairwise_diversity_weight * diversity_metrics["teacher_pairwise_diversity_base"]
    )
    total = camera_metrics["recon_loss"] + slm_smoothness + camera_saturation + code_diversity + teacher_pairwise_diversity
    return {
        "loss": total,
        "camera_recon_loss": camera_metrics["recon_loss"].detach(),
        "camera_fg_mse": camera_metrics["fg_mse"],
        "camera_bg_mse": camera_metrics["bg_mse"],
        "slm_smoothness_loss": slm_smoothness.detach(),
        "slm_tv": slm_tv.detach(),
        "camera_saturation_loss": camera_saturation.detach(),
        "camera_saturation_penalty": camera_saturation_base.detach(),
        "camera_saturation_ratio": _camera_saturation_ratio(camera_prediction.detach()),
        "camera_foreground_dice": _foreground_dice_score(camera_prediction.detach(), foreground_mask),
        "code_diversity_loss": code_diversity.detach(),
        "code_diversity_penalty": diversity_metrics["code_diversity_base"].detach(),
        "teacher_pairwise_diversity_loss": teacher_pairwise_diversity.detach(),
        "teacher_pairwise_diversity_penalty": diversity_metrics["teacher_pairwise_diversity_base"].detach(),
        "teacher_pairwise_mse": diversity_metrics["teacher_pairwise_mse"].detach(),
        "student_pairwise_mse": diversity_metrics["student_pairwise_mse"].detach(),
        "diversity_pair_count": diversity_metrics["diversity_pair_count"].detach(),
        "fg_pixel_ratio": foreground_mask.mean(),
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


def diffusion_per_sample_losses(
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
    return F.mse_loss(predicted_noise, noise, reduction="none").flatten(1).mean(dim=1)


def _resolve_diffusion_loss_weights(
    class_labels: torch.Tensor,
    class_weight_labels: tuple[int, ...] = (),
    class_weight: float = 1.0,
) -> torch.Tensor:
    weights = torch.ones(class_labels.shape[0], device=class_labels.device, dtype=torch.float32)
    if not class_weight_labels or class_weight == 1.0:
        return weights

    weak_class_mask = torch.zeros_like(class_labels, dtype=torch.bool)
    for label in class_weight_labels:
        weak_class_mask |= class_labels == int(label)
    return torch.where(weak_class_mask, weights.new_full((), float(class_weight)), weights)


def diffusion_loss(
    first_stage: OpticalPathVAE,
    diffusion_model: LatentDiffusionUNet,
    scheduler: LinearNoiseScheduler,
    batch: torch.Tensor,
    class_labels: torch.Tensor,
    latent_scale: torch.Tensor,
    latent_source: str = "mu",
    class_weight_labels: tuple[int, ...] = (),
    class_weight: float = 1.0,
) -> torch.Tensor:
    per_sample_losses = diffusion_per_sample_losses(
        first_stage,
        diffusion_model,
        scheduler,
        batch,
        class_labels,
        latent_scale,
        latent_source=latent_source,
    )
    weights = _resolve_diffusion_loss_weights(class_labels, class_weight_labels, class_weight)
    return (per_sample_losses * weights).sum() / weights.sum().clamp_min(1.0)


def resolve_denoising_trace_steps(
    total_timesteps: int,
    trace_timesteps: tuple[int, ...] = DEFAULT_DENOISING_TRACE_TIMESTEPS,
) -> list[dict[str, int | str]]:
    if total_timesteps <= 0:
        raise ValueError(f"total_timesteps must be positive, got {total_timesteps}")

    steps: list[dict[str, int | str]] = []
    used_actual_timesteps: set[int] = set()
    for nominal_timestep in trace_timesteps:
        if nominal_timestep < 0:
            raise ValueError(f"trace timestep must be non-negative, got {nominal_timestep}")
        actual_timestep = int(
            round(nominal_timestep * float(total_timesteps - 1) / float(REFERENCE_DENOISING_TIMESTEPS - 1))
        )
        actual_timestep = max(0, min(total_timesteps - 1, actual_timestep))
        if nominal_timestep != 0 and actual_timestep == 0:
            continue
        if actual_timestep in used_actual_timesteps:
            continue
        used_actual_timesteps.add(actual_timestep)
        steps.append(
            {
                "label": f"t{nominal_timestep:03d}",
                "nominal_timestep": nominal_timestep,
                "actual_timestep": actual_timestep,
            }
        )

    if 0 not in used_actual_timesteps:
        steps.append({"label": "t000", "nominal_timestep": 0, "actual_timestep": 0})
    return steps


def _validate_sampling_inputs(
    shape: tuple[int, int, int, int],
    class_labels: torch.Tensor,
    guidance_scale: float,
) -> None:
    if guidance_scale < 0.0:
        raise ValueError(f"guidance_scale must be >= 0, got {guidance_scale}")
    if class_labels.shape[0] != shape[0]:
        raise ValueError(
            f"class_labels batch size ({class_labels.shape[0]}) must match sample batch size ({shape[0]})."
        )


def _sample_latents_impl(
    diffusion_model: LatentDiffusionUNet,
    scheduler: LinearNoiseScheduler,
    shape: tuple[int, int, int, int],
    device: torch.device,
    class_labels: torch.Tensor,
    guidance_scale: float,
    trace_timesteps: tuple[int, ...] | None,
    initial_noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _validate_sampling_inputs(shape, class_labels, guidance_scale)

    if initial_noise is None:
        sample = torch.randn(shape, device=device)
    else:
        if tuple(initial_noise.shape) != tuple(shape):
            raise ValueError(f"initial_noise shape must match sample shape {shape}, got {tuple(initial_noise.shape)}")
        sample = initial_noise.to(device=device)
    trace: dict[str, torch.Tensor] = {}
    trace_steps_by_timestep: dict[int, str] = {}
    if trace_timesteps is not None:
        trace["noise"] = sample.detach().clone()
        trace_steps_by_timestep = {
            int(step["actual_timestep"]): str(step["label"])
            for step in resolve_denoising_trace_steps(scheduler.timesteps, trace_timesteps)
        }

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
        trace_label = trace_steps_by_timestep.get(timestep)
        if trace_label is not None:
            trace[trace_label] = sample.detach().clone()
    return sample, trace


@torch.no_grad()
def sample_latents(
    diffusion_model: LatentDiffusionUNet,
    scheduler: LinearNoiseScheduler,
    shape: tuple[int, int, int, int],
    device: torch.device,
    class_labels: torch.Tensor,
    guidance_scale: float = 3.0,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    sample, _ = _sample_latents_impl(
        diffusion_model,
        scheduler,
        shape,
        device,
        class_labels,
        guidance_scale,
        trace_timesteps=None,
        initial_noise=initial_noise,
    )
    return sample


@torch.no_grad()
def sample_latents_with_trace(
    diffusion_model: LatentDiffusionUNet,
    scheduler: LinearNoiseScheduler,
    shape: tuple[int, int, int, int],
    device: torch.device,
    class_labels: torch.Tensor,
    guidance_scale: float = 3.0,
    trace_timesteps: tuple[int, ...] = DEFAULT_DENOISING_TRACE_TIMESTEPS,
    initial_noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _sample_latents_impl(
        diffusion_model,
        scheduler,
        shape,
        device,
        class_labels,
        guidance_scale,
        trace_timesteps=trace_timesteps,
        initial_noise=initial_noise,
    )
