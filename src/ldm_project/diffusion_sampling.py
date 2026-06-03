from __future__ import annotations

import torch

from .diffusion import LatentDiffusionUNet, LinearNoiseScheduler
from .diffusion_losses import apply_classifier_free_guidance


DEFAULT_DENOISING_TRACE_TIMESTEPS = (750, 500, 250, 100, 50, 0)
REFERENCE_DENOISING_TIMESTEPS = 1000


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
    sampling_steps: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _validate_sampling_inputs(shape, class_labels, guidance_scale)
    timestep_sequence = _resolve_sampling_timestep_sequence(scheduler.timesteps, sampling_steps)
    use_full_ddpm_steps = len(timestep_sequence) == scheduler.timesteps

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
    for index, timestep in enumerate(timestep_sequence):
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
        if use_full_ddpm_steps:
            sample = scheduler.step(predicted_noise, current_t, sample)
        else:
            next_timestep = timestep_sequence[index + 1] if index + 1 < len(timestep_sequence) else -1
            sample = _ddim_step_to_timestep(scheduler, predicted_noise, current_t, next_timestep, sample)
        trace_label = trace_steps_by_timestep.get(timestep)
        if trace_label is not None:
            trace[trace_label] = sample.detach().clone()
    return sample, trace


def _resolve_sampling_timestep_sequence(total_timesteps: int, sampling_steps: int | None) -> list[int]:
    if sampling_steps is None or sampling_steps >= total_timesteps:
        return list(range(total_timesteps - 1, -1, -1))
    if sampling_steps < 2:
        raise ValueError(f"sampling_steps must be at least 2 when set, got {sampling_steps}")
    raw_steps = torch.linspace(total_timesteps - 1, 0, steps=sampling_steps).round().long().tolist()
    sequence: list[int] = []
    seen: set[int] = set()
    for step in raw_steps:
        step = int(step)
        if step not in seen:
            sequence.append(step)
            seen.add(step)
    if sequence[-1] != 0:
        sequence.append(0)
    return sequence


def _expand_scheduler_scalar(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    while value.ndim < reference.ndim:
        value = value.unsqueeze(-1)
    return value


def _ddim_step_to_timestep(
    scheduler: LinearNoiseScheduler,
    model_output: torch.Tensor,
    timesteps: torch.Tensor,
    next_timestep: int,
    sample: torch.Tensor,
) -> torch.Tensor:
    alpha_bar_t = _expand_scheduler_scalar(scheduler.alphas_cumprod[timesteps], sample)
    if next_timestep >= 0:
        alpha_bar_prev = scheduler.alphas_cumprod[next_timestep].expand(sample.shape[0])
    else:
        alpha_bar_prev = scheduler.alphas_cumprod.new_ones((sample.shape[0],))
    alpha_bar_prev = _expand_scheduler_scalar(alpha_bar_prev.to(device=sample.device), sample)
    pred_x0 = (sample - torch.sqrt(1.0 - alpha_bar_t) * model_output) / torch.sqrt(alpha_bar_t)
    return torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1.0 - alpha_bar_prev) * model_output


@torch.no_grad()
def sample_latents(
    diffusion_model: LatentDiffusionUNet,
    scheduler: LinearNoiseScheduler,
    shape: tuple[int, int, int, int],
    device: torch.device,
    class_labels: torch.Tensor,
    guidance_scale: float = 3.0,
    initial_noise: torch.Tensor | None = None,
    sampling_steps: int | None = None,
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
        sampling_steps=sampling_steps,
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
    sampling_steps: int | None = None,
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
        sampling_steps=sampling_steps,
    )
