from __future__ import annotations

import math


DEFAULT_LATENT_SCALE_BATCHES = 4


def resolve_device(torch_module, requested: str):
    if requested == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(requested)


def validate_class_label_values(labels: list[int] | tuple[int, ...], num_classes: int, field_name: str) -> tuple[int, ...]:
    resolved = tuple(int(label) for label in labels)
    for label in resolved:
        if label < 0 or label >= num_classes:
            raise ValueError(f"{field_name} must be in [0, {num_classes - 1}], got {label}")
    return resolved


def should_save_epoch_visual(epoch_index: int, interval: int) -> bool:
    if interval <= 0:
        raise ValueError(f"visual_save_epoch_interval must be positive, got {interval}")
    return epoch_index == 0 or (epoch_index + 1) % interval == 0


def should_validate_epoch(epoch_index: int, total_epochs: int, interval: int) -> bool:
    if total_epochs <= 0:
        raise ValueError(f"total_epochs must be positive, got {total_epochs}")
    if interval <= 0:
        raise ValueError(f"validation_epoch_interval must be positive, got {interval}")
    return epoch_index == 0 or epoch_index == total_epochs - 1 or (epoch_index + 1) % interval == 0


def should_log_step(step_index: int, interval: int, is_epoch_last_batch: bool) -> bool:
    if interval <= 0:
        raise ValueError(f"step_log_interval must be positive, got {interval}")
    # 首步、固定间隔和每个 epoch 的最后一步都保留，既方便看趋势，也避免日志过密。
    return step_index == 0 or (step_index + 1) % interval == 0 or is_epoch_last_batch


def resolve_image_fusion_weight(first_stage_config, epoch_index: int) -> float:
    if epoch_index < 0:
        raise ValueError(f"epoch_index must be non-negative, got {epoch_index}")
    if first_stage_config.encoder_fusion_mode != "gated_dual_path":
        return 0.0
    decay_epochs = first_stage_config.image_fusion_decay_epochs
    if decay_epochs <= 1:
        return float(first_stage_config.image_fusion_end_weight)
    progress = min(float(epoch_index) / float(decay_epochs - 1), 1.0)
    start = float(first_stage_config.image_fusion_start_weight)
    end = float(first_stage_config.image_fusion_end_weight)
    schedule = getattr(first_stage_config, "image_fusion_schedule", "linear")
    if schedule == "linear":
        decay_factor = progress
    elif schedule == "cosine":
        decay_factor = 0.5 - 0.5 * math.cos(math.pi * progress)
    else:
        raise ValueError(f"Unsupported image_fusion_schedule: {schedule}; expected linear or cosine.")
    return start + (end - start) * decay_factor


def set_first_stage_image_fusion_weight(first_stage, weight: float) -> None:
    if hasattr(first_stage, "set_image_fusion_weight"):
        first_stage.set_image_fusion_weight(weight)


def is_best_optical_candidate(first_stage_config, image_fusion_weight: float) -> bool:
    if first_stage_config.encoder_fusion_mode == "optical_only":
        return True
    if first_stage_config.encoder_fusion_mode != "gated_dual_path":
        return False
    return math.isclose(
        float(image_fusion_weight),
        0.0,
        rel_tol=0.0,
        abs_tol=1.0e-8,
    )


def require_strict_optical_best_checkpoint(best_checkpoint_path: str | None) -> str:
    if best_checkpoint_path is None:
        raise RuntimeError(
            "Full diffusion training requires a strict optical-only first-stage best checkpoint. "
            "Train first_stage with encoder_fusion_mode=optical_only or gated_dual_path at image_fusion_weight=0.0 first."
        )
    return best_checkpoint_path


def should_update_best_optical(score: float, best_score: float | None) -> bool:
    return best_score is None or score > best_score


def resolve_logged_step(batch_index: int, stage_step: int) -> dict[str, int]:
    if batch_index < 0:
        raise ValueError(f"batch_index must be non-negative, got {batch_index}")
    if stage_step < 0:
        raise ValueError(f"stage_step must be non-negative, got {stage_step}")
    # 日志里的 global_step 按当前这轮 train 重新计数，绝对优化步数单独保留，避免跨轮混淆。
    return {
        "global_step": batch_index,
        "stage_step": stage_step,
    }


def validate_stage_requirements(
    stage: str,
    first_stage_checkpoint: str | None,
    diffusion_checkpoint: str | None = None,
) -> None:
    if stage == "diffusion" and first_stage_checkpoint is None:
        raise ValueError("--stage diffusion requires --first-stage-checkpoint to load a trained first stage.")
    if stage == "sample":
        if first_stage_checkpoint is None:
            raise ValueError("--stage sample requires --first-stage-checkpoint to load a trained first stage.")
        if diffusion_checkpoint is None:
            raise ValueError("--stage sample requires --diffusion-checkpoint to load a trained diffusion model.")


def resolve_stage_total_steps(loader_length: int, epochs: int) -> int:
    if loader_length <= 0:
        raise ValueError("Training loader is empty.")
    if epochs <= 0:
        raise ValueError(f"stage epochs must be positive, got {epochs}")
    return loader_length * epochs


def resolve_stage_epoch_plan(stage: str, first_stage_epochs: int, diffusion_epochs: int) -> dict[str, int]:
    if first_stage_epochs < 0:
        raise ValueError(f"first_stage_epochs must be non-negative, got {first_stage_epochs}")
    if diffusion_epochs < 0:
        raise ValueError(f"diffusion_epochs must be non-negative, got {diffusion_epochs}")

    if stage == "first_stage":
        return {"first_stage": first_stage_epochs, "diffusion": 0}
    if stage == "diffusion":
        return {"first_stage": 0, "diffusion": diffusion_epochs}
    if stage == "full":
        return {"first_stage": first_stage_epochs, "diffusion": diffusion_epochs}
    if stage == "sample":
        return {"first_stage": 0, "diffusion": 0}
    raise ValueError(f"Unsupported stage mode: {stage}")


def apply_cli_overrides(train_cfg, args) -> None:
    if args.stage is not None:
        train_cfg.mode = args.stage
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.subset_fraction is not None:
        train_cfg.data.subset_fraction = args.subset_fraction
    subset_size = getattr(args, "subset_size", None)
    if subset_size is not None:
        if subset_size <= 0:
            raise ValueError(f"data.subset_size must be positive, got {subset_size}")
        train_cfg.data.subset_size = subset_size
    if args.validation_fraction is not None:
        if args.validation_fraction < 0.0 or args.validation_fraction >= 1.0:
            raise ValueError(
                "data.validation_fraction must be in [0.0, 1.0), "
                f"got {args.validation_fraction}"
            )
        train_cfg.data.validation_fraction = args.validation_fraction
    num_workers = getattr(args, "num_workers", None)
    if num_workers is not None:
        if num_workers < 0:
            raise ValueError(f"data.num_workers must be non-negative, got {num_workers}")
        train_cfg.data.num_workers = num_workers
    if args.first_stage_epochs is not None:
        train_cfg.first_stage_epochs = args.first_stage_epochs
    if args.diffusion_epochs is not None:
        train_cfg.diffusion_epochs = args.diffusion_epochs
    guidance_scale = getattr(args, "guidance_scale", None)
    if guidance_scale is not None:
        if guidance_scale < 0.0:
            raise ValueError(f"diffusion.guidance_scale must be non-negative, got {guidance_scale}")
        train_cfg.diffusion.guidance_scale = guidance_scale
    cond_dropout_prob = getattr(args, "cond_dropout_prob", None)
    if cond_dropout_prob is not None:
        if cond_dropout_prob < 0.0 or cond_dropout_prob > 1.0:
            raise ValueError(
                "diffusion.cond_dropout_prob must be in [0.0, 1.0], "
                f"got {cond_dropout_prob}"
            )
        train_cfg.diffusion.cond_dropout_prob = cond_dropout_prob
    sample_class_labels = getattr(args, "sample_class_labels", None)
    if sample_class_labels is not None:
        train_cfg.diffusion.sample_class_labels = validate_class_label_values(
            sample_class_labels,
            train_cfg.diffusion.num_classes,
            "sample_class_labels",
        )
    sample_repeats_per_class = getattr(args, "sample_repeats_per_class", None)
    if sample_repeats_per_class is not None:
        if sample_repeats_per_class <= 0:
            raise ValueError(
                "diffusion.sample_repeats_per_class must be positive, "
                f"got {sample_repeats_per_class}"
            )
        train_cfg.diffusion.sample_repeats_per_class = sample_repeats_per_class
    diffusion_class_weight_labels = getattr(args, "diffusion_class_weight_labels", None)
    if diffusion_class_weight_labels is not None:
        train_cfg.diffusion.class_weight_labels = validate_class_label_values(
            diffusion_class_weight_labels,
            train_cfg.diffusion.num_classes,
            "diffusion_class_weight_labels",
        )
    diffusion_class_weight = getattr(args, "diffusion_class_weight", None)
    if diffusion_class_weight is not None:
        if diffusion_class_weight <= 0.0:
            raise ValueError(f"diffusion.class_weight must be positive, got {diffusion_class_weight}")
        train_cfg.diffusion.class_weight = diffusion_class_weight
    camera_artifact_weight = getattr(args, "camera_artifact_weight", None)
    if camera_artifact_weight is not None:
        if camera_artifact_weight < 0.0:
            raise ValueError(
                "first_stage.camera_artifact_weight must be non-negative, "
                f"got {camera_artifact_weight}"
            )
        train_cfg.first_stage.camera_artifact_weight = camera_artifact_weight
    encoder_fusion_mode = getattr(args, "encoder_fusion_mode", None)
    if encoder_fusion_mode is not None:
        train_cfg.first_stage.encoder_fusion_mode = encoder_fusion_mode
    image_fusion_weight = getattr(args, "image_fusion_weight", None)
    if image_fusion_weight is not None:
        if image_fusion_weight < 0.0 or image_fusion_weight > 1.0:
            raise ValueError(
                "first_stage.image_fusion_weight override must be in [0.0, 1.0], "
                f"got {image_fusion_weight}"
            )
        train_cfg.first_stage.image_fusion_start_weight = image_fusion_weight
        train_cfg.first_stage.image_fusion_end_weight = image_fusion_weight
        train_cfg.first_stage.image_fusion_decay_epochs = 1
    optical_encoder_input_mode = getattr(args, "optical_encoder_input_mode", None)
    if optical_encoder_input_mode is not None:
        train_cfg.first_stage.optical_encoder_input_mode = optical_encoder_input_mode
    optical_encoder_arch = getattr(args, "optical_encoder_arch", None)
    if optical_encoder_arch is not None:
        train_cfg.first_stage.optical_encoder_arch = optical_encoder_arch
    metasurface_decoder_arch = getattr(args, "metasurface_decoder_arch", None)
    if metasurface_decoder_arch is not None:
        train_cfg.first_stage.metasurface_decoder_arch = metasurface_decoder_arch


def estimate_latent_scale_from_loader(
    torch_module,
    first_stage,
    loader,
    device,
    use_pin_memory: bool,
    latent_source: str,
    max_batches: int = DEFAULT_LATENT_SCALE_BATCHES,
):
    from ldm_project.training import select_diffusion_latents

    if max_batches <= 0:
        raise ValueError(f"max_batches must be positive, got {max_batches}")

    total_square = None
    total_count = 0
    first_stage.eval()
    with torch_module.no_grad():
        for batch_index, (images, _) in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = images.to(device, non_blocking=use_pin_memory)
            encoded = first_stage.encode(batch)
            latents = select_diffusion_latents(encoded, latent_source).detach()
            square_sum = latents.pow(2).sum()
            total_square = square_sum if total_square is None else total_square + square_sum
            total_count += latents.numel()

    if total_square is None or total_count == 0:
        raise ValueError("Training loader is empty.")
    return (total_square / float(total_count)).sqrt().clamp_min(1e-6)


def load_latent_scale_from_diffusion_checkpoint(torch_module, payload, device):
    latent_scale_payload = payload.get("latent_scale")
    if latent_scale_payload is None:
        raise ValueError("Incompatible diffusion checkpoint: missing latent_scale.")
    if hasattr(latent_scale_payload, "to"):
        latent_scale = latent_scale_payload.to(device=device)
    else:
        latent_scale = torch_module.tensor(latent_scale_payload, device=device)
    if latent_scale.numel() != 1:
        raise ValueError("Incompatible diffusion checkpoint: latent_scale must be scalar.")
    return latent_scale.reshape(()).float().clamp_min(1e-6)
