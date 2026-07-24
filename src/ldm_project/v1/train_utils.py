from __future__ import annotations

import math
from pathlib import Path


DEFAULT_LATENT_SCALE_BATCHES = 4
STAGE_ALIASES = {
    "first_stage": "teacher_first_stage",
    "diffusion": "teacher_diffusion",
    "full": "teacher_full",
    "sample": "teacher_sample",
}
CANONICAL_STAGES = {
    "teacher_first_stage",
    "teacher_diffusion",
    "teacher_full",
    "teacher_sample",
    "teacher_distill_export",
    "teacher_distill_filter",
    "teacher_distill_label_audit",
    "teacher_distill_label_strict_filter",
    "teacher_distill_readable_split",
    "teacher_first_stage_oracle_audit",
    "teacher_diffusion_latent_audit",
    "one_shot",
    "one_shot_distill",
    "one_shot_distill_eval",
    "one_shot_distill_eval_diagnostics",
    "one_shot_optical_identity_audit",
    "one_shot_sample",
}
CLI_STAGE_CHOICES = tuple(sorted(CANONICAL_STAGES | set(STAGE_ALIASES)))


def canonicalize_stage(stage: str) -> str:
    canonical_stage = STAGE_ALIASES.get(stage, stage)
    if canonical_stage not in CANONICAL_STAGES:
        raise ValueError(f"Unsupported stage mode: {stage}")
    return canonical_stage


def is_teacher_first_stage_mode(stage: str) -> bool:
    return canonicalize_stage(stage) in {"teacher_first_stage", "teacher_full"}


def is_teacher_diffusion_mode(stage: str) -> bool:
    return canonicalize_stage(stage) in {"teacher_diffusion", "teacher_full"}


def resolve_device(torch_module, requested: str):
    if requested == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(requested)


def apply_training_seed(torch_module, seed: int | None) -> int | None:
    if seed is None:
        return None
    resolved_seed = int(seed)
    if resolved_seed < 0:
        raise ValueError(f"training.seed must be non-negative, got {resolved_seed}")
    torch_module.manual_seed(resolved_seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(resolved_seed)
    return resolved_seed


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
    one_shot_checkpoint: str | None = None,
    distill_dataset: str | None = None,
) -> None:
    canonical_stage = canonicalize_stage(stage)
    if canonical_stage == "teacher_diffusion" and first_stage_checkpoint is None:
        raise ValueError("--stage teacher_diffusion requires --first-stage-checkpoint to load a trained first stage.")
    if canonical_stage in {"teacher_sample", "teacher_distill_export", "teacher_diffusion_latent_audit"}:
        if first_stage_checkpoint is None:
            raise ValueError(f"--stage {canonical_stage} requires --first-stage-checkpoint to load a trained first stage.")
        if diffusion_checkpoint is None:
            raise ValueError(f"--stage {canonical_stage} requires --diffusion-checkpoint to load a trained diffusion model.")
    if canonical_stage in {"teacher_first_stage_oracle_audit"} and first_stage_checkpoint is None:
        raise ValueError(
            f"--stage {canonical_stage} requires --first-stage-checkpoint to load a trained first stage."
        )
    if canonical_stage in {"one_shot_sample", "one_shot_distill_eval"} and one_shot_checkpoint is None:
        raise ValueError(f"--stage {canonical_stage} requires --one-shot-checkpoint to load a trained one-shot model.")
    if canonical_stage in {
        "one_shot_distill",
        "one_shot_distill_eval",
        "teacher_distill_filter",
        "teacher_distill_label_audit",
        "teacher_distill_label_strict_filter",
        "teacher_distill_readable_split",
    } and distill_dataset is None:
        raise ValueError(f"--stage {canonical_stage} requires --distill-dataset to load teacher distillation targets.")


def _resolve_checkpoint_from_final_pt(
    explicit_checkpoint: str | None,
    final_pt_dir: str | Path,
    checkpoint_name: str,
) -> tuple[str | None, str | None]:
    if explicit_checkpoint is not None:
        return explicit_checkpoint, "cli"

    candidate = Path(final_pt_dir) / checkpoint_name
    if candidate.is_file():
        return str(candidate), "final_pt"
    return None, None


def resolve_final_pt_stage_checkpoints(
    stage: str,
    first_stage_checkpoint: str | None,
    diffusion_checkpoint: str | None,
    final_pt_dir: str | Path,
    first_stage_checkpoint_name: str,
    diffusion_checkpoint_name: str,
) -> dict[str, str | None]:
    canonical_stage = canonicalize_stage(stage)
    resolved_first_stage = first_stage_checkpoint
    resolved_first_stage_source = "cli" if first_stage_checkpoint is not None else None
    resolved_diffusion = diffusion_checkpoint
    resolved_diffusion_source = "cli" if diffusion_checkpoint is not None else None

    if canonical_stage in {
        "teacher_diffusion",
        "teacher_sample",
        "teacher_distill_export",
        "teacher_first_stage_oracle_audit",
        "teacher_diffusion_latent_audit",
    }:
        # 缺省 checkpoint 优先从最终验收目录取，避免误用旧 run 或重复传路径。
        resolved_first_stage, resolved_first_stage_source = _resolve_checkpoint_from_final_pt(
            first_stage_checkpoint,
            final_pt_dir,
            first_stage_checkpoint_name,
        )
    if canonical_stage in {"teacher_sample", "teacher_distill_export", "teacher_diffusion_latent_audit"}:
        resolved_diffusion, resolved_diffusion_source = _resolve_checkpoint_from_final_pt(
            diffusion_checkpoint,
            final_pt_dir,
            diffusion_checkpoint_name,
        )

    return {
        "first_stage_checkpoint": resolved_first_stage,
        "first_stage_checkpoint_source": resolved_first_stage_source,
        "diffusion_checkpoint": resolved_diffusion,
        "diffusion_checkpoint_source": resolved_diffusion_source,
    }


def resolve_stage_total_steps(loader_length: int, epochs: int) -> int:
    if loader_length <= 0:
        raise ValueError("Training loader is empty.")
    if epochs <= 0:
        raise ValueError(f"stage epochs must be positive, got {epochs}")
    return loader_length * epochs


def resolve_stage_epoch_plan(
    stage: str,
    first_stage_epochs: int,
    diffusion_epochs: int,
    one_shot_epochs: int = 0,
) -> dict[str, int]:
    if first_stage_epochs < 0:
        raise ValueError(f"first_stage_epochs must be non-negative, got {first_stage_epochs}")
    if diffusion_epochs < 0:
        raise ValueError(f"diffusion_epochs must be non-negative, got {diffusion_epochs}")
    if one_shot_epochs < 0:
        raise ValueError(f"one_shot_epochs must be non-negative, got {one_shot_epochs}")

    canonical_stage = canonicalize_stage(stage)
    if canonical_stage == "teacher_first_stage":
        return {"first_stage": first_stage_epochs, "diffusion": 0, "one_shot": 0}
    if canonical_stage == "teacher_diffusion":
        return {"first_stage": 0, "diffusion": diffusion_epochs, "one_shot": 0}
    if canonical_stage == "teacher_full":
        return {"first_stage": first_stage_epochs, "diffusion": diffusion_epochs, "one_shot": 0}
    if canonical_stage in {
        "teacher_sample",
        "teacher_distill_export",
        "teacher_distill_filter",
        "teacher_distill_label_audit",
        "teacher_distill_label_strict_filter",
        "teacher_distill_readable_split",
        "teacher_first_stage_oracle_audit",
        "teacher_diffusion_latent_audit",
        "one_shot_sample",
        "one_shot_distill_eval",
        "one_shot_distill_eval_diagnostics",
        "one_shot_optical_identity_audit",
    }:
        return {"first_stage": 0, "diffusion": 0, "one_shot": 0}
    if canonical_stage in {"one_shot", "one_shot_distill"}:
        return {"first_stage": 0, "diffusion": 0, "one_shot": one_shot_epochs}
    raise ValueError(f"Unsupported stage mode: {stage}")


def _optional_arg(args, name: str):
    return getattr(args, name, None)


def _set_optional(target, attribute: str, value) -> None:
    if value is not None:
        setattr(target, attribute, value)


def _require_positive(value, message_prefix: str) -> None:
    if value <= 0:
        raise ValueError(f"{message_prefix} must be positive, got {value}")


def apply_cli_overrides(train_cfg, args) -> None:
    if args.stage is not None:
        train_cfg.mode = canonicalize_stage(args.stage)
    else:
        train_cfg.mode = canonicalize_stage(train_cfg.mode)
    _set_optional(train_cfg, "batch_size", args.batch_size)
    seed = _optional_arg(args, "seed")
    if seed is not None:
        if seed < 0:
            raise ValueError(f"training.seed must be non-negative, got {seed}")
        train_cfg.seed = int(seed)
    _set_optional(train_cfg.data, "subset_fraction", args.subset_fraction)
    subset_size = _optional_arg(args, "subset_size")
    if subset_size is not None:
        _require_positive(subset_size, "data.subset_size")
        train_cfg.data.subset_size = subset_size
    if args.validation_fraction is not None:
        if args.validation_fraction < 0.0 or args.validation_fraction >= 1.0:
            raise ValueError(
                "data.validation_fraction must be in [0.0, 1.0), "
                f"got {args.validation_fraction}"
            )
        train_cfg.data.validation_fraction = args.validation_fraction
    num_workers = _optional_arg(args, "num_workers")
    if num_workers is not None:
        if num_workers < 0:
            raise ValueError(f"data.num_workers must be non-negative, got {num_workers}")
        train_cfg.data.num_workers = num_workers
    _set_optional(train_cfg, "first_stage_epochs", args.first_stage_epochs)
    _set_optional(train_cfg, "diffusion_epochs", args.diffusion_epochs)
    _set_optional(train_cfg, "one_shot_epochs", _optional_arg(args, "one_shot_epochs"))
    distill_samples_per_class = _optional_arg(args, "distill_samples_per_class")
    if distill_samples_per_class is not None:
        _require_positive(distill_samples_per_class, "distillation.samples_per_class")
        train_cfg.distillation.samples_per_class = distill_samples_per_class
    distill_total_samples = _optional_arg(args, "distill_total_samples")
    if distill_total_samples is not None:
        _require_positive(distill_total_samples, "distillation.total_samples")
        train_cfg.distillation.total_samples = distill_total_samples
    distill_train_subset_size = _optional_arg(args, "distill_train_subset_size")
    if distill_train_subset_size is not None:
        _require_positive(distill_train_subset_size, "distillation.train_subset_size")
        train_cfg.distillation.train_subset_size = distill_train_subset_size
    _set_optional(train_cfg.distillation, "output_path", _optional_arg(args, "distill_output"))
    guidance_scale = _optional_arg(args, "guidance_scale")
    if guidance_scale is not None:
        if guidance_scale < 0.0:
            raise ValueError(f"diffusion.guidance_scale must be non-negative, got {guidance_scale}")
        train_cfg.diffusion.guidance_scale = guidance_scale
    cond_dropout_prob = _optional_arg(args, "cond_dropout_prob")
    if cond_dropout_prob is not None:
        if cond_dropout_prob < 0.0 or cond_dropout_prob > 1.0:
            raise ValueError(
                "diffusion.cond_dropout_prob must be in [0.0, 1.0], "
                f"got {cond_dropout_prob}"
            )
        train_cfg.diffusion.cond_dropout_prob = cond_dropout_prob
    sample_class_labels = _optional_arg(args, "sample_class_labels")
    if sample_class_labels is not None:
        diffusion_sample_class_labels = validate_class_label_values(
            sample_class_labels,
            train_cfg.diffusion.num_classes,
            "sample_class_labels",
        )
        one_shot_sample_class_labels = validate_class_label_values(
            sample_class_labels,
            train_cfg.one_shot.num_classes,
            "one_shot.sample_class_labels",
        )
        train_cfg.diffusion.sample_class_labels = diffusion_sample_class_labels
        train_cfg.one_shot.sample_class_labels = one_shot_sample_class_labels
    sample_repeats_per_class = _optional_arg(args, "sample_repeats_per_class")
    if sample_repeats_per_class is not None:
        _require_positive(sample_repeats_per_class, "sample_repeats_per_class")
        train_cfg.diffusion.sample_repeats_per_class = sample_repeats_per_class
        train_cfg.one_shot.sample_repeats_per_class = sample_repeats_per_class
    diffusion_class_weight_labels = _optional_arg(args, "diffusion_class_weight_labels")
    if diffusion_class_weight_labels is not None:
        train_cfg.diffusion.class_weight_labels = validate_class_label_values(
            diffusion_class_weight_labels,
            train_cfg.diffusion.num_classes,
            "diffusion_class_weight_labels",
        )
    diffusion_class_weight = _optional_arg(args, "diffusion_class_weight")
    if diffusion_class_weight is not None:
        _require_positive(diffusion_class_weight, "diffusion.class_weight")
        train_cfg.diffusion.class_weight = diffusion_class_weight
    camera_artifact_weight = _optional_arg(args, "camera_artifact_weight")
    if camera_artifact_weight is not None:
        if camera_artifact_weight < 0.0:
            raise ValueError(
                "first_stage.camera_artifact_weight must be non-negative, "
                f"got {camera_artifact_weight}"
            )
        train_cfg.first_stage.camera_artifact_weight = camera_artifact_weight
    _set_optional(train_cfg.first_stage, "encoder_fusion_mode", _optional_arg(args, "encoder_fusion_mode"))
    image_fusion_weight = _optional_arg(args, "image_fusion_weight")
    if image_fusion_weight is not None:
        if image_fusion_weight < 0.0 or image_fusion_weight > 1.0:
            raise ValueError(
                "first_stage.image_fusion_weight override must be in [0.0, 1.0], "
                f"got {image_fusion_weight}"
            )
        train_cfg.first_stage.image_fusion_start_weight = image_fusion_weight
        train_cfg.first_stage.image_fusion_end_weight = image_fusion_weight
        train_cfg.first_stage.image_fusion_decay_epochs = 1
    _set_optional(train_cfg.first_stage, "optical_encoder_input_mode", _optional_arg(args, "optical_encoder_input_mode"))
    _set_optional(train_cfg.first_stage, "optical_encoder_arch", _optional_arg(args, "optical_encoder_arch"))
    _set_optional(train_cfg.first_stage, "metasurface_decoder_arch", _optional_arg(args, "metasurface_decoder_arch"))


def estimate_latent_scale_from_loader(
    torch_module,
    first_stage,
    loader,
    device,
    use_pin_memory: bool,
    latent_source: str,
    max_batches: int = DEFAULT_LATENT_SCALE_BATCHES,
):
    from .teacher.first_stage_losses import select_diffusion_latents

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
