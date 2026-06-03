from __future__ import annotations

from .one_shot_stage_common import ONE_SHOT_DISTILL_EVAL_PREVIEW_PER_CLASS, _sample_one_shot_codes
from .sampling import select_post_train_sample_batch


ONE_SHOT_HISTORY_LOSS_KEYS = (
    "loss",
    "camera_recon_loss",
    "camera_fg_mse",
    "camera_bg_mse",
    "camera_foreground_dice",
    "slm_smoothness_loss",
    "slm_tv",
    "camera_saturation_loss",
    "camera_saturation_ratio",
    "code_diversity_loss",
    "code_diversity_penalty",
    "teacher_pairwise_diversity_loss",
    "teacher_pairwise_diversity_penalty",
    "teacher_pairwise_mse",
    "student_pairwise_mse",
    "diversity_ratio",
    "diversity_pair_count",
    "cross_class_pairwise_loss",
    "cross_class_pairwise_penalty",
    "cross_class_teacher_pairwise_mse",
    "cross_class_student_pairwise_mse",
    "cross_class_diversity_ratio",
    "cross_class_pair_count",
    "slm_pairwise_mse",
    "slm_per_class_std_mean",
    "fg_pixel_ratio",
)
ONE_SHOT_DISTILL_HISTORY_LOSS_KEYS = (
    "loss",
    "camera_recon_loss",
    "camera_fg_mse",
    "camera_bg_mse",
    "camera_foreground_dice",
    "shape_loss",
    "shape_penalty",
    "edge_loss",
    "edge_penalty",
    "class_weight_mean",
    "slm_smoothness_loss",
    "slm_tv",
    "camera_saturation_loss",
    "camera_saturation_ratio",
    "code_diversity_loss",
    "code_diversity_penalty",
    "teacher_pairwise_diversity_loss",
    "teacher_pairwise_diversity_penalty",
    "teacher_pairwise_mse",
    "student_pairwise_mse",
    "diversity_ratio",
    "diversity_pair_count",
    "cross_class_pairwise_loss",
    "cross_class_pairwise_penalty",
    "cross_class_teacher_pairwise_mse",
    "cross_class_student_pairwise_mse",
    "cross_class_diversity_ratio",
    "cross_class_pair_count",
    "slm_pairwise_mse",
    "slm_per_class_std_mean",
    "fg_pixel_ratio",
)


def _scalar_loss_items(losses, keys: tuple[str, ...]) -> dict[str, float]:
    return {key: losses[key].item() for key in keys}

def _one_shot_d2nn_crop_retention_metrics(outputs: dict[str, object]) -> dict[str, float]:
    """Extract compact D2NN crop energy retention scalars from model outputs."""
    metrics: dict[str, float] = {}
    branch_means: list[float] = []
    for branch, key in (
        ("encoder", "diffractive_encoder_crop_crop_retention"),
        ("decoder", "diffractive_decoder_crop_crop_retention"),
    ):
        tensor = outputs.get(key)
        if tensor is None or not callable(getattr(tensor, "detach", None)):
            continue
        retention = tensor.detach().float()
        if retention.numel() == 0:
            continue
        branch_mean = float(retention.mean().item())
        branch_means.append(branch_mean)
        metrics[f"{branch}_crop_energy_retention_mean"] = branch_mean
        if retention.ndim >= 2:
            by_layer = retention.reshape(retention.shape[0], -1)
            for layer_index in range(by_layer.shape[1]):
                metrics[f"{branch}_l{layer_index + 1}_crop_energy_retention"] = float(
                    by_layer[:, layer_index].mean().item()
                )
    if branch_means:
        metrics["crop_energy_retention_mean"] = sum(branch_means) / float(len(branch_means))
    return metrics

def _one_shot_loss_kwargs(train_cfg) -> dict[str, object]:
    return {
        "foreground_threshold": train_cfg.one_shot.recon_foreground_threshold,
        "foreground_weight": train_cfg.one_shot.recon_foreground_weight,
        "background_weight": train_cfg.one_shot.recon_background_weight,
        "class_recon_weight_labels": train_cfg.one_shot.class_recon_weight_labels,
        "class_recon_weight": train_cfg.one_shot.class_recon_weight,
        "shape_loss_weight": train_cfg.one_shot.shape_loss_weight,
        "edge_loss_weight": train_cfg.one_shot.edge_loss_weight,
        "slm_smoothness_weight": train_cfg.one_shot.slm_smoothness_weight,
        "slm_contrast_weight": train_cfg.one_shot.slm_contrast_weight,
        "camera_saturation_weight": train_cfg.one_shot.camera_saturation_weight,
        "camera_topology_weight": train_cfg.one_shot.camera_topology_weight,
        "camera_artifact_weight": train_cfg.one_shot.camera_artifact_weight,
        "code_diversity_weight": train_cfg.one_shot.code_diversity_weight,
        "teacher_pairwise_diversity_weight": train_cfg.one_shot.teacher_pairwise_diversity_weight,
        "cross_class_pairwise_weight": train_cfg.one_shot.cross_class_pairwise_weight,
        "diversity_min_teacher_mse": train_cfg.one_shot.diversity_min_teacher_mse,
    }

def _select_one_shot_visual_batch(torch_module, validation_bundle, train_bundle, train_cfg, device):
    sample_labels, reference_images, _, _ = select_post_train_sample_batch(
        torch_module,
        validation_bundle,
        train_bundle,
        train_cfg.one_shot.num_classes,
        device,
        sample_class_labels=train_cfg.one_shot.sample_class_labels,
        sample_repeats_per_class=ONE_SHOT_DISTILL_EVAL_PREVIEW_PER_CLASS,
    )
    visual_codes = _sample_one_shot_codes(torch_module, int(sample_labels.shape[0]), train_cfg, device)
    return visual_codes, sample_labels, reference_images

def _evaluate_one_shot(
    torch_module,
    one_shot_model,
    one_shot_loss_fn,
    loader,
    train_cfg,
    device,
    use_pin_memory: bool,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    one_shot_model.eval()
    with torch_module.no_grad():
        for images, labels in loader:
            target_batch = images.to(device, non_blocking=use_pin_memory)
            class_labels = labels.to(device, non_blocking=use_pin_memory)
            codes = _sample_one_shot_codes(torch_module, target_batch.shape[0], train_cfg, device)
            outputs = one_shot_model(codes, class_labels)
            losses = one_shot_loss_fn(outputs, target_batch, class_labels=class_labels, **_one_shot_loss_kwargs(train_cfg))
            batch_size = int(target_batch.shape[0])
            count += batch_size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().item()) * batch_size
    if count <= 0:
        raise ValueError("Validation loader is empty.")
    return {key: value / float(count) for key, value in totals.items()}

def _evaluate_one_shot_distill(
    torch_module,
    one_shot_model,
    one_shot_loss_fn,
    loader,
    train_cfg,
    device,
    use_pin_memory: bool,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    crop_totals: dict[str, float] = {}
    crop_count = 0
    count = 0
    one_shot_model.eval()
    with torch_module.no_grad():
        for codes, labels, targets in loader:
            code_batch = codes.to(device, non_blocking=use_pin_memory)
            class_labels = labels.to(device, non_blocking=use_pin_memory)
            target_batch = targets.to(device, non_blocking=use_pin_memory)
            outputs = one_shot_model(code_batch, class_labels)
            losses = one_shot_loss_fn(outputs, target_batch, class_labels=class_labels, **_one_shot_loss_kwargs(train_cfg))
            batch_size = int(target_batch.shape[0])
            count += batch_size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().item()) * batch_size
            crop_metrics = _one_shot_d2nn_crop_retention_metrics(outputs)
            if crop_metrics:
                crop_count += batch_size
                for key, value in crop_metrics.items():
                    crop_totals[key] = crop_totals.get(key, 0.0) + value * batch_size
    if count <= 0:
        raise ValueError("Distillation validation loader is empty.")
    result: dict[str, float] = {key: value / float(count) for key, value in totals.items()}
    if crop_totals:
        for key, value in crop_totals.items():
            result[key] = value / float(crop_count)
    return result

def _build_one_shot_optimizer(torch_module, one_shot_model, train_cfg):
    phase_params = []
    other_params = []
    for name, param in one_shot_model.named_parameters():
        if "phase" in name:
            phase_params.append(param)
        else:
            other_params.append(param)
    phase_lr = float(train_cfg.one_shot.phase_learning_rate)
    base_lr = float(train_cfg.learning_rate)
    if phase_params:
        print(f"[optimizer] phase params: {sum(p.numel() for p in phase_params)} total, lr={phase_lr:.1e}")
    print(f"[optimizer] other params: {sum(p.numel() for p in other_params)} total, lr={base_lr:.1e}")
    return torch_module.optim.Adam(
        [
            {"params": other_params, "lr": base_lr},
            {"params": phase_params, "lr": phase_lr},
        ]
    )

def _phase_grad_summary(model) -> str:
    parts: list[str] = []
    for name, param in model.named_parameters():
        if "phase" in name and param.grad is not None:
            g = param.grad
            parts.append(f"{name.split('.')[0]}_grad={g.abs().mean().item():.2e}")
    return " ".join(parts)

def _log_phase_gradients(model) -> None:
    summary = _phase_grad_summary(model)
    if summary:
        print(f"[phase_grad] {summary}")
