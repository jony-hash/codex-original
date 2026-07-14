from __future__ import annotations

import torch
import torch.nn.functional as F

from .metrics import _eval_correlation, _eval_foreground_dice
from ..generation.codes import (
    ONE_SHOT_DISTILL_EVAL_PREVIEW_PER_CLASS,
    _sample_one_shot_codes,
)
from ..sampling_reports import select_post_train_sample_batch


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
    "camera_low_saturation_ratio",
    "camera_high_saturation_ratio",
    "camera_readability_loss",
    "camera_readability_penalty",
    "camera_background_speckle_loss",
    "camera_background_speckle_penalty",
    "background_speckle_ratio",
    "camera_background_hotspot_loss",
    "camera_background_hotspot_penalty",
    "background_hotspot_ratio",
    "camera_background_highfreq_loss",
    "camera_background_highfreq_penalty",
    "background_highfreq_energy",
    "camera_readable_contrast_loss",
    "camera_readable_contrast_penalty",
    "readable_contrast",
    "student_label_margin_loss",
    "student_label_margin_penalty",
    "student_label_margin_mean",
    "student_label_correct_ratio",
    "student_prototype_ce_loss",
    "student_prototype_ce_penalty",
    "student_prototype_ce_correct_ratio",
    "student_prototype_ce_count",
    "student_negative_label_margin_loss",
    "student_negative_label_margin_penalty",
    "student_negative_label_margin_mean",
    "student_negative_label_margin_count",
    "student_roi_prototype_ce_loss",
    "student_roi_prototype_ce_penalty",
    "student_roi_prototype_ce_correct_ratio",
    "student_roi_prototype_ce_count",
    "student_roi_negative_label_margin_loss",
    "student_roi_negative_label_margin_penalty",
    "student_roi_negative_label_margin_mean",
    "student_roi_negative_label_margin_count",
    "aux_semantic_loss",
    "aux_semantic_prototype_ce_loss",
    "aux_semantic_prototype_ce_penalty",
    "aux_semantic_prototype_ce_correct_ratio",
    "aux_semantic_prototype_ce_count",
    "aux_semantic_negative_label_margin_loss",
    "aux_semantic_negative_label_margin_penalty",
    "aux_semantic_negative_label_margin_mean",
    "aux_semantic_negative_label_margin_count",
    "aux_topology_loss",
    "aux_topology_dice",
    "aux_topology_mse",
    "aux_topology_target_fg_ratio",
    "code_diversity_loss",
    "code_diversity_penalty",
    "sample_code_diversity_loss",
    "sample_code_diversity_penalty",
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
    "encoder_input_topology_loss",
    "encoder_input_topology_dice",
    "encoder_input_topology_mse",
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
    "camera_low_saturation_ratio",
    "camera_high_saturation_ratio",
    "camera_readability_loss",
    "camera_readability_penalty",
    "camera_background_speckle_loss",
    "camera_background_speckle_penalty",
    "background_speckle_ratio",
    "camera_background_hotspot_loss",
    "camera_background_hotspot_penalty",
    "background_hotspot_ratio",
    "camera_background_highfreq_loss",
    "camera_background_highfreq_penalty",
    "background_highfreq_energy",
    "camera_readable_contrast_loss",
    "camera_readable_contrast_penalty",
    "readable_contrast",
    "student_label_margin_loss",
    "student_label_margin_penalty",
    "student_label_margin_mean",
    "student_label_correct_ratio",
    "student_prototype_ce_loss",
    "student_prototype_ce_penalty",
    "student_prototype_ce_correct_ratio",
    "student_prototype_ce_count",
    "student_negative_label_margin_loss",
    "student_negative_label_margin_penalty",
    "student_negative_label_margin_mean",
    "student_negative_label_margin_count",
    "student_roi_prototype_ce_loss",
    "student_roi_prototype_ce_penalty",
    "student_roi_prototype_ce_correct_ratio",
    "student_roi_prototype_ce_count",
    "student_roi_negative_label_margin_loss",
    "student_roi_negative_label_margin_penalty",
    "student_roi_negative_label_margin_mean",
    "student_roi_negative_label_margin_count",
    "aux_semantic_loss",
    "aux_semantic_prototype_ce_loss",
    "aux_semantic_prototype_ce_penalty",
    "aux_semantic_prototype_ce_correct_ratio",
    "aux_semantic_prototype_ce_count",
    "aux_semantic_negative_label_margin_loss",
    "aux_semantic_negative_label_margin_penalty",
    "aux_semantic_negative_label_margin_mean",
    "aux_semantic_negative_label_margin_count",
    "aux_topology_loss",
    "aux_topology_dice",
    "aux_topology_mse",
    "aux_topology_target_fg_ratio",
    "code_diversity_loss",
    "code_diversity_penalty",
    "sample_code_diversity_loss",
    "sample_code_diversity_penalty",
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
    "clean_foreground_dice",
    "clean_dice_loss",
    "clean_dice_penalty",
    "clean_fg_hole_loss",
    "clean_fg_hole_penalty",
    "clean_fg_hole_ratio",
    "clean_bg_speckle_loss",
    "clean_bg_speckle_penalty",
    "clean_bg_speckle_ratio",
    "clean_bg_hotspot_loss",
    "clean_bg_hotspot_penalty",
    "clean_bg_hotspot_ratio",
    "clean_bg_highfreq_loss",
    "clean_bg_highfreq_penalty",
    "clean_bg_highfreq_energy",
    "clean_target_fg_ratio",
    "encoder_input_topology_loss",
    "encoder_input_topology_dice",
    "encoder_input_topology_mse",
    "student_pair_contrast_loss",
    "student_pair_contrast_penalty",
    "student_pair_contrast_count",
    "display_speckle_loss",
    "display_speckle_penalty",
    "display_speckle_ratio",
    "display_highfreq_energy",
    "aux_detail_shape_loss",
    "aux_detail_shape_penalty",
    "aux_detail_edge_loss",
    "aux_detail_edge_penalty",
    "aux_detail_gate_fg_ratio",
    "aux_detail_active_fg_ratio",
    "residual_l1_loss",
    "residual_l1_penalty",
    "residual_background_loss",
    "residual_background_penalty",
    "residual_background_mask_ratio",
    "generator_residual_abs_mean",
    "generator_residual_bg_abs_mean",
    "student_base_mse",
    "student_base_foreground_mse",
    "student_base_background_mse",
)


def _scalar_loss_items(losses, keys: tuple[str, ...]) -> dict[str, float]:
    return {key: losses[key].item() for key in keys}


def _one_shot_d2nn_crop_retention_metrics(
    outputs: dict[str, object],
) -> dict[str, float]:
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
        metrics["crop_energy_retention_mean"] = sum(branch_means) / float(
            len(branch_means)
        )
    return metrics


def _one_shot_loss_kwargs(
    train_cfg, student_label_margin_prototypes=None
) -> dict[str, object]:
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
        "camera_readability_weight": train_cfg.one_shot.camera_readability_weight,
        "camera_background_speckle_weight": train_cfg.one_shot.camera_background_speckle_weight,
        "camera_background_hotspot_weight": train_cfg.one_shot.camera_background_hotspot_weight,
        "camera_background_hotspot_threshold": train_cfg.one_shot.camera_background_hotspot_threshold,
        "camera_background_highfreq_weight": train_cfg.one_shot.camera_background_highfreq_weight,
        "camera_readable_contrast_weight": train_cfg.one_shot.camera_readable_contrast_weight,
        "camera_readable_contrast_target": train_cfg.one_shot.camera_readable_contrast_target,
        "camera_recon_weight": train_cfg.one_shot.camera_recon_weight,
        "clean_target_enabled": train_cfg.one_shot.clean_target_enabled,
        "clean_foreground_dice_weight": train_cfg.one_shot.clean_foreground_dice_weight,
        "clean_foreground_hole_weight": train_cfg.one_shot.clean_foreground_hole_weight,
        "clean_background_speckle_weight": train_cfg.one_shot.clean_background_speckle_weight,
        "clean_background_hotspot_weight": train_cfg.one_shot.clean_background_hotspot_weight,
        "clean_background_highfreq_weight": train_cfg.one_shot.clean_background_highfreq_weight,
        "encoder_input_topology_weight": train_cfg.one_shot.encoder_input_topology_weight,
        "encoder_input_topology_dice_weight": train_cfg.one_shot.encoder_input_topology_dice_weight,
        "encoder_input_topology_mse_weight": train_cfg.one_shot.encoder_input_topology_mse_weight,
        "student_label_margin_weight": train_cfg.one_shot.student_label_margin_weight,
        "student_label_margin_target": train_cfg.one_shot.student_label_margin_target,
        "student_label_margin_labels": train_cfg.one_shot.student_label_margin_labels,
        "student_label_margin_focus_label": train_cfg.one_shot.student_label_margin_focus_label,
        "student_label_margin_focus_weight": train_cfg.one_shot.student_label_margin_focus_weight,
        "student_label_margin_prototypes": student_label_margin_prototypes,
        "student_prototype_ce_weight": train_cfg.one_shot.student_prototype_ce_weight,
        "student_prototype_ce_labels": train_cfg.one_shot.student_prototype_ce_labels,
        "student_prototype_ce_temperature": train_cfg.one_shot.student_prototype_ce_temperature,
        "student_negative_label_margin_weight": train_cfg.one_shot.student_negative_label_margin_weight,
        "student_negative_label_margin_config": train_cfg.one_shot.student_negative_label_margin_config,
        "student_roi_prototype_ce_weight": train_cfg.one_shot.student_roi_prototype_ce_weight,
        "student_roi_prototype_ce_labels": train_cfg.one_shot.student_roi_prototype_ce_labels,
        "student_roi_prototype_ce_temperature": train_cfg.one_shot.student_roi_prototype_ce_temperature,
        "student_roi_negative_label_margin_weight": train_cfg.one_shot.student_roi_negative_label_margin_weight,
        "student_roi_negative_label_margin_config": train_cfg.one_shot.student_roi_negative_label_margin_config,
        "student_roi_foreground_threshold": train_cfg.one_shot.student_roi_foreground_threshold,
        "aux_semantic_plane_key": train_cfg.one_shot.aux_semantic_plane_key,
        "aux_semantic_weight": train_cfg.one_shot.aux_semantic_weight,
        "aux_semantic_prototype_ce_weight": train_cfg.one_shot.aux_semantic_prototype_ce_weight,
        "aux_semantic_negative_label_margin_weight": train_cfg.one_shot.aux_semantic_negative_label_margin_weight,
        "aux_topology_plane_key": train_cfg.one_shot.aux_topology_plane_key,
        "aux_topology_weight": train_cfg.one_shot.aux_topology_weight,
        "aux_topology_dice_weight": train_cfg.one_shot.aux_topology_dice_weight,
        "aux_topology_mse_weight": train_cfg.one_shot.aux_topology_mse_weight,
        "aux_topology_target_transform": train_cfg.one_shot.aux_topology_target_transform,
        "code_diversity_weight": train_cfg.one_shot.code_diversity_weight,
        "sample_code_diversity_weight": train_cfg.one_shot.sample_code_diversity_weight,
        "teacher_pairwise_diversity_weight": train_cfg.one_shot.teacher_pairwise_diversity_weight,
        "cross_class_pairwise_weight": train_cfg.one_shot.cross_class_pairwise_weight,
        "diversity_min_teacher_mse": train_cfg.one_shot.diversity_min_teacher_mse,
        "student_pair_contrast_config": train_cfg.one_shot.student_pair_contrast_config,
        "student_display_speckle_weight": train_cfg.one_shot.student_display_speckle_weight,
        "aux_detail_edge_weight": train_cfg.one_shot.aux_detail_edge_weight,
        "aux_detail_shape_weight": train_cfg.one_shot.aux_detail_shape_weight,
        "aux_detail_foreground_gate_dilation": train_cfg.one_shot.aux_detail_foreground_gate_dilation,
        "residual_l1_weight": train_cfg.one_shot.residual_l1_weight,
        "residual_background_weight": train_cfg.one_shot.residual_background_weight,
    }


def _select_one_shot_visual_batch(
    torch_module, validation_bundle, train_bundle, train_cfg, device
):
    sample_labels, reference_images, _, _ = select_post_train_sample_batch(
        torch_module,
        validation_bundle,
        train_bundle,
        train_cfg.one_shot.num_classes,
        device,
        sample_class_labels=train_cfg.one_shot.sample_class_labels,
        sample_repeats_per_class=ONE_SHOT_DISTILL_EVAL_PREVIEW_PER_CLASS,
        allow_train_reference_fallback=True,
    )
    visual_codes = _sample_one_shot_codes(
        torch_module, int(sample_labels.shape[0]), train_cfg, device
    )
    return visual_codes, sample_labels, reference_images


def _evaluate_one_shot(
    torch_module,
    one_shot_model,
    one_shot_loss_fn,
    loader,
    train_cfg,
    device,
    use_pin_memory: bool,
    student_label_margin_prototypes=None,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    one_shot_model.eval()
    with torch_module.no_grad():
        for images, labels in loader:
            target_batch = images.to(device, non_blocking=use_pin_memory)
            class_labels = labels.to(device, non_blocking=use_pin_memory)
            codes = _sample_one_shot_codes(
                torch_module, target_batch.shape[0], train_cfg, device
            )
            outputs = one_shot_model(codes, class_labels)
            losses = one_shot_loss_fn(
                outputs,
                target_batch,
                class_labels=class_labels,
                **_one_shot_loss_kwargs(train_cfg, student_label_margin_prototypes),
            )
            batch_size = int(target_batch.shape[0])
            count += batch_size
            for key, value in losses.items():
                totals[key] = (
                    totals.get(key, 0.0) + float(value.detach().item()) * batch_size
                )
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
    student_label_margin_prototypes=None,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    crop_totals: dict[str, float] = {}
    crop_count = 0
    count = 0
    # Per-class accumulators
    per_class_holes: dict[int, float] = {}
    per_class_counts: dict[int, int] = {}
    per_class_label_totals: dict[str, float] = {}
    per_class_label_counts: dict[str, int] = {}
    # Paired metric accumulators (CPU, detached).
    all_student_camera: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    one_shot_model.eval()
    with torch_module.no_grad():
        for batch in loader:
            codes, labels, targets, *aux_items = batch
            code_batch = codes.to(device, non_blocking=use_pin_memory)
            class_labels = labels.to(device, non_blocking=use_pin_memory)
            target_batch = targets.to(device, non_blocking=use_pin_memory)
            aux_detail_batch = (
                aux_items[0].to(device, non_blocking=use_pin_memory)
                if aux_items
                else None
            )
            outputs = one_shot_model(code_batch, class_labels)
            camera_prediction = outputs["camera_plane"]
            losses = one_shot_loss_fn(
                outputs,
                target_batch,
                class_labels=class_labels,
                aux_detail_target=aux_detail_batch,
                **_one_shot_loss_kwargs(train_cfg, student_label_margin_prototypes),
            )
            batch_size = int(target_batch.shape[0])
            count += batch_size
            for key, value in losses.items():
                totals[key] = (
                    totals.get(key, 0.0) + float(value.detach().item()) * batch_size
                )
            # Accumulate student/target for paired corr/dice (same as strict eval).
            all_student_camera.append(
                camera_prediction.detach().float().clamp(0.0, 1.0).cpu()
            )
            all_targets.append(target_batch.detach().float().clamp(0.0, 1.0).cpu())
            crop_metrics = _one_shot_d2nn_crop_retention_metrics(outputs)
            if crop_metrics:
                crop_count += batch_size
                for key, value in crop_metrics.items():
                    crop_totals[key] = crop_totals.get(key, 0.0) + value * batch_size
            # Per-class label audit using prototypes (accumulates sums and counts).
            if student_label_margin_prototypes is not None and class_labels is not None:
                per_class_label, per_class_n = _per_class_label_audit_metrics(
                    camera_prediction,
                    class_labels,
                    student_label_margin_prototypes,
                    train_cfg.one_shot.num_classes,
                )
                for key, value in per_class_label.items():
                    per_class_label_totals[key] = (
                        per_class_label_totals.get(key, 0.0) + value
                    )
                for key, n in per_class_n.items():
                    per_class_label_counts[key] = per_class_label_counts.get(key, 0) + n

            # Per-class hole ratio aggregation.
            if class_labels is not None:
                device_labels = class_labels.detach().long().cpu()
                fg_mask = (
                    (
                        target_batch.detach()
                        > train_cfg.one_shot.recon_foreground_threshold
                    )
                    .float()
                    .cpu()
                )
                student_clamped = (
                    camera_prediction.detach().float().clamp(0.0, 1.0).cpu()
                )
                for c in range(int(train_cfg.one_shot.num_classes)):
                    mask = device_labels == c
                    if not mask.any():
                        continue
                    student_class = student_clamped[mask]
                    fg_class = fg_mask[mask]
                    hole_mask = (student_class < 0.10).float()
                    fg_area = fg_class.float().sum(dim=list(range(1, fg_class.ndim)))
                    hole_area = (hole_mask * fg_class).sum(
                        dim=list(range(1, hole_mask.ndim))
                    )
                    hole_frac = (hole_area / fg_area.clamp_min(1.0)).mean().item()
                    per_class_holes[c] = per_class_holes.get(c, 0.0) + hole_frac * int(
                        mask.sum().item()
                    )
                    per_class_counts[c] = per_class_counts.get(c, 0) + int(
                        mask.sum().item()
                    )
    if count <= 0:
        raise ValueError("Distillation validation loader is empty.")
    result: dict[str, float] = {
        key: value / float(count) for key, value in totals.items()
    }
    if crop_totals:
        for key, value in crop_totals.items():
            result[key] = value / float(crop_count)
    # Merge per-class metrics into result.
    for c, total_hole in per_class_holes.items():
        if per_class_counts.get(c, 0) > 0:
            result[f"per_class_{c}_foreground_hole_ratio"] = total_hole / float(
                per_class_counts[c]
            )
    # Merge per-class label audit metrics (divide sums by counts).
    for c in range(int(train_cfg.one_shot.num_classes)):
        count_key = f"per_class_{c}_count"
        if per_class_label_counts.get(count_key, 0) > 0:
            n = per_class_label_counts[count_key]
            correct_sum = per_class_label_totals.get(
                f"per_class_{c}_label_correct_count", 0.0
            )
            margin_sum = per_class_label_totals.get(
                f"per_class_{c}_label_margin_sum", 0.0
            )
            result[f"per_class_{c}_label_correct_ratio"] = correct_sum / float(n)
            result[f"per_class_{c}_label_margin_mean"] = margin_sum / float(n)
    # Paired corr/dice — computed across all accumulated batches (same as strict eval).
    if all_student_camera and all_targets:
        student_all = torch.cat(all_student_camera, dim=0)
        target_all = torch.cat(all_targets, dim=0)
        result["paired_corr"] = float(
            _eval_correlation(torch_module, student_all, target_all).item()
        )
        result["paired_dice"] = float(
            _eval_foreground_dice(student_all, target_all).item()
        )
    return result


def _per_class_label_audit_metrics(
    camera_prediction: torch.Tensor,
    class_labels: torch.Tensor,
    prototypes: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    """Compute per-class label correct ratio and margin mean from camera predictions.

    Uses the same feature normalization and prototype correlation as
    _student_label_margin_metrics, but returns per-class scalars instead of
    a single overall loss term.
    """
    from .one_shot_losses import _normalize_for_label_margin

    result: dict[str, float] = {}
    per_class_counts: dict[str, int] = {}
    import torch

    with torch.no_grad():
        pred = camera_prediction.detach().float()
        labels = class_labels.detach().long().reshape(-1)
        proto = prototypes.to(device=pred.device, dtype=pred.dtype)
        if proto.shape[-2:] != pred.shape[-2:]:
            proto = F.interpolate(
                proto, size=pred.shape[-2:], mode="bilinear", align_corners=False
            )

        sample_features = _normalize_for_label_margin(pred)
        prototype_features = _normalize_for_label_margin(proto)
        correlations = sample_features @ prototype_features.t()
        nearest_labels = correlations.argmax(dim=1)
        target_corr = correlations.gather(1, labels.view(-1, 1)).squeeze(1)
        other_corr = correlations.clone()
        other_corr.scatter_(1, labels.view(-1, 1), -2.0)
        best_other = other_corr.max(dim=1).values
        margins = target_corr - best_other

        for c in range(int(num_classes)):
            mask = labels == c
            if not mask.any():
                continue
            n = int(mask.sum().item())
            correct = (nearest_labels[mask] == labels[mask]).float()
            class_margins = margins[mask]
            # Return sums and counts for proper cross-batch averaging.
            result[f"per_class_{c}_label_correct_count"] = float(correct.sum().item())
            result[f"per_class_{c}_label_margin_sum"] = float(
                class_margins.sum().item()
            )
            per_class_counts[f"per_class_{c}_count"] = n
            # Confusion diagnostics (not averaged, just from this batch).
            confusion = nearest_labels[mask]
            confusion_counts: dict[str, int] = {}
            for predicted in confusion.tolist():
                key = str(predicted)
                confusion_counts[key] = confusion_counts.get(key, 0) + 1
            top_confused = (
                max(confusion_counts, key=lambda k: confusion_counts[k])
                if confusion_counts
                else "?"
            )
            result[f"per_class_{c}_confusion_top"] = (
                float(top_confused) if top_confused != "?" else 0.0
            )
            result[f"per_class_{c}_confusion_top_count"] = float(
                confusion_counts.get(top_confused, 0)
            )
    return result, per_class_counts


def _one_shot_parameter_groups(one_shot_model):
    groups = {
        "other": [],
        "generator_base": [],
        "generator_residual": [],
        "phase": [],
        "amplitude": [],
    }
    for name, param in one_shot_model.named_parameters():
        if "amplitude_logits" in name:
            groups["amplitude"].append((name, param))
        elif "phase" in name:
            groups["phase"].append((name, param))
        elif name.startswith("generator_e.residual_net."):
            groups["generator_residual"].append((name, param))
        elif name.startswith("generator_e."):
            groups["generator_base"].append((name, param))
        else:
            groups["other"].append((name, param))
    return groups


def _one_shot_group_learning_rates(train_cfg) -> dict[str, float]:
    phase_lr = float(train_cfg.one_shot.phase_learning_rate)
    amplitude_lr = (
        phase_lr
        if train_cfg.one_shot.amplitude_learning_rate is None
        else float(train_cfg.one_shot.amplitude_learning_rate)
    )
    base_lr = float(train_cfg.learning_rate)
    generator_base_lr = base_lr * float(train_cfg.one_shot.generator_base_lr_scale)
    generator_residual_lr = base_lr * float(
        train_cfg.one_shot.generator_residual_lr_scale
    )
    return {
        "other": base_lr,
        "generator_base": generator_base_lr,
        "generator_residual": generator_residual_lr,
        "phase": phase_lr,
        "amplitude": amplitude_lr,
    }


def _snapshot_frozen_one_shot_parameters(one_shot_model, train_cfg):
    groups = _one_shot_parameter_groups(one_shot_model)
    learning_rates = _one_shot_group_learning_rates(train_cfg)
    return {
        group_name: {
            name: parameter.detach().cpu().clone()
            for name, parameter in named_parameters
        }
        for group_name, named_parameters in groups.items()
        if learning_rates[group_name] == 0.0 and named_parameters
    }


def _verify_frozen_one_shot_parameters(one_shot_model, snapshots) -> dict[str, object]:
    current = dict(one_shot_model.named_parameters())
    group_reports: dict[str, object] = {}
    overall_max_abs_change = 0.0
    for group_name, group_snapshots in snapshots.items():
        group_max_abs_change = 0.0
        parameter_count = 0
        for name, before in group_snapshots.items():
            if name not in current:
                raise RuntimeError(
                    f"Frozen one-shot parameter disappeared during training: {name}"
                )
            after = current[name].detach().cpu()
            max_abs_change = float((after - before).abs().max().item())
            group_max_abs_change = max(group_max_abs_change, max_abs_change)
            parameter_count += int(after.numel())
        group_reports[group_name] = {
            "tensor_count": len(group_snapshots),
            "parameter_count": parameter_count,
            "max_abs_change": group_max_abs_change,
        }
        overall_max_abs_change = max(overall_max_abs_change, group_max_abs_change)
    report = {
        "passed": overall_max_abs_change == 0.0,
        "max_abs_change": overall_max_abs_change,
        "groups": group_reports,
    }
    if not report["passed"]:
        raise RuntimeError(
            "Frozen one-shot parameters changed despite zero effective learning rate: "
            f"max_abs_change={overall_max_abs_change}"
        )
    return report


def _build_one_shot_optimizer(torch_module, one_shot_model, train_cfg):
    groups = _one_shot_parameter_groups(one_shot_model)
    if bool(getattr(train_cfg.one_shot, "decoder_only_training", False)):
        skipped = 0
        for group_name, named_parameters in groups.items():
            kept = [
                (name, param)
                for name, param in named_parameters
                if name.startswith("decoder.")
            ]
            skipped += len(named_parameters) - len(kept)
            groups[group_name] = kept
        print(
            f"[optimizer] decoder-only training active; skipped {skipped} non-decoder tensors"
        )
    if bool(
        getattr(train_cfg.one_shot, "target_code_bridge_trainability_audit", False)
    ):
        skipped = 0
        for group_name, named_parameters in groups.items():
            kept = [
                (name, param)
                for name, param in named_parameters
                if name.startswith("target_code_bridge_adapter.")
            ]
            skipped += len(named_parameters) - len(kept)
            groups[group_name] = kept
        print(
            f"[optimizer] target-code bridge audit active; skipped {skipped} non-bridge tensors"
        )
    learning_rates = _one_shot_group_learning_rates(train_cfg)
    display_names = {
        "phase": "optical core phase",
        "amplitude": "optical core amplitude",
        "generator_base": "generator base",
        "generator_residual": "generator residual",
        "other": "other",
    }
    for group_name in ("phase", "amplitude", "generator_base", "generator_residual"):
        named_parameters = groups[group_name]
        if not named_parameters:
            continue
        print(
            f"[optimizer] {display_names[group_name]} params: "
            f"{sum(param.numel() for _, param in named_parameters)} total, "
            f"lr={learning_rates[group_name]:.1e}"
        )
    other_parameters = groups["other"]
    print(
        "[optimizer] other params: "
        f"{sum(param.numel() for _, param in other_parameters)} total, "
        f"lr={learning_rates['other']:.1e}"
    )
    param_groups = [
        {
            "params": [parameter for _, parameter in groups[group_name]],
            "lr": learning_rates[group_name],
        }
        for group_name in (
            "other",
            "generator_base",
            "generator_residual",
            "phase",
            "amplitude",
        )
        if groups[group_name]
    ]
    if not param_groups:
        raise ValueError("one-shot optimizer has no trainable parameters.")
    return torch_module.optim.Adam(param_groups)


def _tensor_norm_and_count(tensors) -> dict[str, float | int]:
    total_sq = 0.0
    count = 0
    max_abs = 0.0
    for tensor in tensors:
        value = tensor.detach().float().cpu()
        total_sq += float(value.pow(2).sum().item())
        count += int(value.numel())
        max_abs = max(
            max_abs, float(value.abs().max().item()) if value.numel() else 0.0
        )
    return {
        "parameter_count": count,
        "l2_norm": total_sq**0.5,
        "max_abs": max_abs,
    }


def _init_decoder_trainability_audit(one_shot_model, optimizer, train_cfg):
    if not bool(getattr(train_cfg.one_shot, "decoder_trainability_audit", False)):
        return None
    named = list(one_shot_model.named_parameters())
    decoder_named = [
        (name, param) for name, param in named if name.startswith("decoder.")
    ]
    upstream_named = [
        (name, param) for name, param in named if not name.startswith("decoder.")
    ]
    id_to_name = {id(param): name for name, param in named}
    optimizer_groups = []
    for group_index, group in enumerate(optimizer.param_groups):
        names = [
            id_to_name.get(id(param), "<unknown>") for param in group.get("params", [])
        ]
        optimizer_groups.append(
            {
                "group_index": group_index,
                "lr": float(group.get("lr", 0.0)),
                "tensor_count": len(names),
                "parameter_count": int(
                    sum(param.numel() for param in group.get("params", []))
                ),
                "decoder_tensor_count": sum(
                    1 for name in names if name.startswith("decoder.")
                ),
                "upstream_tensor_count": sum(
                    1 for name in names if not name.startswith("decoder.")
                ),
                "names": names,
            }
        )
    audit = {
        "enabled": True,
        "decoder_only_training": bool(
            getattr(train_cfg.one_shot, "decoder_only_training", False)
        ),
        "initial": {
            "decoder": _tensor_norm_and_count([param for _, param in decoder_named]),
            "upstream": _tensor_norm_and_count([param for _, param in upstream_named]),
        },
        "optimizer_param_groups": optimizer_groups,
        "gradients": {
            "steps": 0,
            "decoder_grad_l2_sum": 0.0,
            "decoder_grad_l2_max": 0.0,
            "decoder_grad_abs_max": 0.0,
            "decoder_grad_nonfinite_count": 0,
            "upstream_grad_l2_sum": 0.0,
            "upstream_grad_l2_max": 0.0,
            "upstream_grad_abs_max": 0.0,
            "upstream_grad_nonfinite_count": 0,
        },
        "_snapshots": {
            "decoder": {
                name: param.detach().cpu().clone() for name, param in decoder_named
            },
            "upstream": {
                name: param.detach().cpu().clone() for name, param in upstream_named
            },
        },
    }
    return audit


def _accumulate_decoder_trainability_gradients(audit, one_shot_model) -> None:
    if audit is None:
        return
    gradients = audit["gradients"]
    gradients["steps"] += 1
    decoder_sq = 0.0
    upstream_sq = 0.0
    decoder_abs_max = 0.0
    upstream_abs_max = 0.0
    decoder_nonfinite = 0
    upstream_nonfinite = 0
    for name, param in one_shot_model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        finite = torch.isfinite(grad)
        nonfinite_count = int((~finite).sum().item())
        safe_grad = torch.where(finite, grad, torch.zeros_like(grad))
        grad_sq = float(safe_grad.pow(2).sum().item())
        grad_abs = float(safe_grad.abs().max().item()) if safe_grad.numel() else 0.0
        if name.startswith("decoder."):
            decoder_sq += grad_sq
            decoder_abs_max = max(decoder_abs_max, grad_abs)
            decoder_nonfinite += nonfinite_count
        else:
            upstream_sq += grad_sq
            upstream_abs_max = max(upstream_abs_max, grad_abs)
            upstream_nonfinite += nonfinite_count
    decoder_l2 = decoder_sq**0.5
    upstream_l2 = upstream_sq**0.5
    gradients["decoder_grad_l2_sum"] += decoder_l2
    gradients["decoder_grad_l2_max"] = max(
        float(gradients["decoder_grad_l2_max"]), decoder_l2
    )
    gradients["decoder_grad_abs_max"] = max(
        float(gradients["decoder_grad_abs_max"]), decoder_abs_max
    )
    gradients["decoder_grad_nonfinite_count"] += decoder_nonfinite
    gradients["upstream_grad_l2_sum"] += upstream_l2
    gradients["upstream_grad_l2_max"] = max(
        float(gradients["upstream_grad_l2_max"]), upstream_l2
    )
    gradients["upstream_grad_abs_max"] = max(
        float(gradients["upstream_grad_abs_max"]), upstream_abs_max
    )
    gradients["upstream_grad_nonfinite_count"] += upstream_nonfinite


def _finalize_decoder_trainability_audit(audit, one_shot_model):
    if audit is None:
        return None
    current = dict(one_shot_model.named_parameters())
    final_summary: dict[str, object] = {}
    failure_reasons: list[str] = []
    for group_name, snapshots in audit.pop("_snapshots").items():
        update_sq = 0.0
        max_abs_update = 0.0
        final_tensors = []
        for name, before in snapshots.items():
            if name not in current:
                failure_reasons.append(f"missing-final-parameter:{name}")
                continue
            after = current[name].detach().cpu()
            diff = (after - before).float()
            update_sq += float(diff.pow(2).sum().item())
            max_abs_update = max(
                max_abs_update, float(diff.abs().max().item()) if diff.numel() else 0.0
            )
            final_tensors.append(after)
        final_summary[group_name] = {
            **_tensor_norm_and_count(final_tensors),
            "update_l2_norm": update_sq**0.5,
            "update_max_abs": max_abs_update,
        }
    gradients = audit["gradients"]
    steps = max(int(gradients["steps"]), 1)
    gradients["decoder_grad_l2_mean"] = float(gradients["decoder_grad_l2_sum"]) / float(
        steps
    )
    gradients["upstream_grad_l2_mean"] = float(
        gradients["upstream_grad_l2_sum"]
    ) / float(steps)
    decoder_final = final_summary.get("decoder", {})
    upstream_final = final_summary.get("upstream", {})
    if float(decoder_final.get("update_l2_norm", 0.0)) <= 0.0:
        failure_reasons.append("decoder-update-zero")
    if int(gradients.get("decoder_grad_nonfinite_count", 0)) > 0:
        failure_reasons.append("decoder-gradient-nonfinite")
    if float(gradients.get("decoder_grad_l2_max", 0.0)) <= 0.0:
        failure_reasons.append("decoder-gradient-zero")
    if (
        audit.get("decoder_only_training")
        and float(upstream_final.get("update_max_abs", 0.0)) != 0.0
    ):
        failure_reasons.append("upstream-updated-during-decoder-only")
    audit["final"] = final_summary
    audit["failure_reasons"] = failure_reasons
    audit["passed"] = not failure_reasons
    return audit


def _init_target_code_bridge_trainability_audit(one_shot_model, optimizer, train_cfg):
    if not bool(
        getattr(train_cfg.one_shot, "target_code_bridge_trainability_audit", False)
    ):
        return None
    named = list(one_shot_model.named_parameters())
    bridge_named = [
        (name, param)
        for name, param in named
        if name.startswith("target_code_bridge_adapter.")
    ]
    non_bridge_named = [
        (name, param)
        for name, param in named
        if not name.startswith("target_code_bridge_adapter.")
    ]
    id_to_name = {id(param): name for name, param in named}
    optimizer_groups = []
    for group_index, group in enumerate(optimizer.param_groups):
        names = [
            id_to_name.get(id(param), "<unknown>") for param in group.get("params", [])
        ]
        optimizer_groups.append(
            {
                "group_index": group_index,
                "lr": float(group.get("lr", 0.0)),
                "tensor_count": len(names),
                "parameter_count": int(
                    sum(param.numel() for param in group.get("params", []))
                ),
                "bridge_tensor_count": sum(
                    1
                    for name in names
                    if name.startswith("target_code_bridge_adapter.")
                ),
                "non_bridge_tensor_count": sum(
                    1
                    for name in names
                    if not name.startswith("target_code_bridge_adapter.")
                ),
                "names": names,
            }
        )
    return {
        "enabled": True,
        "target_code_bridge_source": getattr(
            train_cfg.one_shot, "target_code_bridge_source", "none"
        ),
        "target_code_bridge_arch": getattr(
            train_cfg.one_shot, "target_code_bridge_arch", "shallow"
        ),
        "target_code_diagnostic_mode": getattr(
            train_cfg.one_shot, "target_code_diagnostic_mode", "none"
        ),
        "initial": {
            "bridge": _tensor_norm_and_count([param for _, param in bridge_named]),
            "non_bridge": _tensor_norm_and_count(
                [param for _, param in non_bridge_named]
            ),
        },
        "optimizer_param_groups": optimizer_groups,
        "gradients": {
            "steps": 0,
            "bridge_grad_l2_sum": 0.0,
            "bridge_grad_l2_max": 0.0,
            "bridge_grad_abs_max": 0.0,
            "bridge_grad_nonfinite_count": 0,
            "non_bridge_grad_l2_sum": 0.0,
            "non_bridge_grad_l2_max": 0.0,
            "non_bridge_grad_abs_max": 0.0,
            "non_bridge_grad_nonfinite_count": 0,
        },
        "_snapshots": {
            "bridge": {
                name: param.detach().cpu().clone() for name, param in bridge_named
            },
            "non_bridge": {
                name: param.detach().cpu().clone() for name, param in non_bridge_named
            },
        },
    }


def _accumulate_target_code_bridge_trainability_gradients(
    audit, one_shot_model
) -> None:
    if audit is None:
        return
    gradients = audit["gradients"]
    gradients["steps"] += 1
    bridge_sq = 0.0
    non_bridge_sq = 0.0
    bridge_abs_max = 0.0
    non_bridge_abs_max = 0.0
    bridge_nonfinite = 0
    non_bridge_nonfinite = 0
    for name, param in one_shot_model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        finite = torch.isfinite(grad)
        nonfinite_count = int((~finite).sum().item())
        safe_grad = torch.where(finite, grad, torch.zeros_like(grad))
        grad_sq = float(safe_grad.pow(2).sum().item())
        grad_abs = float(safe_grad.abs().max().item()) if safe_grad.numel() else 0.0
        if name.startswith("target_code_bridge_adapter."):
            bridge_sq += grad_sq
            bridge_abs_max = max(bridge_abs_max, grad_abs)
            bridge_nonfinite += nonfinite_count
        else:
            non_bridge_sq += grad_sq
            non_bridge_abs_max = max(non_bridge_abs_max, grad_abs)
            non_bridge_nonfinite += nonfinite_count
    bridge_l2 = bridge_sq**0.5
    non_bridge_l2 = non_bridge_sq**0.5
    gradients["bridge_grad_l2_sum"] += bridge_l2
    gradients["bridge_grad_l2_max"] = max(
        float(gradients["bridge_grad_l2_max"]), bridge_l2
    )
    gradients["bridge_grad_abs_max"] = max(
        float(gradients["bridge_grad_abs_max"]), bridge_abs_max
    )
    gradients["bridge_grad_nonfinite_count"] += bridge_nonfinite
    gradients["non_bridge_grad_l2_sum"] += non_bridge_l2
    gradients["non_bridge_grad_l2_max"] = max(
        float(gradients["non_bridge_grad_l2_max"]), non_bridge_l2
    )
    gradients["non_bridge_grad_abs_max"] = max(
        float(gradients["non_bridge_grad_abs_max"]), non_bridge_abs_max
    )
    gradients["non_bridge_grad_nonfinite_count"] += non_bridge_nonfinite


def _finalize_target_code_bridge_trainability_audit(audit, one_shot_model):
    if audit is None:
        return None
    current = dict(one_shot_model.named_parameters())
    final_summary: dict[str, object] = {}
    failure_reasons: list[str] = []
    for group_name, snapshots in audit.pop("_snapshots").items():
        update_sq = 0.0
        max_abs_update = 0.0
        final_tensors = []
        for name, before in snapshots.items():
            if name not in current:
                failure_reasons.append(f"missing-final-parameter:{name}")
                continue
            after = current[name].detach().cpu()
            diff = (after - before).float()
            update_sq += float(diff.pow(2).sum().item())
            max_abs_update = max(
                max_abs_update, float(diff.abs().max().item()) if diff.numel() else 0.0
            )
            final_tensors.append(after)
        final_summary[group_name] = {
            **_tensor_norm_and_count(final_tensors),
            "update_l2_norm": update_sq**0.5,
            "update_max_abs": max_abs_update,
        }
    gradients = audit["gradients"]
    steps = max(int(gradients["steps"]), 1)
    gradients["bridge_grad_l2_mean"] = float(gradients["bridge_grad_l2_sum"]) / float(
        steps
    )
    gradients["non_bridge_grad_l2_mean"] = float(
        gradients["non_bridge_grad_l2_sum"]
    ) / float(steps)
    bridge_final = final_summary.get("bridge", {})
    non_bridge_final = final_summary.get("non_bridge", {})
    if int(bridge_final.get("parameter_count", 0)) <= 0:
        failure_reasons.append("bridge-parameter-missing")
    if float(bridge_final.get("update_l2_norm", 0.0)) <= 0.0:
        failure_reasons.append("bridge-update-zero")
    if int(gradients.get("bridge_grad_nonfinite_count", 0)) > 0:
        failure_reasons.append("bridge-gradient-nonfinite")
    if float(gradients.get("bridge_grad_l2_max", 0.0)) <= 0.0:
        failure_reasons.append("bridge-gradient-zero")
    if float(non_bridge_final.get("update_max_abs", 0.0)) != 0.0:
        failure_reasons.append("non-bridge-updated-during-bridge-only")
    audit["final"] = final_summary
    audit["failure_reasons"] = failure_reasons
    audit["passed"] = not failure_reasons
    return audit


def _phase_grad_summary(model) -> str:
    parts: list[str] = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        module_name = name.split(".")[0]
        if name.endswith(".phase"):
            parts.append(
                f"{module_name}_phase_grad={param.grad.abs().mean().item():.2e}"
            )
        elif name.endswith(".amplitude_logits"):
            parts.append(f"{module_name}_amp_grad={param.grad.abs().mean().item():.2e}")
    return " ".join(parts)


def _mask_amplitude_summary(model) -> str:
    values = []
    for module_name in ("encoder", "decoder"):
        module = getattr(model, module_name, None)
        if module is None or not hasattr(module, "phase_transfers"):
            continue
        amplitude = module.phase_transfers().detach().abs().float()
        values.append(amplitude.flatten())
    if not values:
        return ""
    masks = torch.cat(values)
    return (
        f"mask_amp_mean={masks.mean().item():.4f} "
        f"mask_amp_std={masks.std(unbiased=False).item():.4f} "
        f"mask_amp_min={masks.min().item():.4f} "
        f"mask_amp_max={masks.max().item():.4f}"
    )


def _log_phase_gradients(model) -> None:
    summary = _phase_grad_summary(model)
    if summary:
        print(f"[phase_grad] {summary}")
