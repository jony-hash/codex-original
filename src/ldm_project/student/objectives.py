# ruff: noqa: F401
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..distillation.targets import build_clean_readable_target
from ..training_loss_common import (
    _camera_artifact_metrics,
    _camera_high_saturation_ratio,
    _camera_low_saturation_ratio,
    _camera_saturation_loss,
    _camera_saturation_ratio,
    _camera_topology_loss,
    _foreground_dice_score,
    _foreground_shape_loss_for_samples,
    _masked_mean,
    _masked_mean_per_sample,
    _sobel_magnitude,
    _total_variation,
    _weighted_sample_mean,
    _zero_like_loss,
)
from .objective_composition import compose_one_shot_total
from .objective_terms import formal_reconstruction_terms

from .objective_terms.diversity import (
    _diversity_terms,
    _one_shot_cross_class_pairwise_metrics,
    _one_shot_pairwise_diversity_metrics,
    _pairwise_mse_summary,
    _per_class_std_summary,
)
from .objective_terms.readability import (
    _camera_background_highfreq_metrics,
    _camera_background_hotspot_metrics,
    _camera_background_speckle_metrics,
    _camera_hotspot_ratio,
    _camera_readability_metrics,
    _camera_readability_terms,
    _camera_readable_contrast_metrics,
    _clean_target_regularization_terms,
)
from .objective_terms.residual import (
    _generator_residual_terms,
    _student_display_speckle_metrics,
    _student_pair_contrast_metrics,
)
from .objective_terms.semantic import (
    _aux_semantic_objective_terms,
    _normalize_for_label_margin,
    _resize_aux_semantic_plane,
    _student_label_margin_metrics,
    _student_label_margin_terms,
    _student_negative_label_margin_metrics,
    _student_prototype_ce_metrics,
    _student_prototype_correlations,
    _student_prototype_objective_terms,
    _student_roi_negative_label_margin_metrics,
    _student_roi_prototype_ce_metrics,
    _student_roi_prototype_correlations,
)
from .objective_terms.topology import (
    _aux_detail_terms,
    _aux_topology_metrics,
    _aux_topology_terms,
    _dilate_foreground_mask,
    _encoder_input_topology_metrics,
    _one_shot_edge_shape_loss,
    _resolve_clean_target_masks,
    _resolve_sample_weights,
    transform_aux_topology_target,
)


# Legacy private import retained for existing objective and test callers.
_build_clean_readable_target = build_clean_readable_target


def _reconstruction_terms(
    outputs: dict[str, torch.Tensor],
    target_batch: torch.Tensor,
    class_labels: torch.Tensor | None,
    *,
    foreground_threshold: float,
    foreground_weight: float,
    background_weight: float,
    class_weight_labels: tuple[int, ...],
    class_weight: float,
) -> dict[str, torch.Tensor]:
    return formal_reconstruction_terms(
        outputs,
        target_batch,
        class_labels,
        foreground_threshold=foreground_threshold,
        foreground_weight=foreground_weight,
        background_weight=background_weight,
        class_weight_labels=class_weight_labels,
        class_weight=class_weight,
        sample_weight_resolver=_resolve_sample_weights,
    )


def one_shot_loss(
    outputs: dict[str, torch.Tensor],
    target_batch: torch.Tensor,
    class_labels: torch.Tensor | None = None,
    aux_detail_target: torch.Tensor | None = None,
    foreground_threshold: float = 0.05,
    foreground_weight: float = 3.0,
    background_weight: float = 1.0,
    class_recon_weight_labels: tuple[int, ...] = (),
    class_recon_weight: float = 1.0,
    shape_loss_weight: float = 0.0,
    edge_loss_weight: float = 0.0,
    slm_smoothness_weight: float = 0.0,
    slm_contrast_weight: float = 0.0,
    camera_saturation_weight: float = 0.0,
    camera_topology_weight: float = 0.0,
    camera_artifact_weight: float = 0.0,
    camera_readability_weight: float = 0.0,
    camera_background_speckle_weight: float = 0.0,
    camera_background_hotspot_weight: float = 0.0,
    camera_background_hotspot_threshold: float = 0.08,
    camera_background_highfreq_weight: float = 0.0,
    camera_readable_contrast_weight: float = 0.0,
    camera_readable_contrast_target: float = 4.5,
    camera_recon_weight: float = 1.0,
    clean_target_enabled: bool = False,
    clean_foreground_dice_weight: float = 0.0,
    clean_foreground_hole_weight: float = 0.0,
    clean_background_speckle_weight: float = 0.0,
    clean_background_hotspot_weight: float = 0.0,
    clean_background_highfreq_weight: float = 0.0,
    encoder_input_topology_weight: float = 0.0,
    encoder_input_topology_dice_weight: float = 1.0,
    encoder_input_topology_mse_weight: float = 1.0,
    student_label_margin_weight: float = 0.0,
    student_label_margin_target: float = 0.03,
    student_label_margin_labels: tuple[int, ...] = (),
    student_label_margin_focus_label: int | None = None,
    student_label_margin_focus_weight: float = 1.0,
    student_label_margin_prototypes: torch.Tensor | None = None,
    student_prototype_ce_weight: float = 0.0,
    student_prototype_ce_labels: tuple[int, ...] = (),
    student_prototype_ce_temperature: float = 0.07,
    student_negative_label_margin_weight: float = 0.0,
    student_negative_label_margin_config: tuple[tuple[int, int, float], ...] = (),
    student_roi_prototype_ce_weight: float = 0.0,
    student_roi_prototype_ce_labels: tuple[int, ...] = (),
    student_roi_prototype_ce_temperature: float = 0.07,
    student_roi_negative_label_margin_weight: float = 0.0,
    student_roi_negative_label_margin_config: tuple[tuple[int, int, float], ...] = (),
    student_roi_foreground_threshold: float = 0.20,
    aux_semantic_plane_key: str = "",
    aux_semantic_weight: float = 0.0,
    aux_semantic_prototype_ce_weight: float = 0.0,
    aux_semantic_negative_label_margin_weight: float = 0.0,
    aux_topology_plane_key: str = "",
    aux_topology_weight: float = 0.0,
    aux_topology_dice_weight: float = 1.0,
    aux_topology_mse_weight: float = 1.0,
    aux_topology_target_transform: str = "none",
    student_pair_contrast_config: tuple[tuple[int, int, float], ...] = (),
    student_display_speckle_weight: float = 0.0,
    aux_detail_edge_weight: float = 0.0,
    aux_detail_shape_weight: float = 0.0,
    aux_detail_foreground_gate_dilation: int = 0,
    residual_l1_weight: float = 0.0,
    residual_background_weight: float = 0.0,
    code_diversity_weight: float = 0.0,
    sample_code_diversity_weight: float = 0.0,
    teacher_pairwise_diversity_weight: float = 0.0,
    cross_class_pairwise_weight: float = 0.0,
    diversity_min_teacher_mse: float = 1.0e-4,
) -> dict[str, torch.Tensor]:
    reconstruction = _reconstruction_terms(
        outputs,
        target_batch,
        class_labels,
        foreground_threshold=foreground_threshold,
        foreground_weight=foreground_weight,
        background_weight=background_weight,
        class_weight_labels=class_recon_weight_labels,
        class_weight=class_recon_weight,
    )
    foreground_mask = reconstruction["foreground_mask"]
    background_mask = reconstruction["background_mask"]
    camera_prediction = reconstruction["camera_prediction"]
    camera_loss_prediction = reconstruction["camera_loss_prediction"]
    sample_weights = reconstruction["sample_weights"]
    camera_metrics = reconstruction["metrics"]
    zero = reconstruction["zero"]
    (
        clean_target,
        clean_foreground_mask,
        clean_background_mask,
        clean_target_fg_ratio,
    ) = _resolve_clean_target_masks(
        target_batch,
        foreground_mask,
        background_mask,
        enabled=clean_target_enabled,
        zero=zero,
    )

    shape_base = (
        _foreground_shape_loss_for_samples(
            camera_prediction, clean_foreground_mask, sample_weights
        )
        if shape_loss_weight != 0.0
        else zero
    )
    shape_loss = shape_loss_weight * shape_base
    edge_base = (
        _one_shot_edge_shape_loss(
            camera_prediction, clean_target, clean_foreground_mask, sample_weights
        )
        if edge_loss_weight != 0.0
        else zero
    )
    edge_loss = edge_loss_weight * edge_base
    slm_tv = _total_variation(outputs["slm_amplitude"])
    slm_smoothness = slm_smoothness_weight * slm_tv
    slm_amplitude = outputs["slm_amplitude"]
    if slm_contrast_weight != 0.0:
        grid_tensor = outputs.get("encoder_input_grid_amplitude")
        if grid_tensor is not None:
            grid_h, grid_w = grid_tensor.shape[-2:]
            slm_grid = F.interpolate(slm_amplitude, size=(grid_h, grid_w), mode="area")
        else:
            slm_grid = slm_amplitude
        per_sample_slm_std = slm_grid.reshape(slm_amplitude.shape[0], -1).std(dim=1)
        slm_contrast_base = F.relu(0.12 - per_sample_slm_std).mean()
        slm_contrast = slm_contrast_weight * slm_contrast_base
    else:
        per_sample_slm_std = slm_amplitude.reshape(slm_amplitude.shape[0], -1).std(
            dim=1
        )
        slm_contrast_base = zero
        slm_contrast = zero
    camera_saturation_base = _camera_saturation_loss(camera_loss_prediction)
    camera_saturation = camera_saturation_weight * camera_saturation_base
    # 拓扑约束：保持笔画连接和弯折的结构，对复杂数字(3/4/5)尤为重要。
    topology_target = clean_target if clean_target_enabled else target_batch
    topology_fg = clean_foreground_mask if clean_target_enabled else foreground_mask
    camera_topology_base = (
        _camera_topology_loss(camera_prediction, topology_target, topology_fg)
        if camera_topology_weight != 0.0
        else zero
    )
    camera_topology = camera_topology_weight * camera_topology_base
    # 抗黑斑：惩罚前景笔画内部的暗斑和粗糙纹理。
    artifact_metrics = _camera_artifact_metrics(
        camera_loss_prediction, target_batch, foreground_mask
    )
    camera_artifact_base = artifact_metrics["artifact_base"]
    foreground_hole_ratio = artifact_metrics["foreground_hole_ratio"]
    camera_artifact = camera_artifact_weight * camera_artifact_base
    clean_terms = _clean_target_regularization_terms(
        camera_prediction,
        camera_loss_prediction,
        clean_target,
        clean_foreground_mask,
        clean_background_mask,
        enabled=clean_target_enabled,
        foreground_dice_weight=clean_foreground_dice_weight,
        foreground_hole_weight=clean_foreground_hole_weight,
        background_speckle_weight=clean_background_speckle_weight,
        background_hotspot_weight=clean_background_hotspot_weight,
        background_highfreq_weight=clean_background_highfreq_weight,
        background_hotspot_threshold=camera_background_hotspot_threshold,
        zero=zero,
    )
    clean_dice_base = clean_terms["dice_base"]
    clean_dice_loss = clean_terms["dice_loss"]
    clean_fg_hole_base = clean_terms["foreground_hole_base"]
    clean_fg_hole_loss = clean_terms["foreground_hole_loss"]
    clean_fg_hole_ratio = clean_terms["foreground_hole_ratio"]
    clean_bg_speckle_base = clean_terms["background_speckle_base"]
    clean_bg_speckle_loss = clean_terms["background_speckle_loss"]
    clean_bg_speckle_ratio = clean_terms["background_speckle_ratio"]
    clean_bg_hotspot_base = clean_terms["background_hotspot_base"]
    clean_bg_hotspot_loss = clean_terms["background_hotspot_loss"]
    clean_bg_hotspot_ratio = clean_terms["background_hotspot_ratio"]
    clean_bg_highfreq_base = clean_terms["background_highfreq_base"]
    clean_bg_highfreq_loss = clean_terms["background_highfreq_loss"]
    clean_bg_highfreq_energy = clean_terms["background_highfreq_energy"]

    if encoder_input_topology_weight != 0.0:
        encoder_topology_metrics = _encoder_input_topology_metrics(
            outputs, topology_target, topology_fg
        )
        encoder_input_topology_dice = encoder_topology_metrics["dice"]
        encoder_input_topology_mse = encoder_topology_metrics["mse"]
        encoder_input_topology_base = (
            float(encoder_input_topology_dice_weight)
            * encoder_topology_metrics["dice_penalty"]
            + float(encoder_input_topology_mse_weight) * encoder_input_topology_mse
        )
    else:
        encoder_input_topology_dice = zero
        encoder_input_topology_mse = zero
        encoder_input_topology_base = zero
    encoder_input_topology = encoder_input_topology_weight * encoder_input_topology_base

    readability_terms = _camera_readability_terms(
        camera_prediction,
        camera_loss_prediction,
        target_batch,
        foreground_mask,
        background_mask,
        readability_weight=camera_readability_weight,
        background_speckle_weight=camera_background_speckle_weight,
        background_hotspot_weight=camera_background_hotspot_weight,
        background_hotspot_threshold=camera_background_hotspot_threshold,
        background_highfreq_weight=camera_background_highfreq_weight,
        readable_contrast_weight=camera_readable_contrast_weight,
        readable_contrast_target=camera_readable_contrast_target,
        zero=zero,
    )
    camera_readability_penalty = readability_terms["readability_penalty"]
    camera_readability = readability_terms["readability_loss"]
    camera_background_speckle_base = readability_terms["background_speckle_base"]
    background_speckle_ratio = readability_terms["background_speckle_ratio"]
    camera_background_speckle = readability_terms["background_speckle_loss"]
    camera_background_hotspot_base = readability_terms["background_hotspot_base"]
    background_hotspot_ratio = readability_terms["background_hotspot_ratio"]
    camera_background_hotspot = readability_terms["background_hotspot_loss"]
    camera_background_highfreq_base = readability_terms["background_highfreq_base"]
    background_highfreq_energy = readability_terms["background_highfreq_energy"]
    camera_background_highfreq = readability_terms["background_highfreq_loss"]
    camera_readable_contrast_base = readability_terms["readable_contrast_base"]
    camera_readable_contrast_value = readability_terms["readable_contrast"]
    camera_readable_contrast = readability_terms["readable_contrast_loss"]
    label_margin_terms = _student_label_margin_terms(
        camera_prediction,
        class_labels,
        student_label_margin_prototypes,
        weight=student_label_margin_weight,
        labels=student_label_margin_labels,
        target=student_label_margin_target,
        focus_label=student_label_margin_focus_label,
        focus_weight=student_label_margin_focus_weight,
        zero=zero,
    )
    student_label_margin_base = label_margin_terms["base"]
    student_label_margin = label_margin_terms["loss"]
    student_label_margin_mean = label_margin_terms["margin_mean"]
    student_label_correct_ratio = label_margin_terms["correct_ratio"]
    prototype_terms = _student_prototype_objective_terms(
        camera_prediction,
        class_labels,
        student_label_margin_prototypes,
        prototype_ce_weight=student_prototype_ce_weight,
        prototype_ce_labels=student_prototype_ce_labels,
        prototype_ce_temperature=student_prototype_ce_temperature,
        negative_margin_weight=student_negative_label_margin_weight,
        negative_margin_config=student_negative_label_margin_config,
        roi_prototype_ce_weight=student_roi_prototype_ce_weight,
        roi_prototype_ce_labels=student_roi_prototype_ce_labels,
        roi_prototype_ce_temperature=student_roi_prototype_ce_temperature,
        roi_negative_margin_weight=student_roi_negative_label_margin_weight,
        roi_negative_margin_config=student_roi_negative_label_margin_config,
        roi_foreground_threshold=student_roi_foreground_threshold,
        zero=zero,
    )
    student_prototype_ce_base = prototype_terms["prototype_base"]
    student_prototype_ce_loss = prototype_terms["prototype_loss"]
    student_prototype_ce_correct_ratio = prototype_terms["prototype_correct_ratio"]
    student_prototype_ce_count = prototype_terms["prototype_count"]
    student_negative_label_margin_base = prototype_terms["negative_base"]
    student_negative_label_margin_loss = prototype_terms["negative_loss"]
    student_negative_label_margin_mean = prototype_terms["negative_mean"]
    student_negative_label_margin_count = prototype_terms["negative_count"]
    student_roi_prototype_ce_base = prototype_terms["roi_prototype_base"]
    student_roi_prototype_ce_loss = prototype_terms["roi_prototype_loss"]
    student_roi_prototype_ce_correct_ratio = prototype_terms[
        "roi_prototype_correct_ratio"
    ]
    student_roi_prototype_ce_count = prototype_terms["roi_prototype_count"]
    student_roi_negative_label_margin_base = prototype_terms["roi_negative_base"]
    student_roi_negative_label_margin_loss = prototype_terms["roi_negative_loss"]
    student_roi_negative_label_margin_mean = prototype_terms["roi_negative_mean"]
    student_roi_negative_label_margin_count = prototype_terms["roi_negative_count"]
    aux_semantic_terms = _aux_semantic_objective_terms(
        outputs,
        target_batch,
        class_labels,
        student_label_margin_prototypes,
        plane_key=aux_semantic_plane_key,
        weight=aux_semantic_weight,
        prototype_ce_weight=aux_semantic_prototype_ce_weight,
        negative_margin_weight=aux_semantic_negative_label_margin_weight,
        prototype_ce_labels=student_prototype_ce_labels,
        prototype_ce_temperature=student_prototype_ce_temperature,
        negative_margin_config=student_negative_label_margin_config,
        zero=zero,
    )
    aux_semantic_loss = aux_semantic_terms["loss"]
    aux_semantic_prototype_ce_base = aux_semantic_terms["prototype_base"]
    aux_semantic_prototype_ce_loss = aux_semantic_terms["prototype_loss"]
    aux_semantic_prototype_ce_correct_ratio = aux_semantic_terms[
        "prototype_correct_ratio"
    ]
    aux_semantic_prototype_ce_count = aux_semantic_terms["prototype_count"]
    aux_semantic_negative_label_margin_base = aux_semantic_terms["negative_base"]
    aux_semantic_negative_label_margin_loss = aux_semantic_terms["negative_loss"]
    aux_semantic_negative_label_margin_mean = aux_semantic_terms["negative_mean"]
    aux_semantic_negative_label_margin_count = aux_semantic_terms["negative_count"]
    aux_topology_terms = _aux_topology_terms(
        outputs,
        topology_target,
        plane_key=aux_topology_plane_key,
        weight=aux_topology_weight,
        dice_weight=aux_topology_dice_weight,
        mse_weight=aux_topology_mse_weight,
        target_transform=aux_topology_target_transform,
        foreground_threshold=foreground_threshold,
        zero=zero,
    )
    aux_topology_target_info = aux_topology_terms["target_info"]
    aux_topology_dice = aux_topology_terms["dice"]
    aux_topology_mse = aux_topology_terms["mse"]
    aux_topology_loss = aux_topology_terms["loss"]
    # Student pair contrast: penalize confusion between problematic class pairs.
    if student_pair_contrast_config:
        pair_contrast_metrics = _student_pair_contrast_metrics(
            camera_prediction,
            class_labels,
            student_pair_contrast_config,
        )
        student_pair_contrast_base = pair_contrast_metrics["student_pair_contrast_base"]
        student_pair_contrast_count = pair_contrast_metrics[
            "student_pair_contrast_count"
        ]
    else:
        student_pair_contrast_base = zero
        student_pair_contrast_count = zero
    student_pair_contrast_loss = (
        student_pair_contrast_base  # no extra weight; base is the mean penalty
    )
    # Student display-speckle: penalize background noise in display-normalized domain.
    if student_display_speckle_weight != 0.0:
        display_background_mask = (
            1.0 - clean_foreground_mask if clean_target_enabled else background_mask
        )
        display_speckle_metrics = _student_display_speckle_metrics(
            camera_prediction,
            foreground_mask,
            display_background_mask,
        )
        display_speckle_base = display_speckle_metrics["display_speckle_base"]
        display_speckle_ratio = display_speckle_metrics["display_speckle_ratio"]
        display_highfreq_energy = display_speckle_metrics["display_highfreq_energy"]
    else:
        display_speckle_base = zero
        display_speckle_ratio = zero
        display_highfreq_energy = zero
    display_speckle_loss = student_display_speckle_weight * display_speckle_base
    aux_detail_terms = _aux_detail_terms(
        camera_prediction,
        target_batch,
        aux_detail_target,
        clean_foreground_mask,
        sample_weights,
        foreground_threshold=foreground_threshold,
        gate_dilation=aux_detail_foreground_gate_dilation,
        shape_weight=aux_detail_shape_weight,
        edge_weight=aux_detail_edge_weight,
        zero=zero,
    )
    aux_detail_gate_mask = aux_detail_terms["gate_mask"]
    aux_detail_gate_fg_ratio = aux_detail_terms["gate_foreground_ratio"]
    aux_detail_active_fg_ratio = aux_detail_terms["active_foreground_ratio"]
    aux_detail_shape_base = aux_detail_terms["shape_base"]
    aux_detail_shape_loss = aux_detail_terms["shape_loss"]
    aux_detail_edge_base = aux_detail_terms["edge_base"]
    aux_detail_edge_loss = aux_detail_terms["edge_loss"]
    residual_terms = _generator_residual_terms(
        outputs,
        camera_prediction,
        foreground_mask,
        background_mask,
        clean_background_mask,
        aux_detail_gate_mask,
        l1_weight=residual_l1_weight,
        background_weight=residual_background_weight,
        zero=zero,
    )
    residual_l1_base = residual_terms["l1_base"]
    residual_l1_loss = residual_terms["l1_loss"]
    residual_background_base = residual_terms["background_base"]
    residual_background_loss = residual_terms["background_loss"]
    residual_background_mask_ratio = residual_terms["background_mask_ratio"]
    generator_residual_abs_mean = residual_terms["residual_abs_mean"]
    generator_residual_bg_abs_mean = residual_terms["residual_background_abs_mean"]
    student_base_mse = residual_terms["student_base_mse"]
    student_base_foreground_mse = residual_terms["student_base_foreground_mse"]
    student_base_background_mse = residual_terms["student_base_background_mse"]
    diversity_terms = _diversity_terms(
        camera_prediction,
        target_batch,
        class_labels,
        slm_amplitude,
        code_weight=code_diversity_weight,
        sample_code_weight=sample_code_diversity_weight,
        teacher_pairwise_weight=teacher_pairwise_diversity_weight,
        cross_class_weight=cross_class_pairwise_weight,
        minimum_teacher_mse=diversity_min_teacher_mse,
    )
    diversity_metrics = diversity_terms["pairwise"]
    cross_class_metrics = diversity_terms["cross_class"]
    code_diversity = diversity_terms["code_loss"]
    sample_code_diversity = diversity_terms["sample_code_loss"]
    teacher_pairwise_diversity = diversity_terms["teacher_pairwise_loss"]
    cross_class_pairwise = diversity_terms["cross_class_loss"]
    slm_pairwise_mse = diversity_terms["slm_pairwise_mse"]
    slm_per_class_std_mean = diversity_terms["slm_per_class_std_mean"]
    total = compose_one_shot_total(
        {
            "camera_recon_loss": camera_recon_weight * camera_metrics["recon_loss"],
            "shape_loss": shape_loss,
            "edge_loss": edge_loss,
            "slm_smoothness_loss": slm_smoothness,
            "slm_contrast_loss": slm_contrast,
            "camera_saturation_loss": camera_saturation,
            "camera_topology_loss": camera_topology,
            "camera_artifact_loss": camera_artifact,
            "camera_readability_loss": camera_readability,
            "camera_background_speckle_loss": camera_background_speckle,
            "camera_background_hotspot_loss": camera_background_hotspot,
            "camera_background_highfreq_loss": camera_background_highfreq,
            "camera_readable_contrast_loss": camera_readable_contrast,
            "student_label_margin_loss": student_label_margin,
            "student_prototype_ce_loss": student_prototype_ce_loss,
            "student_negative_label_margin_loss": student_negative_label_margin_loss,
            "student_roi_prototype_ce_loss": student_roi_prototype_ce_loss,
            "student_roi_negative_label_margin_loss": student_roi_negative_label_margin_loss,
            "aux_semantic_loss": aux_semantic_loss,
            "aux_topology_loss": aux_topology_loss,
            "code_diversity_loss": code_diversity,
            "sample_code_diversity_loss": sample_code_diversity,
            "teacher_pairwise_diversity_loss": teacher_pairwise_diversity,
            "cross_class_pairwise_loss": cross_class_pairwise,
            "clean_foreground_dice_loss": clean_dice_loss,
            "clean_foreground_hole_loss": clean_fg_hole_loss,
            "clean_background_speckle_loss": clean_bg_speckle_loss,
            "clean_background_hotspot_loss": clean_bg_hotspot_loss,
            "clean_background_highfreq_loss": clean_bg_highfreq_loss,
            "encoder_input_topology_loss": encoder_input_topology,
            "student_pair_contrast_loss": student_pair_contrast_loss,
            "display_speckle_loss": display_speckle_loss,
            "aux_detail_shape_loss": aux_detail_shape_loss,
            "aux_detail_edge_loss": aux_detail_edge_loss,
            "residual_l1_loss": residual_l1_loss,
            "residual_background_loss": residual_background_loss,
        }
    )
    return {
        "loss": total,
        "camera_recon_loss": camera_metrics["recon_loss"].detach(),
        "camera_fg_mse": camera_metrics["fg_mse"],
        "camera_bg_mse": camera_metrics["bg_mse"],
        "shape_loss": shape_loss.detach(),
        "shape_penalty": shape_base.detach(),
        "edge_loss": edge_loss.detach(),
        "edge_penalty": edge_base.detach(),
        "slm_smoothness_loss": slm_smoothness.detach(),
        "slm_tv": slm_tv.detach(),
        "slm_contrast_loss": slm_contrast.detach(),
        "slm_contrast_penalty": slm_contrast_base.detach(),
        "slm_per_sample_std_mean": per_sample_slm_std.mean().detach(),
        "slm_per_sample_std_min": per_sample_slm_std.min().detach(),
        "camera_saturation_loss": camera_saturation.detach(),
        "camera_saturation_penalty": camera_saturation_base.detach(),
        "camera_saturation_ratio": _camera_saturation_ratio(camera_prediction.detach()),
        "camera_low_saturation_ratio": _camera_low_saturation_ratio(
            camera_prediction.detach()
        ),
        "camera_high_saturation_ratio": _camera_high_saturation_ratio(
            camera_prediction.detach()
        ),
        "camera_topology_loss": camera_topology.detach(),
        "camera_artifact_loss": camera_artifact.detach(),
        "camera_readability_loss": camera_readability.detach(),
        "camera_readability_penalty": camera_readability_penalty.detach(),
        "camera_background_speckle_loss": camera_background_speckle.detach(),
        "camera_background_speckle_penalty": camera_background_speckle_base.detach(),
        "background_speckle_ratio": background_speckle_ratio.detach(),
        "camera_background_hotspot_loss": camera_background_hotspot.detach(),
        "camera_background_hotspot_penalty": camera_background_hotspot_base.detach(),
        "background_hotspot_ratio": background_hotspot_ratio.detach(),
        "camera_background_highfreq_loss": camera_background_highfreq.detach(),
        "camera_background_highfreq_penalty": camera_background_highfreq_base.detach(),
        "background_highfreq_energy": background_highfreq_energy.detach(),
        "camera_readable_contrast_loss": camera_readable_contrast.detach(),
        "camera_readable_contrast_penalty": camera_readable_contrast_base.detach(),
        "readable_contrast": camera_readable_contrast_value.detach(),
        "student_label_margin_loss": student_label_margin.detach(),
        "student_label_margin_penalty": student_label_margin_base.detach(),
        "student_label_margin_mean": student_label_margin_mean.detach(),
        "student_label_correct_ratio": student_label_correct_ratio.detach(),
        "student_prototype_ce_loss": student_prototype_ce_loss.detach(),
        "student_prototype_ce_penalty": student_prototype_ce_base.detach(),
        "student_prototype_ce_correct_ratio": student_prototype_ce_correct_ratio.detach(),
        "student_prototype_ce_count": student_prototype_ce_count.detach(),
        "student_negative_label_margin_loss": student_negative_label_margin_loss.detach(),
        "student_negative_label_margin_penalty": student_negative_label_margin_base.detach(),
        "student_negative_label_margin_mean": student_negative_label_margin_mean.detach(),
        "student_negative_label_margin_count": student_negative_label_margin_count.detach(),
        "student_roi_prototype_ce_loss": student_roi_prototype_ce_loss.detach(),
        "student_roi_prototype_ce_penalty": student_roi_prototype_ce_base.detach(),
        "student_roi_prototype_ce_correct_ratio": student_roi_prototype_ce_correct_ratio.detach(),
        "student_roi_prototype_ce_count": student_roi_prototype_ce_count.detach(),
        "student_roi_negative_label_margin_loss": student_roi_negative_label_margin_loss.detach(),
        "student_roi_negative_label_margin_penalty": student_roi_negative_label_margin_base.detach(),
        "student_roi_negative_label_margin_mean": student_roi_negative_label_margin_mean.detach(),
        "student_roi_negative_label_margin_count": student_roi_negative_label_margin_count.detach(),
        "aux_semantic_loss": aux_semantic_loss.detach(),
        "aux_semantic_prototype_ce_loss": aux_semantic_prototype_ce_loss.detach(),
        "aux_semantic_prototype_ce_penalty": aux_semantic_prototype_ce_base.detach(),
        "aux_semantic_prototype_ce_correct_ratio": aux_semantic_prototype_ce_correct_ratio.detach(),
        "aux_semantic_prototype_ce_count": aux_semantic_prototype_ce_count.detach(),
        "aux_semantic_negative_label_margin_loss": aux_semantic_negative_label_margin_loss.detach(),
        "aux_semantic_negative_label_margin_penalty": aux_semantic_negative_label_margin_base.detach(),
        "aux_semantic_negative_label_margin_mean": aux_semantic_negative_label_margin_mean.detach(),
        "aux_semantic_negative_label_margin_count": aux_semantic_negative_label_margin_count.detach(),
        "aux_topology_loss": aux_topology_loss.detach(),
        "aux_topology_dice": aux_topology_dice.detach(),
        "aux_topology_mse": aux_topology_mse.detach(),
        "aux_topology_target_fg_ratio": aux_topology_target_info[
            "foreground_ratio"
        ].detach(),
        "foreground_hole_ratio": foreground_hole_ratio.detach()
        if hasattr(foreground_hole_ratio, "detach")
        else foreground_hole_ratio,
        "camera_foreground_dice": _foreground_dice_score(
            camera_prediction.detach(), foreground_mask
        ),
        "clean_foreground_dice": _foreground_dice_score(
            camera_prediction.detach(), clean_foreground_mask
        ),
        "clean_dice_loss": clean_dice_loss.detach(),
        "clean_dice_penalty": clean_dice_base.detach(),
        "clean_fg_hole_loss": clean_fg_hole_loss.detach(),
        "clean_fg_hole_penalty": clean_fg_hole_base.detach(),
        "clean_fg_hole_ratio": clean_fg_hole_ratio.detach()
        if hasattr(clean_fg_hole_ratio, "detach")
        else clean_fg_hole_ratio,
        "clean_bg_speckle_loss": clean_bg_speckle_loss.detach(),
        "clean_bg_speckle_penalty": clean_bg_speckle_base.detach(),
        "clean_bg_speckle_ratio": clean_bg_speckle_ratio.detach()
        if hasattr(clean_bg_speckle_ratio, "detach")
        else clean_bg_speckle_ratio,
        "clean_bg_hotspot_loss": clean_bg_hotspot_loss.detach(),
        "clean_bg_hotspot_penalty": clean_bg_hotspot_base.detach(),
        "clean_bg_hotspot_ratio": clean_bg_hotspot_ratio.detach()
        if hasattr(clean_bg_hotspot_ratio, "detach")
        else clean_bg_hotspot_ratio,
        "clean_bg_highfreq_loss": clean_bg_highfreq_loss.detach(),
        "clean_bg_highfreq_penalty": clean_bg_highfreq_base.detach(),
        "clean_bg_highfreq_energy": clean_bg_highfreq_energy.detach()
        if hasattr(clean_bg_highfreq_energy, "detach")
        else clean_bg_highfreq_energy,
        "clean_target_fg_ratio": clean_target_fg_ratio.detach()
        if hasattr(clean_target_fg_ratio, "detach")
        else clean_target_fg_ratio,
        "encoder_input_topology_loss": encoder_input_topology.detach(),
        "encoder_input_topology_dice": encoder_input_topology_dice.detach(),
        "encoder_input_topology_mse": encoder_input_topology_mse.detach(),
        "student_pair_contrast_loss": student_pair_contrast_loss.detach(),
        "student_pair_contrast_penalty": student_pair_contrast_base.detach()
        if hasattr(student_pair_contrast_base, "detach")
        else zero,
        "student_pair_contrast_count": student_pair_contrast_count.detach()
        if hasattr(student_pair_contrast_count, "detach")
        else zero,
        "display_speckle_loss": display_speckle_loss.detach(),
        "display_speckle_penalty": display_speckle_base.detach()
        if hasattr(display_speckle_base, "detach")
        else zero,
        "display_speckle_ratio": display_speckle_ratio.detach()
        if hasattr(display_speckle_ratio, "detach")
        else zero,
        "display_highfreq_energy": display_highfreq_energy.detach()
        if hasattr(display_highfreq_energy, "detach")
        else zero,
        "aux_detail_shape_loss": aux_detail_shape_loss.detach(),
        "aux_detail_shape_penalty": aux_detail_shape_base.detach(),
        "aux_detail_edge_loss": aux_detail_edge_loss.detach(),
        "aux_detail_edge_penalty": aux_detail_edge_base.detach(),
        "aux_detail_gate_fg_ratio": aux_detail_gate_fg_ratio.detach(),
        "aux_detail_active_fg_ratio": aux_detail_active_fg_ratio.detach(),
        "residual_l1_loss": residual_l1_loss.detach(),
        "residual_l1_penalty": residual_l1_base.detach(),
        "residual_background_loss": residual_background_loss.detach(),
        "residual_background_penalty": residual_background_base.detach(),
        "residual_background_mask_ratio": residual_background_mask_ratio.detach(),
        "generator_residual_abs_mean": generator_residual_abs_mean.detach(),
        "generator_residual_bg_abs_mean": generator_residual_bg_abs_mean.detach(),
        "student_base_mse": student_base_mse.detach(),
        "student_base_foreground_mse": student_base_foreground_mse.detach(),
        "student_base_background_mse": student_base_background_mse.detach(),
        "code_diversity_loss": code_diversity.detach(),
        "code_diversity_penalty": diversity_metrics["code_diversity_base"].detach(),
        "sample_code_diversity_loss": sample_code_diversity.detach(),
        "sample_code_diversity_penalty": diversity_metrics[
            "code_diversity_base"
        ].detach(),
        "teacher_pairwise_diversity_loss": teacher_pairwise_diversity.detach(),
        "teacher_pairwise_diversity_penalty": diversity_metrics[
            "teacher_pairwise_diversity_base"
        ].detach(),
        "teacher_pairwise_mse": diversity_metrics["teacher_pairwise_mse"].detach(),
        "student_pairwise_mse": diversity_metrics["student_pairwise_mse"].detach(),
        "diversity_ratio": diversity_metrics["diversity_ratio"].detach(),
        "diversity_pair_count": diversity_metrics["diversity_pair_count"].detach(),
        "cross_class_pairwise_loss": cross_class_pairwise.detach(),
        "cross_class_pairwise_penalty": cross_class_metrics[
            "cross_class_pairwise_base"
        ].detach(),
        "cross_class_teacher_pairwise_mse": cross_class_metrics[
            "cross_class_teacher_pairwise_mse"
        ].detach(),
        "cross_class_student_pairwise_mse": cross_class_metrics[
            "cross_class_student_pairwise_mse"
        ].detach(),
        "cross_class_diversity_ratio": cross_class_metrics[
            "cross_class_diversity_ratio"
        ].detach(),
        "cross_class_pair_count": cross_class_metrics[
            "cross_class_pair_count"
        ].detach(),
        "slm_pairwise_mse": slm_pairwise_mse.detach(),
        "slm_per_class_std_mean": slm_per_class_std_mean.detach(),
        "class_weight_mean": sample_weights.mean().detach(),
        "fg_pixel_ratio": foreground_mask.mean(),
    }
