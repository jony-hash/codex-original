from __future__ import annotations

from .stage_constants import (
    DIFFUSION_BEST_CHECKPOINT,
    DIFFUSION_BEST_METRIC,
    FIRST_STAGE_BEST_OPTICAL_CHECKPOINT,
    FIRST_STAGE_BEST_OPTICAL_METRIC,
)
from .train_utils import (
    is_best_optical_candidate,
    should_update_best_optical,
)

_FIRST_STAGE_STEP_PRINT_SPEC: tuple[tuple[str, str, str], ...] = (
    ("loss", "loss", ".6f"),
    ("recon_loss", "recon_w", ".6f"),
    ("image_recon_loss", "image_recon", ".6f"),
    ("camera_recon_loss", "camera_recon", ".6f"),
    ("image_fg_mse", "image_fg_mse", ".6f"),
    ("camera_fg_mse", "camera_fg_mse", ".6f"),
    ("camera_foreground_dice", "camera_dice", ".4f"),
    ("camera_hotspot_ratio", "hotspot", ".4f"),
    ("camera_topology_loss", "topology", ".6f"),
    ("structure_loss", "structure", ".6f"),
    ("shape_loss", "shape", ".6f"),
    ("field_smoothness_loss", "field", ".6f"),
    ("phase_tv", "phase_tv", ".6f"),
    ("field_energy", "field_energy", ".6f"),
    ("camera_saturation_ratio", "sat", ".4f"),
    ("camera_artifact_loss", "artifact", ".6f"),
    ("camera_foreground_hole_ratio", "hole", ".4f"),
    ("kl_loss", "kl", ".6f"),
)

FIRST_STAGE_HISTORY_LOSS_KEYS: tuple[str, ...] = (
    "loss",
    "recon_loss",
    "image_recon_loss",
    "image_fg_mse",
    "image_bg_mse",
    "camera_recon_loss",
    "camera_fg_mse",
    "camera_bg_mse",
    "camera_foreground_dice_loss",
    "camera_foreground_dice",
    "structure_loss",
    "image_structure_loss",
    "camera_structure_loss",
    "shape_loss",
    "image_shape_loss",
    "camera_shape_loss",
    "field_smoothness_loss",
    "field_tv",
    "phase_smoothness_loss",
    "phase_tv",
    "field_energy_loss",
    "field_energy",
    "background_leakage_loss",
    "camera_background_leakage",
    "camera_hotspot_loss",
    "camera_hotspot_ratio",
    "camera_saturation_ratio",
    "camera_topology_loss",
    "camera_topology_base",
    "camera_saturation_loss",
    "camera_saturation_penalty",
    "camera_artifact_loss",
    "camera_foreground_hole_loss",
    "camera_foreground_hole_ratio",
    "camera_foreground_roughness",
    "kl_loss",
    "fg_pixel_ratio",
    "pred_fg_energy_ratio",
    "beta",
)


def _record_first_stage_train_step(
    run_manager,
    *,
    epoch_number: int,
    step_record: dict[str, int],
    losses: dict[str, object],
    image_fusion_weight: float,
) -> None:
    parts = [
        f"[first_stage] epoch={epoch_number} global_step={step_record['global_step']} "
        f"stage_step={step_record['stage_step']}"
    ]
    for key, label, fmt in _FIRST_STAGE_STEP_PRINT_SPEC:
        parts.append(f"{label}={losses[key].item():{fmt}}")
    parts.append(f"image_fusion={image_fusion_weight:.4f}")
    print(" ".join(parts))
    run_manager.append_history(
        "first_stage",
        {
            "epoch": epoch_number,
            **step_record,
            **{key: losses[key].item() for key in FIRST_STAGE_HISTORY_LOSS_KEYS},
            "image_fusion_weight": image_fusion_weight,
        },
    )


def _record_diffusion_train_step(
    run_manager,
    *,
    epoch_number: int,
    step_record: dict[str, int],
    loss,
    latent_scale,
) -> None:
    loss_val = loss.item()
    scale_val = latent_scale.item()
    print(
        f"[diffusion] epoch={epoch_number} global_step={step_record['global_step']} "
        f"stage_step={step_record['stage_step']} "
        f"loss={loss_val:.6f} latent_scale={scale_val:.6f}"
    )
    run_manager.append_history(
        "diffusion",
        {
            "epoch": epoch_number,
            **step_record,
            "loss": loss_val,
            "latent_scale": scale_val,
        },
    )


def _record_first_stage_validation(run_manager, *, epoch_number: int, validation_metrics, per_class_metrics) -> None:
    print(
        f"[first_stage][val] epoch={epoch_number} loss={validation_metrics['loss']:.6f} "
        f"recon_w={validation_metrics['recon_loss']:.6f} "
        f"image_recon={validation_metrics['image_recon_loss']:.6f} "
        f"camera_recon={validation_metrics['camera_recon_loss']:.6f} "
        f"image_fg_mse={validation_metrics['image_fg_mse']:.6f} "
        f"camera_fg_mse={validation_metrics['camera_fg_mse']:.6f} "
        f"camera_dice={validation_metrics['camera_foreground_dice']:.4f} "
        f"hotspot={validation_metrics['camera_hotspot_ratio']:.4f} "
        f"topology={validation_metrics['camera_topology_loss']:.6f} "
        f"structure={validation_metrics['structure_loss']:.6f} "
        f"shape={validation_metrics['shape_loss']:.6f} "
        f"field={validation_metrics['field_smoothness_loss']:.6f} "
        f"phase_tv={validation_metrics['phase_tv']:.6f} "
        f"field_energy={validation_metrics['field_energy']:.6f} "
        f"sat={validation_metrics['camera_saturation_ratio']:.4f} "
        f"artifact={validation_metrics['camera_artifact_loss']:.6f} "
        f"hole={validation_metrics['camera_foreground_hole_ratio']:.4f} "
        f"camera_corr={validation_metrics['camera_corr']:.4f} "
        f"camera_edge_corr={validation_metrics['camera_edge_corr']:.4f} "
        f"min_class_corr={validation_metrics['min_class_camera_corr']:.4f} "
        f"balanced={validation_metrics['camera_balanced_score']:.4f} "
        f"mnist_score={validation_metrics['camera_mnist_score']:.4f} "
        f"min_class_mnist={validation_metrics['min_class_camera_mnist_score']:.4f} "
        f"class_balanced_mnist={validation_metrics['camera_class_balanced_mnist_score']:.4f} "
        f"complex_corr={validation_metrics['complex_camera_corr']:.4f} "
        f"complex_mnist={validation_metrics['complex_camera_mnist_score']:.4f} "
        f"complex_balanced_mnist={validation_metrics['camera_complex_balanced_mnist_score']:.4f} "
        f"kl={validation_metrics['kl_loss']:.6f}"
    )
    run_manager.append_history(
        "first_stage_validation",
        {
            "epoch": epoch_number,
            **validation_metrics,
        },
    )
    run_manager.append_history(
        "first_stage_validation_per_class",
        {
            "epoch": epoch_number,
            "classes": per_class_metrics,
        },
    )


def _update_first_stage_best_checkpoint(
    run_manager,
    *,
    first_stage,
    train_cfg,
    run_meta: dict[str, object],
    physics,
    epoch_number: int,
    validation_metrics,
    image_fusion_weight: float,
    best_optical_score,
    best_optical_checkpoint_path,
    first_stage_model_version: str,
):
    optical_candidate = is_best_optical_candidate(train_cfg.first_stage, image_fusion_weight)
    selection_score = validation_metrics[FIRST_STAGE_BEST_OPTICAL_METRIC]
    updated_best = optical_candidate and should_update_best_optical(selection_score, best_optical_score)
    if updated_best:
        best_optical_score = selection_score
        best_optical_checkpoint_path = run_manager.save_checkpoint(
            FIRST_STAGE_BEST_OPTICAL_CHECKPOINT,
            {
                "model_state_dict": first_stage.state_dict(),
                "stage": "first_stage",
                "model_version": first_stage_model_version,
                "physics_config": physics,
                "train_config": train_cfg,
                "epoch": epoch_number,
                "selection_metric": FIRST_STAGE_BEST_OPTICAL_METRIC,
                "selection_score": selection_score,
                "validation_metrics": validation_metrics,
                "image_fusion_weight": image_fusion_weight,
            },
        )
        run_meta.update(
            {
                "first_stage_best_optical_checkpoint": best_optical_checkpoint_path,
                "first_stage_best_optical_epoch": epoch_number,
                "first_stage_best_optical_score": best_optical_score,
            }
        )
        run_manager.write_meta(run_meta)
    return best_optical_score, best_optical_checkpoint_path, optical_candidate, selection_score, updated_best


def _record_first_stage_best_history(
    run_manager,
    *,
    epoch_number: int,
    optical_candidate: bool,
    updated_best: bool,
    selection_score,
    best_optical_score,
    image_fusion_weight: float,
    best_optical_checkpoint_path,
) -> None:
    run_manager.append_history(
        "first_stage_best_optical",
        {
            "epoch": epoch_number,
            "eligible": optical_candidate,
            "updated": updated_best,
            "selection_metric": FIRST_STAGE_BEST_OPTICAL_METRIC,
            "selection_score": selection_score,
            "best_score": best_optical_score,
            "image_fusion_weight": image_fusion_weight,
            "checkpoint": best_optical_checkpoint_path,
        },
    )


def _record_diffusion_validation(run_manager, *, epoch_number: int, validation_metrics, per_class_metrics) -> None:
    print(
        f"[diffusion][val] epoch={epoch_number} loss={validation_metrics['loss']:.6f} "
        f"latent_scale={validation_metrics['latent_scale']:.6f}"
    )
    run_manager.append_history(
        "diffusion_validation",
        {
            "epoch": epoch_number,
            **validation_metrics,
        },
    )
    run_manager.append_history(
        "diffusion_validation_per_class",
        {
            "epoch": epoch_number,
            "classes": per_class_metrics,
        },
    )


def _update_diffusion_best_checkpoint(
    run_manager,
    *,
    diffusion_model,
    train_cfg,
    run_meta: dict[str, object],
    physics,
    epoch_number: int,
    validation_metrics,
    latent_scale,
    best_diffusion_loss,
    best_diffusion_checkpoint_path,
    first_stage_model_version: str,
    diffusion_model_version: str,
):
    selection_loss = validation_metrics["loss"]
    updated_best_diffusion = best_diffusion_loss is None or selection_loss < best_diffusion_loss
    if updated_best_diffusion:
        best_diffusion_loss = selection_loss
        best_diffusion_checkpoint_path = run_manager.save_checkpoint(
            DIFFUSION_BEST_CHECKPOINT,
            {
                "model_state_dict": diffusion_model.state_dict(),
                "latent_scale": latent_scale.detach().cpu(),
                "latent_source": train_cfg.diffusion.latent_source,
                "stage": "diffusion",
                "model_version": diffusion_model_version,
                "first_stage_model_version": first_stage_model_version,
                "conditioning": diffusion_model.conditioning_metadata(),
                "physics_config": physics,
                "train_config": train_cfg,
                "epoch": epoch_number,
                "selection_metric": DIFFUSION_BEST_METRIC,
                "selection_loss": selection_loss,
                "validation_metrics": validation_metrics,
            },
        )
        run_meta.update(
            {
                "diffusion_best_checkpoint": best_diffusion_checkpoint_path,
                "diffusion_best_epoch": epoch_number,
                "diffusion_best_loss": best_diffusion_loss,
            }
        )
        run_manager.write_meta(run_meta)
    return best_diffusion_loss, best_diffusion_checkpoint_path, selection_loss, updated_best_diffusion


def _record_diffusion_best_history(
    run_manager,
    *,
    epoch_number: int,
    updated_best_diffusion: bool,
    selection_loss,
    best_diffusion_loss,
    best_diffusion_checkpoint_path,
) -> None:
    run_manager.append_history(
        "diffusion_best",
        {
            "epoch": epoch_number,
            "updated": updated_best_diffusion,
            "selection_metric": DIFFUSION_BEST_METRIC,
            "selection_loss": selection_loss,
            "best_loss": best_diffusion_loss,
            "checkpoint": best_diffusion_checkpoint_path,
        },
    )
