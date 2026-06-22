from __future__ import annotations

import sys

from ldm_project.stage_constants import (
    DIFFUSION_BEST_METRIC,
    ONE_SHOT_BEST_METRIC,
    ONE_SHOT_DISTILL_BEST_METRIC,
)


def save_visual_config_snapshots(
    run_manager,
    physics,
    train_cfg,
    *,
    first_stage_visual_labels,
    first_stage_visual_source,
    diffusion_visual_labels,
    diffusion_visual_reference_labels,
    diffusion_visual_source,
) -> None:
    run_manager.save_config_snapshot("physics_config", physics)
    run_manager.save_config_snapshot("train_config", train_cfg)
    if first_stage_visual_labels is not None:
        run_manager.save_config_snapshot(
            "first_stage_visual_config",
            {
                "class_labels": first_stage_visual_labels.detach().cpu().tolist(),
                "reference_source": first_stage_visual_source,
                "source": "train.py_fixed_first_stage_visualization",
            },
        )
    if diffusion_visual_labels is not None:
        run_manager.save_config_snapshot(
            "diffusion_visual_config",
            {
                "class_labels": diffusion_visual_labels.detach().cpu().tolist(),
                "reference_labels": diffusion_visual_reference_labels.detach().cpu().tolist(),
                "reference_source": diffusion_visual_source,
                "source": "train.py_fixed_diffusion_visualization",
            },
        )


def d2nn_metadata(d2nn_cfg) -> dict | None:
    if d2nn_cfg is None:
        return None
    return {
        "wavelength": d2nn_cfg.wavelength,
        "pixel_size": d2nn_cfg.pixel_size,
        "grid_size": d2nn_cfg.grid_size,
        "propagation_grid_size": d2nn_cfg.propagation_grid_size,
        "encoder_layers": d2nn_cfg.encoder_layers,
        "decoder_layers": d2nn_cfg.decoder_layers,
        "layer_spacing": d2nn_cfg.layer_spacing,
        "modulation": d2nn_cfg.modulation,
        "phase_init": d2nn_cfg.phase_init,
        "amplitude_logit_init": d2nn_cfg.amplitude_logit_init,
    }


def build_run_meta(
    *,
    args,
    train_cfg,
    device,
    checkpoint_resolution,
    dataset_splits,
    train_bundle,
    validation_bundle,
    epoch_plan,
    run_manager,
    diffusion_model,
    first_stage_model_version,
    one_shot_model_version,
) -> dict[str, object]:
    return {
        "script": "train.py",
        "argv": sys.argv,
        "device": str(device),
        "mode": train_cfg.mode,
        "physics_config_path": args.physics_config,
        "train_config_path": args.train_config,
        "final_pt_dir": args.final_pt_dir,
        "training_seed": train_cfg.seed,
        "first_stage_checkpoint": args.first_stage_checkpoint,
        "first_stage_checkpoint_source": checkpoint_resolution["first_stage_checkpoint_source"],
        "diffusion_checkpoint": args.diffusion_checkpoint,
        "diffusion_checkpoint_source": checkpoint_resolution["diffusion_checkpoint_source"],
        "one_shot_checkpoint": args.one_shot_checkpoint,
        "one_shot_generator_checkpoint": getattr(args, "one_shot_generator_checkpoint", None),
        "one_shot_optical_checkpoint": getattr(args, "one_shot_optical_checkpoint", None),
        "distill_dataset": args.distill_dataset,
        "distill_validation_dataset": getattr(args, "distill_validation_dataset", None),
        "run_dir": run_manager.run_dir,
        "dataset": train_cfg.data.dataset,
        "dataset_root": train_cfg.data.root,
        "dataset_total_size": dataset_splits.total_size,
        "dataset_effective_size": train_bundle.effective_size,
        "train_split_size": train_bundle.total_size,
        "train_effective_size": train_bundle.effective_size,
        "validation_size": 0 if validation_bundle is None else validation_bundle.effective_size,
        "mnist_aux_train_split_size": train_bundle.total_size,
        "mnist_aux_train_effective_size": train_bundle.effective_size,
        "mnist_aux_validation_size": 0 if validation_bundle is None else validation_bundle.effective_size,
        "validation_fraction": train_cfg.data.validation_fraction,
        "validation_split_seed": train_cfg.data.validation_split_seed,
        "subset_fraction": train_cfg.data.subset_fraction,
        "subset_size": train_cfg.data.subset_size,
        "subset_fraction_applies_to": "train_only",
        "planned_first_stage_epochs": epoch_plan["first_stage"],
        "planned_diffusion_epochs": epoch_plan["diffusion"],
        "planned_one_shot_epochs": epoch_plan["one_shot"],
        "visual_save_epoch_interval": train_cfg.visual_save_epoch_interval,
        "validation_epoch_interval": train_cfg.validation_epoch_interval,
        "first_stage_model_version": first_stage_model_version,
        "first_stage_encoder_fusion_mode": train_cfg.first_stage.encoder_fusion_mode,
        "first_stage_optical_encoder_input_mode": train_cfg.first_stage.optical_encoder_input_mode,
        "first_stage_optical_encoder_arch": train_cfg.first_stage.optical_encoder_arch,
        "first_stage_metasurface_decoder_arch": train_cfg.first_stage.metasurface_decoder_arch,
        "first_stage_linear_metasurface_grid_size": train_cfg.first_stage.linear_metasurface_grid_size,
        "first_stage_linear_metasurface_layer_spacing": train_cfg.first_stage.linear_metasurface_layer_spacing,
        "first_stage_field_energy_weight": train_cfg.first_stage.field_energy_weight,
        "first_stage_hard_class_weight_labels": list(train_cfg.first_stage.hard_class_weight_labels),
        "first_stage_hard_class_weight": train_cfg.first_stage.hard_class_weight,
        "first_stage_camera_label_margin_weight": train_cfg.first_stage.camera_label_margin_weight,
        "first_stage_camera_label_margin_target": train_cfg.first_stage.camera_label_margin_target,
        "first_stage_camera_label_margin_labels": list(train_cfg.first_stage.camera_label_margin_labels),
        "first_stage_camera_label_margin_prototypes_per_class": (
            train_cfg.first_stage.camera_label_margin_prototypes_per_class
        ),
        "image_fusion_start_weight": train_cfg.first_stage.image_fusion_start_weight,
        "image_fusion_end_weight": train_cfg.first_stage.image_fusion_end_weight,
        "image_fusion_decay_epochs": train_cfg.first_stage.image_fusion_decay_epochs,
        "first_stage_best_optical_checkpoint": None,
        "first_stage_best_optical_epoch": None,
        "first_stage_best_optical_metric": train_cfg.first_stage.checkpoint_selection_metric,
        "first_stage_best_optical_score": None,
        "first_stage_best_optical_loaded_for_diffusion": False,
        "diffusion_latent_source": train_cfg.diffusion.latent_source,
        "diffusion_guidance_scale": train_cfg.diffusion.guidance_scale,
        "diffusion_class_weight_labels": list(train_cfg.diffusion.class_weight_labels),
        "diffusion_class_weight": train_cfg.diffusion.class_weight,
        "diffusion_best_checkpoint": None,
        "diffusion_best_epoch": None,
        "diffusion_best_metric": DIFFUSION_BEST_METRIC,
        "diffusion_best_loss": None,
        "diffusion_checkpoint_loaded_for_training": False,
        "diffusion_best_loaded_for_sampling": False,
        "latent_scale_source": None,
        "sample_class_labels": list(train_cfg.diffusion.sample_class_labels),
        "sample_repeats_per_class": train_cfg.diffusion.sample_repeats_per_class,
        "sample_metric_source": None,
        "conditioning": diffusion_model.conditioning_metadata(),
        "one_shot_model_version": one_shot_model_version,
        "one_shot_optical_core_arch": train_cfg.one_shot.optical_core_arch,
        "one_shot_generator_e_arch": train_cfg.one_shot.generator_e_arch,
        "one_shot_generator_command_transform": train_cfg.one_shot.generator_command_transform,
        "one_shot_generator_template_init": train_cfg.one_shot.generator_template_init,
        "one_shot_generator_code_conditioning_active": (
            train_cfg.one_shot.generator_e_arch != "label_template"
        ),
        "one_shot_code_dim": train_cfg.one_shot.code_dim,
        "one_shot_num_classes": train_cfg.one_shot.num_classes,
        "one_shot_ols_kind": train_cfg.one_shot.ols_kind,
        "one_shot_metasurface_grid_size": train_cfg.one_shot.metasurface_grid_size,
        "one_shot_diffractive_grid_size": train_cfg.one_shot.diffractive_grid_size,
        "one_shot_diffractive_propagation_grid_size": train_cfg.one_shot.diffractive_propagation_grid_size,
        "one_shot_encoder_diffractive_layers": train_cfg.one_shot.encoder_diffractive_layers,
        "one_shot_decoder_diffractive_layers": train_cfg.one_shot.decoder_diffractive_layers,
        "one_shot_diffractive_layer_spacing": train_cfg.one_shot.diffractive_layer_spacing,
        "one_shot_diffractive_phase_init": train_cfg.one_shot.diffractive_phase_init,
        "one_shot_d2nn": d2nn_metadata(train_cfg.one_shot.d2nn),
        "one_shot_sample_class_labels": list(train_cfg.one_shot.sample_class_labels),
        "one_shot_sample_repeats_per_class": train_cfg.one_shot.sample_repeats_per_class,
        "one_shot_best_checkpoint": None,
        "one_shot_best_epoch": None,
        "one_shot_best_metric": ONE_SHOT_BEST_METRIC,
        "one_shot_best_loss": None,
        "one_shot_distill_best_checkpoint": None,
        "one_shot_distill_best_epoch": None,
        "one_shot_distill_best_metric": ONE_SHOT_DISTILL_BEST_METRIC,
        "one_shot_distill_best_loss": None,
        "one_shot_sample_metric_source": None,
        "one_shot_distill_eval_metric_source": None,
        "teacher_distill_dataset": None,
        "teacher_distill_sample_count": None,
    }
