from __future__ import annotations

import argparse
import sys

from ldm_project.evaluation import (
    average_scalar_metrics,
    compute_camera_mnist_score,
    compute_first_stage_balanced_metrics,
    compute_input_complexity_scores,
    evaluate_diffusion_with_per_class,
    evaluate_first_stage,
    evaluate_first_stage_with_per_class,
    first_stage_quality_metrics,
    select_complex_subset_mask,
)
from ldm_project.sampling import (
    DEFAULT_DENOISING_TRACE_TIMESTEPS,
    DEFAULT_NEAREST_REFERENCE_LIMIT,
    build_denoising_trace_visualization_tensors,
    build_diffusion_visualization_tensors,
    build_first_stage_visualization_tensors,
    build_one_shot_visualization_tensors,
    build_post_train_sample_labels,
    build_reference_image_bank_for_labels,
    build_visualization_labels,
    compute_sample_quality_metrics,
    format_output_title,
    save_post_train_samples,
    select_first_stage_visual_batch,
    select_post_train_sample_batch,
    select_reference_images_for_labels,
)
from ldm_project.stage_runners import (
    DIFFUSION_BEST_CHECKPOINT,
    DIFFUSION_BEST_METRIC,
    FIRST_STAGE_BEST_OPTICAL_CHECKPOINT,
    FIRST_STAGE_BEST_OPTICAL_METRIC,
    ONE_SHOT_BEST_CHECKPOINT,
    ONE_SHOT_BEST_METRIC,
    ONE_SHOT_DISTILL_BEST_CHECKPOINT,
    ONE_SHOT_DISTILL_BEST_METRIC,
    run_one_shot_distill_eval_stage,
    run_diffusion_training,
    run_first_stage_training,
    run_one_shot_distill_training,
    run_one_shot_sample_stage,
    run_one_shot_training,
    run_sample_stage,
    run_teacher_distill_export,
    run_teacher_distill_filter,
)
from ldm_project.train_utils import (
    CLI_STAGE_CHOICES,
    DEFAULT_LATENT_SCALE_BATCHES,
    apply_cli_overrides,
    canonicalize_stage,
    estimate_latent_scale_from_loader,
    is_teacher_diffusion_mode,
    is_teacher_first_stage_mode,
    is_best_optical_candidate,
    load_latent_scale_from_diffusion_checkpoint,
    require_strict_optical_best_checkpoint,
    resolve_device,
    resolve_final_pt_stage_checkpoints,
    resolve_image_fusion_weight,
    resolve_logged_step,
    resolve_stage_epoch_plan,
    resolve_stage_total_steps,
    set_first_stage_image_fusion_weight,
    should_log_step,
    should_save_epoch_visual,
    should_update_best_optical,
    should_validate_epoch,
    validate_class_label_values,
    validate_stage_requirements,
)

__all__ = [
    "DEFAULT_DENOISING_TRACE_TIMESTEPS",
    "DEFAULT_LATENT_SCALE_BATCHES",
    "DEFAULT_NEAREST_REFERENCE_LIMIT",
    "DIFFUSION_BEST_CHECKPOINT",
    "DIFFUSION_BEST_METRIC",
    "FIRST_STAGE_BEST_OPTICAL_CHECKPOINT",
    "FIRST_STAGE_BEST_OPTICAL_METRIC",
    "ONE_SHOT_BEST_CHECKPOINT",
    "ONE_SHOT_BEST_METRIC",
    "ONE_SHOT_DISTILL_BEST_CHECKPOINT",
    "ONE_SHOT_DISTILL_BEST_METRIC",
    "apply_cli_overrides",
    "average_scalar_metrics",
    "build_denoising_trace_visualization_tensors",
    "build_diffusion_visualization_tensors",
    "build_first_stage_visualization_tensors",
    "build_one_shot_visualization_tensors",
    "build_post_train_sample_labels",
    "build_reference_image_bank_for_labels",
    "build_visualization_labels",
    "compute_camera_mnist_score",
    "compute_first_stage_balanced_metrics",
    "compute_input_complexity_scores",
    "compute_sample_quality_metrics",
    "evaluate_diffusion_with_per_class",
    "evaluate_first_stage",
    "evaluate_first_stage_with_per_class",
    "estimate_latent_scale_from_loader",
    "first_stage_quality_metrics",
    "format_output_title",
    "canonicalize_stage",
    "is_teacher_diffusion_mode",
    "is_teacher_first_stage_mode",
    "is_best_optical_candidate",
    "load_latent_scale_from_diffusion_checkpoint",
    "main",
    "parse_args",
    "require_strict_optical_best_checkpoint",
    "resolve_device",
    "resolve_final_pt_stage_checkpoints",
    "resolve_image_fusion_weight",
    "resolve_logged_step",
    "resolve_stage_epoch_plan",
    "resolve_stage_total_steps",
    "run_one_shot_distill_eval_stage",
    "run_diffusion_training",
    "run_first_stage_training",
    "run_one_shot_distill_training",
    "run_one_shot_sample_stage",
    "run_one_shot_training",
    "run_sample_stage",
    "run_teacher_distill_export",
    "run_teacher_distill_filter",
    "save_post_train_samples",
    "select_complex_subset_mask",
    "select_first_stage_visual_batch",
    "select_post_train_sample_batch",
    "select_reference_images_for_labels",
    "set_first_stage_image_fusion_weight",
    "should_log_step",
    "should_save_epoch_visual",
    "should_update_best_optical",
    "should_validate_epoch",
    "validate_class_label_values",
    "validate_stage_requirements",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Optical-Path VAE LDM scaffold.")
    parser.add_argument("--physics-config", default="configs/physics_cfg.yaml", help="Path to physics_cfg.yaml")
    parser.add_argument("--train-config", default="configs/train_ldm.yaml", help="Path to train_ldm.yaml")
    parser.add_argument("--stage", choices=CLI_STAGE_CHOICES, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--subset-fraction", type=float, default=None, help="Override dataset subset fraction, e.g. 0.1667")
    parser.add_argument("--subset-size", type=int, default=None, help="Override exact train subset size, e.g. 2000")
    parser.add_argument("--validation-fraction", type=float, default=None, help="Override validation split fraction.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader worker processes.")
    parser.add_argument("--first-stage-epochs", type=int, default=None, help="Override first-stage training epochs.")
    parser.add_argument("--diffusion-epochs", type=int, default=None, help="Override diffusion training epochs.")
    parser.add_argument("--one-shot-epochs", type=int, default=None, help="Override one-shot student training epochs.")
    parser.add_argument(
        "--distill-samples-per-class",
        type=int,
        default=None,
        help="Override teacher distillation export samples per class.",
    )
    parser.add_argument(
        "--distill-total-samples",
        type=int,
        default=None,
        help="Override total teacher distillation samples; labels cycle over selected classes.",
    )
    parser.add_argument(
        "--distill-output",
        default=None,
        help="Optional path for the exported teacher distillation tensor bundle.",
    )
    parser.add_argument("--guidance-scale", type=float, default=None, help="Override classifier-free guidance scale.")
    parser.add_argument("--cond-dropout-prob", type=float, default=None, help="Override diffusion conditioning dropout.")
    parser.add_argument(
        "--sample-class-labels",
        nargs="+",
        type=int,
        default=None,
        help="Class labels to generate in sample outputs, e.g. --sample-class-labels 3 4 5.",
    )
    parser.add_argument(
        "--sample-repeats-per-class",
        type=int,
        default=None,
        help="Repeat each requested sample class this many times.",
    )
    parser.add_argument(
        "--diffusion-class-weight-labels",
        nargs="*",
        type=int,
        default=None,
        help="Class labels whose diffusion training loss should be up-weighted.",
    )
    parser.add_argument(
        "--diffusion-class-weight",
        type=float,
        default=None,
        help="Positive multiplier for --diffusion-class-weight-labels during diffusion training.",
    )
    parser.add_argument(
        "--camera-artifact-weight",
        type=float,
        default=None,
        help="Override first-stage camera artifact regularization weight.",
    )
    parser.add_argument(
        "--encoder-fusion-mode",
        choices=["optical_only", "gated_dual_path", "dual_path_concat"],
        default=None,
        help="Override first-stage encoder fusion mode for controlled ablations.",
    )
    parser.add_argument(
        "--image-fusion-weight",
        type=float,
        default=None,
        help="Fix gated image fusion start/end weight for the whole first-stage run.",
    )
    parser.add_argument(
        "--optical-encoder-input-mode",
        choices=["pf2_intensity", "pf2_intensity_multiscale", "pf2_intensity_multiscale_v2", "pf2_complex_field"],
        default=None,
        help="Override first-stage optical encoder input built from PF2 intensity.",
    )
    parser.add_argument(
        "--optical-encoder-arch",
        choices=["conv", "global_context", "linear_metasurface"],
        default=None,
        help="Override first-stage optical encoder architecture.",
    )
    parser.add_argument(
        "--metasurface-decoder-arch",
        choices=["conv", "linear_metasurface"],
        default=None,
        help="Override first-stage metasurface decoder architecture.",
    )
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cuda:0, or cpu")
    parser.add_argument("--output-root", default="outputs/runs", help="Root directory for timestamped run outputs")
    parser.add_argument(
        "--final-pt-dir",
        default="outputs/final_pt",
        help="Final accepted checkpoint directory used to auto-fill missing diffusion/sample checkpoints.",
    )
    parser.add_argument("--run-name", default=None, help="Optional suffix for the timestamped run directory")
    parser.add_argument(
        "--first-stage-checkpoint",
        default=None,
        help="Required for --stage diffusion to load a trained first-stage model.",
    )
    parser.add_argument(
        "--diffusion-checkpoint",
        default=None,
        help="Required for --stage sample to load a trained diffusion model.",
    )
    parser.add_argument(
        "--one-shot-checkpoint",
        default=None,
        help="Required for --stage one_shot_sample to load a trained one-shot optical student.",
    )
    parser.add_argument(
        "--distill-dataset",
        default=None,
        help="Required for --stage one_shot_distill to load offline teacher targets.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to run train.py.") from exc

    from ldm_project import (
        FIRST_STAGE_MODEL_VERSION,
        LatentDiffusionUNet,
        LinearNoiseScheduler,
        ONE_SHOT_MODEL_VERSION,
        OpticalPathVAE,
        OneShotOpticalGenerator,
        RunManager,
        build_train_validation_loaders,
        load_first_stage_checkpoint,
        load_one_shot_checkpoint,
        load_physics_config,
        load_training_config,
    )
    from ldm_project.diffusion import DIFFUSION_MODEL_VERSION, load_diffusion_checkpoint
    from ldm_project.training import (
        diffusion_loss,
        diffusion_per_sample_losses,
        first_stage_loss,
        one_shot_loss,
        resolve_denoising_trace_steps,
        sample_latents,
        sample_latents_with_trace,
    )

    args = parse_args()
    physics = load_physics_config(args.physics_config)
    train_cfg = load_training_config(args.train_config)
    apply_cli_overrides(train_cfg, args)

    checkpoint_resolution = resolve_final_pt_stage_checkpoints(
        train_cfg.mode,
        args.first_stage_checkpoint,
        args.diffusion_checkpoint,
        args.final_pt_dir,
        FIRST_STAGE_BEST_OPTICAL_CHECKPOINT,
        DIFFUSION_BEST_CHECKPOINT,
    )
    args.first_stage_checkpoint = checkpoint_resolution["first_stage_checkpoint"]
    args.diffusion_checkpoint = checkpoint_resolution["diffusion_checkpoint"]
    validate_stage_requirements(
        train_cfg.mode,
        args.first_stage_checkpoint,
        args.diffusion_checkpoint,
        args.one_shot_checkpoint,
        args.distill_dataset,
    )

    device = resolve_device(torch, args.device)
    use_pin_memory = train_cfg.data.pin_memory and device.type == "cuda"
    train_cfg.data.pin_memory = use_pin_memory

    run_manager = RunManager.create(
        output_root=args.output_root,
        purpose="train",
        run_name=args.run_name or train_cfg.mode,
    )
    first_stage = OpticalPathVAE(physics, train_cfg.first_stage).to(device)
    diffusion_model = LatentDiffusionUNet(train_cfg.first_stage.latent_channels, train_cfg.diffusion).to(device)
    one_shot_model = OneShotOpticalGenerator(
        physics,
        train_cfg.one_shot,
        image_size=train_cfg.first_stage.image_size,
    ).to(device)
    scheduler = LinearNoiseScheduler(
        timesteps=train_cfg.diffusion.timesteps,
        beta_start=train_cfg.diffusion.beta_start,
        beta_end=train_cfg.diffusion.beta_end,
    ).to(device)

    if args.first_stage_checkpoint is not None:
        first_stage_payload = torch.load(args.first_stage_checkpoint, map_location=device, weights_only=False)
        load_first_stage_checkpoint(first_stage, first_stage_payload)
        if checkpoint_resolution["first_stage_checkpoint_source"] == "final_pt":
            print(f"using final_pt first-stage checkpoint: {args.first_stage_checkpoint}")

    dataset_splits = build_train_validation_loaders(
        data_config=train_cfg.data,
        first_stage_config=train_cfg.first_stage,
        batch_size=train_cfg.batch_size,
    )
    train_bundle = dataset_splits.train
    validation_bundle = dataset_splits.validation
    first_stage_visual_batch = None
    first_stage_visual_labels = None
    first_stage_visual_source = None
    diffusion_visual_labels = None
    diffusion_visual_reference_images = None
    diffusion_visual_reference_labels = None
    diffusion_visual_source = None
    if is_teacher_first_stage_mode(train_cfg.mode):
        first_stage_visual_batch, first_stage_visual_labels, first_stage_visual_source = select_first_stage_visual_batch(
            torch,
            validation_bundle,
            train_bundle,
            train_cfg.diffusion.num_classes,
            device,
        )
    if is_teacher_diffusion_mode(train_cfg.mode):
        (
            diffusion_visual_labels,
            diffusion_visual_reference_images,
            diffusion_visual_reference_labels,
            diffusion_visual_source,
        ) = select_post_train_sample_batch(
            torch,
            validation_bundle,
            train_bundle,
            train_cfg.diffusion.num_classes,
            device,
        )

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
    epoch_plan = resolve_stage_epoch_plan(
        train_cfg.mode,
        train_cfg.first_stage_epochs,
        train_cfg.diffusion_epochs,
        train_cfg.one_shot_epochs,
    )
    run_meta = {
        "script": "train.py",
        "argv": sys.argv,
        "device": str(device),
        "mode": train_cfg.mode,
        "physics_config_path": args.physics_config,
        "train_config_path": args.train_config,
        "final_pt_dir": args.final_pt_dir,
        "first_stage_checkpoint": args.first_stage_checkpoint,
        "first_stage_checkpoint_source": checkpoint_resolution["first_stage_checkpoint_source"],
        "diffusion_checkpoint": args.diffusion_checkpoint,
        "diffusion_checkpoint_source": checkpoint_resolution["diffusion_checkpoint_source"],
        "one_shot_checkpoint": args.one_shot_checkpoint,
        "distill_dataset": args.distill_dataset,
        "run_dir": run_manager.run_dir,
        "dataset": train_cfg.data.dataset,
        "dataset_root": train_cfg.data.root,
        "dataset_total_size": dataset_splits.total_size,
        "dataset_effective_size": train_bundle.effective_size,
        "train_split_size": train_bundle.total_size,
        "train_effective_size": train_bundle.effective_size,
        "validation_size": 0 if validation_bundle is None else validation_bundle.effective_size,
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
        "first_stage_model_version": FIRST_STAGE_MODEL_VERSION,
        "first_stage_encoder_fusion_mode": train_cfg.first_stage.encoder_fusion_mode,
        "first_stage_optical_encoder_input_mode": train_cfg.first_stage.optical_encoder_input_mode,
        "first_stage_optical_encoder_arch": train_cfg.first_stage.optical_encoder_arch,
        "first_stage_metasurface_decoder_arch": train_cfg.first_stage.metasurface_decoder_arch,
        "first_stage_linear_metasurface_grid_size": train_cfg.first_stage.linear_metasurface_grid_size,
        "first_stage_linear_metasurface_layer_spacing": train_cfg.first_stage.linear_metasurface_layer_spacing,
        "first_stage_field_energy_weight": train_cfg.first_stage.field_energy_weight,
        "image_fusion_start_weight": train_cfg.first_stage.image_fusion_start_weight,
        "image_fusion_end_weight": train_cfg.first_stage.image_fusion_end_weight,
        "image_fusion_decay_epochs": train_cfg.first_stage.image_fusion_decay_epochs,
        "first_stage_best_optical_checkpoint": None,
        "first_stage_best_optical_epoch": None,
        "first_stage_best_optical_metric": FIRST_STAGE_BEST_OPTICAL_METRIC,
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
        "diffusion_best_loaded_for_sampling": False,
        "latent_scale_source": None,
        "sample_class_labels": list(train_cfg.diffusion.sample_class_labels),
        "sample_repeats_per_class": train_cfg.diffusion.sample_repeats_per_class,
        "sample_metric_source": None,
        "conditioning": diffusion_model.conditioning_metadata(),
        "one_shot_model_version": ONE_SHOT_MODEL_VERSION,
        "one_shot_code_dim": train_cfg.one_shot.code_dim,
        "one_shot_num_classes": train_cfg.one_shot.num_classes,
        "one_shot_ols_kind": train_cfg.one_shot.ols_kind,
        "one_shot_metasurface_grid_size": train_cfg.one_shot.metasurface_grid_size,
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
    run_manager.write_meta(run_meta)
    print(
        f"planned epochs: first_stage={epoch_plan['first_stage']} "
        f"diffusion={epoch_plan['diffusion']} one_shot={epoch_plan['one_shot']}"
    )

    if train_cfg.mode == "teacher_sample":
        if checkpoint_resolution["diffusion_checkpoint_source"] == "final_pt":
            print(f"using final_pt diffusion checkpoint: {args.diffusion_checkpoint}")
        run_sample_stage(
            torch,
            first_stage=first_stage,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            train_bundle=train_bundle,
            validation_bundle=validation_bundle,
            train_cfg=train_cfg,
            run_manager=run_manager,
            run_meta=run_meta,
            device=device,
            diffusion_checkpoint=args.diffusion_checkpoint,
            first_stage_model_version=FIRST_STAGE_MODEL_VERSION,
            load_diffusion_checkpoint_fn=load_diffusion_checkpoint,
            sample_latents_fn=sample_latents,
            sample_latents_with_trace_fn=sample_latents_with_trace,
            resolve_denoising_trace_steps_fn=resolve_denoising_trace_steps,
        )

    if train_cfg.mode == "teacher_distill_export":
        if checkpoint_resolution["diffusion_checkpoint_source"] == "final_pt":
            print(f"using final_pt diffusion checkpoint: {args.diffusion_checkpoint}")
        run_teacher_distill_export(
            torch,
            first_stage=first_stage,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            train_cfg=train_cfg,
            run_manager=run_manager,
            run_meta=run_meta,
            device=device,
            diffusion_checkpoint=args.diffusion_checkpoint,
            first_stage_model_version=FIRST_STAGE_MODEL_VERSION,
            load_diffusion_checkpoint_fn=load_diffusion_checkpoint,
            sample_latents_fn=sample_latents,
        )

    if train_cfg.mode == "teacher_distill_filter":
        run_teacher_distill_filter(
            torch,
            train_cfg=train_cfg,
            run_manager=run_manager,
            run_meta=run_meta,
            distill_dataset=args.distill_dataset,
        )

    if train_cfg.mode == "one_shot_sample":
        run_one_shot_sample_stage(
            torch,
            one_shot_model=one_shot_model,
            train_bundle=train_bundle,
            validation_bundle=validation_bundle,
            train_cfg=train_cfg,
            run_manager=run_manager,
            run_meta=run_meta,
            device=device,
            one_shot_checkpoint=args.one_shot_checkpoint,
            load_one_shot_checkpoint_fn=load_one_shot_checkpoint,
        )

    if train_cfg.mode == "one_shot_distill_eval":
        run_one_shot_distill_eval_stage(
            torch,
            one_shot_model=one_shot_model,
            train_cfg=train_cfg,
            run_manager=run_manager,
            run_meta=run_meta,
            device=device,
            one_shot_checkpoint=args.one_shot_checkpoint,
            distill_dataset=args.distill_dataset,
            load_one_shot_checkpoint_fn=load_one_shot_checkpoint,
        )

    if is_teacher_first_stage_mode(train_cfg.mode):
        run_first_stage_training(
            torch,
            first_stage=first_stage,
            first_stage_loss_fn=first_stage_loss,
            first_stage_visual_batch=first_stage_visual_batch,
            run_manager=run_manager,
            train_bundle=train_bundle,
            validation_bundle=validation_bundle,
            train_cfg=train_cfg,
            run_meta=run_meta,
            physics=physics,
            device=device,
            use_pin_memory=use_pin_memory,
            first_stage_model_version=FIRST_STAGE_MODEL_VERSION,
        )

    if is_teacher_diffusion_mode(train_cfg.mode):
        run_diffusion_training(
            torch,
            first_stage=first_stage,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            train_bundle=train_bundle,
            validation_bundle=validation_bundle,
            train_cfg=train_cfg,
            run_meta=run_meta,
            run_manager=run_manager,
            physics=physics,
            device=device,
            use_pin_memory=use_pin_memory,
            first_stage_model_version=FIRST_STAGE_MODEL_VERSION,
            diffusion_model_version=DIFFUSION_MODEL_VERSION,
            diffusion_visual_labels=diffusion_visual_labels,
            diffusion_visual_reference_images=diffusion_visual_reference_images,
            load_first_stage_checkpoint_fn=load_first_stage_checkpoint,
            load_diffusion_checkpoint_fn=load_diffusion_checkpoint,
            diffusion_loss_fn=diffusion_loss,
            diffusion_per_sample_losses_fn=diffusion_per_sample_losses,
            sample_latents_fn=sample_latents,
            sample_latents_with_trace_fn=sample_latents_with_trace,
            resolve_denoising_trace_steps_fn=resolve_denoising_trace_steps,
        )

    if train_cfg.mode == "one_shot":
        run_one_shot_training(
            torch,
            one_shot_model=one_shot_model,
            one_shot_loss_fn=one_shot_loss,
            run_manager=run_manager,
            train_bundle=train_bundle,
            validation_bundle=validation_bundle,
            train_cfg=train_cfg,
            run_meta=run_meta,
            physics=physics,
            device=device,
            use_pin_memory=use_pin_memory,
            one_shot_model_version=ONE_SHOT_MODEL_VERSION,
        )

    if train_cfg.mode == "one_shot_distill":
        run_one_shot_distill_training(
            torch,
            one_shot_model=one_shot_model,
            one_shot_loss_fn=one_shot_loss,
            run_manager=run_manager,
            train_cfg=train_cfg,
            run_meta=run_meta,
            physics=physics,
            device=device,
            use_pin_memory=use_pin_memory,
            one_shot_model_version=ONE_SHOT_MODEL_VERSION,
            distill_dataset=args.distill_dataset,
        )

    print(f"train dataset size: {train_bundle.effective_size}/{train_bundle.total_size}")
    if validation_bundle is None:
        print("validation dataset size: disabled")
    else:
        print(f"validation dataset size: {validation_bundle.effective_size}/{validation_bundle.total_size}")
    print(f"dataset total size: {dataset_splits.total_size}")
    print(f"run outputs saved to: {run_manager.run_dir}")


if __name__ == "__main__":
    main()
