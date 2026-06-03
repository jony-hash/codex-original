from __future__ import annotations

import argparse
import sys

from ldm_project.sampling_reports import (
    select_first_stage_visual_batch,
    select_post_train_sample_batch,
)
from ldm_project.one_shot_eval import (
    run_one_shot_distill_eval_diagnostics_stage,
)
from ldm_project.stage_constants import (
    DIFFUSION_BEST_CHECKPOINT,
    FIRST_STAGE_BEST_OPTICAL_CHECKPOINT,
)
from ldm_project.train_orchestration import (
    StageContext,
    StageData,
    StageHooks,
    StageModels,
    StageVersions,
    run_selected_stage,
)
from ldm_project.train_run_context import build_run_meta, save_visual_config_snapshots
from ldm_project.train_utils import (
    CLI_STAGE_CHOICES,
    apply_cli_overrides,
    is_teacher_diffusion_mode,
    is_teacher_first_stage_mode,
    resolve_device,
    resolve_final_pt_stage_checkpoints,
    resolve_stage_epoch_plan,
    validate_class_label_values,
    validate_stage_requirements,
)

__all__ = [
    "main",
    "parse_args",
    "run_one_shot_distill_eval_diagnostics_stage",
    "_create_diagnostics_run",
    "_validate_diagnostics_args",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Optical-Path VAE LDM scaffold.")
    parser.add_argument("--physics-config", default="configs/physics_cfg.yaml", help="Path to physics_cfg.yaml")
    parser.add_argument(
        "--train-config",
        default="configs/train_ldm.yaml",
        help="Path to training YAML; use configs/train_one_shot_distill_d2nn_mnist_3k_fast200.yaml for D2NN.",
    )
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
        "--distill-train-subset-size",
        type=int,
        default=None,
        help="Use a deterministic subset of the offline distillation dataset for one_shot_distill.",
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
        help="For --stage one_shot_sample / one_shot_distill_eval: required to load a trained one-shot model. "
        "For --stage one_shot_distill: optional model-only warm-start (fresh optimizer, epoch starts at 1).",
    )
    parser.add_argument(
        "--distill-dataset",
        default=None,
        help="Required for --stage one_shot_distill to load offline teacher targets.",
    )
    parser.add_argument(
        "--eval-tensor-bundle",
        default=None,
        help="Required for --stage one_shot_distill_eval_diagnostics to load eval tensor artifacts.",
    )
    parser.add_argument(
        "--eval-metrics",
        default=None,
        help="Required for --stage one_shot_distill_eval_diagnostics to load eval metric JSON.",
    )
    parser.add_argument(
        "--diagnostic-class-labels",
        nargs="+",
        type=int,
        default=list(range(10)),
        help="Class labels to include in one-shot eval focus diagnostics.",
    )
    parser.add_argument(
        "--diagnostic-max-per-class",
        type=int,
        default=2,
        help="Maximum samples per class in one-shot eval focus diagnostics.",
    )
    return parser.parse_args()


def _validate_diagnostics_args(args, train_cfg) -> tuple[int, ...]:
    if args.eval_tensor_bundle is None:
        raise ValueError("--stage one_shot_distill_eval_diagnostics requires --eval-tensor-bundle.")
    if args.eval_metrics is None:
        raise ValueError("--stage one_shot_distill_eval_diagnostics requires --eval-metrics.")
    if args.diagnostic_max_per_class <= 0:
        raise ValueError(
            "diagnostic_max_per_class must be positive, "
            f"got {args.diagnostic_max_per_class}"
        )
    return validate_class_label_values(
        args.diagnostic_class_labels,
        train_cfg.one_shot.num_classes,
        "diagnostic_class_labels",
    )


def _create_diagnostics_run(torch_module, *, args, physics, train_cfg, device, run_manager_cls) -> None:
    diagnostic_labels = _validate_diagnostics_args(args, train_cfg)
    run_manager = run_manager_cls.create(
        output_root=args.output_root,
        purpose="train",
        run_name=args.run_name or train_cfg.mode,
    )
    run_meta = {
        "script": "train.py",
        "argv": sys.argv,
        "device": str(device),
        "mode": train_cfg.mode,
        "physics_config_path": args.physics_config,
        "train_config_path": args.train_config,
        "eval_tensor_bundle": args.eval_tensor_bundle,
        "eval_metrics": args.eval_metrics,
        "diagnostic_class_labels": list(diagnostic_labels),
        "diagnostic_max_per_class": args.diagnostic_max_per_class,
        "run_dir": run_manager.run_dir,
    }
    run_manager.write_meta(run_meta)
    run_manager.save_config_snapshot("physics_config", physics)
    run_manager.save_config_snapshot("train_config", train_cfg)
    run_one_shot_distill_eval_diagnostics_stage(
        torch_module,
        run_manager=run_manager,
        run_meta=run_meta,
        eval_tensor_bundle=args.eval_tensor_bundle,
        eval_metrics=args.eval_metrics,
        diagnostic_class_labels=diagnostic_labels,
        diagnostic_max_per_class=args.diagnostic_max_per_class,
    )
    print(f"run outputs saved to: {run_manager.run_dir}")








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
    from ldm_project.diffusion_losses import diffusion_loss, diffusion_per_sample_losses
    from ldm_project.diffusion_sampling import (
        resolve_denoising_trace_steps,
        sample_latents,
        sample_latents_with_trace,
    )
    from ldm_project.first_stage_losses import first_stage_loss
    from ldm_project.one_shot_losses import one_shot_loss

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
    if train_cfg.mode == "one_shot_distill_eval_diagnostics":
        _create_diagnostics_run(
            torch,
            args=args,
            physics=physics,
            train_cfg=train_cfg,
            device=device,
            run_manager_cls=RunManager,
        )
        return

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
            sample_repeats_per_class=2,
        )

    save_visual_config_snapshots(
        run_manager,
        physics,
        train_cfg,
        first_stage_visual_labels=first_stage_visual_labels,
        first_stage_visual_source=first_stage_visual_source,
        diffusion_visual_labels=diffusion_visual_labels,
        diffusion_visual_reference_labels=diffusion_visual_reference_labels,
        diffusion_visual_source=diffusion_visual_source,
    )
    epoch_plan = resolve_stage_epoch_plan(
        train_cfg.mode,
        train_cfg.first_stage_epochs,
        train_cfg.diffusion_epochs,
        train_cfg.one_shot_epochs,
    )
    run_meta = build_run_meta(
        args=args,
        train_cfg=train_cfg,
        device=device,
        checkpoint_resolution=checkpoint_resolution,
        dataset_splits=dataset_splits,
        train_bundle=train_bundle,
        validation_bundle=validation_bundle,
        epoch_plan=epoch_plan,
        run_manager=run_manager,
        diffusion_model=diffusion_model,
        first_stage_model_version=FIRST_STAGE_MODEL_VERSION,
        one_shot_model_version=ONE_SHOT_MODEL_VERSION,
    )
    run_manager.write_meta(run_meta)
    print(
        f"planned epochs: first_stage={epoch_plan['first_stage']} "
        f"diffusion={epoch_plan['diffusion']} one_shot={epoch_plan['one_shot']}"
    )

    run_selected_stage(
        StageContext(
            torch_module=torch,
            args=args,
            checkpoint_resolution=checkpoint_resolution,
            train_cfg=train_cfg,
            run_manager=run_manager,
            run_meta=run_meta,
            physics=physics,
            device=device,
            use_pin_memory=use_pin_memory,
            models=StageModels(
                first_stage=first_stage,
                diffusion_model=diffusion_model,
                one_shot_model=one_shot_model,
                scheduler=scheduler,
            ),
            data=StageData(
                train_bundle=train_bundle,
                validation_bundle=validation_bundle,
                first_stage_visual_batch=first_stage_visual_batch,
                first_stage_visual_labels=first_stage_visual_labels,
                diffusion_visual_labels=diffusion_visual_labels,
                diffusion_visual_reference_images=diffusion_visual_reference_images,
            ),
            hooks=StageHooks(
                load_first_stage_checkpoint=load_first_stage_checkpoint,
                load_diffusion_checkpoint=load_diffusion_checkpoint,
                load_one_shot_checkpoint=load_one_shot_checkpoint,
                first_stage_loss=first_stage_loss,
                diffusion_loss=diffusion_loss,
                diffusion_per_sample_losses=diffusion_per_sample_losses,
                one_shot_loss=one_shot_loss,
                sample_latents=sample_latents,
                sample_latents_with_trace=sample_latents_with_trace,
                resolve_denoising_trace_steps=resolve_denoising_trace_steps,
            ),
            versions=StageVersions(
                first_stage=FIRST_STAGE_MODEL_VERSION,
                diffusion=DIFFUSION_MODEL_VERSION,
                one_shot=ONE_SHOT_MODEL_VERSION,
            ),
        )
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
