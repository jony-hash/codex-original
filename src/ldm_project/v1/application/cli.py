from __future__ import annotations

import argparse
import sys

from ..student.evaluation import (
    run_one_shot_distill_eval_diagnostics_stage,
)
from ..stage_constants import (
    DIFFUSION_BEST_CHECKPOINT,
    FIRST_STAGE_BEST_OPTICAL_CHECKPOINT,
)
from .output_layout import resolve_output_layout
from .stages import (
    StageRequest,
    run_selected_stage,
)
from ..train_utils import (
    CLI_STAGE_CHOICES,
    apply_cli_overrides,
    apply_training_seed,
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
    parser.add_argument("--physics-config", default="configs/v1/formal/physics_cfg.yaml", help="Path to physics_cfg.yaml")
    parser.add_argument(
        "--train-config",
        default="configs/v1/formal/train_ldm.yaml",
        help="Path to training YAML; formal D2NN runs require an approved coherent-relay physical-forward config (see docs/ldm_migration/stage_guide.md).",
    )
    parser.add_argument("--stage", choices=CLI_STAGE_CHOICES, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override training random seed for reproducible runs.")
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
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root directory for timestamped run outputs; defaults to outputs/YYYYMMDD_<version>/runs.",
    )
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
        "--one-shot-generator-checkpoint",
        default=None,
        help="Optional one_shot_distill warm-start for generator_e.* parameters only.",
    )
    parser.add_argument(
        "--one-shot-optical-checkpoint",
        default=None,
        help="Optional one_shot_distill warm-start for optical encoder/decoder parameters only.",
    )
    parser.add_argument(
        "--distill-dataset",
        default=None,
        help="Required for --stage one_shot_distill to load offline teacher targets.",
    )
    parser.add_argument(
        "--distill-validation-dataset",
        default=None,
        help="Optional held-out distillation validation bundle for --stage one_shot_distill.",
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
        "output_version": args.output_version,
        "output_root": args.output_root,
        "output_root_explicit": args.output_root_explicit,
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

    # 通过根包公开契约读取依赖，保留既有 CLI 的可替换测试/集成接口。
    from ldm_project import (
        DIFFUSION_MODEL_VERSION,
        FIRST_STAGE_MODEL_VERSION,
        ONE_SHOT_MODEL_VERSION,
        RunManager,
        load_physics_config,
        load_training_config,
        validate_physical_forward_contract,
    )

    args = parse_args()
    physics = load_physics_config(args.physics_config)
    train_cfg = load_training_config(args.train_config)
    if train_cfg.physics is not None:
        physics = train_cfg.physics
    apply_cli_overrides(train_cfg, args)
    validate_physical_forward_contract(train_cfg, physics)
    output_layout = resolve_output_layout(
        args.output_root,
        run_name=args.run_name,
        train_config=args.train_config,
    )
    args.output_root = str(output_layout.output_root)
    args.output_version = output_layout.version
    args.output_root_explicit = output_layout.explicit

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
    applied_training_seed = apply_training_seed(torch, train_cfg.seed)
    if applied_training_seed is not None:
        train_cfg.seed = applied_training_seed
        print(f"using training seed: {applied_training_seed}")

    run_manager = RunManager.create(
        output_root=args.output_root,
        purpose="train",
        run_name=args.run_name or train_cfg.mode,
    )
    epoch_plan = resolve_stage_epoch_plan(
        train_cfg.mode,
        train_cfg.first_stage_epochs,
        train_cfg.diffusion_epochs,
        train_cfg.one_shot_epochs,
    )
    print(
        f"planned epochs: first_stage={epoch_plan['first_stage']} "
        f"diffusion={epoch_plan['diffusion']} one_shot={epoch_plan['one_shot']}"
    )
    stage_context = run_selected_stage(
        StageRequest(
            torch_module=torch,
            args=args,
            checkpoint_resolution=checkpoint_resolution,
            train_cfg=train_cfg,
            run_manager=run_manager,
            physics=physics,
            device=device,
            use_pin_memory=use_pin_memory,
            first_stage_version=FIRST_STAGE_MODEL_VERSION,
            diffusion_version=DIFFUSION_MODEL_VERSION,
            one_shot_version=ONE_SHOT_MODEL_VERSION,
        )
    )
    if stage_context.data.train_bundle is None:
        print("mnist auxiliary dataset: not constructed for this stage")
    else:
        train_bundle = stage_context.data.train_bundle
        validation_bundle = stage_context.data.validation_bundle
        print(f"mnist auxiliary train dataset size: {train_bundle.effective_size}/{train_bundle.total_size}")
        if validation_bundle is None:
            print("mnist auxiliary validation dataset size: disabled")
        else:
            print(f"mnist auxiliary validation dataset size: {validation_bundle.effective_size}/{validation_bundle.total_size}")
        print(f"mnist auxiliary dataset total size: {stage_context.run_meta['dataset_total_size']}")
    print(f"run outputs saved to: {run_manager.run_dir}")


if __name__ == "__main__":
    main()
