from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Callable

from .config import TrainingConfig
from .one_shot_eval import (
    run_one_shot_distill_eval_stage,
    run_one_shot_sample_stage,
)
from .one_shot_identity_audit import run_one_shot_optical_identity_audit_stage
from .one_shot_distill_training import run_one_shot_distill_training
from .one_shot_training import run_one_shot_training
from .teacher_distill import (
    run_teacher_distill_export,
    run_teacher_distill_filter,
    run_teacher_distill_label_audit,
    run_teacher_distill_label_strict_filter,
    run_teacher_distill_readable_split,
)
from .teacher_fault_localization import (
    run_teacher_diffusion_latent_audit,
    run_teacher_first_stage_oracle_audit,
)
from .teacher_sampling import run_sample_stage
from .teacher_training import (
    run_diffusion_training,
    run_first_stage_training,
)
from .train_utils import is_teacher_diffusion_mode, is_teacher_first_stage_mode


@dataclass(frozen=True)
class StageModels:
    first_stage: Any
    diffusion_model: Any
    one_shot_model: Any
    scheduler: Any


@dataclass(frozen=True)
class StageData:
    train_bundle: Any
    validation_bundle: Any
    first_stage_visual_batch: Any
    first_stage_visual_labels: Any
    diffusion_visual_labels: Any
    diffusion_visual_reference_images: Any


@dataclass(frozen=True)
class StageHooks:
    load_first_stage_checkpoint: Callable[..., Any]
    load_diffusion_checkpoint: Callable[..., Any]
    load_one_shot_checkpoint: Callable[..., Any]
    first_stage_loss: Callable[..., Any]
    diffusion_loss: Callable[..., Any]
    diffusion_per_sample_losses: Callable[..., Any]
    one_shot_loss: Callable[..., Any]
    sample_latents: Callable[..., Any]
    sample_latents_with_trace: Callable[..., Any]
    resolve_denoising_trace_steps: Callable[..., Any]


@dataclass(frozen=True)
class StageVersions:
    first_stage: str
    diffusion: str
    one_shot: str


@dataclass(frozen=True)
class StageContext:
    torch_module: Any
    args: Namespace
    checkpoint_resolution: dict[str, str | None]
    train_cfg: TrainingConfig
    run_manager: Any
    run_meta: dict[str, Any]
    physics: Any
    device: Any
    use_pin_memory: bool
    models: StageModels
    data: StageData
    hooks: StageHooks
    versions: StageVersions


def run_selected_stage(context: StageContext) -> None:
    mode = context.train_cfg.mode
    if _run_teacher_artifact_stage(context, mode):
        return
    if _run_one_shot_artifact_stage(context, mode):
        return
    _run_training_stage(context, mode)


def _run_teacher_artifact_stage(context: StageContext, mode: str) -> bool:
    args = context.args
    data = context.data
    hooks = context.hooks
    models = context.models
    versions = context.versions

    if mode == "teacher_sample":
        if context.checkpoint_resolution["diffusion_checkpoint_source"] == "final_pt":
            print(f"using final_pt diffusion checkpoint: {args.diffusion_checkpoint}")
        run_sample_stage(
            context.torch_module,
            first_stage=models.first_stage,
            diffusion_model=models.diffusion_model,
            scheduler=models.scheduler,
            train_bundle=data.train_bundle,
            validation_bundle=data.validation_bundle,
            train_cfg=context.train_cfg,
            run_manager=context.run_manager,
            run_meta=context.run_meta,
            device=context.device,
            diffusion_checkpoint=args.diffusion_checkpoint,
            first_stage_model_version=versions.first_stage,
            load_diffusion_checkpoint_fn=hooks.load_diffusion_checkpoint,
            sample_latents_fn=hooks.sample_latents,
            sample_latents_with_trace_fn=hooks.sample_latents_with_trace,
            resolve_denoising_trace_steps_fn=hooks.resolve_denoising_trace_steps,
        )
        return True

    if mode == "teacher_distill_export":
        if context.checkpoint_resolution["diffusion_checkpoint_source"] == "final_pt":
            print(f"using final_pt diffusion checkpoint: {args.diffusion_checkpoint}")
        run_teacher_distill_export(
            context.torch_module,
            first_stage=models.first_stage,
            diffusion_model=models.diffusion_model,
            scheduler=models.scheduler,
            train_cfg=context.train_cfg,
            run_manager=context.run_manager,
            run_meta=context.run_meta,
            device=context.device,
            diffusion_checkpoint=args.diffusion_checkpoint,
            first_stage_model_version=versions.first_stage,
            load_diffusion_checkpoint_fn=hooks.load_diffusion_checkpoint,
            sample_latents_fn=hooks.sample_latents,
        )
        return True

    if mode == "teacher_distill_filter":
        run_teacher_distill_filter(
            context.torch_module,
            train_cfg=context.train_cfg,
            run_manager=context.run_manager,
            run_meta=context.run_meta,
            distill_dataset=args.distill_dataset,
        )
        return True

    if mode == "teacher_distill_label_audit":
        run_teacher_distill_label_audit(
            context.torch_module,
            train_cfg=context.train_cfg,
            run_manager=context.run_manager,
            run_meta=context.run_meta,
            distill_dataset=args.distill_dataset,
            device=context.device,
        )
        return True

    if mode == "teacher_distill_label_strict_filter":
        run_teacher_distill_label_strict_filter(
            context.torch_module,
            train_cfg=context.train_cfg,
            run_manager=context.run_manager,
            run_meta=context.run_meta,
            distill_dataset=args.distill_dataset,
            device=context.device,
        )
        return True

    if mode == "teacher_distill_readable_split":
        run_teacher_distill_readable_split(
            context.torch_module,
            train_cfg=context.train_cfg,
            run_manager=context.run_manager,
            run_meta=context.run_meta,
            distill_dataset=args.distill_dataset,
            device=context.device,
        )
        return True

    if mode == "teacher_first_stage_oracle_audit":
        run_teacher_first_stage_oracle_audit(
            context.torch_module,
            first_stage=models.first_stage,
            train_cfg=context.train_cfg,
            run_manager=context.run_manager,
            run_meta=context.run_meta,
            device=context.device,
        )
        return True

    if mode == "teacher_diffusion_latent_audit":
        if context.checkpoint_resolution["diffusion_checkpoint_source"] == "final_pt":
            print(f"using final_pt diffusion checkpoint: {args.diffusion_checkpoint}")
        run_teacher_diffusion_latent_audit(
            context.torch_module,
            first_stage=models.first_stage,
            diffusion_model=models.diffusion_model,
            scheduler=models.scheduler,
            train_cfg=context.train_cfg,
            run_manager=context.run_manager,
            run_meta=context.run_meta,
            device=context.device,
            diffusion_checkpoint=args.diffusion_checkpoint,
            first_stage_model_version=versions.first_stage,
            load_diffusion_checkpoint_fn=hooks.load_diffusion_checkpoint,
            sample_latents_fn=hooks.sample_latents,
        )
        return True

    return False


def _run_one_shot_artifact_stage(context: StageContext, mode: str) -> bool:
    args = context.args
    data = context.data
    hooks = context.hooks
    models = context.models

    if mode == "one_shot_sample":
        run_one_shot_sample_stage(
            context.torch_module,
            one_shot_model=models.one_shot_model,
            train_bundle=data.train_bundle,
            validation_bundle=data.validation_bundle,
            train_cfg=context.train_cfg,
            run_manager=context.run_manager,
            run_meta=context.run_meta,
            device=context.device,
            one_shot_checkpoint=args.one_shot_checkpoint,
            load_one_shot_checkpoint_fn=hooks.load_one_shot_checkpoint,
        )
        return True

    if mode == "one_shot_distill_eval":
        run_one_shot_distill_eval_stage(
            context.torch_module,
            one_shot_model=models.one_shot_model,
            train_cfg=context.train_cfg,
            run_manager=context.run_manager,
            run_meta=context.run_meta,
            device=context.device,
            one_shot_checkpoint=args.one_shot_checkpoint,
            distill_dataset=args.distill_dataset,
            load_one_shot_checkpoint_fn=hooks.load_one_shot_checkpoint,
        )
        return True

    if mode == "one_shot_optical_identity_audit":
        run_one_shot_optical_identity_audit_stage(
            context.torch_module,
            one_shot_model=models.one_shot_model,
            train_cfg=context.train_cfg,
            run_manager=context.run_manager,
            run_meta=context.run_meta,
            device=context.device,
            one_shot_generator_checkpoint=getattr(args, "one_shot_generator_checkpoint", None),
        )
        return True

    return False


def _run_training_stage(context: StageContext, mode: str) -> None:
    args = context.args
    data = context.data
    hooks = context.hooks
    models = context.models
    versions = context.versions

    if is_teacher_first_stage_mode(mode):
        run_first_stage_training(
            context.torch_module,
            first_stage=models.first_stage,
            first_stage_loss_fn=hooks.first_stage_loss,
            first_stage_visual_batch=data.first_stage_visual_batch,
            first_stage_visual_labels=data.first_stage_visual_labels,
            run_manager=context.run_manager,
            train_bundle=data.train_bundle,
            validation_bundle=data.validation_bundle,
            train_cfg=context.train_cfg,
            run_meta=context.run_meta,
            physics=context.physics,
            device=context.device,
            use_pin_memory=context.use_pin_memory,
            first_stage_model_version=versions.first_stage,
        )

    if is_teacher_diffusion_mode(mode):
        run_diffusion_training(
            context.torch_module,
            first_stage=models.first_stage,
            diffusion_model=models.diffusion_model,
            scheduler=models.scheduler,
            train_bundle=data.train_bundle,
            validation_bundle=data.validation_bundle,
            train_cfg=context.train_cfg,
            run_meta=context.run_meta,
            run_manager=context.run_manager,
            physics=context.physics,
            device=context.device,
            use_pin_memory=context.use_pin_memory,
            first_stage_model_version=versions.first_stage,
            diffusion_model_version=versions.diffusion,
            diffusion_visual_labels=data.diffusion_visual_labels,
            diffusion_visual_reference_images=data.diffusion_visual_reference_images,
            load_first_stage_checkpoint_fn=hooks.load_first_stage_checkpoint,
            load_diffusion_checkpoint_fn=hooks.load_diffusion_checkpoint,
            diffusion_loss_fn=hooks.diffusion_loss,
            diffusion_per_sample_losses_fn=hooks.diffusion_per_sample_losses,
            sample_latents_fn=hooks.sample_latents,
            sample_latents_with_trace_fn=hooks.sample_latents_with_trace,
            resolve_denoising_trace_steps_fn=hooks.resolve_denoising_trace_steps,
        )

    if mode == "one_shot":
        run_one_shot_training(
            context.torch_module,
            one_shot_model=models.one_shot_model,
            one_shot_loss_fn=hooks.one_shot_loss,
            run_manager=context.run_manager,
            train_bundle=data.train_bundle,
            validation_bundle=data.validation_bundle,
            train_cfg=context.train_cfg,
            run_meta=context.run_meta,
            physics=context.physics,
            device=context.device,
            use_pin_memory=context.use_pin_memory,
            one_shot_model_version=versions.one_shot,
        )
        return

    if mode == "one_shot_distill":
        run_one_shot_distill_training(
            context.torch_module,
            one_shot_model=models.one_shot_model,
            one_shot_loss_fn=hooks.one_shot_loss,
            run_manager=context.run_manager,
            train_cfg=context.train_cfg,
            run_meta=context.run_meta,
            physics=context.physics,
            device=context.device,
            use_pin_memory=context.use_pin_memory,
            one_shot_model_version=versions.one_shot,
            distill_dataset=args.distill_dataset,
            distill_validation_dataset=args.distill_validation_dataset,
            one_shot_checkpoint=args.one_shot_checkpoint,
            one_shot_generator_checkpoint=getattr(args, "one_shot_generator_checkpoint", None),
            one_shot_optical_checkpoint=getattr(args, "one_shot_optical_checkpoint", None),
            load_one_shot_checkpoint_fn=hooks.load_one_shot_checkpoint,
        )
