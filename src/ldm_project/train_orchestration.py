from __future__ import annotations

from dataclasses import dataclass

from .one_shot_eval import (
    run_one_shot_distill_eval_stage,
    run_one_shot_sample_stage,
)
from .one_shot_distill_training import run_one_shot_distill_training
from .one_shot_training import run_one_shot_training
from .teacher_distill import run_teacher_distill_export, run_teacher_distill_filter
from .teacher_sampling import run_sample_stage
from .teacher_training import (
    run_diffusion_training,
    run_first_stage_training,
)
from .train_utils import is_teacher_diffusion_mode, is_teacher_first_stage_mode


@dataclass(frozen=True)
class StageModels:
    first_stage: object
    diffusion_model: object
    one_shot_model: object
    scheduler: object


@dataclass(frozen=True)
class StageData:
    train_bundle: object
    validation_bundle: object
    first_stage_visual_batch: object
    first_stage_visual_labels: object
    diffusion_visual_labels: object
    diffusion_visual_reference_images: object


@dataclass(frozen=True)
class StageHooks:
    load_first_stage_checkpoint: object
    load_diffusion_checkpoint: object
    load_one_shot_checkpoint: object
    first_stage_loss: object
    diffusion_loss: object
    diffusion_per_sample_losses: object
    one_shot_loss: object
    sample_latents: object
    sample_latents_with_trace: object
    resolve_denoising_trace_steps: object


@dataclass(frozen=True)
class StageVersions:
    first_stage: str
    diffusion: str
    one_shot: str


@dataclass(frozen=True)
class StageContext:
    torch_module: object
    args: object
    checkpoint_resolution: dict[str, str | None]
    train_cfg: object
    run_manager: object
    run_meta: dict[str, object]
    physics: object
    device: object
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
            one_shot_checkpoint=args.one_shot_checkpoint,
            load_one_shot_checkpoint_fn=hooks.load_one_shot_checkpoint,
        )
