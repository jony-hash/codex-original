from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Callable

from ..config import TrainingConfig
from ..train_utils import is_teacher_diffusion_mode, is_teacher_first_stage_mode
from .stage_runtimes import StageRuntime, build_stage_runtime


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
    # 未被当前 stage 声明的 hook 固定为 None；StageContext 仍保留旧字段形状。
    load_first_stage_checkpoint: Callable[..., Any] | None
    load_diffusion_checkpoint: Callable[..., Any] | None
    load_one_shot_checkpoint: Callable[..., Any] | None
    first_stage_loss: Callable[..., Any] | None
    diffusion_loss: Callable[..., Any] | None
    diffusion_per_sample_losses: Callable[..., Any] | None
    one_shot_loss: Callable[..., Any] | None
    sample_latents: Callable[..., Any] | None
    sample_latents_with_trace: Callable[..., Any] | None
    resolve_denoising_trace_steps: Callable[..., Any] | None


@dataclass(frozen=True)
class StageVersions:
    first_stage: str
    diffusion: str
    one_shot: str


@dataclass(frozen=True)
class StageRequest:
    """Lightweight stage input; handlers construct only their own dependencies."""

    torch_module: Any
    args: Namespace
    checkpoint_resolution: dict[str, str | None]
    train_cfg: TrainingConfig
    run_manager: Any
    physics: Any
    device: Any
    use_pin_memory: bool
    first_stage_version: str
    diffusion_version: str
    one_shot_version: str


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
    runtime: StageRuntime | None = None


_NO_DATA_STAGES = frozenset(
    {
        "teacher_distill_filter",
        "teacher_distill_label_audit",
        "teacher_distill_label_strict_filter",
        "teacher_distill_readable_split",
        "teacher_first_stage_oracle_audit",
        "teacher_diffusion_latent_audit",
        "one_shot_distill",
        "one_shot_distill_eval",
        "one_shot_optical_identity_audit",
    }
)
_FIRST_STAGE_STAGES = frozenset(
    {
        "teacher_sample",
        "teacher_distill_export",
        "teacher_first_stage_oracle_audit",
        "teacher_diffusion_latent_audit",
    }
)
_DIFFUSION_STAGES = frozenset({"teacher_sample", "teacher_distill_export", "teacher_diffusion_latent_audit"})
_STUDENT_STAGES = frozenset({"one_shot", "one_shot_distill", "one_shot_sample", "one_shot_distill_eval", "one_shot_optical_identity_audit"})

# Registry is intentionally declarative so a stage's model/data footprint is auditable.
STAGE_REGISTRY = {
    mode: {"data": mode not in _NO_DATA_STAGES, "first_stage": mode in _FIRST_STAGE_STAGES or is_teacher_first_stage_mode(mode) or is_teacher_diffusion_mode(mode), "diffusion": mode in _DIFFUSION_STAGES or is_teacher_diffusion_mode(mode), "student": mode in _STUDENT_STAGES}
    for mode in (
        "first_stage", "diffusion", "teacher_first_stage", "teacher_diffusion", "teacher_full",
        "teacher_sample", "teacher_distill_export", "teacher_distill_filter",
        "teacher_distill_label_audit", "teacher_distill_label_strict_filter", "teacher_distill_readable_split",
        "teacher_first_stage_oracle_audit", "teacher_diffusion_latent_audit", "one_shot", "one_shot_distill",
        "one_shot_sample", "one_shot_distill_eval", "one_shot_optical_identity_audit",
    )
}


def _build_stage_context(request: StageRequest, runtime: StageRuntime | None = None) -> StageContext:
    """Compatibility adapter around the canonical scoped runtime materializer."""
    from .stage_runtimes import materialize_stage_runtime

    if runtime is None:
        runtime = build_stage_runtime(request, STAGE_REGISTRY)
    materialized = materialize_stage_runtime(request, runtime)
    hooks = materialized.hooks
    return StageContext(
        torch_module=materialized.torch_module,
        args=request.args,
        checkpoint_resolution=request.checkpoint_resolution,
        train_cfg=request.train_cfg,
        run_manager=request.run_manager,
        run_meta=materialized.run_meta,
        physics=request.physics,
        device=request.device,
        use_pin_memory=request.use_pin_memory,
        models=StageModels(
            first_stage=materialized.first_stage,
            diffusion_model=materialized.diffusion_model,
            one_shot_model=materialized.one_shot_model,
            scheduler=materialized.scheduler,
        ),
        data=StageData(
            train_bundle=materialized.train_bundle,
            validation_bundle=materialized.validation_bundle,
            first_stage_visual_batch=materialized.first_stage_visual_batch,
            first_stage_visual_labels=materialized.first_stage_visual_labels,
            diffusion_visual_labels=materialized.diffusion_visual_labels,
            diffusion_visual_reference_images=materialized.diffusion_visual_reference_images,
        ),
        hooks=StageHooks(**hooks),
        versions=StageVersions(**materialized.versions),
        runtime=runtime,
    )

def run_selected_stage(context: StageRequest | StageContext) -> StageContext:
    """Dispatch a request with lazy construction; accept StageContext for legacy callers."""
    if isinstance(context, StageRequest):
        # 先登记 stage-specific footprint，再通过兼容适配器物化旧 handler payload。
        # 该登记本身不导入模型，确保 data-only/student stage 的边界可测试。
        runtime = build_stage_runtime(context, STAGE_REGISTRY)
        context = _build_stage_context(context, runtime=runtime)
    mode = context.train_cfg.mode
    handler = STAGE_HANDLER_REGISTRY.get(mode)
    if handler is None:
        raise ValueError(f"unsupported stage: {mode!r}")
    handler(context)
    return context


def _run_teacher_artifact_stage(context: StageContext, mode: str) -> bool:
    from ..teacher.distill import (
        run_teacher_distill_export,
        run_teacher_distill_filter,
        run_teacher_distill_label_audit,
        run_teacher_distill_label_strict_filter,
        run_teacher_distill_readable_split,
    )
    from ..teacher.fault_localization import (
        run_teacher_diffusion_latent_audit,
        run_teacher_first_stage_oracle_audit,
    )
    from ..teacher.sampling import run_sample_stage

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
    from ..diagnostics.identity_audit import run_one_shot_optical_identity_audit_stage
    from ..student.evaluation import (
        run_one_shot_distill_eval_stage,
        run_one_shot_sample_stage,
    )

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
    from ..student.baseline_training import run_one_shot_training
    from ..student.training import run_one_shot_distill_training
    from ..teacher.training import run_diffusion_training, run_first_stage_training

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


def _handle_teacher_artifact_stage(context: StageContext) -> None:
    _run_teacher_artifact_stage(context, context.train_cfg.mode)


def _handle_one_shot_artifact_stage(context: StageContext) -> None:
    _run_one_shot_artifact_stage(context, context.train_cfg.mode)


def _handle_training_stage(context: StageContext) -> None:
    _run_training_stage(context, context.train_cfg.mode)


# Explicit registry: dispatch ownership is auditable without constructing a
# model to discover which branch a stage will use.  StageContext remains only
# the compatibility payload for the existing runners and tests.
STAGE_HANDLER_REGISTRY: dict[str, Callable[[StageContext], None]] = {
    **{
        mode: _handle_teacher_artifact_stage
        for mode in {
            "teacher_sample",
            "teacher_distill_export",
            "teacher_distill_filter",
            "teacher_distill_label_audit",
            "teacher_distill_label_strict_filter",
            "teacher_distill_readable_split",
            "teacher_first_stage_oracle_audit",
            "teacher_diffusion_latent_audit",
        }
    },
    **{
        mode: _handle_one_shot_artifact_stage
        for mode in {"one_shot_sample", "one_shot_distill_eval", "one_shot_optical_identity_audit"}
    },
    **{
        mode: _handle_training_stage
        for mode in {
            "first_stage",
            "diffusion",
            "teacher_first_stage",
            "teacher_diffusion",
            "teacher_full",
            "one_shot",
            "one_shot_distill",
        }
    },
}
