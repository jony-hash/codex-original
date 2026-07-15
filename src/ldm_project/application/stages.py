from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Callable

from ..config import TrainingConfig
from ..student.evaluation import (
    run_one_shot_distill_eval_stage,
    run_one_shot_sample_stage,
)
from ..diagnostics.identity_audit import run_one_shot_optical_identity_audit_stage
from ..student.training import run_one_shot_distill_training
from ..student.baseline_training import run_one_shot_training
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
from ..teacher.training import (
    run_diffusion_training,
    run_first_stage_training,
)
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
    """Construct the legacy handler context lazily from a stage request."""
    # 保留根包的可注入公开接口：既兼容既有 CLI 测试替身，也不改变实际实现的归属。
    from .. import (
        DIFFUSION_MODEL_VERSION,
        FIRST_STAGE_MODEL_VERSION,
        LatentDiffusionUNet,
        LinearNoiseScheduler,
        OpticalPathVAE,
        build_train_validation_loaders,
        load_diffusion_checkpoint,
        load_first_stage_checkpoint,
        load_one_shot_checkpoint,
    )
    from ..sampling_reports import select_first_stage_visual_batch, select_post_train_sample_batch
    from .model_factory import build_student_model
    from ..student.objectives import one_shot_loss
    from ..teacher.diffusion_losses import diffusion_loss, diffusion_per_sample_losses
    from ..teacher.diffusion_sampling import resolve_denoising_trace_steps, sample_latents, sample_latents_with_trace
    from ..teacher.first_stage_losses import first_stage_loss
    from .run_context import build_run_meta, save_visual_config_snapshots
    from ..train_utils import resolve_stage_epoch_plan

    mode = request.train_cfg.mode
    footprint = STAGE_REGISTRY.get(mode)
    if footprint is None:
        raise ValueError(f"unsupported stage for lazy construction: {mode!r}")
    torch_module = request.torch_module
    first_stage = None
    diffusion_model = None
    one_shot_model = None
    scheduler = None
    if footprint["first_stage"]:
        first_stage = OpticalPathVAE(request.physics, request.train_cfg.first_stage).to(request.device)
        if request.args.first_stage_checkpoint is not None:
            payload = torch_module.load(request.args.first_stage_checkpoint, map_location=request.device, weights_only=False)
            load_first_stage_checkpoint(first_stage, payload)
            if request.checkpoint_resolution["first_stage_checkpoint_source"] == "final_pt":
                print(f"using final_pt first-stage checkpoint: {request.args.first_stage_checkpoint}")
    if footprint["diffusion"]:
        diffusion_model = LatentDiffusionUNet(request.train_cfg.first_stage.latent_channels, request.train_cfg.diffusion).to(request.device)
        scheduler = LinearNoiseScheduler(
            timesteps=request.train_cfg.diffusion.timesteps,
            beta_start=request.train_cfg.diffusion.beta_start,
            beta_end=request.train_cfg.diffusion.beta_end,
        ).to(request.device)
    if footprint["student"]:
        one_shot_model = build_student_model(
            request.physics,
            request.train_cfg.one_shot,
            image_size=request.train_cfg.first_stage.image_size,
            device=request.device,
        )

    dataset_splits = train_bundle = validation_bundle = None
    first_stage_visual_batch = first_stage_visual_labels = None
    first_stage_visual_source = None
    diffusion_visual_labels = diffusion_visual_reference_images = None
    diffusion_visual_reference_labels = diffusion_visual_source = None
    if footprint["data"]:
        dataset_splits = build_train_validation_loaders(
            data_config=request.train_cfg.data, first_stage_config=request.train_cfg.first_stage, batch_size=request.train_cfg.batch_size
        )
        train_bundle = dataset_splits.train
        validation_bundle = dataset_splits.validation
        if is_teacher_first_stage_mode(mode):
            first_stage_visual_batch, first_stage_visual_labels, first_stage_visual_source = select_first_stage_visual_batch(
                torch_module, validation_bundle, train_bundle, request.train_cfg.diffusion.num_classes, request.device, allow_train_reference_fallback=True
            )
        if is_teacher_diffusion_mode(mode):
            diffusion_visual_labels, diffusion_visual_reference_images, diffusion_visual_reference_labels, diffusion_visual_source = select_post_train_sample_batch(
                torch_module, validation_bundle, train_bundle, request.train_cfg.diffusion.num_classes, request.device, sample_repeats_per_class=2, allow_train_reference_fallback=True
            )

    save_visual_config_snapshots(
        request.run_manager, request.physics, request.train_cfg,
        first_stage_visual_labels=first_stage_visual_labels, first_stage_visual_source=first_stage_visual_source,
        diffusion_visual_labels=diffusion_visual_labels, diffusion_visual_reference_labels=diffusion_visual_reference_labels,
        diffusion_visual_source=diffusion_visual_source,
    )
    epoch_plan = resolve_stage_epoch_plan(mode, request.train_cfg.first_stage_epochs, request.train_cfg.diffusion_epochs, request.train_cfg.one_shot_epochs)
    run_meta = build_run_meta(
        args=request.args, train_cfg=request.train_cfg, device=request.device, checkpoint_resolution=request.checkpoint_resolution,
        dataset_splits=dataset_splits, train_bundle=train_bundle, validation_bundle=validation_bundle, epoch_plan=epoch_plan,
        run_manager=request.run_manager, diffusion_model=diffusion_model,
        first_stage_model_version=FIRST_STAGE_MODEL_VERSION, one_shot_model_version=request.one_shot_version,
    )
    request.run_manager.write_meta(run_meta)
    actual_models = tuple(
        name for name, value in (
            ("first_stage", first_stage),
            ("diffusion_model", diffusion_model),
            ("scheduler", scheduler),
            ("one_shot_model", one_shot_model),
        ) if value is not None
    )
    if runtime is not None and actual_models != runtime.footprint.constructs_models:
        raise RuntimeError(
            "stage runtime footprint disagrees with constructed models: "
            f"declared={runtime.footprint.constructs_models}, actual={actual_models}"
        )
    return StageContext(
        torch_module=torch_module, args=request.args, checkpoint_resolution=request.checkpoint_resolution,
        train_cfg=request.train_cfg, run_manager=request.run_manager, run_meta=run_meta, physics=request.physics,
        device=request.device, use_pin_memory=request.use_pin_memory,
        models=StageModels(first_stage=first_stage, diffusion_model=diffusion_model, one_shot_model=one_shot_model, scheduler=scheduler),
        data=StageData(train_bundle=train_bundle, validation_bundle=validation_bundle, first_stage_visual_batch=first_stage_visual_batch,
                       first_stage_visual_labels=first_stage_visual_labels, diffusion_visual_labels=diffusion_visual_labels,
                       diffusion_visual_reference_images=diffusion_visual_reference_images),
        hooks=StageHooks(load_first_stage_checkpoint=load_first_stage_checkpoint, load_diffusion_checkpoint=load_diffusion_checkpoint,
                         load_one_shot_checkpoint=load_one_shot_checkpoint, first_stage_loss=first_stage_loss,
                         diffusion_loss=diffusion_loss, diffusion_per_sample_losses=diffusion_per_sample_losses,
                         one_shot_loss=one_shot_loss, sample_latents=sample_latents,
                         sample_latents_with_trace=sample_latents_with_trace, resolve_denoising_trace_steps=resolve_denoising_trace_steps),
        versions=StageVersions(first_stage=FIRST_STAGE_MODEL_VERSION, diffusion=DIFFUSION_MODEL_VERSION, one_shot=request.one_shot_version),
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
