"""Stage-scoped runtime contracts.

The runtime classification is deliberately independent of model classes.  It gives the
CLI and tests an auditable dependency footprint before the compatibility StageContext
adapter materializes legacy handler payloads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RuntimeMaterialization:
    """Concrete dependencies for one stage, built from its declared footprint."""

    torch_module: Any
    run_meta: dict[str, Any]
    first_stage: Any
    diffusion_model: Any
    one_shot_model: Any
    scheduler: Any
    train_bundle: Any
    validation_bundle: Any
    first_stage_visual_batch: Any
    first_stage_visual_labels: Any
    diffusion_visual_labels: Any
    diffusion_visual_reference_images: Any
    hooks: dict[str, Any]
    versions: dict[str, str]


@dataclass(frozen=True)
class RuntimeFootprint:
    constructs_models: tuple[str, ...]
    constructs_data: bool
    constructs_loader: bool


@dataclass(frozen=True)
class StageRuntime:
    request: Any
    footprint: RuntimeFootprint


@dataclass(frozen=True)
class DataOnlyRuntime(StageRuntime):
    """Filter/split/audit runtime; no model or dataset loader is constructed."""


@dataclass(frozen=True)
class StudentRuntime(StageRuntime):
    """Student-only runtime; teacher networks and schedulers are absent."""


@dataclass(frozen=True)
class TeacherRuntime(StageRuntime):
    """Teacher runtime with only the networks declared by its stage."""


def build_stage_runtime(request: Any, registry: Mapping[str, Mapping[str, bool]]) -> StageRuntime:
    """Return a scoped runtime descriptor without importing or constructing models."""
    mode = request.train_cfg.mode
    footprint = registry.get(mode)
    if footprint is None:
        raise ValueError(f"unsupported stage for lazy construction: {mode!r}")

    models = []
    if footprint["first_stage"]:
        models.append("first_stage")
    if footprint["diffusion"]:
        models.extend(("diffusion_model", "scheduler"))
    if footprint["student"]:
        models.append("one_shot_model")
    runtime_footprint = RuntimeFootprint(
        constructs_models=tuple(models),
        constructs_data=bool(footprint["data"]),
        constructs_loader=bool(footprint["data"]),
    )
    if not models and not footprint["data"]:
        return DataOnlyRuntime(request, runtime_footprint)
    if footprint["student"] and not (footprint["first_stage"] or footprint["diffusion"]):
        return StudentRuntime(request, runtime_footprint)
    return TeacherRuntime(request, runtime_footprint)


def materialize_stage_runtime(request: Any, runtime: StageRuntime) -> RuntimeMaterialization:
    """Construct exactly the dependencies declared by ``runtime``.

    This is the canonical construction boundary.  The legacy ``StageContext``
    remains an adapter in ``stages.py`` and does not own model construction.
    """
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
    from ..train_utils import resolve_stage_epoch_plan, is_teacher_diffusion_mode, is_teacher_first_stage_mode

    mode = request.train_cfg.mode
    footprint = runtime.footprint
    torch_module = request.torch_module
    model_names = set(footprint.constructs_models)
    first_stage = diffusion_model = one_shot_model = scheduler = None
    if "first_stage" in model_names:
        first_stage = OpticalPathVAE(request.physics, request.train_cfg.first_stage).to(request.device)
        if request.args.first_stage_checkpoint is not None:
            payload = torch_module.load(request.args.first_stage_checkpoint, map_location=request.device, weights_only=False)
            load_first_stage_checkpoint(first_stage, payload)
            if request.checkpoint_resolution["first_stage_checkpoint_source"] == "final_pt":
                print(f"using final_pt first-stage checkpoint: {request.args.first_stage_checkpoint}")
    if "diffusion_model" in model_names:
        diffusion_model = LatentDiffusionUNet(request.train_cfg.first_stage.latent_channels, request.train_cfg.diffusion).to(request.device)
    if "scheduler" in model_names:
        scheduler = LinearNoiseScheduler(
            timesteps=request.train_cfg.diffusion.timesteps,
            beta_start=request.train_cfg.diffusion.beta_start,
            beta_end=request.train_cfg.diffusion.beta_end,
        ).to(request.device)
    if "one_shot_model" in model_names:
        one_shot_model = build_student_model(request.physics, request.train_cfg.one_shot, image_size=request.train_cfg.first_stage.image_size, device=request.device)

    dataset_splits = train_bundle = validation_bundle = None
    first_stage_visual_batch = first_stage_visual_labels = None
    first_stage_visual_source = None
    diffusion_visual_labels = diffusion_visual_reference_images = None
    diffusion_visual_reference_labels = diffusion_visual_source = None
    if footprint.constructs_data:
        dataset_splits = build_train_validation_loaders(
            data_config=request.train_cfg.data,
            first_stage_config=request.train_cfg.first_stage,
            batch_size=request.train_cfg.batch_size,
        )
        train_bundle, validation_bundle = dataset_splits.train, dataset_splits.validation
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
    actual_models = tuple(name for name, value in (("first_stage", first_stage), ("diffusion_model", diffusion_model), ("scheduler", scheduler), ("one_shot_model", one_shot_model)) if value is not None)
    if actual_models != footprint.constructs_models:
        raise RuntimeError(f"stage runtime footprint disagrees with constructed models: declared={footprint.constructs_models}, actual={actual_models}")
    return RuntimeMaterialization(
        torch_module=torch_module, run_meta=run_meta, first_stage=first_stage, diffusion_model=diffusion_model,
        one_shot_model=one_shot_model, scheduler=scheduler, train_bundle=train_bundle, validation_bundle=validation_bundle,
        first_stage_visual_batch=first_stage_visual_batch, first_stage_visual_labels=first_stage_visual_labels,
        diffusion_visual_labels=diffusion_visual_labels, diffusion_visual_reference_images=diffusion_visual_reference_images,
        hooks={"load_first_stage_checkpoint": load_first_stage_checkpoint, "load_diffusion_checkpoint": load_diffusion_checkpoint,
               "load_one_shot_checkpoint": load_one_shot_checkpoint, "first_stage_loss": first_stage_loss,
               "diffusion_loss": diffusion_loss, "diffusion_per_sample_losses": diffusion_per_sample_losses,
               "one_shot_loss": one_shot_loss, "sample_latents": sample_latents, "sample_latents_with_trace": sample_latents_with_trace,
               "resolve_denoising_trace_steps": resolve_denoising_trace_steps},
        versions={"first_stage": FIRST_STAGE_MODEL_VERSION, "diffusion": DIFFUSION_MODEL_VERSION, "one_shot": request.one_shot_version},
    )


__all__ = [
    "DataOnlyRuntime",
    "RuntimeFootprint",
    "StageRuntime",
    "StudentRuntime",
    "TeacherRuntime",
    "build_stage_runtime",
    "RuntimeMaterialization",
    "materialize_stage_runtime",
]
