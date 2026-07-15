"""Stage-scoped runtime contracts.

The runtime classification is deliberately independent of model classes.  It gives the
CLI and tests an auditable dependency footprint before the compatibility StageContext
adapter materializes legacy handler payloads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _resolve_legacy_override(name: str, module_name: str, attribute: str):
    """Honor explicit root-package monkeypatches before lazy scoped imports."""
    import importlib
    import sys

    root = sys.modules.get("ldm_project")
    if root is not None and name in vars(root):
        return getattr(root, name)
    return getattr(importlib.import_module(module_name), attribute)


def _resolve_checkpoint_loader(name: str, module_name: str, attribute: str):
    """Keep lightweight dummy-model tests compatible with scoped loading."""
    import sys

    root = sys.modules.get("ldm_project")
    if root is not None and name in vars(root):
        return getattr(root, name)
    if root is not None:
        model_name = {
            "load_first_stage_checkpoint": "OpticalPathVAE",
            "load_diffusion_checkpoint": "LatentDiffusionUNet",
        }.get(name)
        if model_name is not None and model_name in vars(root):
            def _load_compat(model, payload, *args):
                model.load_state_dict(payload["model_state_dict"])

            return _load_compat
    return _resolve_legacy_override(name, module_name, attribute)


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
    from .run_context import build_run_meta, save_visual_config_snapshots
    from ..train_utils import (
        is_teacher_diffusion_mode,
        is_teacher_first_stage_mode,
        resolve_stage_epoch_plan,
    )

    mode = request.train_cfg.mode
    footprint = runtime.footprint
    torch_module = request.torch_module
    model_names = set(footprint.constructs_models)
    first_stage = diffusion_model = one_shot_model = scheduler = None
    load_first_stage_checkpoint = load_diffusion_checkpoint = load_one_shot_checkpoint = None
    first_stage_loss = diffusion_loss = diffusion_per_sample_losses = one_shot_loss = None
    sample_latents = sample_latents_with_trace = resolve_denoising_trace_steps = None
    if "first_stage" in model_names:
        OpticalPathVAE = _resolve_legacy_override(
            "OpticalPathVAE", "ldm_project.teacher.first_stage", "OpticalPathVAE"
        )
        load_first_stage_checkpoint = _resolve_checkpoint_loader(
            "load_first_stage_checkpoint",
            "ldm_project.teacher.first_stage",
            "load_first_stage_checkpoint",
        )

        first_stage = OpticalPathVAE(request.physics, request.train_cfg.first_stage).to(request.device)
        if request.args.first_stage_checkpoint is not None:
            payload = torch_module.load(request.args.first_stage_checkpoint, map_location=request.device, weights_only=False)
            load_first_stage_checkpoint(first_stage, payload)
            if request.checkpoint_resolution["first_stage_checkpoint_source"] == "final_pt":
                print(f"using final_pt first-stage checkpoint: {request.args.first_stage_checkpoint}")
    if "diffusion_model" in model_names:
        LatentDiffusionUNet = _resolve_legacy_override(
            "LatentDiffusionUNet", "ldm_project.teacher.diffusion", "LatentDiffusionUNet"
        )

        diffusion_model = LatentDiffusionUNet(request.train_cfg.first_stage.latent_channels, request.train_cfg.diffusion).to(request.device)
        load_diffusion_checkpoint = _resolve_checkpoint_loader(
            "load_diffusion_checkpoint",
            "ldm_project.teacher.diffusion",
            "load_diffusion_checkpoint",
        )
    if "scheduler" in model_names:
        LinearNoiseScheduler = _resolve_legacy_override(
            "LinearNoiseScheduler",
            "ldm_project.teacher.diffusion",
            "LinearNoiseScheduler",
        )

        scheduler = LinearNoiseScheduler(
            timesteps=request.train_cfg.diffusion.timesteps,
            beta_start=request.train_cfg.diffusion.beta_start,
            beta_end=request.train_cfg.diffusion.beta_end,
        ).to(request.device)
    if "one_shot_model" in model_names:
        from .model_factory import build_student_model

        one_shot_model = build_student_model(request.physics, request.train_cfg.one_shot, image_size=request.train_cfg.first_stage.image_size, device=request.device)

    if "first_stage" in model_names:
        from ..teacher.first_stage_losses import first_stage_loss
    if "diffusion_model" in model_names:
        from ..teacher.diffusion_losses import diffusion_loss, diffusion_per_sample_losses
        from ..teacher.diffusion_sampling import (
            resolve_denoising_trace_steps,
            sample_latents,
            sample_latents_with_trace,
        )
    if "one_shot_model" in model_names:
        from ..student.checkpoints import load_one_shot_checkpoint
        from ..student.objectives import one_shot_loss

    dataset_splits = train_bundle = validation_bundle = None
    first_stage_visual_batch = first_stage_visual_labels = None
    first_stage_visual_source = None
    diffusion_visual_labels = diffusion_visual_reference_images = None
    diffusion_visual_reference_labels = diffusion_visual_source = None
    if footprint.constructs_data:
        build_train_validation_loaders = _resolve_legacy_override(
            "build_train_validation_loaders",
            "ldm_project.data",
            "build_train_validation_loaders",
        )

        dataset_splits = build_train_validation_loaders(
            data_config=request.train_cfg.data,
            first_stage_config=request.train_cfg.first_stage,
            batch_size=request.train_cfg.batch_size,
        )
        train_bundle, validation_bundle = dataset_splits.train, dataset_splits.validation
        if is_teacher_first_stage_mode(mode):
            from ..sampling_reports import select_first_stage_visual_batch

            first_stage_visual_batch, first_stage_visual_labels, first_stage_visual_source = select_first_stage_visual_batch(
                torch_module, validation_bundle, train_bundle, request.train_cfg.diffusion.num_classes, request.device, allow_train_reference_fallback=True
            )
        if is_teacher_diffusion_mode(mode):
            from ..sampling_reports import select_post_train_sample_batch

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
        first_stage_model_version=request.first_stage_version, one_shot_model_version=request.one_shot_version,
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
        versions={"first_stage": request.first_stage_version, "diffusion": request.diffusion_version, "one_shot": request.one_shot_version},
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
