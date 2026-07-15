from .config import (
    DataConfig,
    DiffusionConfig,
    DistillationConfig,
    ElectronicDomainConfig,
    FirstStageConfig,
    MacroOpticsConfig,
    OneShotConfig,
    PhysicsConfig,
    TrainingConfig,
    load_physics_config,
    load_training_config,
    validate_physical_forward_contract,
)
from .run_manager import RunManager
from importlib import import_module

__all__ = [
    "DataConfig",
    "DiffusionConfig",
    "DistillationConfig",
    "ElectronicDomainConfig",
    "FirstStageConfig",
    "MacroOpticsConfig",
    "OneShotConfig",
    "PhysicsConfig",
    "RunManager",
    "TrainingConfig",
    "load_physics_config",
    "load_training_config",
    "validate_physical_forward_contract",
]

_LAZY_EXPORTS = {
    "DatasetBundle": (".data", "DatasetBundle"),
    "DatasetSplits": (".data", "DatasetSplits"),
    "build_train_validation_loaders": (".data", "build_train_validation_loaders"),
    "build_training_loader": (".data", "build_training_loader"),
    "DIFFUSION_MODEL_VERSION": (".diffusion", "DIFFUSION_MODEL_VERSION"),
    "ClassConditionEncoder": (".diffusion", "ClassConditionEncoder"),
    "LatentDiffusionUNet": (".diffusion", "LatentDiffusionUNet"),
    "LinearNoiseScheduler": (".diffusion", "LinearNoiseScheduler"),
    "load_diffusion_checkpoint": (".diffusion", "load_diffusion_checkpoint"),
    "FIRST_STAGE_MODEL_VERSION": (".first_stage", "FIRST_STAGE_MODEL_VERSION"),
    "OpticalPathVAE": (".first_stage", "OpticalPathVAE"),
    "load_first_stage_checkpoint": (".first_stage", "load_first_stage_checkpoint"),
    "validate_first_stage_checkpoint_payload": (".first_stage", "validate_first_stage_checkpoint_payload"),
    "ONE_SHOT_MODEL_VERSION": (".one_shot", "ONE_SHOT_MODEL_VERSION"),
    "ElectronicGeneratorE": (".one_shot", "ElectronicGeneratorE"),
    "OneShotOpticalGenerator": (".one_shot", "OneShotOpticalGenerator"),
    "load_one_shot_checkpoint": (".one_shot", "load_one_shot_checkpoint"),
    "validate_one_shot_checkpoint_payload": (".one_shot", "validate_one_shot_checkpoint_payload"),
    "FormalOneShotOpticalGenerator": (".student.model", "FormalOneShotOpticalGenerator"),
    "compute_first_stage_balanced_metrics": (".evaluation.core", "compute_first_stage_balanced_metrics"),
    "evaluate_diffusion_with_per_class": (".evaluation.core", "evaluate_diffusion_with_per_class"),
    "evaluate_first_stage_with_per_class": (".evaluation.core", "evaluate_first_stage_with_per_class"),
    "DEFAULT_DENOISING_TRACE_TIMESTEPS": (".sampling", "DEFAULT_DENOISING_TRACE_TIMESTEPS"),
    "DEFAULT_NEAREST_REFERENCE_LIMIT": (".sampling", "DEFAULT_NEAREST_REFERENCE_LIMIT"),
    "build_post_train_sample_labels": (".sampling_reports", "build_post_train_sample_labels"),
    "compute_sample_quality_metrics": (".sampling_reports", "compute_sample_quality_metrics"),
    "save_post_train_samples": (".sampling_reports", "save_post_train_samples"),
    "select_first_stage_visual_batch": (".sampling_reports", "select_first_stage_visual_batch"),
    "select_post_train_sample_batch": (".sampling_reports", "select_post_train_sample_batch"),
    "apply_cli_overrides": (".train_utils", "apply_cli_overrides"),
    "canonicalize_stage": (".train_utils", "canonicalize_stage"),
    "resolve_device": (".train_utils", "resolve_device"),
    "resolve_stage_epoch_plan": (".train_utils", "resolve_stage_epoch_plan"),
    "validate_stage_requirements": (".train_utils", "validate_stage_requirements"),
}

__all__.extend(_LAZY_EXPORTS)


def __getattr__(name: str):
    """Load legacy root exports only when a caller actually requests them."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
