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
    # Keep the historic root API, but never initialize v1 merely because a
    # caller imports ``ldm_project.v2``.  The package root is shared by both
    # runtimes, so eager v1 compatibility exports are an otherwise invisible
    # v2 startup dependency.
    "DataConfig": (".v1.config", "DataConfig"),
    "DiffusionConfig": (".v1.config", "DiffusionConfig"),
    "DistillationConfig": (".v1.config", "DistillationConfig"),
    "ElectronicDomainConfig": (".v1.config", "ElectronicDomainConfig"),
    "FirstStageConfig": (".v1.config", "FirstStageConfig"),
    "MacroOpticsConfig": (".v1.config", "MacroOpticsConfig"),
    "OneShotConfig": (".v1.config", "OneShotConfig"),
    "PhysicsConfig": (".v1.config", "PhysicsConfig"),
    "RunManager": (".v1.application.run_manager", "RunManager"),
    "TrainingConfig": (".v1.config", "TrainingConfig"),
    "load_physics_config": (".v1.config", "load_physics_config"),
    "load_training_config": (".v1.config", "load_training_config"),
    "validate_physical_forward_contract": (".v1.config", "validate_physical_forward_contract"),
    "DatasetBundle": (".v1.data", "DatasetBundle"), "DatasetSplits": (".v1.data", "DatasetSplits"),
    "build_train_validation_loaders": (".v1.data", "build_train_validation_loaders"), "build_training_loader": (".v1.data", "build_training_loader"),
    "DIFFUSION_MODEL_VERSION": (".v1.teacher.diffusion", "DIFFUSION_MODEL_VERSION"), "ClassConditionEncoder": (".v1.teacher.diffusion", "ClassConditionEncoder"),
    "LatentDiffusionUNet": (".v1.teacher.diffusion", "LatentDiffusionUNet"), "LinearNoiseScheduler": (".v1.teacher.diffusion", "LinearNoiseScheduler"), "load_diffusion_checkpoint": (".v1.teacher.diffusion", "load_diffusion_checkpoint"),
    "FIRST_STAGE_MODEL_VERSION": (".v1.teacher.first_stage", "FIRST_STAGE_MODEL_VERSION"), "OpticalPathVAE": (".v1.teacher.first_stage", "OpticalPathVAE"),
    "load_first_stage_checkpoint": (".v1.teacher.first_stage", "load_first_stage_checkpoint"), "validate_first_stage_checkpoint_payload": (".v1.teacher.first_stage", "validate_first_stage_checkpoint_payload"),
    "ONE_SHOT_MODEL_VERSION": (".v1.one_shot", "ONE_SHOT_MODEL_VERSION"), "ElectronicGeneratorE": (".v1.one_shot", "ElectronicGeneratorE"),
    "OneShotOpticalGenerator": (".v1.one_shot", "OneShotOpticalGenerator"), "load_one_shot_checkpoint": (".v1.one_shot", "load_one_shot_checkpoint"), "validate_one_shot_checkpoint_payload": (".v1.one_shot", "validate_one_shot_checkpoint_payload"),
    "FormalOneShotOpticalGenerator": (".v1.student.model", "FormalOneShotOpticalGenerator"),
    "compute_first_stage_balanced_metrics": (".v1.evaluation.core", "compute_first_stage_balanced_metrics"), "evaluate_diffusion_with_per_class": (".v1.evaluation.core", "evaluate_diffusion_with_per_class"), "evaluate_first_stage_with_per_class": (".v1.evaluation.core", "evaluate_first_stage_with_per_class"),
    "DEFAULT_DENOISING_TRACE_TIMESTEPS": (".v1.sampling", "DEFAULT_DENOISING_TRACE_TIMESTEPS"), "DEFAULT_NEAREST_REFERENCE_LIMIT": (".v1.sampling", "DEFAULT_NEAREST_REFERENCE_LIMIT"),
    "build_post_train_sample_labels": (".v1.sampling_reports", "build_post_train_sample_labels"), "compute_sample_quality_metrics": (".v1.sampling_reports", "compute_sample_quality_metrics"), "save_post_train_samples": (".v1.sampling_reports", "save_post_train_samples"), "select_first_stage_visual_batch": (".v1.sampling_reports", "select_first_stage_visual_batch"), "select_post_train_sample_batch": (".v1.sampling_reports", "select_post_train_sample_batch"),
    "apply_cli_overrides": (".v1.train_utils", "apply_cli_overrides"), "canonicalize_stage": (".v1.train_utils", "canonicalize_stage"), "resolve_device": (".v1.train_utils", "resolve_device"), "resolve_stage_epoch_plan": (".v1.train_utils", "resolve_stage_epoch_plan"), "validate_stage_requirements": (".v1.train_utils", "validate_stage_requirements"),
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
