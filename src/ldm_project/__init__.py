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

try:
    from .data import (
        DatasetBundle as DatasetBundle,
        DatasetSplits as DatasetSplits,
        build_train_validation_loaders as build_train_validation_loaders,
        build_training_loader as build_training_loader,
    )
    from .diffusion import (
        DIFFUSION_MODEL_VERSION as DIFFUSION_MODEL_VERSION,
        ClassConditionEncoder as ClassConditionEncoder,
        LatentDiffusionUNet as LatentDiffusionUNet,
        LinearNoiseScheduler as LinearNoiseScheduler,
        load_diffusion_checkpoint as load_diffusion_checkpoint,
    )
    from .first_stage import (
        FIRST_STAGE_MODEL_VERSION as FIRST_STAGE_MODEL_VERSION,
        OpticalPathVAE as OpticalPathVAE,
        load_first_stage_checkpoint as load_first_stage_checkpoint,
        validate_first_stage_checkpoint_payload as validate_first_stage_checkpoint_payload,
    )
    from .one_shot import (
        ONE_SHOT_MODEL_VERSION as ONE_SHOT_MODEL_VERSION,
        ElectronicGeneratorE as ElectronicGeneratorE,
        OneShotOpticalGenerator as OneShotOpticalGenerator,
        load_one_shot_checkpoint as load_one_shot_checkpoint,
        validate_one_shot_checkpoint_payload as validate_one_shot_checkpoint_payload,
    )
    from .student.model import FormalOneShotOpticalGenerator as FormalOneShotOpticalGenerator
    from .evaluation import (
        compute_first_stage_balanced_metrics as compute_first_stage_balanced_metrics,
        evaluate_diffusion_with_per_class as evaluate_diffusion_with_per_class,
        evaluate_first_stage_with_per_class as evaluate_first_stage_with_per_class,
    )
    from .sampling import (
        DEFAULT_DENOISING_TRACE_TIMESTEPS as DEFAULT_DENOISING_TRACE_TIMESTEPS,
        DEFAULT_NEAREST_REFERENCE_LIMIT as DEFAULT_NEAREST_REFERENCE_LIMIT,
        build_post_train_sample_labels as build_post_train_sample_labels,
        compute_sample_quality_metrics as compute_sample_quality_metrics,
        save_post_train_samples as save_post_train_samples,
        select_first_stage_visual_batch as select_first_stage_visual_batch,
        select_post_train_sample_batch as select_post_train_sample_batch,
    )
    from .train_utils import (
        apply_cli_overrides as apply_cli_overrides,
        canonicalize_stage as canonicalize_stage,
        resolve_device as resolve_device,
        resolve_stage_epoch_plan as resolve_stage_epoch_plan,
        validate_stage_requirements as validate_stage_requirements,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.extend(
        [
            "ClassConditionEncoder",
            "DEFAULT_DENOISING_TRACE_TIMESTEPS",
            "DEFAULT_NEAREST_REFERENCE_LIMIT",
            "DIFFUSION_MODEL_VERSION",
            "DatasetBundle",
            "DatasetSplits",
            "ElectronicGeneratorE",
            "FormalOneShotOpticalGenerator",
            "FIRST_STAGE_MODEL_VERSION",
            "LatentDiffusionUNet",
            "LinearNoiseScheduler",
            "ONE_SHOT_MODEL_VERSION",
            "OpticalPathVAE",
            "OneShotOpticalGenerator",
            "apply_cli_overrides",
            "build_post_train_sample_labels",
            "build_train_validation_loaders",
            "build_training_loader",
            "canonicalize_stage",
            "compute_first_stage_balanced_metrics",
            "compute_sample_quality_metrics",
            "evaluate_diffusion_with_per_class",
            "evaluate_first_stage_with_per_class",
            "load_diffusion_checkpoint",
            "load_first_stage_checkpoint",
            "load_one_shot_checkpoint",
            "resolve_device",
            "resolve_stage_epoch_plan",
            "save_post_train_samples",
            "select_first_stage_visual_batch",
            "select_post_train_sample_batch",
            "validate_stage_requirements",
            "validate_first_stage_checkpoint_payload",
            "validate_one_shot_checkpoint_payload",
        ]
    )
