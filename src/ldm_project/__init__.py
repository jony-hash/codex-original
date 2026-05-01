from .config import (
    DataConfig,
    DiffusionConfig,
    ElectronicDomainConfig,
    FirstStageConfig,
    MacroOpticsConfig,
    PhysicsConfig,
    TrainingConfig,
    load_physics_config,
    load_training_config,
)
from .run_manager import RunManager

__all__ = [
    "DataConfig",
    "DiffusionConfig",
    "ElectronicDomainConfig",
    "FirstStageConfig",
    "MacroOpticsConfig",
    "PhysicsConfig",
    "RunManager",
    "TrainingConfig",
    "load_physics_config",
    "load_training_config",
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
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.extend(
        [
            "ClassConditionEncoder",
            "DIFFUSION_MODEL_VERSION",
            "DatasetBundle",
            "DatasetSplits",
            "FIRST_STAGE_MODEL_VERSION",
            "LatentDiffusionUNet",
            "LinearNoiseScheduler",
            "OpticalPathVAE",
            "build_train_validation_loaders",
            "build_training_loader",
            "load_diffusion_checkpoint",
            "load_first_stage_checkpoint",
            "validate_first_stage_checkpoint_payload",
        ]
    )
