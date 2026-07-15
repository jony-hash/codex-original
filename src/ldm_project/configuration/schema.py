"""Configuration dataclass ownership.

This module is intentionally a stable scoped import while the legacy loader
continues to provide the exact field defaults and dataclass constructors.
"""
from .legacy import (
    DataConfig,
    D2NNConfig,
    DiffusionConfig,
    DistillationConfig,
    ElectronicDomainConfig,
    FirstStageConfig,
    GeometryConfig,
    MacroOpticsConfig,
    OneShotConfig,
    OpticsConfig,
    PhysicsConfig,
    ReadoutConfig,
    TrainingConfig,
)

__all__ = [
    "DataConfig",
    "D2NNConfig",
    "DiffusionConfig",
    "DistillationConfig",
    "ElectronicDomainConfig",
    "FirstStageConfig",
    "GeometryConfig",
    "MacroOpticsConfig",
    "OneShotConfig",
    "OpticsConfig",
    "PhysicsConfig",
    "ReadoutConfig",
    "TrainingConfig",
]
