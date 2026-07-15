"""Stage-scoped runtime contracts.

The runtime classification is deliberately independent of model classes.  It gives the
CLI and tests an auditable dependency footprint before the compatibility StageContext
adapter materializes legacy handler payloads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


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


__all__ = [
    "DataOnlyRuntime",
    "RuntimeFootprint",
    "StageRuntime",
    "StudentRuntime",
    "TeacherRuntime",
    "build_stage_runtime",
]

