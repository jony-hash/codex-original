from __future__ import annotations

import ast
from pathlib import Path

from ldm_project.application.stages import STAGE_HANDLER_REGISTRY, STAGE_REGISTRY
from ldm_project.configuration.loading import load_training_config
from ldm_project.configuration.schema import TrainingConfig
from ldm_project.generation import project_codes_to_teacher_noise
from ldm_project.teacher import distill
from ldm_project.teacher.export import run_teacher_distill_export


ROOT = Path(__file__).resolve().parents[1]


def test_config_facade_and_scoped_schema_preserve_identity() -> None:
    from ldm_project.config import TrainingConfig as FacadeTrainingConfig

    assert FacadeTrainingConfig is TrainingConfig
    assert load_training_config is not None


def test_stage_registry_covers_every_declared_stage() -> None:
    assert set(STAGE_REGISTRY) <= set(STAGE_HANDLER_REGISTRY)


def test_teacher_distill_facade_preserves_public_callable_identity() -> None:
    assert distill.run_teacher_distill_export is run_teacher_distill_export
    assert distill._project_codes_to_teacher_noise is project_codes_to_teacher_noise


def test_legacy_layers_are_reexports_only() -> None:
    path = ROOT / "src" / "ldm_project" / "student" / "layers.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        for node in tree.body
    )
