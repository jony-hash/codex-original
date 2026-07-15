from __future__ import annotations

import ast
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from ldm_project.application.stages import StageRequest, _build_stage_context
from ldm_project.config import PhysicsConfig, TrainingConfig
from ldm_project.student.model import FormalOneShotOpticalGenerator


ROOT = Path(__file__).resolve().parents[1]


class _RunManager:
    run_dir = Path("/tmp/architecture-boundary-run")

    def save_config_snapshot(self, *_args, **_kwargs):
        return self.run_dir / "config.json"

    def write_meta(self, _meta):
        return None


def _request(mode: str) -> StageRequest:
    args = Namespace(
        physics_config="configs/physics_cfg.yaml",
        train_config="configs/train_ldm.yaml",
        final_pt_dir="outputs/final_pt",
        first_stage_checkpoint=None,
        diffusion_checkpoint=None,
        one_shot_checkpoint=None,
        one_shot_generator_checkpoint=None,
        one_shot_optical_checkpoint=None,
        distill_dataset="distill.pt",
        distill_validation_dataset=None,
        output_root="outputs/vtest/runs",
        output_version="vtest",
        output_root_explicit=False,
    )
    return StageRequest(
        torch_module=__import__("torch"),
        args=args,
        checkpoint_resolution={"first_stage_checkpoint_source": None, "diffusion_checkpoint_source": None},
        train_cfg=TrainingConfig(mode=mode),
        run_manager=_RunManager(),
        physics=PhysicsConfig(),
        device=__import__("torch").device("cpu"),
        use_pin_memory=False,
        first_stage_version="test-first-stage",
        diffusion_version="test-diffusion",
        one_shot_version="test-student",
    )


def _imports_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def test_formal_domain_import_boundaries() -> None:
    for path in (ROOT / "src" / "ldm_project" / "optics").glob("*.py"):
        imports = _imports_in(path)
        assert not any(name.startswith(("student", "teacher", "application", "diagnostics")) for name in imports), path
    for path in (ROOT / "src" / "ldm_project" / "teacher").glob("*.py"):
        imports = _imports_in(path)
        assert not any(name.startswith("student") for name in imports), path
    for path in (ROOT / "src" / "ldm_project" / "student").glob("*.py"):
        imports = _imports_in(path)
        assert not any(name.startswith("teacher") for name in imports), path


def test_student_layer_shim_contains_no_implementation() -> None:
    path = ROOT / "src" / "ldm_project" / "student" / "layers.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert not any(isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body)


def test_generator_and_legacy_imports_share_identity() -> None:
    from ldm_project.one_shot import ElectronicGeneratorE as root_generator
    from ldm_project.student.generator import ElectronicGeneratorE
    from ldm_project.student.layers import ElectronicGeneratorE as shim_generator

    assert root_generator is ElectronicGeneratorE
    assert shim_generator is ElectronicGeneratorE


def test_neutral_domains_do_not_depend_on_execution_domains() -> None:
    forbidden = ("student", "teacher", "application", "diagnostics")
    for domain in ("distillation", "generation", "evaluation"):
        for path in (ROOT / "src" / "ldm_project" / domain).glob("*.py"):
            imports = _imports_in(path)
            assert not any(name.startswith(forbidden) for name in imports), path


def test_teacher_workflows_do_not_depend_on_common_facade() -> None:
    for name in ("export.py", "label_audit.py", "filter.py", "readable.py"):
        imports = _imports_in(ROOT / "src" / "ldm_project" / "teacher" / name)
        assert "common" not in imports, name
    common_tree = ast.parse(
        (ROOT / "src" / "ldm_project" / "teacher" / "common.py").read_text(
            encoding="utf-8"
        )
    )
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in common_tree.body
    )


def test_objective_terms_do_not_depend_on_objective_facade() -> None:
    terms_root = ROOT / "src" / "ldm_project" / "student" / "objective_terms"
    for path in terms_root.glob("*.py"):
        assert "objectives" not in _imports_in(path), path


def test_filter_stage_constructs_no_model_or_dataset() -> None:
    context = _build_stage_context(_request("teacher_distill_filter"))
    assert context.models.first_stage is None
    assert context.models.diffusion_model is None
    assert context.models.one_shot_model is None
    assert context.models.scheduler is None
    assert context.data.train_bundle is None


def test_student_distillation_does_not_construct_teacher_models() -> None:
    class _Student:
        def to(self, _device):
            return self

    with (
        patch("ldm_project.student.model.OneShotOpticalGenerator", return_value=_Student()),
        patch("ldm_project.teacher.first_stage.OpticalPathVAE", side_effect=AssertionError("teacher model constructed")),
        patch("ldm_project.teacher.diffusion.LatentDiffusionUNet", side_effect=AssertionError("teacher model constructed")),
    ):
        context = _build_stage_context(_request("one_shot_distill"))
    assert isinstance(context.models.one_shot_model, _Student)
    assert context.models.first_stage is None
    assert context.models.diffusion_model is None
    assert context.models.scheduler is None


def test_stage_modules_do_not_eagerly_import_execution_models() -> None:
    import subprocess
    import sys

    script = """
import sys
import ldm_project.application.stages
assert 'ldm_project.diffusion' not in sys.modules
assert 'ldm_project.first_stage' not in sys.modules
assert 'ldm_project.student.model' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == ""


def test_student_has_no_diagnostics_imports() -> None:
    for path in (ROOT / "src" / "ldm_project" / "student").rglob("*.py"):
        imports = _imports_in(path)
        assert not any(name.startswith("diagnostics") for name in imports), path


def test_teacher_scoped_modules_use_public_support_symbols() -> None:
    for name in ("export.py", "label_audit.py", "filter.py", "readable.py", "fault_localization.py"):
        path = ROOT / "src" / "ldm_project" / "teacher" / name
        text = path.read_text(encoding="utf-8")
        assert "from .distill import" not in text, path
        assert "from .common import" not in text, path


def test_formal_student_rejects_diagnostic_switches() -> None:
    physics = PhysicsConfig(
        front_path_model="coherent_imaging_4f",
        back_path_model="free_space",
        physical_forward_contract=True,
    )
    config = TrainingConfig().one_shot
    config.decoder_diagnostic_mode = "bypass_decoder"
    try:
        FormalOneShotOpticalGenerator(physics, config, image_size=200)
    except ValueError as exc:
        assert "decoder_diagnostic_mode" in str(exc)
    else:
        raise AssertionError("formal student accepted a decoder diagnostic switch")
