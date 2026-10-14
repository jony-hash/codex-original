from __future__ import annotations

import ast
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from ldm_project.application.stages import StageRequest, _build_stage_context
from ldm_project.config import PhysicsConfig, TrainingConfig
from ldm_project.student.model import FormalOneShotOpticalGenerator
from ldm_project.application.model_factory import build_student_model
from ldm_project.configuration.schema import D2NNConfig, OneShotConfig


ROOT = Path(__file__).resolve().parents[1]


class _RunManager:
    run_dir = Path("/tmp/architecture-boundary-run")

    def save_config_snapshot(self, *_args, **_kwargs):
        return self.run_dir / "config.json"

    def write_meta(self, _meta):
        return None


def _request(mode: str) -> StageRequest:
    args = Namespace(
        physics_config="configs/v1/formal/physics_cfg.yaml",
        train_config="configs/v1/formal/train_ldm.yaml",
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


def _python_files_under(root: Path) -> tuple[Path, ...]:
    assert root.is_dir(), f"canonical implementation directory is missing: {root}"
    paths = tuple(sorted(root.rglob("*.py")))
    assert paths, f"canonical implementation directory is empty: {root}"
    return paths


def _package_name(path: Path) -> str:
    parts = path.resolve().parts
    source_index = max(index for index, part in enumerate(parts) if part == "src")
    relative = Path(*parts[source_index + 1 :]).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    package = _package_name(path)
    current_package = package if path.name == "__init__.py" else package.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = current_package.split(".")
                base = parts[: len(parts) - (node.level - 1)]
                module = ".".join((*base, *(node.module.split(".") if node.module else ())))
            else:
                module = node.module or ""
            if module:
                imported.add(module)
            # `from package import child` can be a submodule import. Preserve
            # the full candidate so package-boundary rules see it as well.
            imported.update(f"{module}.{alias.name}" for alias in node.names if module and alias.name != "*")
    return imported


def _matches(imported: set[str], forbidden: tuple[str, ...]) -> bool:
    return any(
        name == prefix or name.startswith(f"{prefix}.")
        for name in imported
        for prefix in forbidden
    )


def _canonical_runtime_files(runtime: str) -> tuple[Path, ...]:
    return _python_files_under(ROOT / "src" / "ldm_project" / runtime)


def _assert_import_boundary(paths: tuple[Path, ...], forbidden: tuple[str, ...]) -> None:
    assert paths, "architecture scan must not be empty"
    for path in paths:
        assert not _matches(_imports_in(path), forbidden), path


def test_formal_domain_import_boundaries() -> None:
    v1_root = ROOT / "src" / "ldm_project" / "v1"
    for runtime in ("v1", "v2"):
        optics = ROOT / "src" / "ldm_project" / runtime / "optics"
        _assert_import_boundary(_python_files_under(optics), (
            "ldm_project.v1.student", "ldm_project.v1.teacher", "ldm_project.v1.application", "ldm_project.v1.diagnostics",
            "ldm_project.v2.student", "ldm_project.v2.application",
            "ldm_project.student", "ldm_project.teacher", "ldm_project.application", "ldm_project.diagnostics",
        ))
    _assert_import_boundary(_python_files_under(v1_root / "teacher"), ("ldm_project.v1.student", "ldm_project.student"))
    _assert_import_boundary(_python_files_under(v1_root / "student"), ("ldm_project.v1.teacher", "ldm_project.teacher"))


def test_canonical_runtime_scan_is_nonempty_and_v2_stays_teacher_free() -> None:
    v1_files = _canonical_runtime_files("v1")
    v2_files = _canonical_runtime_files("v2")
    assert v1_files and v2_files
    _assert_import_boundary(v1_files, ("ldm_project.v2",))
    _assert_import_boundary(v2_files, ("ldm_project.v1.teacher", "ldm_project.teacher", "ldm_project.first_stage", "ldm_project.diffusion"))

    current_v2_boundaries = _python_files_under(ROOT / "src/ldm_project/v2/application/current") + (
        ROOT / "src/ldm_project/v2/student/identity_generator.py",
        ROOT / "src/ldm_project/v2/student/spatial_quadrature_generator.py",
        ROOT / "src/ldm_project/v2/student/checkpoints.py",
        ROOT / "src/ldm_project/v2/student/conditional.py",
        ROOT / "src/ldm_project/v2/student/evaluator.py",
    )
    forbidden_current_v2 = (
        "ldm_project.v2.teacher",
        "ldm_project.v2.application.diagnostic",
        "ldm_project.v2.application.legacy.teacher",
        "ldm_project.v2.application.research",
        "ldm_project.v2.student.direct_template",
        "ldm_project.v2.student.capacity",
        "ldm_project.v2.student.conditional_generator",
        "ldm_project.v2.student.generator.LabelPrototypeBasisGenerator",
    )
    for path in current_v2_boundaries:
        assert path.is_file(), f"current v2 boundary is missing: {path}"
    _assert_import_boundary(current_v2_boundaries, forbidden_current_v2)


def test_import_resolver_handles_absolute_relative_and_empty_scans(tmp_path: Path) -> None:
    package = tmp_path / "src" / "ldm_project" / "v2" / "application"
    package.mkdir(parents=True)
    source = package / "sample.py"
    source.write_text(
        "from ldm_project.v1.teacher import targets\nfrom ..student import capacity\n",
        encoding="utf-8",
    )
    imported = _imports_in(source)
    assert "ldm_project.v1.teacher.targets" in imported
    assert "ldm_project.v2.student.capacity" in imported
    with pytest.raises(AssertionError):
        _assert_import_boundary((source,), ("ldm_project.v1.teacher",))
    with pytest.raises(AssertionError):
        _assert_import_boundary((source,), ("ldm_project.v2.student.capacity",))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AssertionError, match="empty"):
        _python_files_under(empty)
    with pytest.raises(AssertionError, match="missing"):
        _python_files_under(tmp_path / "missing")
    with pytest.raises(AssertionError, match="empty"):
        _assert_import_boundary((), ("ldm_project.v1.teacher",))


def test_student_layer_shim_contains_no_implementation() -> None:
    path = ROOT / "src" / "ldm_project" / "v1" / "student" / "layers.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert not any(isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body)


def test_generator_and_legacy_imports_share_identity() -> None:
    from ldm_project.one_shot import ElectronicGeneratorE as root_generator
    from ldm_project.student.generator import ElectronicGeneratorE
    from ldm_project.student.layers import ElectronicGeneratorE as shim_generator

    assert root_generator is ElectronicGeneratorE
    assert shim_generator is ElectronicGeneratorE


def test_neutral_domains_do_not_depend_on_execution_domains() -> None:
    forbidden = tuple(f"ldm_project.{prefix}{domain}" for prefix in ("", "v1.", "v2.") for domain in ("student", "teacher", "application", "diagnostics"))
    for domain in ("distillation", "generation", "evaluation"):
        _assert_import_boundary(_python_files_under(ROOT / "src" / "ldm_project" / "v1" / domain), forbidden)


def test_teacher_workflows_do_not_depend_on_common_facade() -> None:
    for name in ("export.py", "label_audit.py", "filter.py", "readable.py"):
        imports = _imports_in(ROOT / "src" / "ldm_project" / "v1" / "teacher" / name)
        assert not _matches(imports, ("ldm_project.v1.teacher.common", "ldm_project.teacher.common")), name
    common_tree = ast.parse(
        (ROOT / "src" / "ldm_project" / "v1" / "teacher" / "common.py").read_text(
            encoding="utf-8"
        )
    )
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in common_tree.body
    )


def test_objective_terms_do_not_depend_on_objective_facade() -> None:
    terms_root = ROOT / "src" / "ldm_project" / "v1" / "student" / "objective_terms"
    _assert_import_boundary(_python_files_under(terms_root), ("ldm_project.v1.student.objectives", "ldm_project.student.objectives"))


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
    _assert_import_boundary(_python_files_under(ROOT / "src" / "ldm_project" / "v1" / "student"), ("ldm_project.v1.diagnostics", "ldm_project.diagnostics"))


def test_teacher_scoped_modules_use_public_support_symbols() -> None:
    for name in ("export.py", "label_audit.py", "filter.py", "readable.py", "fault_localization.py"):
        path = ROOT / "src" / "ldm_project" / "v1" / "teacher" / name
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


def test_transparent_physical_stage_routes_to_diagnostic_student() -> None:
    physics = PhysicsConfig(physical_forward_contract=True)
    config = OneShotConfig(
        physical_forward_contract=True,
        generator_e_arch="label_template",
        generator_command_transform="intensity_to_amplitude",
        generator_template_init="mnist_prototype",
        distill_target_source="mnist_prototype",
        d2nn=D2NNConfig(
            grid_size=4,
            propagation_grid_size=8,
            pixel_size=2.0e-6,
            layer_spacing=1.0e-5,
            exterior_boundary="transparent_identity",
            transparent_propagation_method="padded_band_limited_asm",
            transparent_fft_grid_size=16,
            transparent_memory_mode="checkpointed_active_capture",
        ),
    )
    model = build_student_model(physics, config, image_size=8, device="cpu")
    assert not isinstance(model, FormalOneShotOpticalGenerator)
    assert model.config.d2nn.exterior_boundary == "transparent_identity"
