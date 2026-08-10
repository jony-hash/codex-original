"""Behavioral regressions for the v2 lifecycle-boundary refactor."""

from __future__ import annotations

from pathlib import Path

import pytest

from ldm_project.v2.application import cli
from ldm_project.v2.application.legacy import cli_impl
from ldm_project.v2.config import load_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "v2"
CONFIG_CASES = tuple(
    path.relative_to(CONFIG_ROOT).as_posix()
    for path in sorted(CONFIG_ROOT.rglob("*.yaml"))
    if path.name != "path_migration_manifest.yaml"
)


@pytest.mark.parametrize("relative", CONFIG_CASES, ids=CONFIG_CASES)
def test_active_v2_configurations_load_with_explicit_architecture(relative: str) -> None:
    config = load_config(CONFIG_ROOT / relative)
    assert config.schema_version == 2
    if relative.startswith("current/"):
        assert config.architecture == "conditional_identity_logit_amplitude_v2"


@pytest.mark.parametrize(
    "argv",
    [
        ["student", "train", "--config", "identity.yaml"],
        [
            "student",
            "calibrate-prototypes",
            "--config",
            "legacy.yaml",
            "--clean-target-manifest",
            "clean.json",
        ],
        [
            "audit",
            "conditional-pca-transport-admission",
            "--config",
            "diagnostic.yaml",
            "--dataset-root",
            "data",
            "--evaluator-checkpoint",
            "evaluator.pt",
            "--semantic-checkpoint",
            "semantic.pt",
            "--safe-scale-evidence",
            "scale.json",
        ],
    ],
    ids=["current-train", "legacy-calibration", "diagnostic-admission"],
)
def test_legacy_cli_module_delegates_to_the_canonical_parser(argv: list[str]) -> None:
    assert vars(cli_impl._parser().parse_args(argv)) == vars(cli._parser().parse_args(argv))


def test_moved_diagnostic_import_paths_resolve_to_the_new_implementations() -> None:
    from ldm_project.v2.application.conditional_pca_transport import _metrics
    from ldm_project.v2.application.diagnostic.pca_transport import _metrics as moved

    assert _metrics is moved


def test_current_and_diagnostic_recovery_share_the_same_workflow_implementation() -> None:
    from ldm_project.v2.application.current import recovery as current
    from ldm_project.v2.application.diagnostic import recovery as diagnostic

    for name in ("_conditional_optimizer", "controllability_smoke", "semantic_recovery_smoke"):
        current_workflow = getattr(current, name)
        diagnostic_workflow = getattr(diagnostic, name)
        assert current_workflow.__module__.endswith("_conditional_recovery_shared")
        assert diagnostic_workflow.__module__ == current_workflow.__module__
        assert diagnostic_workflow.__code__ is current_workflow.__code__
