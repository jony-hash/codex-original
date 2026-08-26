from __future__ import annotations

import json

import pytest

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research.electronic_distribution_recovery import _gfa_gate
from ldm_project.v2.application.research.teacher_distribution_admission import (
    _config,
    _validate_manifest,
)


CONFIG = "configs/research/teacher_distribution_admission_v1.yaml"


def _args() -> list[str]:
    return [
        "research",
        "teacher-distribution-admission",
        "validate",
        "--config", CONFIG,
        "--dataset-root", "data",
        "--teacher-bank-manifest", "teacher_banks.json",
        "--primary-evaluator-checkpoint", "primary.pt",
        "--secondary-evaluator-checkpoint", "secondary.pt",
        "--primary-qualification-evidence", "primary.json",
        "--secondary-qualification-evidence", "secondary.json",
        "--package-a-gate-evidence", "gate.json",
        "--package-a-closeout-evidence", "closeout.json",
        "--device", "cuda",
        "--output", "validation.json",
    ]


def test_package_b_cli_is_registered() -> None:
    parser = _parser()
    for action in ("validate", "audit", "report", "closeout"):
        argv = _args()
        argv[2] = action
        if action == "audit":
            argv.extend(["--validation-evidence", "validation.json"])
        elif action == "report":
            argv.extend(["--audit-evidence", "audit.json"])
        elif action == "closeout":
            argv.extend(["--report-evidence", "report.json"])
        parsed = parser.parse_args(argv)
        assert parsed.research_program == "teacher-distribution-admission"
        assert parsed.research_action == action


def test_package_b_manifest_rejects_non_audit_artifact(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": "wrong", "status": "candidate"}), encoding="utf-8")
    with pytest.raises(ValueError, match="audit artifact"):
        _validate_manifest(str(path), _config(CONFIG))


def _passing_metrics() -> dict[str, object]:
    evaluator = {
        "accuracy": 1.0,
        "minimum_recall": 1.0,
        "diversity": [1.0] * 10,
        "coverage": 1.0,
        "memorization": {"per_class": [{"anomalous_rate": 0.0} for _ in range(10)]},
    }
    return {"primary": evaluator, "secondary": evaluator, "agreement": 1.0}


def test_package_b_gate_accepts_only_complete_dual_evaluator_metrics() -> None:
    config = _config(CONFIG)
    assert _gfa_gate(_passing_metrics(), config)
    failed = _passing_metrics()
    failed["secondary"]["minimum_recall"] = 0.899
    assert not _gfa_gate(failed, config)
