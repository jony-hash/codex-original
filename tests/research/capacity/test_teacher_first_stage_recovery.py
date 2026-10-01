from __future__ import annotations

import json
from pathlib import Path

import pytest

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research.capacity import teacher_first_stage_recovery as b3


CONFIG = "configs/research/capacity/teacher_first_stage_recovery_v1.yaml"


def _args(action: str) -> list[str]:
    values = [
        "research", "teacher-first-stage-recovery", action,
        "--config", CONFIG,
        "--dataset-root", "data",
        "--reconstruction-manifest", "reconstruction.json",
        "--primary-evaluator-checkpoint", "primary.pt",
        "--secondary-evaluator-checkpoint", "secondary.pt",
        "--primary-qualification-evidence", "primary.json",
        "--secondary-qualification-evidence", "secondary.json",
        "--device", "cuda", "--output", "evidence.json",
    ]
    if action == "audit":
        values += ["--validation-evidence", "validation.json"]
    elif action == "report":
        values += ["--audit-evidence", "audit.json"]
    elif action == "closeout":
        values += ["--report-evidence", "report.json"]
    return values


def test_b3_cli_is_registered_for_every_lifecycle_action() -> None:
    parser = _parser()
    for action in ("validate", "audit", "report", "closeout"):
        parsed = parser.parse_args(_args(action))
        assert parsed.research_program == "teacher-first-stage-recovery"
        assert parsed.research_action == action


def test_b3_classification_is_fail_closed() -> None:
    assert b3._classify(all_banks_passed=True, negative_control_rejected=False)[0] == "invalid_harness"
    assert b3._classify(all_banks_passed=False, negative_control_rejected=True)[:2] == (
        "first_stage_reconstruction_failed", "first_stage_reconstruction_distribution_gate"
    )
    assert b3._classify(all_banks_passed=True, negative_control_rejected=True)[0] == "first_stage_reconstruction_qualified"


def test_b3_manifest_rejects_stale_or_wrong_schema_before_shards(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "wrong", "status": "candidate"}), encoding="utf-8")
    config = b3._config(CONFIG)
    with pytest.raises(ValueError, match="reconstruction manifest"):
        b3._validate_upstream(config, str(manifest), "data")


def test_b3_explicit_evaluator_paths_are_locked() -> None:
    config = b3._config(CONFIG)
    parsed = b3._parser().parse_args(_args("validate")[2:])
    parsed.primary_evaluator_checkpoint = "wrong.pt"
    with pytest.raises(ValueError, match="primary_evaluator_checkpoint"):
        b3._args_match_config(parsed, config)


def test_b3_consumer_source_does_not_import_v1_producer() -> None:
    source = Path(b3.__file__).read_text(encoding="utf-8")
    assert "ldm_project.v1" not in source
    assert "package_b3_vae" not in source
