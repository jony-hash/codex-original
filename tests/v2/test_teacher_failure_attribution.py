from __future__ import annotations

import json

import pytest
import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research import teacher_failure_attribution as b2
from ldm_project.v2.application.research.electronic_distribution_recovery import _classification_metrics


CONFIG = "configs/research/teacher_failure_attribution_v1.yaml"


def _args(action: str) -> list[str]:
    values = [
        "research",
        "teacher-failure-attribution",
        action,
        "--config", CONFIG,
        "--dataset-root", "data",
        "--attribution-artifact", "attribution.json",
        "--primary-evaluator-checkpoint", "primary.pt",
        "--secondary-evaluator-checkpoint", "secondary.pt",
        "--primary-qualification-evidence", "primary.json",
        "--secondary-qualification-evidence", "secondary.json",
        "--device", "cuda",
        "--output", "evidence.json",
    ]
    if action == "audit":
        values.extend(["--validation-evidence", "validation.json"])
    elif action == "report":
        values.extend(["--audit-evidence", "audit.json"])
    elif action == "closeout":
        values.extend(["--report-evidence", "report.json"])
    return values


def test_b2_cli_is_registered_for_every_lifecycle_action() -> None:
    parser = _parser()
    for action in ("validate", "audit", "report", "closeout"):
        parsed = parser.parse_args(_args(action))
        assert parsed.research_program == "teacher-failure-attribution"
        assert parsed.research_action == action


def test_b2_classification_branches_are_fail_closed() -> None:
    assert b2._classify(
        real_positive_control_passed=False,
        label_shift_rejected=True,
        reconstruction_passed=True,
        diffusion_passed=True,
        package_b_replay_consistent=True,
    )[0] == "invalid_harness"
    assert b2._classify(
        real_positive_control_passed=True,
        label_shift_rejected=False,
        reconstruction_passed=True,
        diffusion_passed=True,
        package_b_replay_consistent=True,
    )[0] == "invalid_harness"
    assert b2._classify(
        real_positive_control_passed=True,
        label_shift_rejected=True,
        reconstruction_passed=False,
        diffusion_passed=False,
        package_b_replay_consistent=True,
    )[:2] == ("teacher_first_stage_distribution_failed", "first_stage_reconstruction_distribution_gate")
    assert b2._classify(
        real_positive_control_passed=True,
        label_shift_rejected=True,
        reconstruction_passed=True,
        diffusion_passed=False,
        package_b_replay_consistent=True,
    )[:2] == ("teacher_diffusion_distribution_failed", "diffusion_distribution_gate")
    assert b2._classify(
        real_positive_control_passed=True,
        label_shift_rejected=True,
        reconstruction_passed=True,
        diffusion_passed=True,
        package_b_replay_consistent=False,
    )[:2] == ("teacher_composition_harness_invalid", "package_b_replay_binding")


def test_b2_label_shift_negative_control_is_rejected() -> None:
    prediction = torch.arange(10).repeat(25)
    shifted = (prediction + 1) % 10
    metrics = _classification_metrics(prediction, shifted)
    assert metrics["accuracy"] == 0.0
    assert metrics["minimum_recall"] == 0.0
    assert not (
        metrics["accuracy"] >= 0.95 and metrics["minimum_recall"] >= 0.90
    )


def test_b2_artifact_schema_rejects_tamper_before_shard_loading(tmp_path) -> None:
    path = tmp_path / "attribution.json"
    path.write_text(json.dumps({"schema": "wrong", "status": "candidate"}), encoding="utf-8")
    with pytest.raises(ValueError, match="attribution artifact"):
        b2._validate_artifact(str(path), b2._config(CONFIG))


def test_b2_explicit_evaluator_paths_are_locked() -> None:
    config = b2._config(CONFIG)
    parsed = b2._parser().parse_args(
        [
            "validate",
            "--config", CONFIG,
            "--dataset-root", "data",
            "--attribution-artifact", "attribution.json",
            "--primary-evaluator-checkpoint", "wrong.pt",
            "--secondary-evaluator-checkpoint", config["secondary_evaluator"],
            "--primary-qualification-evidence", config["primary_qualification"],
            "--secondary-qualification-evidence", config["secondary_qualification"],
            "--output", "validation.json",
        ]
    )
    with pytest.raises(ValueError, match="primary_evaluator_checkpoint"):
        b2._args_match_config(parsed, config)
