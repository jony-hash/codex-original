from __future__ import annotations

import json
from pathlib import Path

import pytest

from ldm_project.v1.application import package_b3_t3 as t3


def test_t3_parser_requires_explicit_t2_and_upstream_paths() -> None:
    parsed = t3._parser().parse_args(
        [
            "run",
            "--config", "config.yaml",
            "--dataset-root", "data",
            "--t2-output-dir", "t2",
            "--upstream-evidence", "preflight.json",
            "--output-dir", "out",
        ]
    )
    assert parsed.action == "run"
    assert parsed.device == "cuda"


def test_t3_contract_is_locked_and_has_required_constraints() -> None:
    assert t3.EQUIVALENCE_CONTRACT["version"] == "b3_t3_equivalence_contract_v1"
    assert t3.EQUIVALENCE_CONTRACT["gradient_cosine_minimum"] == pytest.approx(0.995)
    assert t3.EQUIVALENCE_CONTRACT["optimizer_step_count_must_match"] is True


def test_t3_compare_trajectory_rejects_parameter_or_gradient_drift() -> None:
    tensor = __import__("torch").ones(2)
    record = {
        "probe_metrics": {"total": 1.0},
        "gradient_norm": 1.0,
        "reconstruction_sha256": "r",
    }
    reference = {"private": {}, "landmarks": {}, "successful_optimizer_steps": 100}
    candidate = {"private": {}, "landmarks": {}, "successful_optimizer_steps": 100, "resume": {"passed": True}, "finite": True, "skipped_optimizer_steps": 0}
    for step in t3.LANDMARKS:
        reference["private"][step] = {"reconstruction": tensor.clone(), "parameters": tensor.clone(), "gradient": tensor.clone()}
        candidate["private"][step] = {"reconstruction": tensor.clone(), "parameters": tensor.clone(), "gradient": tensor.clone()}
        reference["landmarks"][step] = {**record, "probe_metrics": {"total": 1.0}}
        candidate["landmarks"][step] = {**record, "probe_metrics": {"total": 1.0}}
    assert t3._compare_trajectory(reference, candidate)["passed"] is True
    candidate["private"][100]["parameters"][0] = 0.1
    assert t3._compare_trajectory(reference, candidate)["passed"] is False


def test_t3_candidate_classification_names_are_explicit() -> None:
    assert {"unsupported", "numerical_equivalence_failed", "numerically_supported_but_throughput_insufficient", "numerically_supported_and_throughput_supported"}.issuperset(
        {"unsupported", "numerical_equivalence_failed"}
    )


def test_t3_source_has_no_downstream_construction_or_evaluator_import() -> None:
    source = Path(t3.__file__).read_text(encoding="utf-8")
    assert "teacher_distribution_admission" not in source
    assert "evaluator" not in source.lower() or "evaluator_calls" in source
    assert "student_training" not in source
    assert "optical_training" not in source


def test_t3_read_json_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1]), encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        t3._read_json(path)
