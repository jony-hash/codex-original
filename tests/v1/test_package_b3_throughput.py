from __future__ import annotations

import json
from pathlib import Path

import pytest

from ldm_project.v1.application import package_b3_throughput as profiler


def test_b3_throughput_summary_has_required_distribution_fields() -> None:
    summary = profiler._summary([1.0, 2.0, 3.0])
    assert summary["count"] == 3
    assert summary["total_ms"] == 6.0
    assert summary["median_ms"] == 2.0
    assert summary["p95_ms"] == pytest.approx(2.9)


def test_b3_throughput_parser_requires_explicit_upstream_and_output() -> None:
    parsed = profiler._parser().parse_args(
        [
            "baseline",
            "--config",
            "config.yaml",
            "--dataset-root",
            "data",
            "--output-dir",
            "out",
            "--upstream-evidence",
            "preflight.json",
        ]
    )
    assert parsed.action == "baseline"
    assert parsed.device == "cuda"


def test_b3_throughput_upstream_rejects_stale_preflight(tmp_path: Path) -> None:
    path = tmp_path / "preflight.json"
    path.write_text(
        json.dumps(
            {
                "schema": "teacher_first_stage_recovery_preflight.v1",
                "passed": False,
                "git_head": "head",
                "config_sha256": "config",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stale or failed"):
        profiler._validate_upstream({}, str(path), head="head", config_sha="config")


def test_b3_throughput_accepts_only_complete_failed_throughput_baseline(tmp_path: Path) -> None:
    path = tmp_path / "preflight.json"
    path.write_text(
        json.dumps(
            {
                "schema": "teacher_first_stage_recovery_preflight.v1",
                "passed": False,
                "classification": "invalid_harness",
                "first_failed_gate": "preflight_contract",
                "throughput_gate_passed": False,
                "resume_exact": True,
                "cuda_peak_bytes": 1,
                "host_rss_peak_bytes": 1,
                "git_head": "a" * 40,
                "config_sha256": "config",
                "evaluator_calls": 0,
            }
        ),
        encoding="utf-8",
    )
    accepted = profiler._validate_upstream(
        {"cuda_memory_limit_bytes": 2, "host_rss_limit_bytes": 2},
        str(path),
        head="a" * 40,
        config_sha="config",
    )
    assert accepted["head_relation"] == "same"


def test_b3_throughput_source_does_not_construct_downstream_models() -> None:
    source = Path(profiler.__file__).read_text(encoding="utf-8")
    assert "teacher_distribution_admission" not in source
    assert "teacher_failure_attribution" not in source
    assert "student_training" not in source
    assert "optical_training" not in source
