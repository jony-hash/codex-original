from __future__ import annotations

import json
from pathlib import Path

import pytest

from ldm_project.v1.application import package_b3_t2 as t2


def test_t2_parser_requires_explicit_lineage_and_timing_paths() -> None:
    parsed = t2._parser().parse_args(
        [
            "run",
            "--config",
            "config.yaml",
            "--dataset-root",
            "data",
            "--upstream-evidence",
            "preflight.json",
            "--t1-output-dir",
            "t1",
            "--output-dir",
            "t2",
        ]
    )
    assert parsed.action == "run"
    assert parsed.device == "cuda"


def test_t2_relative_tolerance_and_replay_classification() -> None:
    t1 = {
        "profile_training_window_ms": 100.0,
        "development_evaluation": {"total_ms": 20.0},
        "phases": {name: {"total_ms": 10.0} for name in ("backward", "model_forward", "cuda_synchronization_wait", "reconstruction_loss", "camera_preprocessing_gpu", "host_to_device")},
    }
    replay = {
        "profile_training_window_ms": 110.0,
        "development_evaluation": {"total_ms": 22.0},
        "phases": {name: {"total_ms": 11.0} for name in ("backward", "model_forward", "cuda_synchronization_wait", "reconstruction_loss", "camera_preprocessing_gpu", "host_to_device")},
    }
    comparison = t2._timing_replay_comparison(t1, replay)
    assert comparison["passed"] is True
    replay["phases"]["backward"]["total_ms"] = 14.0
    assert t2._timing_replay_comparison(t1, replay)["passed"] is False


def test_t2_optimization_matrix_rejects_unsafe_and_unjustified_candidates() -> None:
    replay = {
        "phase_percentages": {
            "backward": 76.0,
            "model_forward": 16.0,
            "cuda_synchronization_wait": 8.0,
            "reconstruction_loss": 1.2,
            "cpu_preprocessing": 0.0,
            "dataloader_wait": 0.0,
            "host_to_device": 0.0,
        },
        "phases": {
            "cuda_synchronization_wait": {"mean_ms": 1.0, "total_ms": 10.0},
            "reconstruction_loss": {"total_ms": 2.0},
        },
    }
    matrix = t2._optimization_matrix(replay)
    assert matrix["accepted_interventions"] == []
    assert any("safety" in row["reason"] for row in matrix["rows"])


def test_t2_equivalence_is_fail_closed_when_no_candidate_is_accepted() -> None:
    replay = {"equivalence": {"bitwise_identical": True}}
    matrix = {"accepted_interventions": []}
    evidence = t2._equivalence_evidence(replay, matrix)
    assert evidence["combined_equivalence"]["bitwise_equal"] is True
    assert evidence["candidate_count"] == 0
    assert evidence["evaluator_calls"] == 0


def test_t2_sync_inventory_contains_categories() -> None:
    rows = t2._static_sync_inventory()
    categories = {row["category"] for row in rows}
    assert "required_for_fail_closed_safety" in categories
    assert "profiling_only" in categories


def test_t2_source_does_not_open_downstream_stages() -> None:
    source = Path(t2.__file__).read_text(encoding="utf-8")
    assert "teacher_distribution_admission" not in source
    assert "student_training" in source
    assert "torch.cuda.amp" not in source


def test_t2_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        t2._read_json(path)
