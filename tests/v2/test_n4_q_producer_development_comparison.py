"""Focused contract checks for the one-shot N4 q-producer comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ldm_project.v2.evidence import file_sha256, safe_torch_load, tensor_sha256
from scripts.research import execute_n4_q_producer_development_comparison as execution


def test_authority_hash_mismatch_is_rejected_before_consumption() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    expected = dict(execution.AUTHORITY_SHA256)
    first_path = next(iter(expected))
    expected[first_path] = "0" * 64

    with pytest.raises(execution.InvalidHarnessError, match="authority hash changed"):
        execution.validate_authority_hashes(repo_root, expected)


def test_fold_contamination_is_rejected() -> None:
    fit_positions = torch.tensor([0, 1, 2], dtype=torch.int64)
    heldout_positions = torch.tensor([2, 3], dtype=torch.int64)

    with pytest.raises(execution.InvalidHarnessError, match="overlap"):
        execution.validate_fit_holdout_boundary(fit_positions, heldout_positions)


def test_heldout_fit_leakage_is_rejected_even_when_each_side_is_unique() -> None:
    fit_positions = torch.tensor([4, 5, 6], dtype=torch.int64)
    heldout_positions = torch.tensor([0, 6, 7], dtype=torch.int64)

    with pytest.raises(execution.InvalidHarnessError, match="overlap"):
        execution.validate_fit_holdout_boundary(fit_positions, heldout_positions)


def test_sealed_access_boundary_is_fail_closed() -> None:
    boundary = {
        "status": "RESERVED_NOT_MATERIALIZED",
        "consumed": False,
        "materialized": False,
        "sample_values_loaded": True,
    }

    with pytest.raises(execution.InvalidHarnessError, match="values are loaded"):
        execution.validate_sealed_boundary(boundary)


def test_registry_mutation_after_freeze_is_rejected() -> None:
    with pytest.raises(execution.InvalidHarnessError, match="registry changed"):
        execution.validate_registry_frozen("before", "after")


def test_evaluator_exposure_overflow_is_rejected() -> None:
    records = [
        {
            "candidate_id": execution.EXPECTED_CANDIDATE_IDS[0],
            "fold": 0,
            "evaluator_call_number": call_number,
            "result_caused_code_or_config_change": False,
        }
        for call_number in range(1, execution.MAX_EVALUATOR_CALLS + 2)
    ]

    with pytest.raises(execution.InvalidHarnessError, match="call count exceeded"):
        execution.validate_exposure_ledger(records, execution.EXPECTED_CANDIDATE_IDS)


def test_post_evaluation_configuration_mutation_is_rejected() -> None:
    with pytest.raises(execution.InvalidHarnessError, match="changed after evaluation"):
        execution.validate_runtime_snapshot({"seed": 11}, {"seed": 12})


@pytest.mark.parametrize("field", ["best_of_n", "seed_retry", "heldout_seed_tuning"])
def test_resampling_and_best_of_n_are_rejected(field: str) -> None:
    record = {
        "best_of_n": False,
        "seed_retry": False,
        "heldout_seed_tuning": False,
        "sample_count": execution.MODEL_SAMPLE_COUNT,
    }
    record[field] = True

    with pytest.raises(execution.InvalidHarnessError):
        execution.validate_sampling_record(record)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selected_producer", "SHRINKAGE_ELLIPTICAL_GAUSSIAN_AMPLITUDE_V1"),
        ("training_authorized", True),
        ("formal_promotion", True),
        ("generation_readiness", "ready"),
        ("sealed_confirmation_consumed", True),
    ],
)
def test_n4_governance_flags_cannot_open_after_comparison(field: str, value: object) -> None:
    governance = {
        "selected_producer": None,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "sealed_confirmation_consumed": False,
    }
    governance[field] = value

    with pytest.raises(execution.InvalidHarnessError, match="governance boundary"):
        execution.validate_final_governance(governance)


def test_published_n4_result_has_no_supported_candidate_and_closed_governance() -> None:
    report_path = execution.OUTPUT_DIR / execution.REPORT_NAME
    if not report_path.is_file():
        pytest.skip("N4 comparison report is published by the bounded implementation task")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == execution.REPORT_SCHEMA
    assert report["status"] == "valid_completed"
    assert report["result"] == "VALID_COMPLETED"
    assert report["classification"] == execution.CLASSIFICATION_NO_SUPPORTED
    assert report["candidate_count"] == len(execution.EXPECTED_CANDIDATE_IDS)
    assert report["development_supported_candidates"] == []
    assert sorted(report["development_unsupported_candidates"]) == sorted(execution.EXPECTED_CANDIDATE_IDS)
    assert report["evaluator_exposure"]["total_evaluator_calls"] == execution.MAX_EVALUATOR_CALLS
    assert report["evaluator_exposure"]["adaptive_development_exposure"] is False
    assert report["governance"] == {
        "selected_producer": None,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "sealed_confirmation_consumed": False,
    }


def test_published_n4_artifacts_are_hash_bound_and_partition_safe() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / execution.OUTPUT_DIR
    report_path = output_dir / execution.REPORT_NAME
    if not report_path.is_file():
        pytest.skip("N4 comparison artifacts are published by the bounded implementation task")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    registry_path = repo_root / execution.REGISTRY_PATH
    registry_sha256 = file_sha256(registry_path)
    fit_manifest = json.loads(
        (output_dir / execution.FIT_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    compatibility = json.loads(
        (output_dir / execution.FOLD_COMPATIBILITY_NAME).read_text(encoding="utf-8")
    )
    ledger = json.loads((output_dir / execution.LEDGER_NAME).read_text(encoding="utf-8"))
    lineage = json.loads((output_dir / execution.LINEAGE_NAME).read_text(encoding="utf-8"))

    assert lineage["all_artifacts_write_once"] is True
    assert lineage["source_row_lineage_recorded"] is True
    assert len(lineage["artifacts"]) == (
        len(execution.EXPECTED_CANDIDATE_IDS) * execution.FOLD_COUNT * 2 + 3
    )
    execution.validate_exposure_ledger(
        ledger["call_records"], execution.EXPECTED_CANDIDATE_IDS
    )
    assert ledger["total_evaluator_calls"] == execution.MAX_EVALUATOR_CALLS
    assert ledger["registry"]["sha256"] == registry_sha256

    fit_records = {
        (record["candidate_id"], int(record["fold"])): record
        for record in fit_manifest["fit_records"]
    }
    fold_records = {
        (record["candidate_id"], int(fold["fold"])): fold
        for record in compatibility["candidate_folds"]
        for fold in record["folds"]
    }
    artifact_keys: set[tuple[str, int, str]] = set()
    for reference in lineage["artifacts"]:
        role = str(reference["role"])
        artifact_path = repo_root / reference["path"]
        assert file_sha256(artifact_path) == reference["sha256"]
        if role not in {"fitted producer", "sampled model q"}:
            assert role in {"fit manifest", "fold compatibility", "evaluator exposure ledger"}
            continue

        candidate_id = str(reference["candidate_id"])
        fold = int(reference["fold"])
        key = (candidate_id, fold, role)
        assert key not in artifact_keys
        artifact_keys.add(key)
        payload = safe_torch_load(
            artifact_path,
            expected_sha256=reference["sha256"],
            map_location="cpu",
        )
        assert payload["registry_sha256"] == registry_sha256
        assert payload["candidate_id"] == candidate_id
        assert payload["fold"] == fold
        assert payload["source_lineage"]["q_artifact"]["sha256"] == execution.AUTHORITY_SHA256[
            execution.Q_ARTIFACT.as_posix()
        ]

        fold_record = fold_records[(candidate_id, fold)]
        if role == "fitted producer":
            fit_source = payload["fit_source"]
            fit_positions = set(fit_source["row_positions"])
            heldout_positions = set(fold_record["data"]["row_positions"])
            assert len(fit_positions) == execution.FIT_COUNT_PER_FOLD
            assert len(heldout_positions) == execution.HELDOUT_COUNT_PER_FOLD
            assert fit_positions.isdisjoint(heldout_positions)
            assert fit_records[(candidate_id, fold)]["fit_artifact"]["sha256"] == reference[
                "sha256"
            ]
            assert fold_record["fit"]["fit_artifact"]["sha256"] == reference["sha256"]
        else:
            assert role == "sampled model q"
            model_q = payload["model_q"]
            assert tuple(model_q.shape) == (execution.MODEL_SAMPLE_COUNT, execution.DIMENSION)
            assert model_q.dtype == torch.float64
            assert bool(torch.isfinite(model_q).all())
            assert bool(model_q.ge(0.0).all())
            assert tensor_sha256(model_q) == payload["model_q_sha256"]
            execution.validate_sampling_record(payload["sampling"])
            assert payload["heldout_values_used_for_sampling"] is False
            assert payload["best_of_n"] is False
            assert payload["resampling_after_evaluation"] is False
            assert fold_record["model"]["sample_artifact"]["sha256"] == reference[
                "sha256"
            ]

    assert len(artifact_keys) == len(execution.EXPECTED_CANDIDATE_IDS) * execution.FOLD_COUNT * 2
    for item in report["artifact_manifests"].values():
        path = repo_root / item["path"]
        assert file_sha256(path) == item["sha256"]
