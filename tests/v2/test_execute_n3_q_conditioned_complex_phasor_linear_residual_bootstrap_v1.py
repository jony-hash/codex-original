"""Focused checks for the frozen complex-phasor residual-bootstrap consumer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ldm_project.v2.evidence import file_sha256, safe_torch_load, tensor_sha256
from scripts.research import execute_n3_q_conditioned_complex_phasor_linear_residual_bootstrap_v1 as execution


def test_contract_and_gate_constants_are_frozen() -> None:
    assert execution.EXPECTED_BRANCH == "research/lightgen-ols-native-semantic-generation-v1"
    assert execution.EXPECTED_STARTING_HEAD == "a5c07512614944d4be6392ae29d48e0b189bea4b"
    assert execution.EXPECTED_CONTRACT_SHA256 == "8f19c3dbb6b9bc17560276e7f1cf61004fd4b0583a3c8a15792b4a86fa05c29f"
    assert execution.CANDIDATE_ID == "Q_CONDITIONED_COMPLEX_PHASOR_LINEAR_RESIDUAL_BOOTSTRAP_V1"
    assert execution.PRIMARY_METRIC == "energy_distance"
    assert execution.OLS_RCOND == 1e-12
    assert execution.PHASOR_PROJECTION_EPSILON == 1e-12
    assert execution.MODEL_BANK_COUNT == 4
    assert execution.MODEL_BANK_SIZE == 200
    assert execution.CALIBRATION_TRIAL_COUNT == 999
    assert execution.PER_FOLD_ALPHA == 0.01
    assert execution.FOLD_COUNT == 5


def test_seed_streams_are_deterministic_and_disjoint() -> None:
    seeds = [
        execution._derive_seed(role, fold, trial, bank)
        for fold in range(execution.FOLD_COUNT)
        for role, trial, bank in (
            ("CANDIDATE_Q_SAMPLE", None, 0),
            ("CANDIDATE_COMPLETE_RESIDUAL_SAMPLE", None, 0),
            ("CANDIDATE_MODEL_BANK", None, 0),
            ("CANDIDATE_PSEUDO_DATA", 0, None),
        )
    ]
    assert len(seeds) == len(set(seeds))
    assert all(0 <= seed <= execution.MAX_SIGNED_63 for seed in seeds)
    assert seeds == [
        execution._derive_seed(role, fold, trial, bank)
        for fold in range(execution.FOLD_COUNT)
        for role, trial, bank in (
            ("CANDIDATE_Q_SAMPLE", None, 0),
            ("CANDIDATE_COMPLETE_RESIDUAL_SAMPLE", None, 0),
            ("CANDIDATE_MODEL_BANK", None, 0),
            ("CANDIDATE_PSEUDO_DATA", 0, None),
        )
    ]


def test_complex_phasor_ols_uses_full_rank_101_column_design() -> None:
    x = torch.zeros((execution.FIT_COUNT, execution.DIMENSION), dtype=torch.float64)
    x[: execution.DIMENSION] = torch.eye(execution.DIMENSION, dtype=torch.float64)
    phase = torch.zeros_like(x)
    fitted = execution._fit_complex_phasor_model(x, phase, fold=0)
    assert fitted["rank"] == execution.DESIGN_FEATURES == 101
    assert fitted["design"].shape == (execution.FIT_COUNT, execution.DESIGN_FEATURES)
    assert fitted["target"].dtype == torch.complex128
    assert fitted["coefficients"].shape == (execution.DESIGN_FEATURES, execution.DIMENSION)
    assert torch.isfinite(fitted["coefficients"].real).all()
    assert torch.isfinite(fitted["coefficients"].imag).all()


def test_rank_deficiency_is_invalid_harness() -> None:
    x = torch.zeros((execution.FIT_COUNT, execution.DIMENSION), dtype=torch.float64)
    phase = torch.zeros_like(x)
    with pytest.raises(execution.InvalidHarnessError, match="INVALID_HARNESS"):
        execution._fit_complex_phasor_model(x, phase, fold=0)


def test_bounded_diagnostic_sample_is_deterministic_and_non_decisional() -> None:
    values = torch.arange(1000, dtype=torch.float64)
    sampled = execution._bounded_diagnostic_sample(values, max_count=16)
    assert sampled.shape == (16,)
    assert torch.equal(sampled, execution._bounded_diagnostic_sample(values, max_count=16))
    assert sampled[0] == values[0]
    assert sampled[-1] == values[-1]


def test_published_result_is_hash_bound_when_available() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / execution.OUTPUT_DIR
    report_path = output_dir / execution.REPORT_NAME
    if not report_path.is_file():
        pytest.skip("q-conditioned complex-phasor residual-bootstrap execution has not been published")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == execution.REPORT_SCHEMA
    assert report["result"] == "VALID_COMPLETED"
    assert report["classification"] in {
        execution.CLASSIFICATION_SUPPORTED,
        execution.CLASSIFICATION_NOT_SUPPORTED,
    }
    assert report["contract"]["sha256"] == execution.EXPECTED_CONTRACT_SHA256
    assert report["candidate"]["id"] == execution.CANDIDATE_ID
    assert report["candidate"]["aggregate"]["fold_count"] == execution.FOLD_COUNT
    assert len(report["candidate"]["folds"]) == execution.FOLD_COUNT
    assert report["control"]["reused_without_rerun"] is True
    assert report["control"]["aggregate"]["strict_five_of_five"] is True
    assert report["governance"]["sealed_confirmation_reused"] is False
    assert report["governance"]["training_authorized"] is False
    assert report["governance"]["formal_promotion"] is False
    assert report["governance"]["generation_readiness"] == "not_ready"

    lineage_ref = report["provenance"]["lineage_manifest"]
    lineage_path = repo_root / lineage_ref["path"]
    assert file_sha256(lineage_path) == lineage_ref["sha256"]
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    assert lineage["contract"]["sha256"] == execution.EXPECTED_CONTRACT_SHA256
    assert lineage["candidate"]["aggregate"]["fold_count"] == execution.FOLD_COUNT
    assert lineage["control"]["reused_without_rerun"] is True
    candidate_ref = lineage["candidate"]["artifact"]
    candidate_path = repo_root / candidate_ref["path"]
    assert file_sha256(candidate_path) == candidate_ref["sha256"]
    payload = safe_torch_load(candidate_path, expected_sha256=candidate_ref["sha256"], map_location="cpu")
    assert payload["schema"] == execution.CANDIDATE_ARTIFACT_SCHEMA
    assert tuple(payload["model_banks"].shape) == (
        execution.FOLD_COUNT,
        execution.MODEL_BANK_COUNT,
        execution.MODEL_BANK_SIZE,
        execution.DIMENSION,
    )
    assert tuple(payload["calibration_trials"].shape) == (execution.FOLD_COUNT, execution.CALIBRATION_TRIAL_COUNT)
    assert tensor_sha256(payload["model_banks"]) == payload["model_banks_tensor_sha256"]
    assert payload["governance"]["sealed_confirmation_reused"] is False
