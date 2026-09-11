"""Focused checks for the one-shot K=1 q-conditioned phase-donor consumer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ldm_project.v2.evidence import file_sha256, safe_torch_load, tensor_sha256
from scripts.research import execute_n3_q_conditioned_nearest_phase_donor_v1 as execution


def test_contract_and_gate_constants_are_frozen() -> None:
    assert execution.EXPECTED_BRANCH == "research/lightgen-ols-native-semantic-generation-v1"
    assert execution.EXPECTED_STARTING_HEAD == "a5c07512614944d4be6392ae29d48e0b189bea4b"
    assert execution.EXPECTED_CONTRACT_SHA256 == "14f4f3cd5d1046eb648b3410593d778910f8b8d33262528f9e0f12ba94250434"
    assert execution.CANDIDATE_ID == "Q_CONDITIONED_NEAREST_PHASE_DONOR_V1"
    assert execution.PRIMARY_METRIC == "energy_distance"
    assert execution.CALIBRATION_TRIAL_COUNT == 999
    assert execution.PER_FOLD_ALPHA == 0.01
    assert execution.FOLD_COUNT == 5


def test_seed_streams_are_deterministic_and_disjoint() -> None:
    seeds = [
        execution._derive_seed(role, fold, trial, bank)
        for fold in range(execution.FOLD_COUNT)
        for role, trial, bank in (
            ("CANDIDATE_Q_SAMPLE", None, 0),
            ("CANDIDATE_Q_SAMPLE", 0, None),
            ("CANDIDATE_MODEL_BANK", None, 0),
            ("CANDIDATE_PSEUDO_DATA", 0, None),
        )
    ]
    assert len(seeds) == len(set(seeds))
    assert seeds == [
        execution._derive_seed(role, fold, trial, bank)
        for fold in range(execution.FOLD_COUNT)
        for role, trial, bank in (
            ("CANDIDATE_Q_SAMPLE", None, 0),
            ("CANDIDATE_Q_SAMPLE", 0, None),
            ("CANDIDATE_MODEL_BANK", None, 0),
            ("CANDIDATE_PSEUDO_DATA", 0, None),
        )
    ]


def test_vectorized_gaussianized_q_matches_contract_rule() -> None:
    sorted_values = torch.linspace(0.1, 1.0, execution.DIMENSION, dtype=torch.float64).reshape(1, -1).repeat(4, 1)
    sorted_values[1, 5] = sorted_values[0, 5]
    sorted_values[2, 5] = sorted_values[0, 5]
    sorted_values[:, 6:] = torch.maximum(sorted_values[:, 6:], sorted_values[:, 5:6] + 0.1)
    query = sorted_values[0:1].clone()
    query[0, 0] = 0.0
    query[0, 1] = 2.0
    query[0, 2] = (sorted_values[0, 2] + sorted_values[3, 2]) / 2.0
    expected = execution.contract_definition._gaussianized_q_coordinates(query, sorted_values)
    actual = execution._gaussianized_q_coordinates_batch(query, sorted_values)
    assert actual.dtype == torch.float64
    assert torch.equal(actual, expected)


def test_nearest_donor_tie_uses_bytewise_smallest_identity() -> None:
    donors = torch.zeros((execution.FIT_COUNT, execution.DIMENSION), dtype=torch.float64)
    donor_q = torch.ones_like(donors)
    donor_ids = ["stable-row-z", "stable-row-a"] + [f"stable-row-m-{index:04d}" for index in range(execution.FIT_COUNT - 2)]
    selected, distance, raw_q_distance, ties = execution._nearest_donors_with_q(
        torch.zeros((1, execution.DIMENSION), dtype=torch.float64),
        torch.ones((1, execution.DIMENSION), dtype=torch.float64),
        donors,
        donor_q,
        donor_ids,
    )
    assert int(selected[0]) == 1
    assert float(distance[0]) == 0.0
    assert float(raw_q_distance[0]) == 0.0
    assert int(ties[0]) == execution.FIT_COUNT


def test_published_result_is_hash_bound_when_available() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / execution.OUTPUT_DIR
    report_path = output_dir / execution.REPORT_NAME
    if not report_path.is_file():
        pytest.skip("q-conditioned nearest phase-donor execution has not been published")

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
    assert tuple(payload["model_banks"].shape) == (execution.FOLD_COUNT, execution.MODEL_BANK_COUNT, execution.MODEL_BANK_SIZE, execution.DIMENSION)
    assert tuple(payload["calibration_trials"].shape) == (execution.FOLD_COUNT, execution.CALIBRATION_TRIAL_COUNT)
    assert tensor_sha256(payload["model_banks"]) == payload["model_banks_tensor_sha256"]
    assert payload["governance"]["sealed_confirmation_reused"] is False
