"""Focused tests for the one-shot selected-producer full fit and state."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ldm_project.v2.evidence import file_sha256, safe_torch_load
from scripts.research import execute_n3_selected_producer_full_development_fit_v1 as execution


STATE_SHA256 = "752db16087e7bdf8343e4499202182370b376a1a2d5925a383f411fd4a15bb88"
LINEAGE_SHA256 = "9244df01c95e44005207cb1dc188fc7de2961665e13055beebf55f0f5cdd2267"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _published() -> tuple[Path, dict, dict]:
    root = _repo_root()
    output = root / execution.OUTPUT_DIR
    state_path = output / execution.STATE_NAME
    lineage_path = output / execution.LINEAGE_NAME
    state_sha = file_sha256(state_path)
    state = safe_torch_load(state_path, expected_sha256=state_sha, map_location="cpu")
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    return root, state, lineage


def test_published_state_is_the_single_validated_full_fit_authority() -> None:
    root, state, lineage = _published()
    output = root / execution.OUTPUT_DIR
    state_path = output / execution.STATE_NAME
    lineage_path = output / execution.LINEAGE_NAME

    assert file_sha256(state_path) == STATE_SHA256
    assert file_sha256(lineage_path) == LINEAGE_SHA256
    validation = execution.validate_state(state)
    assert validation["diagonal_error"] <= execution.DIAGONAL_ATOL
    assert state["schema"] == execution.STATE_SCHEMA
    assert state["semantic_authority_id"] == execution.PRODUCER_STATE_SEMANTIC_ID
    assert state["fit_scope"] == {
        "all_development_rows": True,
        "fit_row_count": 5000,
        "fit_rows_per_class": 500,
        "folds_used_for_fit": False,
        "heldout_rows_used": False,
        "fold_evidence_role": "prior model selection only",
        "labels_as_inputs": False,
        "fit_execution_count": 1,
        "device": "cpu",
        "dtype": "torch.float64",
    }
    assert not {"q", "raw_q", "z", "labels", "source_indices", "fold_ids"}.intersection(state)
    assert set(state["sampling_tensors"]) == {"sorted_values", "quantile_table_positions", "cholesky"}
    assert set(state["audit_tensors"]) == {
        "plotting_positions",
        "z_mean",
        "standard_deviation",
        "R_emp",
        "R_shrunk",
        "eigenvalues",
        "diagonal",
    }
    assert all(value.dtype == torch.float64 and value.device.type == "cpu" for value in state["sampling_tensors"].values())
    assert lineage["status"] == "valid_completed"
    assert lineage["result"] == "VALID_COMPLETED"
    assert lineage["classification"] == execution.CLASSIFICATION
    assert lineage["producer_state"]["sha256"] == STATE_SHA256
    assert lineage["producer_state"]["semantic_authority_id"] == execution.PRODUCER_STATE_SEMANTIC_ID
    assert lineage["fit_summary"]["fit_execution_count"] == 1
    assert lineage["fit_summary"]["fit_rows"] == 5000
    assert lineage["fit_summary"]["folds_used_for_fit"] is False
    assert lineage["fit_summary"]["compatibility_evaluation_executed"] is False
    assert lineage["development_input_verification"]["sealed_values_loaded"] is False
    assert lineage["development_input_verification"]["sealed_confirmation_consumed"] is False
    assert lineage["next_action"] == "DEFINE_N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_CONTRACT_V1"


def test_published_state_can_sample_without_q_or_sealed_inputs() -> None:
    _root, state, lineage = _published()
    q_first = execution.sample_from_state(
        state,
        producer_state_sha256=STATE_SHA256,
        requested_sample_count=3,
        rng_namespace=execution.RNG_NAMESPACE,
        seed=execution.VALIDATION_SMOKE_SEED,
    )
    q_second = execution.sample_from_state(
        state,
        producer_state_sha256=STATE_SHA256,
        requested_sample_count=3,
        rng_namespace=execution.RNG_NAMESPACE,
        seed=execution.VALIDATION_SMOKE_SEED,
    )
    assert torch.equal(q_first, q_second)
    assert q_first.shape == (3, 100)
    assert q_first.dtype == torch.float64
    assert q_first.device.type == "cpu"
    assert torch.isfinite(q_first).all()
    assert q_first.ge(0).all()
    assert lineage["serialization_validation"]["standalone_sampler_uses_state_only"] is True
    assert lineage["serialization_validation"]["smoke"]["scientific_evaluation"] is False


def test_stale_contract_rejects_before_q_load_or_output_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = _repo_root()
    monkeypatch.setattr(execution, "EXPECTED_CONTRACT_SHA256", "0" * 64)

    def fail_if_q_is_loaded(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("q was loaded before stale-contract rejection")

    monkeypatch.setattr(execution, "safe_torch_load", fail_if_q_is_loaded)
    target = tmp_path / "stale-contract-output"
    with pytest.raises(execution.InvalidHarnessError, match="selected-producer freeze contract hash changed"):
        execution.preflight(root, target)
    assert not target.exists()
