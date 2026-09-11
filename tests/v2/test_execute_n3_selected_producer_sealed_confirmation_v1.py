"""Focused contract and firewall tests for the one-shot sealed confirmation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.research import execute_n3_selected_producer_sealed_confirmation_v1 as execution
from scripts.research import execute_n3_development_bank_v2_empirical_ceiling as v3_evaluator


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_frozen_observed_bank_seeds_and_contract_are_exact() -> None:
    root = _repo_root()
    contract, identity = execution._validate_contract(root)
    assert identity["contract_sha256"] == execution.CONTRACT_SHA256
    assert execution._seed("OBSERVED_MODEL_BANK", None, 0) == 2702721627146884145
    assert identity["observed_model_bank_seeds"] == [
        2702721627146884145,
        2090324094609360616,
        370415492307778749,
        2559923572555499134,
    ]
    assert contract["monte_carlo_decision"]["B"] == 999
    assert contract["monte_carlo_decision"]["alpha"] == 0.05
    assert contract["sealed_source"]["expected_consumed_on_entry"] is False


def test_presealed_reservation_validation_reads_manifest_only() -> None:
    root = _repo_root()
    contract, _identity = execution._validate_contract(root)
    development = execution._validate_development_and_v3(root, contract)
    sealed = execution._validate_sealed_reservation(
        root,
        contract,
        development["source_manifest"]["source_indices"],
    )
    assert sealed["status_before_consumption"] == "RESERVED_NOT_MATERIALIZED"
    assert sealed["consumed_before"] is False
    assert sealed["protected_content_loaded_before"] is False
    assert sealed["indices"].shape == (500,)
    assert sealed["labels"].shape == (500,)
    assert sealed["labels"].bincount(minlength=10).tolist() == [50] * 10


def test_prebound_sampler_is_bitwise_equal_to_public_sampler() -> None:
    root = _repo_root()
    contract, _identity = execution._validate_contract(root)
    state, producer_identity = execution._validate_producer(root, contract)
    fast = execution._make_fast_sampler(state)
    parity = execution._sampler_parity_check(
        state,
        producer_identity["state_sha256"],
        fast,
    )
    assert parity["bitwise_equal"] is True
    assert parity["public_q_tensor_sha256"] == parity["fast_q_tensor_sha256"]


def test_primary_statistic_is_the_preregistered_arithmetic_mean() -> None:
    generator = torch.Generator(device="cpu").manual_seed(19)
    data = torch.rand((5, 100), generator=generator, dtype=torch.float64)
    banks = [
        torch.rand((3, 100), generator=generator, dtype=torch.float64)
        for _ in range(execution.MODEL_BANK_COUNT)
    ]
    statistic, per_bank = execution._primary_statistic(data, banks)
    assert statistic == sum(per_bank) / execution.MODEL_BANK_COUNT
    assert statistic >= 0.0


def test_descriptive_metrics_match_frozen_v3_formulas() -> None:
    generator = torch.Generator(device="cpu").manual_seed(23)
    left = torch.rand((5, 100), generator=generator, dtype=torch.float64)
    right = torch.rand((3, 100), generator=generator, dtype=torch.float64)
    expected = v3_evaluator._pair_metrics_raw_q(left, right)
    actual = execution._descriptive_pair_metrics(left, right)
    for name in execution.DESCRIPTIVE_ONLY_METRICS:
        assert actual[name] == expected[name]


def test_consumption_marker_is_write_once(tmp_path: Path) -> None:
    sealed = {
        "path": "sealed-manifest.json",
        "sha256": "a" * 64,
        "source_index_sha256": "b" * 64,
        "ordered_row_identity_sha256": "c" * 64,
    }
    first = execution._publish_consumption_marker(
        tmp_path,
        execution.CONTRACT_SHA256,
        execution.PRODUCER_STATE_SHA256,
        sealed,
        "d" * 64,
        "e" * 64,
    )
    assert first["consumed"] is True
    marker = json.loads((tmp_path / execution.CONSUMPTION_NAME).read_text(encoding="utf-8"))
    assert marker["consumed"] is True
    with pytest.raises(execution.InvalidHarnessError, match="consumption marker already exists"):
        execution._publish_consumption_marker(
            tmp_path,
            execution.CONTRACT_SHA256,
            execution.PRODUCER_STATE_SHA256,
            sealed,
            "d" * 64,
            "e" * 64,
        )
