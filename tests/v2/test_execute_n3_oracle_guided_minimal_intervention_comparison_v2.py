from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.research import execute_n3_oracle_guided_minimal_intervention_comparison_v2 as consumer


def _synthetic_m2_fit() -> dict[str, object]:
    dimension = consumer.DIMENSION
    identity = torch.eye(dimension, dtype=torch.float64)
    alpha = torch.full((dimension,), 0.5, dtype=torch.float64)
    return {
        "frozen": {"dependence": {"cholesky": identity}},
        "mu": torch.zeros(dimension, dtype=torch.float64),
        "sigma": torch.ones(dimension, dtype=torch.float64),
        "alpha": alpha,
    }


def test_m2_sampling_is_deterministic_and_nonnegative() -> None:
    fit = _synthetic_m2_fit()
    first, first_record = consumer._sample_candidate_m2(
        fit,
        fold=0,
        role="data-vs-model",
        replicate=0,
    )
    second, second_record = consumer._sample_candidate_m2(
        fit,
        fold=0,
        role="data-vs-model",
        replicate=0,
    )

    assert torch.equal(first, second)
    assert first_record == second_record
    assert bool(torch.isfinite(first).all())
    assert bool(first.ge(0.0).all())
    assert first_record["support_by_construction"] is True
    assert first_record["materially_negative_count"] == 0


def test_m2_sampling_uses_frozen_substream_and_bank_shape() -> None:
    _, record = consumer._sample_candidate_m2(
        _synthetic_m2_fit(),
        fold=4,
        role="model-vs-model",
        replicate=7,
    )

    assert record["candidate"] == consumer.CANDIDATE_M2
    assert record["role"] == "model-vs-model"
    assert record["fold"] == 4
    assert record["replicate"] == 7
    assert record["bank_size"] == consumer.GENERATED_BANK_SIZE
    assert record["proposal_rejection"] is False
    assert record["canonicalization_count"] >= 0


def test_invalid_probability_is_fail_closed() -> None:
    with pytest.raises(consumer._InvalidHarnessError) as raised:
        consumer._finite_probability(
            torch.tensor([[0.0, 0.5]], dtype=torch.float64),
            candidate=consumer.CANDIDATE_M2,
            stage="sampling",
            fold=0,
            role="data-vs-model",
            replicate=0,
            name="test probability",
        )

    assert raised.value.details["failure_semantics"] == "INVALID_HARNESS"
    assert raised.value.details["contract_action"].startswith("stop the candidate/fold")


def test_v2_preflight_replays_authority_and_semantic_diff() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    validated = consumer._validate_contract(repo_root)
    contract = validated["contract"]

    assert contract["schema"] == "n3_oracle_guided_minimal_intervention_contract.v2"
    assert contract["candidate_M2"]["family"] == consumer.CANDIDATE_M2_FAMILY
    assert contract["candidate_D"]["family"] == consumer.CANDIDATE_D_FAMILY
    assert contract["semantic_diff"]["status"] == "passed"
    assert contract["semantic_diff"]["candidate_D_byte_semantic_equivalence"] is True
    for key in (
        "candidate_M_fit",
        "candidate_M_sampling",
        "candidate_D_fit",
        "candidate_D_sampling",
        "new_candidate_results_observed",
        "q_tensor_loaded_by_this_definition_consumer",
        "sealed_confirmation_consumed",
        "training_authorized",
    ):
        assert contract["execution_boundary"][key] is False
