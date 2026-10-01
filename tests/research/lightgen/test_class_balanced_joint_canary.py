from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.research.lightgen import define_lightgen_2d_class_balanced_joint_canary_contract_v1 as contract
from scripts.research.lightgen import execute_lightgen_2d_class_balanced_joint_canary_v1 as canary


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _contract_payload() -> dict[str, object]:
    path = _root() / canary.CONTRACT_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def test_contract_freezes_class_balanced_endpoint_and_production_batch() -> None:
    payload = _contract_payload()
    assert payload["outer_gate"] == "CLASS_BALANCED_JOINT_CANARY_CONTRACT_DEFINED"
    assert payload["protocol"]["start_step"] == 9000
    assert payload["protocol"]["max_updates"] == 300
    assert payload["protocol"]["primary_endpoint"] == 9300
    assert payload["protocol"]["checkpoints"] == [9100, 9200, 9300]
    assert payload["protocol"]["production_batch_semantics"] == "exactly_one_row_per_class"
    assert payload["authorities"]["production_batch_semantics"]["new_sampler_required"] is False
    assert payload["structural_aggregation"]["mode"] == canary.STRUCTURAL_AGGREGATION_MODE
    assert payload["structural_aggregation"]["class_weight"] == pytest.approx(0.1)
    assert payload["execution_boundary"]["scientific_attempt_created"] is False


def test_contract_uses_producer_provenance_not_exact_consumer_lock() -> None:
    payload = _contract_payload()
    binding = canary.validate_contract_payload(_root(), payload)
    assert binding["producer_head_is_provenance_only"] is True
    assert payload["git"]["consumer_head_exact_required"] is False
    assert payload["git"]["producer"]["head"] == contract.HISTORICAL_PRODUCER_HEAD


def test_batch_balance_requires_ordered_one_row_per_class() -> None:
    result = canary._batch_balance(torch.arange(10))
    assert result["counts"] == [1] * 10
    assert result["mode"] == "CLASS_BALANCED_EQUAL_NORMALIZED"
    with pytest.raises(canary.CanaryError, match="exactly class 0..9"):
        canary._batch_balance(torch.tensor([0, 0, 1, 2, 3, 4, 5, 6, 7, 8]))


def test_equal_normalized_aggregation_is_not_magnitude_weighted() -> None:
    gradients = [torch.tensor([100.0, 0.0])]
    gradients.extend(torch.tensor([0.0, 1.0]) for _ in range(9))
    aggregate = canary._aggregate_class_gradients(gradients)
    expected = torch.tensor([0.1, 0.9])
    assert torch.allclose(aggregate["v_struct_CB"], expected, atol=1.0e-6)
    assert torch.allclose(aggregate["u_struct_CB"], expected / torch.linalg.vector_norm(expected), atol=1.0e-6)


def test_classwise_aggregation_fails_closed_for_nonfinite_or_degenerate_values() -> None:
    with pytest.raises(canary.CanaryError, match="nonfinite"):
        canary._aggregate_class_gradients([torch.tensor([float("nan")])] * 10)
    with pytest.raises(canary.CanaryError, match="degenerate"):
        canary._aggregate_class_gradients([torch.tensor([1.0, 0.0]) if index < 5 else torch.tensor([-1.0, 0.0]) for index in range(10)])


def test_class_balanced_backend_and_lifecycle_are_fixed() -> None:
    assert canary.AUTHORIZATION_TOKEN == "AUTHORIZED_EXECUTE_CLASS_BALANCED_JOINT_CANARY_V1"
    assert canary.GRADIENT_ENGINE == "SCALAR_REFERENCE_VJP"
    assert canary.STRUCTURAL_AGGREGATION_MODE == "CLASS_BALANCED_EQUAL_NORMALIZED"
    assert canary.PRIMARY_ENDPOINT == 9300
    assert canary.CHECKPOINT_STEPS == (9100, 9200, 9300)
    assert canary.HISTORICAL_ATTEMPT_DIR != canary.DEFAULT_OUTPUT_DIR
    with pytest.raises(canary.CanaryError, match="training backend"):
        canary._require_training_backend("HYBRID_SEMANTIC_BATCHED_VJP")
