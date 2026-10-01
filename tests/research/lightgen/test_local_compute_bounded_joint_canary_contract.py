from __future__ import annotations

from pathlib import Path

from scripts.research.lightgen import define_lightgen_2d_local_compute_bounded_joint_canary_contract_v1 as canary


def test_bounded_canary_contract_keeps_stage1_and_backend_fixed() -> None:
    assert canary.OUTER_GATE == "LOCAL_COMPUTE_BOUNDED_JOINT_CANARY_CONTRACT_DEFINED"
    assert canary.START_STEP == 9000
    assert canary.MAX_UPDATES == 300
    assert canary.PRIMARY_ENDPOINT == 9300
    assert canary.CHECKPOINT_STEPS == (9100, 9200, 9300)
    assert canary.GRADIENT_ENGINE == "SCALAR_REFERENCE_VJP"
    assert canary.PROPOSED_AUTHORIZATION == "AUTHORIZED_EXECUTE_LOCAL_COMPUTE_BOUNDED_JOINT_CANARY_V1"


def test_bounded_canary_contract_payload_is_non_promoting() -> None:
    payload = canary.build_contract(Path(__file__).resolve().parents[3])

    assert payload["result"] == "READY"
    assert payload["git"]["producer"]["head"] == canary.HISTORICAL_PRODUCER_HEAD
    assert payload["git"]["consumer"]["head"] == canary._git_contract(Path(__file__).resolve().parents[3])["consumer"]["head"]
    assert payload["git"]["consumer_head_exact_required"] is False
    assert payload["protocol"]["stage2"]["auto_continuation"] is False
    assert payload["protocol"]["joint_normalization_unchanged"]["delta_joint"] == canary.DELTA_JOINT
    assert payload["execution_boundary"]["scientific_training_executed"] is False
    assert payload["execution_boundary"]["scientific_attempt_created"] is False
    assert payload["gradient_backend"]["performance_gate"] == "none beyond technical executability and exact equivalence; no +10% speedup requirement"
    assert payload["gradient_backend"]["engineering_measurement_basis"]["scientific_evidence"] is False
    assert payload["gradient_backend"]["engineering_measurement_basis"]["scalar_median_seconds"] < payload["gradient_backend"]["engineering_measurement_basis"]["batched_median_seconds"]
    assert payload["acceptance"]["primary"]["endpoint"] == 9300
    assert payload["claim_boundary"]["new_user_decision_required_after_9300"] is True
