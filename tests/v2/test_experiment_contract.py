from __future__ import annotations

import pytest

from ldm_project.v2.evidence import canonical_sha256
from ldm_project.v2.experiment_contract import (
    ExperimentContract,
    assert_transition,
    checkpoint_manifest,
    validate_checkpoint_manifest,
    validate_preflight_report,
    checkpoint_interval_steps,
    validate_history_steps,
)


def _contract() -> ExperimentContract:
    runtime = {"entrypoint": "test", "device": "cpu", "deterministic": True}
    return ExperimentContract(
        schema_version=1,
        name="contract-test",
        level="L2",
        hypothesis="the candidate is trainable",
        comparison={"control": "A", "variable": "objective"},
        budget={"planned_seconds": 600, "preflight_seconds": 60},
        inputs={"config": "config", "dataset_split": "split", "evaluator": "evaluator"},
        gates={"finite": True},
        stop_policy={"invalid_harness": "repair", "valid_failed": "close", "passed": "measure"},
        runtime=runtime,
        runtime_closure_hash=canonical_sha256(runtime),
    )


def test_contract_digest_and_budget_are_stable() -> None:
    contract = _contract()
    contract.validate()
    assert contract.digest == ExperimentContract.from_dict(contract.to_dict()).digest
    assert contract.preflight_budget_seconds == 30.0


def test_contract_rejects_runtime_or_input_tampering() -> None:
    payload = _contract().to_dict()
    payload["runtime"]["device"] = "cuda"
    with pytest.raises(ValueError, match="closure hash"):
        ExperimentContract.from_dict(payload)


def test_preflight_requires_resume_for_l2() -> None:
    with pytest.raises(ValueError, match="independent and resume"):
        validate_preflight_report(
            _contract(),
            {"finite": True, "all_parameters_updated": True, "evaluator_gradients_clean": True, "independent_match": True, "resume_match": False},
            elapsed_seconds=1,
        )


def test_checkpoint_manifest_is_explicit_and_tamper_evident(tmp_path) -> None:
    contract = _contract()
    checkpoint = tmp_path / "step_10.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = checkpoint_manifest(checkpoint, contract=contract, completed_step=10, candidate="A", seed=1234)
    assert validate_checkpoint_manifest(checkpoint, manifest, contract=contract, candidate="A", seed=1234) == 10
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="tampered"):
        validate_checkpoint_manifest(checkpoint, manifest, contract=contract, candidate="A", seed=1234)


def test_state_machine_rejects_scientific_skip() -> None:
    assert_transition("validated", "preflight_passed")
    with pytest.raises(ValueError, match="invalid experiment state"):
        assert_transition("validated", "passed")


def test_checkpoint_cadence_is_bounded_and_history_is_contiguous() -> None:
    assert checkpoint_interval_steps(984, steps_per_second=2.0) == 50
    validate_history_steps([{"step": 1}, {"step": 2}], 2)
    with pytest.raises(ValueError, match="exactly once"):
        validate_history_steps([{"step": 1}, {"step": 1}], 2)
