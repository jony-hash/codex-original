from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.research.scaled_chen.run_scaled_chen_repaired_coordinate_generative_alignment_canary import ALPHA_REPAIRED, CANARY_END_STEP, CANARY_START_STEP, OLD_ALPHA, T1_ALIGNMENT_DIRECTION_SUPPORTED, T1_ALIGNMENT_HARMFUL, T1_ALIGNMENT_NOT_SUPPORTED, classify_final_result, classify_primary_gate, next_write_once_output, validate_alpha, validate_checkpoint_identity, validate_fixed_end_policy, validate_optimizer_contract, validate_trainable_parameter_names


def test_registered_alpha_is_exact_and_old_alpha_is_rejected() -> None:
    validate_alpha(ALPHA_REPAIRED)
    with pytest.raises(ValueError, match="old alpha"):
        validate_alpha(OLD_ALPHA)
    with pytest.raises(ValueError, match="registered alpha"):
        validate_alpha(ALPHA_REPAIRED * 1.001)


def test_primary_gate_is_strictly_ci_separated_from_parent() -> None:
    parent_r50 = 1.8142740
    parent_r95 = 1.0679024
    assert classify_primary_gate((1.20, 1.80), (0.80, 1.06), parent_r50, parent_r95)
    assert not classify_primary_gate((1.20, parent_r50), (0.80, 1.06), parent_r50, parent_r95)
    assert not classify_primary_gate((1.20, 1.80), (0.80, parent_r95), parent_r50, parent_r95)


def test_final_classification_preserves_harmful_and_invalid_boundaries() -> None:
    assert classify_final_result(True, True, True, True) == T1_ALIGNMENT_DIRECTION_SUPPORTED
    assert classify_final_result(False, True, True, True) == T1_ALIGNMENT_NOT_SUPPORTED
    assert classify_final_result(True, False, True, True) == T1_ALIGNMENT_HARMFUL
    assert classify_final_result(True, True, True, True, valid=False) == "T1_TRAINING_INVALID"


def test_fixed_endpoint_and_parent_checkpoint_identity_are_fail_closed() -> None:
    validate_fixed_end_policy(CANARY_START_STEP, CANARY_END_STEP, 1000)
    with pytest.raises(ValueError, match="1000"):
        validate_fixed_end_policy(CANARY_START_STEP, CANARY_END_STEP + 1, 1000)
    with pytest.raises(ValueError, match="endpoint"):
        validate_fixed_end_policy(CANARY_START_STEP, CANARY_END_STEP, 999)

    payload = {
        "schema": "scaled_chen_bogt_horizon_checkpoint.v1",
        "global_step": 5000,
        "model_state": {"system.encoder.phase": object()},
        "optimizer_state": {"state": {}, "param_groups": []},
    }
    validate_checkpoint_identity(payload, expected_schema=payload["schema"], expected_step=5000)
    with pytest.raises(ValueError, match="checkpoint"):
        validate_checkpoint_identity(payload, expected_schema=payload["schema"], expected_step=6000)


def test_optimizer_continuation_requires_parent_adam_state_and_exact_group() -> None:
    state = {
        "state": {
            0: {"step": 5000, "exp_avg": torch.zeros(2), "exp_avg_sq": torch.zeros(2)},
            1: {"step": 5000, "exp_avg": torch.zeros(2), "exp_avg_sq": torch.zeros(2)},
        },
        "param_groups": [
            {
                "lr": 1.0e-3,
                "betas": (0.9, 0.999),
                "eps": 1.0e-8,
                "weight_decay": 0,
                "amsgrad": False,
                "maximize": False,
                "params": [0, 1],
            }
        ],
    }
    validate_optimizer_contract(state, expected_parameter_count=2, learning_rate=1.0e-3)
    with pytest.raises(ValueError, match="optimizer"):
        validate_optimizer_contract(state | {"param_groups": []}, expected_parameter_count=2, learning_rate=1.0e-3)


def test_trainable_scope_is_exact() -> None:
    names = ("system.encoder.phase", "system.decoder.phase")
    validate_trainable_parameter_names(names)
    with pytest.raises(ValueError, match="trainable"):
        validate_trainable_parameter_names(names + ("ols.transport",))


def test_write_once_output_never_reuses_an_existing_attempt(tmp_path: Path) -> None:
    requested = tmp_path / "lightgen_scaled_chen_repaired_coordinate_generative_alignment_canary_v1_attempt_01"
    first = next_write_once_output(requested)
    first.mkdir()
    second = next_write_once_output(requested)
    assert second.name.endswith("attempt_02")
