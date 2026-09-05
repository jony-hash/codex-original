from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from scripts.research.run_scaled_chen_direct_ols_semantic_decode_canary import (
    CONFIG_PATH,
    CONFIG_SCHEMA,
    DIRECT_GATE,
    DIRECT_NOT_SUPPORTED,
    EVALUATOR_BINDING_INVALID,
    TASK_ID,
    classify_s2,
    compare_fidelity_tensor,
    fixed_label_permutation,
    load_config,
    next_write_once_output,
    stratified_bootstrap_delta_ci,
    validate_authority_bindings,
    validate_direct_coefficients,
)


def test_dosd_config_locks_authorities_statistics_and_closed_global_family() -> None:
    payload = load_config(CONFIG_PATH)
    assert payload["schema"] == CONFIG_SCHEMA
    assert payload["task_id"] == TASK_ID
    assert payload["s1"]["gate"] == "S1_POWER_SEMANTICS_SUPPORTED"
    assert payload["s1"]["minimal_representation"] == "R1_POWER"
    assert payload["s1"]["phase_attribution"] == "PHASE_SEMANTIC_INCREMENT_SUPPORTED"
    assert payload["osera"]["gate"] == "OUTPUT_SEMANTIC_EVALUATOR_REQUALIFICATION_PASS"
    assert payload["statistics"]["bootstrap_seed"] == 1236
    assert payload["statistics"]["bootstrap_replicates"] == 2000
    assert payload["statistics"]["null_seed"] == 2053
    assert payload["statistics"]["paired_preservation_rule"] == (
        "lower95_paired_correctness_delta_greater_equal_zero"
    )
    assert payload["global_prior"]["status"] == "CLOSED_AFTER_THIS_CANARY"
    assert payload["authorization"]["training"] is False
    assert payload["authorization"]["optimizer"] is False
    assert payload["authorization"]["backward"] is False


def test_config_loader_rejects_schema_or_registered_authority_mismatch(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    for key, value in (("schema", "wrong"), ("task_id", "wrong")):
        candidate = dict(payload)
        candidate[key] = value
        path = tmp_path / f"{key}.yaml"
        path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(path)

    candidate = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    candidate["osera"] = dict(candidate["osera"])
    candidate["osera"]["manifest_sha256"] = "0" * 64
    path = tmp_path / "osera-mismatch.yaml"
    path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_authority_binding_rejects_s1_osera_or_checkpoint_mismatch() -> None:
    payload = load_config(CONFIG_PATH)
    validate_authority_bindings(payload)
    for section, key in (("s1", "manifest_sha256"), ("osera", "manifest_sha256"), ("optical", "checkpoint_sha256")):
        candidate = {**payload, section: {**payload[section], key: "0" * 64}}
        with pytest.raises(ValueError):
            validate_authority_bindings(candidate)


def test_direct_input_requires_exact_complex_coefficients_and_no_phase_override() -> None:
    coefficients = torch.ones((2, 100), dtype=torch.complex64)
    assert validate_direct_coefficients(coefficients) is coefficients
    with pytest.raises(ValueError, match="complex"):
        validate_direct_coefficients(coefficients.real)
    with pytest.raises(ValueError, match="phase"):
        validate_direct_coefficients(coefficients, phase=torch.zeros(100))
    with pytest.raises(ValueError, match="shape"):
        validate_direct_coefficients(torch.ones((2, 99), dtype=torch.complex64))


def test_fidelity_is_fixed_and_fail_closed() -> None:
    first = torch.ones((2, 1, 4, 4), dtype=torch.complex64)
    second = first.clone()
    result = compare_fidelity_tensor(first, second, name="decoder_input", rtol=1.0e-5, atol=1.0e-8)
    assert result["allclose"] is True
    assert result["max_abs"] == 0.0
    changed = second + (1.0e-3 + 0.0j)
    result = compare_fidelity_tensor(first, changed, name="decoder_input", rtol=1.0e-5, atol=1.0e-8)
    assert result["allclose"] is False
    assert result["name"] == "decoder_input"


def test_null_and_paired_bootstrap_are_deterministic_and_directional() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    first = fixed_label_permutation(labels, seed=2053)
    second = fixed_label_permutation(labels, seed=2053)
    assert torch.equal(first, second)
    assert torch.equal(torch.bincount(first), torch.bincount(labels))
    delta = torch.ones(labels.shape[0], dtype=torch.float64)
    first_ci = stratified_bootstrap_delta_ci(delta, labels, reps=200, seed=1236)
    second_ci = stratified_bootstrap_delta_ci(delta, labels, reps=200, seed=1236)
    assert first_ci == second_ci
    assert first_ci["lower95"] > 0.0


def test_gate_routes_preserve_interpretation_boundary() -> None:
    supported = classify_s2(
        fidelity_pass=True,
        evaluator_binding_pass=True,
        source_identity_pass=True,
        selection_delta=0.2,
        confirmation_lower95=0.1,
        paired_lower95=0.0,
        state_isolation_pass=True,
        finite=True,
        predictions_identical=True,
        tensors_identical=True,
    )
    assert supported["gate"] == DIRECT_GATE
    assert supported["next_action"] == "PLAN_S3_CONTROL_TO_OLS_WRITABILITY_AUDIT"
    assert supported["semantic_generation_supported"] is False
    not_supported = classify_s2(
        fidelity_pass=True,
        evaluator_binding_pass=True,
        source_identity_pass=True,
        selection_delta=0.0,
        confirmation_lower95=-0.1,
        paired_lower95=0.0,
        state_isolation_pass=True,
        finite=True,
        predictions_identical=False,
        tensors_identical=False,
    )
    assert not_supported["gate"] == DIRECT_NOT_SUPPORTED
    assert not_supported["next_action"] == "REVIEW_OLS_SEMANTIC_DECODE_DISCONTINUITY"
    invalid_fidelity = classify_s2(
        fidelity_pass=False,
        evaluator_binding_pass=True,
        source_identity_pass=True,
        selection_delta=1.0,
        confirmation_lower95=1.0,
        paired_lower95=1.0,
        state_isolation_pass=True,
        finite=True,
        predictions_identical=True,
        tensors_identical=True,
    )
    assert invalid_fidelity["gate"] == "S2_DIRECT_REEMIT_INTERFACE_INVALID"
    invalid_evaluator = classify_s2(
        fidelity_pass=True,
        evaluator_binding_pass=False,
        source_identity_pass=True,
        selection_delta=1.0,
        confirmation_lower95=1.0,
        paired_lower95=1.0,
        state_isolation_pass=True,
        finite=True,
        predictions_identical=True,
        tensors_identical=True,
    )
    assert invalid_evaluator["gate"] == EVALUATOR_BINDING_INVALID


def test_write_once_output_never_overwrites(tmp_path: Path) -> None:
    requested = tmp_path / "direct_ols_attempt_01"
    first = next_write_once_output(requested)
    first.mkdir()
    second = next_write_once_output(requested)
    assert first.name == "direct_ols_attempt_01"
    assert second.name == "direct_ols_attempt_02"
