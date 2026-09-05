from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from scripts.research.audit_scaled_chen_ols_semantic_representation import (
    CONFIG_PATH,
    CONFIG_SCHEMA,
    PHASE_SEMANTIC_INCREMENT_NOT_SUPPORTED,
    PHASE_SEMANTIC_INCREMENT_SUPPORTED,
    R1_POWER,
    R2_AMPLITUDE,
    R3_COMPLEX,
    S1_AMPLITUDE_SEMANTICS_SUPPORTED,
    S1_POWER_SEMANTICS_SUPPORTED,
    classify_s1,
    fixed_label_permutation,
    gauge_distance_squared,
    load_config,
    minimal_representation,
    normalized_representations,
    next_write_once_output,
    one_nn_metrics,
    phase_gauge_audit,
    phase_increment_classification,
    predict_ridge_probe,
    representation_gate,
    fit_ridge_probe,
    stratified_bootstrap_ci,
    validate_amplitude_scale,
)


def test_amplitude_scale_is_sqrt_and_registered_r1_r2_identity() -> None:
    coefficients = torch.zeros((2, 100), dtype=torch.complex64)
    coefficients[:, :2] = torch.tensor(
        [[1.0 + 2.0j, 2.0 - 1.0j], [3.0 + 0.5j, -1.0 + 4.0j]],
        dtype=torch.complex64,
    )
    scale = torch.ones(100, dtype=torch.float32)
    scale[:2] = torch.tensor([4.0, 9.0])
    sqrt_scale = validate_amplitude_scale(scale)
    assert torch.equal(sqrt_scale[:2], torch.tensor([2.0, 3.0]))
    representations = normalized_representations(coefficients, scale)
    assert torch.allclose(
        representations[R1_POWER], representations[R2_AMPLITUDE].square()
    )
    assert torch.allclose(
        representations[R1_POWER], representations[R3_COMPLEX].abs().square()
    )
    assert torch.allclose(
        representations["R2_RAW_AMPLITUDE"], coefficients.abs()
    )


def test_gauge_distance_is_phase_invariant_symmetric_and_self_zero() -> None:
    coefficients = torch.zeros((2, 100), dtype=torch.complex64)
    coefficients[:, :2] = torch.tensor(
        [[1.0 + 2.0j, 2.0 - 1.0j], [3.0 + 0.5j, -1.0 + 4.0j]],
        dtype=torch.complex64,
    )
    phase = torch.exp(1j * torch.tensor([0.7, -1.1]))
    transformed = coefficients * phase[:, None]
    distance = gauge_distance_squared(coefficients, coefficients)
    transformed_distance = gauge_distance_squared(transformed, transformed)
    assert torch.allclose(distance, distance.T, atol=1.0e-5)
    assert torch.allclose(torch.diag(distance), torch.zeros(2, dtype=torch.float64), atol=1.0e-5)
    assert torch.allclose(distance, transformed_distance, atol=1.0e-5)
    audit = phase_gauge_audit(coefficients)
    assert audit["classification"] == "GLOBAL_PHASE_GAUGE_CONFIRMED"
    assert audit["distance_invariant"] is True
    assert audit["power_invariant"] is True


def test_one_nn_excludes_self_and_reports_confusion() -> None:
    support = torch.tensor([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0]])
    labels = torch.tensor([0, 1, 2])
    metrics = one_nn_metrics(
        support,
        support,
        labels,
        labels,
        metric="euclidean",
        exclude_self=True,
    )
    assert metrics["self_excluded"] is True
    assert metrics["nearest_indices"] == [1, 0, 1]
    assert metrics["confusion"][0][1] == 1
    assert metrics["confusion"][2][1] == 1


def test_fixed_null_and_stratified_bootstrap_are_deterministic() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    first = fixed_label_permutation(labels, seed=2053)
    second = fixed_label_permutation(labels, seed=2053)
    assert torch.equal(first, second)
    assert torch.equal(torch.bincount(first), torch.bincount(labels))
    real = torch.ones(labels.shape[0], dtype=torch.float64)
    shuffled = torch.zeros(labels.shape[0], dtype=torch.float64)
    first_ci = stratified_bootstrap_ci(real, shuffled, labels, reps=200, seed=1236)
    second_ci = stratified_bootstrap_ci(real, shuffled, labels, reps=200, seed=1236)
    assert first_ci == second_ci
    assert first_ci["lower95"] > 0.0


def test_ridge_probe_standardizes_fit_only_and_predicts_without_query_leakage() -> None:
    fit_features = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]],
        dtype=torch.float32,
    )
    fit_labels = torch.tensor([0, 0, 1, 1])
    probe = fit_ridge_probe(fit_features, fit_labels, ridge_lambda=1.0)
    assert torch.allclose(probe["mean"], fit_features.double().mean(dim=0))
    assert probe["standardization_source"] == "fit_reference_only"
    predictions = predict_ridge_probe(probe, fit_features)
    assert predictions.tolist() == fit_labels.tolist()


def test_representation_gate_and_minimal_selection_are_fail_closed() -> None:
    power = representation_gate(
        geometry_confirmation_lower95=0.1,
        geometry_selection_delta=0.2,
        probe_confirmation_lower95=0.05,
        probe_selection_delta=0.1,
        physical_contract_valid=True,
        linear_probe_authoritative=True,
    )
    amplitude = representation_gate(
        geometry_confirmation_lower95=0.1,
        geometry_selection_delta=0.2,
        probe_confirmation_lower95=0.05,
        probe_selection_delta=0.1,
        physical_contract_valid=True,
        linear_probe_authoritative=True,
    )
    assert power["supported"] is True
    assert minimal_representation(
        {R1_POWER: power, R2_AMPLITUDE: amplitude, R3_COMPLEX: {"supported": True}}
    ) == S1_POWER_SEMANTICS_SUPPORTED
    assert minimal_representation(
        {R1_POWER: {"supported": False}, R2_AMPLITUDE: amplitude, R3_COMPLEX: {"supported": True}}
    ) == S1_AMPLITUDE_SEMANTICS_SUPPORTED
    assert minimal_representation(
        {R1_POWER: {"supported": False}, R2_AMPLITUDE: {"supported": False}, R3_COMPLEX: {"supported": True}}
    ) == "S1_COMPLEX_SEMANTICS_SUPPORTED"
    assert classify_s1(
        {R1_POWER: {"supported": False}, R2_AMPLITUDE: {"supported": False}, R3_COMPLEX: {"supported": False}},
        measurement_valid=False,
        physical_contract_valid=True,
    ) == "S1_SEMANTICS_UNDERDETERMINED"


def test_phase_increment_requires_paired_confirmation_and_selection_direction() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    amplitude_selection = torch.zeros(8, dtype=torch.float64)
    complex_selection = torch.ones(8, dtype=torch.float64)
    amplitude_confirmation = torch.zeros(8, dtype=torch.float64)
    complex_confirmation = torch.ones(8, dtype=torch.float64)
    result = phase_increment_classification(
        amplitude_selection,
        complex_selection,
        amplitude_confirmation,
        complex_confirmation,
        labels,
        reps=200,
        seed=1236,
    )
    assert result["classification"] == PHASE_SEMANTIC_INCREMENT_SUPPORTED
    not_supported = phase_increment_classification(
        complex_selection,
        amplitude_selection,
        complex_confirmation,
        amplitude_confirmation,
        labels,
        reps=200,
        seed=1236,
    )
    assert not_supported["classification"] == PHASE_SEMANTIC_INCREMENT_NOT_SUPPORTED


def test_s1_loader_rejects_schema_task_or_checkpoint_identity_mismatch(tmp_path: Path) -> None:
    source = Path(CONFIG_PATH)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    assert payload["schema"] == CONFIG_SCHEMA
    for key, value in (("schema", "wrong"), ("task_id", "wrong"), ("hypothesis", "wrong")):
        candidate = dict(payload)
        candidate[key] = value
        target = tmp_path / f"{key}.yaml"
        target.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(target)
    candidate = dict(payload)
    candidate["checkpoint"] = dict(payload["checkpoint"])
    candidate["checkpoint"]["sha256"] = "0" * 64
    target = tmp_path / "checkpoint.yaml"
    target.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(target)


def test_write_once_output_advances_without_overwrite(tmp_path: Path) -> None:
    requested = tmp_path / "semantic_audit_attempt_01"
    first = next_write_once_output(requested)
    first.mkdir()
    second = next_write_once_output(requested)
    assert first.name == "semantic_audit_attempt_01"
    assert second.name == "semantic_audit_attempt_02"
    classify_s1,
