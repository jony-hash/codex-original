from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import yaml

from scripts.research.scaled_chen import requalify_scaled_chen_output_semantic_evaluator as osera


def _balanced_labels(count_per_class: int = 4) -> torch.Tensor:
    return torch.arange(10, dtype=torch.long).repeat_interleave(count_per_class)


def test_per_class_margin_formula_and_dominance() -> None:
    labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    predictions = torch.tensor([0, 0, 1, 1, 1, 0], dtype=torch.long)

    result = osera.compute_per_class_margins(labels, predictions, class_count=2)

    assert result["classes"]["0"]["p_correct"] == pytest.approx(2 / 3)
    assert result["classes"]["0"]["p_wrong_max"] == pytest.approx(1 / 3)
    assert result["classes"]["0"]["margin"] == pytest.approx(1 / 3)
    assert result["classes"]["0"]["dominant_prediction"] == 0
    assert result["classes"]["0"]["diagonal_unique_maximum"] is True


def test_class_collapse_is_rejected() -> None:
    labels = _balanced_labels(3)
    predictions = (labels + 1) % 10

    result = osera.compute_per_class_margins(labels, predictions)

    assert result["point_gate"] is False
    assert result["class_collapse"] is True
    assert result["classes"]["0"]["margin"] < 0


def test_stratified_margin_bootstrap_is_deterministic_and_has_fixed_contract() -> None:
    labels = _balanced_labels(4)
    predictions = labels.clone()

    first = osera.stratified_margin_bootstrap(
        labels, predictions, reps=200, seed=1236
    )
    second = osera.stratified_margin_bootstrap(
        labels, predictions, reps=200, seed=1236
    )

    assert first == second
    assert first["seed"] == 1236
    assert first["replicates"] == 200
    assert first["confidence_level"] == 0.95
    assert all(
        value["lower95"] == pytest.approx(1.0)
        for value in first["classes"].values()
    )


def test_global_null_uses_label_only_permutation_and_is_deterministic() -> None:
    labels = _balanced_labels(3)
    predictions = labels.clone()
    permutation = torch.roll(torch.arange(labels.numel(), dtype=torch.long), 3)

    first = osera.compute_global_shuffled_null(
        labels, predictions, permutation, reps=200, seed=1236
    )
    second = osera.compute_global_shuffled_null(
        labels, predictions, permutation, reps=200, seed=1236
    )

    assert first == second
    assert first["real_accuracy"] == 1.0
    assert first["shuffled_accuracy"] == 0.0
    assert first["point_delta"] == 1.0
    assert first["lower95"] > 0.0
    assert first["class_count_multiset_preserved"] is True


def test_requalification_classification_has_no_absolute_accuracy_threshold() -> None:
    passed = osera.classify_requalification(
        authority_binding=True,
        clean_provenance=True,
        per_class_gate=True,
        global_null_gate=True,
        metrics_finite=True,
        error_model_frozen=True,
        evaluator_state_unchanged=True,
    )
    assert passed == {
        "gate": osera.PASS_GATE,
        "classification": osera.PASS_CLASSIFICATION,
        "reason": None,
        "new_absolute_accuracy_threshold": "NONE",
    }


def test_requalification_failure_routes_are_fail_closed() -> None:
    per_class_failure = osera.classify_requalification(
        authority_binding=True,
        clean_provenance=True,
        per_class_gate=False,
        global_null_gate=True,
        metrics_finite=True,
        error_model_frozen=True,
        evaluator_state_unchanged=True,
    )
    null_failure = osera.classify_requalification(
        authority_binding=True,
        clean_provenance=True,
        per_class_gate=True,
        global_null_gate=False,
        metrics_finite=True,
        error_model_frozen=True,
        evaluator_state_unchanged=True,
    )
    binding_failure = osera.classify_requalification(
        authority_binding=False,
        clean_provenance=True,
        per_class_gate=True,
        global_null_gate=True,
        metrics_finite=True,
        error_model_frozen=True,
        evaluator_state_unchanged=True,
    )

    assert per_class_failure["reason"] == "PER_CLASS_IDENTIFIABILITY"
    assert null_failure["reason"] == "GLOBAL_NULL_SEPARATION"
    assert binding_failure["classification"] == osera.BINDING_FAILURE


def test_error_model_freezes_raw_predictions_without_correction() -> None:
    labels = _balanced_labels(2)
    predictions = labels.clone()

    error_model = osera.build_error_model(labels, predictions)

    assert error_model["schema"] == osera.ERROR_MODEL_SCHEMA
    assert error_model["overall_error_rate"] == 0.0
    assert error_model["prediction_correction_applied"] is False
    assert error_model["posthoc_calibration_applied"] is False
    assert error_model["raw_predictions_authoritative"] is True


def test_config_binds_registered_seeds_and_no_new_accuracy_threshold() -> None:
    config = osera.load_config(osera.CONFIG_PATH)

    assert config["statistics"]["bootstrap_seed"] == 1236
    assert config["statistics"]["bootstrap_replicates"] == 2000
    assert config["statistics"]["null_seed"] == 2053
    assert config["statistics"]["absolute_accuracy_threshold"] is None


def test_registered_vpim_authority_requalifies_for_paired_s2(tmp_path: Path) -> None:
    result = osera.run(output_path=tmp_path / "requalification_attempt_01")

    assert result["classification"] == osera.PASS_CLASSIFICATION
    assert result["primary_gate"] == osera.PASS_GATE
    assert result["prediction_count"] == 5000
    assert result["accuracy"] == 0.9712
    assert result["error_count"] == 144
    assert result["per_class_minimum_lower95"] > 0.0
    assert result["global_null"]["real_accuracy"] == 0.9712
    assert result["global_null"]["shuffled_accuracy"] == 0.1004
    assert result["global_null"]["point_delta"] == pytest.approx(0.8708)
    assert result["global_null"]["lower95"] > 0.0
    assert result["next_action"] == osera.NEXT_ACTION_PASS


def test_config_rejects_prediction_authority_hash_mismatch(tmp_path: Path) -> None:
    config = osera.load_config(osera.CONFIG_PATH)
    config["source"] = dict(config["source"])
    config["source"]["vpim_manifest_sha256"] = "0" * 64
    target = tmp_path / "tampered.yaml"
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="config vpim_manifest_sha256"):
        osera.load_config(target)


def test_write_once_output_selects_next_attempt(tmp_path: Path) -> None:
    requested = tmp_path / "authority_attempt_01"
    requested.mkdir()

    selected = osera.next_write_once_output(requested)

    assert selected.name == "authority_attempt_02"
    assert not selected.exists()


def test_requalification_script_does_not_execute_a_model() -> None:
    source = inspect.getsource(osera)

    assert "FrozenMNISTEvaluator" not in source
    assert "load_frozen_evaluator" not in source
    assert "optimizer.step" not in source
    assert "torch.cuda" not in source
