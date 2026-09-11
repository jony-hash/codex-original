from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.research import diagnose_lightgen_2d_semantic_real_path_control_failure_v1 as diagnosis


def test_balanced_diagnostic_panel_is_deterministic_and_class_balanced() -> None:
    labels = torch.arange(10).repeat_interleave(20)
    role_indices = torch.arange(labels.numel(), dtype=torch.long)

    first = diagnosis._balanced_panel(role_indices, labels, per_class=3, seed=17)
    second = diagnosis._balanced_panel(role_indices, labels, per_class=3, seed=17)

    assert torch.equal(first, second)
    assert first.shape == (30,)
    assert torch.equal(torch.bincount(labels.index_select(0, first), minlength=10), torch.full((10,), 3))


def test_domain_accumulator_reports_preprocessing_boundary_statistics() -> None:
    values = torch.tensor([[[[0.0, 1.0], [0.25, 0.75]]]], dtype=torch.float32)

    accumulator = diagnosis.ImageDomainAccumulator()
    accumulator.update(values)
    result = accumulator.finalize(preprocessing="none")

    assert result["count"] == 1
    assert result["pixel_count"] == 4
    assert result["min"] == 0.0
    assert result["max"] == 1.0
    assert result["zero_fraction"] == 0.25
    assert result["saturation_fraction_ge_1"] == 0.25
    assert result["preprocessing_before_this_measurement"] == "none"


def test_evaluator_accumulator_uses_actual_rows_and_predicted_columns() -> None:
    accumulator = diagnosis.EvaluatorAccumulator()
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(2)
    predictions = labels.clone()
    predictions[1] = 1
    confidence = torch.linspace(0.6, 0.9, labels.numel(), dtype=torch.float32)

    accumulator.update(labels, predictions, confidence)
    result = accumulator.finalize(labels)

    assert result["accuracy"] == pytest.approx(0.95)
    assert result["correct_by_class"][:2] == [1, 2]
    assert result["confusion_matrix_actual_rows_predicted_columns"][0][:2] == [1, 1]
    assert result["confusion_matrix_actual_rows_predicted_columns"][1][:2] == [0, 2]
    assert result["confidence"]["count"] == 20


def test_latent_diagnostics_preserve_complex_ols_and_class_separation() -> None:
    labels = torch.arange(10).repeat_interleave(2)
    z = torch.zeros((20, 100), dtype=torch.complex64)
    z[torch.arange(20), labels] = 1.0e-3
    measurement = diagnosis.RoleMeasurement(
        role="diagnostic",
        indices=torch.arange(20),
        labels=labels,
        z=z,
        original_evaluator={},
        real_path_evaluator={},
        original_domain={},
        raw_domain={},
        target_domain={},
        reconstruction={},
        path_proof={},
    )

    result = diagnosis._latent_diagnostics((measurement,))

    assert result["sample_count"] == 20
    assert len(result["per_port_variance"]["power"]) == 100
    assert len(result["covariance_spectrum"]["gauge_invariant_real_200d"]) == 200
    assert result["class_centroid_separation"]["class_counts"] == [2] * 10


def test_authority_digest_mismatch_is_rejected_before_consumption() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with pytest.raises(diagnosis.BlockedDiagnosis):
        diagnosis._ref(repo_root, Path("docs/CURRENT_STATUS.md"), "authority", "0" * 64)


def test_primary_classification_is_unresolved_without_supported_mechanism() -> None:
    mechanisms = {
        "A": {"status": "unresolved"},
        "B": {"status": "not_supported"},
        "J_mixed_or_unresolved": {"status": "unresolved"},
    }

    assert diagnosis._primary_classification(mechanisms) == diagnosis.UNRESOLVED_CLASSIFICATION
