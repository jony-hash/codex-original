from __future__ import annotations

from pathlib import Path

import torch

from scripts.research.lightgen import execute_lightgen_2d_assess_step6000_conditional_generation_readiness_v1 as audit


def test_consensus_reports_requested_class_and_pairwise_agreement() -> None:
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(2)
    predictions = {
        "E_S": labels.clone(),
        "E_I": labels.clone(),
        "E_C": labels.clone(),
        "E_D": (labels + 1) % 10,
    }
    result = audit._consensus_report(predictions, labels)
    assert result["4_of_4_agreement_fraction"] == 0.0
    assert result["at_least_3_of_4_agreement_fraction"] == 1.0
    assert result["4_of_4_true_or_requested_fraction"] == 0.0
    assert result["at_least_3_of_4_true_or_requested_fraction"] == 1.0
    assert result["pairwise_prediction_agreement_matrix"][0][0] == 1.0


def test_transition_categories_preserve_same_sample_identity() -> None:
    labels = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    before = torch.tensor([9, 1, 8, 3, 6], dtype=torch.long)
    after = torch.tensor([0, 8, 7, 3, 6], dtype=torch.long)
    result = audit._transition_categories(before, after, labels, label_source="requested class")
    categories = result["categories"]
    assert categories["wrong_to_correct"]["count"] == 1
    assert categories["correct_to_wrong"]["count"] == 1
    assert categories["wrong_to_different_wrong"]["count"] == 1
    assert categories["stable_correct"]["count"] == 1
    assert categories["stable_wrong"]["count"] == 1


def test_readiness_executor_has_no_update_boundary() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "torch.autograd.grad" not in source
    assert "--execute" in source
    assert audit.AUTHORIZATION_TOKEN == "AUTHORIZED_ASSESS_STEP6000_CONDITIONAL_GENERATION_READINESS_V1"
