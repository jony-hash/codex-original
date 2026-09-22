from __future__ import annotations

from pathlib import Path

import torch
import pytest

from scripts.research import execute_lightgen_2d_real_vs_sampled_ols_semantic_manifold_gap_v1 as diagnostic


def test_sampled_probe_report_never_adds_ground_truth_accuracy() -> None:
    logits = torch.zeros((20, 10), dtype=torch.float64)
    result = diagnostic._probe_distribution(logits, labels=None)
    assert result["labels_consumed"] is False
    assert "accuracy" not in result
    assert result["represented_class_count"] == 1
    assert result["top1_confidence"]["mean"] == pytest.approx(0.1)


def test_nearest_rows_returns_exact_distance_and_index() -> None:
    query = torch.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=torch.float64)
    reference = torch.tensor([[1.0, 0.0], [3.0, 3.0], [10.0, 10.0]], dtype=torch.float64)
    distances, indices = diagnostic._nearest_rows(query, reference)
    assert torch.equal(indices, torch.tensor([0, 1]))
    assert torch.allclose(distances, torch.tensor([1.0, 1.0], dtype=torch.float64))


def test_component_geometry_keeps_sampled_labels_descriptive() -> None:
    train = torch.eye(10, dtype=torch.float64)
    labels = torch.arange(10, dtype=torch.long)
    dev = train.clone()
    sampled = train[:3].clone()
    predictions = torch.tensor([0, 1, 2], dtype=torch.long)
    report = diagnostic._component_geometry_with_predictions(train, labels, dev, labels, sampled, labels, predictions)
    assert report["real_dev"]["labels_consumed"] is True
    assert report["sampled"]["labels_consumed"] is False
    assert "nearest_centroid_true_class_fraction" not in report["sampled"]


def test_consumer_source_has_no_update_or_resampling_boundary() -> None:
    source = Path(diagnostic.__file__).read_text(encoding="utf-8")
    assert ".backward(" not in source
    assert "torch.autograd.grad" not in source
    assert "torch.optim" not in source
    assert "randn" not in source
    assert "randn_like" not in source
    assert "--execute" in source
