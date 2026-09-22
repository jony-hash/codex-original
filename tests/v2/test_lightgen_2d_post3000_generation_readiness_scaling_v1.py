from __future__ import annotations

from pathlib import Path

import torch

from scripts.research import execute_lightgen_2d_post3000_generation_readiness_scaling_v1 as audit


def test_confusion_and_semantic_summary_use_fixed_ten_class_contract() -> None:
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(500)
    logits = torch.full((5000, 10), -4.0)
    logits[torch.arange(5000), labels] = 4.0
    result = audit._semantic_sample_report(logits, labels=labels)
    assert result["accuracy"] == 1.0
    assert result["macro_recall"] == 1.0
    assert result["represented_class_count"] == 10
    assert result["confusion_matrix"] == (500 * torch.eye(10, dtype=torch.long)).tolist()


def test_feature_centroid_mapping_is_descriptive_and_deterministic() -> None:
    labels = torch.arange(10, dtype=torch.long)
    target = torch.eye(10, 128, dtype=torch.float32)
    report, centroids = audit._feature_realization_report(target, target, labels, include_pair=True)
    assert report["matched_nearest_fraction"] == 1.0
    assert report["nearest_class_histogram"] == [1] * 10
    assert report["paired_target_cmos_feature_distance"]["summary"]["mean"] == 0.0
    assert centroids.shape == (10, 128)


def test_sampled_summary_does_not_introduce_ground_truth_accuracy() -> None:
    logits = torch.zeros((20, 10), dtype=torch.float32)
    result = audit._semantic_sample_report(logits, labels=None)
    assert result["labels_consumed"] is False
    assert "accuracy" not in result
    assert result["represented_class_count"] == 1


def test_consumer_source_has_no_update_boundary() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert ".backward(" not in source
    assert "torch.autograd.grad" not in source
    assert "torch.optim" not in source
    assert "--execute" in source
