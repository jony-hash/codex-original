from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.research.lightgen import execute_lightgen_2d_resolve_cmos_to_smooth_semantic_realization_v1 as resolution


def test_execution_plan_freezes_single_gate_and_rejects_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "outputs" / "run"
    target.parent.mkdir()
    plan = resolution.build_execution_plan(repo_root=tmp_path, output_dir=target)
    assert plan["authorization_token"] == resolution.AUTHORIZATION_TOKEN
    assert plan["outer_gate"] == resolution.OUTER_GATE
    assert plan["optical_forward"] is False
    assert plan["optical_updates"] == 0
    target.mkdir()
    with pytest.raises(resolution.SemanticResolutionError, match="already exists"):
        resolution.build_execution_plan(repo_root=tmp_path, output_dir=target)


def test_smooth_preprocess_is_fixed_for_real_and_cmos_shapes() -> None:
    real = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    real[:, :, 0, 0] = 2.0
    real_view = resolution._cmos_image_view(real)
    assert real_view.shape == (1, 1, 28, 28)
    assert float(real_view.max()) == 1.0

    cmos = torch.zeros((1, 1, 200, 200), dtype=torch.float32)
    cmos[:, :, 0, 0] = 2.0
    cmos_view = resolution._cmos_image_view(cmos)
    assert cmos_view.shape == (1, 1, 28, 28)
    assert float(cmos_view.max()) <= 1.0
    assert float(cmos_view.min()) >= 0.0


def test_reference_indices_are_same_class_or_next_class_without_self() -> None:
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(2)
    same = resolution._same_class_reference_indices(labels)
    different = resolution._different_class_reference_indices(labels)
    assert torch.equal(labels, labels.index_select(0, same))
    assert bool((same != torch.arange(labels.numel())).all())
    assert bool((labels != labels.index_select(0, different)).all())
    assert labels.index_select(0, different).tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 0, 0]


def test_feature_alignment_uses_fixed_normalized_euclidean_geometry() -> None:
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(2)
    real = torch.zeros((20, resolution.FEATURE_DIMENSION), dtype=torch.float32)
    cmos = torch.zeros_like(real)
    for index, label in enumerate(labels.tolist()):
        real[index, label] = 1.0
        cmos[index, label] = 1.0
    alignment = resolution._feature_alignment(real, cmos, labels)
    assert alignment["distance_convention"].startswith("L2-normalize")
    assert alignment["paired_target_cmos_distance"]["mean"] == pytest.approx(0.0)
    assert alignment["centroid_matched_nearest_fraction"] == pytest.approx(1.0)


def test_metrics_and_matched_transitions_partition_rows() -> None:
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(2)
    probe = labels.clone()
    smooth = (labels + 1) % 10
    metrics = resolution._metrics(probe, labels)
    transitions = resolution._matched_transitions(labels, probe, smooth)
    assert metrics["macro_recall"] == pytest.approx(1.0)
    assert sum(transitions[key] for key in ("probe_correct_smooth_correct", "probe_correct_smooth_wrong", "probe_wrong_smooth_correct", "probe_wrong_smooth_wrong")) == labels.numel()
    assert transitions["probe_correct_smooth_wrong"] == labels.numel()


def test_classification_uses_centroid_mismatch_as_feature_geometry_evidence() -> None:
    transitions = {
        "probe_correct_smooth_wrong": 4359,
        "probe_wrong_smooth_correct": 20,
    }
    alignment = {
        "paired_target_cmos_distance": {"mean": 1.01},
        "same_class_different_sample_reference": {"mean": 1.04},
        "centroid_matched_nearest_fraction": 0.1,
    }
    result = resolution._classify(
        {"macro_recall": 0.988},
        {"macro_recall": 0.103},
        {"macro_recall": 0.9708},
        transitions,
        alignment,
        contract_mismatch=False,
    )
    assert result["classification"] == "CMOS_CLASS_CODE_PRESERVED_BUT_IMAGE_SEMANTIC_REALIZATION_FAILED"
    assert result["evidence_flags"]["centroid_alignment_weak"] is True
