from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.research import execute_lightgen_2d_ols_to_cmos_semantic_information_localization_v1 as localization


def test_execution_plan_freezes_single_gate_and_rejects_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "outputs" / "run"
    target.parent.mkdir()
    plan = localization.build_execution_plan(repo_root=tmp_path, output_dir=target)
    assert plan["outer_gate"] == localization.OUTER_GATE
    assert plan["train_forward_rows"] == localization.TRAIN_COUNT
    assert plan["optical_updates"] is False
    assert plan["probe"]["C"] == 1.0
    target.mkdir()
    with pytest.raises(localization.LocalizationError, match="already exists"):
        localization.build_execution_plan(repo_root=tmp_path, output_dir=target)


def test_primary_feature_contracts_are_deterministic() -> None:
    z = torch.zeros((1, 100), dtype=torch.complex64)
    z[0, :2] = torch.tensor([1.0 + 2.0j, -3.0 + 4.0j])
    expected = torch.zeros((1, 200), dtype=torch.float64)
    expected[0, :2] = torch.tensor([1.0, -3.0])
    expected[0, 100:102] = torch.tensor([2.0, 4.0])
    assert torch.equal(localization._ols_features(z), expected)
    raw = torch.zeros((1, 1, 200, 200), dtype=torch.float32)
    raw[:, :, 0, 0] = 1.0
    features = localization._cmos_features(raw)
    assert features.shape == (1, 28 * 28)
    assert float(features.max()) > 0.0
    assert float(features.min()) >= 0.0


def test_streaming_tensor_hash_matches_project_hash() -> None:
    value = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5)
    assert localization._tensor_sha256_streaming(value, rows_per_chunk=1) == localization.tensor_sha256(value)


def test_logistic_probe_and_metrics_are_finite_on_a_balanced_fixture() -> None:
    generator = torch.Generator(device="cpu").manual_seed(7)
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(4)
    features = torch.randn((labels.numel(), 3), generator=generator, dtype=torch.float64)
    features[:, 0] += labels.to(torch.float64) * 3.0
    model, solver = localization._fit_logistic(features, labels)
    evaluation = localization._evaluate_probe(model, features, labels, features, labels)
    assert solver["C"] == 1.0
    assert evaluation["dev"]["rows"] == 40
    assert len(evaluation["dev"]["per_class_recall"]) == 10
    assert torch.isfinite(model["weights"]).all()


def test_matched_transition_counts_partition_dev_rows() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    ols = torch.tensor([0, 1, 1, 0, 2, 0])
    cmos = torch.tensor([0, 0, 0, 1, 2, 2])
    smooth = torch.tensor([0, 0, 1, 1, 0, 2])
    transitions = localization._matched_transitions(labels, ols, cmos, smooth)
    assert sum(transitions[key] for key in ("ols_correct_cmos_correct", "ols_correct_cmos_wrong", "ols_wrong_cmos_correct", "ols_wrong_cmos_wrong")) == 6
    assert sum(transitions["ols_to_cmos_to_smooth_triple_counts"].values()) == 6
    assert len(transitions["per_class"]) == 10
