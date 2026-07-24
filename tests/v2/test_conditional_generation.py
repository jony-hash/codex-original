from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from ldm_project.v2.config import load_config
from ldm_project.v2.student.checkpoints import ConditionalCheckpointManifestV2, load_conditional_checkpoint, save_conditional_checkpoint, state_content_hash, state_signature
from ldm_project.v2.student.conditional import conditional_sinkhorn_loss, fixed_exposure_view
from ldm_project.v2.student.conditional_acceptance import chunked_nearest_distances, memorization_statistics, paired_bootstrap_lower_bound, per_label_feature_diversity_ratio
from ldm_project.v2.student.generator import ConditionalGlobalAmplitudeGenerator
from ldm_project.v2.evidence.reviews import ConditionalManualReviewV2


def test_conditional_generator_is_bounded_zero_code_carrier_and_global() -> None:
    generator = ConditionalGlobalAmplitudeGenerator(shape=(8, 8))
    labels = torch.tensor([2, 7])
    zero = torch.zeros(2, 8)
    command = generator(zero, labels)
    assert command.shape == (2, 1, 8, 8)
    assert bool(((command > 0.0) & (command < 1.0)).all())
    assert torch.equal(command, generator.class_carrier_command(labels))
    changed = generator(torch.tensor([[0.8] * 8, [-0.8] * 8]), labels)
    assert not torch.equal(changed, command)
    assert generator.mode_orthogonality_penalty().isfinite()


def test_fixed_exposure_view_is_not_sample_normalization() -> None:
    raw = torch.stack((torch.ones(1, 56, 56), torch.full((1, 56, 56), 0.5)))
    view = fixed_exposure_view(raw, 2.0)
    assert view.shape == (2, 1, 28, 28)
    assert torch.allclose(view[0], torch.full_like(view[0], 0.5))
    assert torch.allclose(view[1], torch.full_like(view[1], 0.25))


def test_conditional_sinkhorn_has_no_sample_index_pairing() -> None:
    labels = torch.arange(10).repeat_interleave(2)
    generated_features = torch.randn(20, 4)
    generated_view = torch.rand(20, 1, 14, 14)
    real_features = torch.randn(20, 4)
    real_view = torch.rand(20, 1, 14, 14)
    loss = conditional_sinkhorn_loss(generated_features, generated_view, real_features, real_view, labels, epsilon=0.05, iterations=5)
    permutation = torch.cat([torch.nonzero(labels == label).flatten().flip(0) for label in range(10)])
    shuffled = conditional_sinkhorn_loss(generated_features, generated_view, real_features.index_select(0, permutation), real_view.index_select(0, permutation), labels, epsilon=0.05, iterations=5)
    assert torch.allclose(loss, shuffled)


def test_conditional_checkpoint_rejects_architecture_mismatch(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    manifest = ConditionalCheckpointManifestV2(2, "conditional_global_amplitude_v1", "system", "config", "split", "evaluator", 1.0, state_signature(model.state_dict()), state_content_hash(model.state_dict()))
    checkpoint = tmp_path / "conditional.pt"
    save_conditional_checkpoint(checkpoint, model, manifest)
    target = nn.Linear(2, 2)
    load_conditional_checkpoint(checkpoint, target, manifest)
    incompatible = ConditionalCheckpointManifestV2(2, "conditional_global_amplitude_v1", "other-system", "config", "split", "evaluator", 1.0, manifest.state_signature, manifest.state_content_hash)
    with pytest.raises(ValueError, match="does not match"):
        load_conditional_checkpoint(checkpoint, target, incompatible)


def test_conditional_config_rejects_teacher_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """schema_version: 2
architecture: conditional_global_amplitude_v1
system:
  kind: formal_amplitude_finite_crop_identity
conditional_generation: {}
teacher:
  first_stage_checkpoint: first.pt
  diffusion_checkpoint: diffusion.pt
run:
  run_name: unit
  evidence_tier: diagnostic
  device: cpu
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rejects legacy fields"):
        load_config(path)


def test_conditional_manual_review_requires_each_label_to_have_sixteen_slots() -> None:
    review = ConditionalManualReviewV2(review_kind="conditional_generation", reviewed_tensor_sha256="raw", reviewed_preview_sha256="preview", samples_per_label=16, readable_counts=(14,) * 10, reviewer="unit")
    review.validate_contract(tensor_hash="raw", preview_hash="preview")
    with pytest.raises(ValueError, match="sixteen"):
        ConditionalManualReviewV2(review_kind="conditional_generation", reviewed_tensor_sha256="raw", reviewed_preview_sha256="preview", samples_per_label=15, readable_counts=(14,) * 10, reviewer="unit").validate_contract(tensor_hash="raw", preview_hash="preview")


def test_diversity_and_chunked_nearest_neighbor_metrics() -> None:
    labels = torch.arange(10).repeat_interleave(2)
    real = torch.arange(80, dtype=torch.float32).reshape(20, 4)
    generated = real * 1.1
    ratios = per_label_feature_diversity_ratio(generated, real, labels)
    assert all(1.09 < ratio < 1.11 for ratio in ratios)
    distances, indices = chunked_nearest_distances(real[:3], real, chunk_size=1)
    assert torch.allclose(distances, torch.zeros(3))
    assert indices.tolist() == [0, 1, 2]


def test_memorization_copy_is_flagged_and_bootstrap_is_deterministic() -> None:
    real_labels = torch.arange(10).repeat_interleave(4)
    real = torch.arange(160, dtype=torch.float32).reshape(40, 4)
    generated_labels = torch.arange(10)
    generated = torch.stack([real[real_labels == label][0] for label in range(10)])
    report = memorization_statistics(generated, generated_labels, real, real_labels)
    assert report["overall_anomalous_rate"] == 1.0
    values = torch.tensor([1.0, -1.0, 1.0, -1.0])
    assert paired_bootstrap_lower_bound(values, seed=1234, resamples=20) == paired_bootstrap_lower_bound(values, seed=1234, resamples=20)
