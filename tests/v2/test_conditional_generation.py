from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from ldm_project.v2.config import load_config
from ldm_project.v2.student.admission import balanced_code_fraction
from ldm_project.v2.student.checkpoints import ConditionalCheckpointManifestV2, load_conditional_checkpoint, save_conditional_checkpoint, state_content_hash, state_signature
from ldm_project.v2.student.conditional import conditional_sinkhorn_loss, fixed_exposure_view, sinkhorn_cost_view
from ldm_project.v2.student.conditional_acceptance import chunked_nearest_distances, memorization_statistics, paired_bootstrap_lower_bound, per_label_feature_diversity_ratio
from ldm_project.v2.student.generator import ConditionalGlobalAmplitudeGenerator, IdentityAnchoredConditionalAmplitudeGenerator
from ldm_project.v2.evidence.reviews import ConditionalManualReviewV2
from ldm_project.v2.application.conditional import _first_hundred_schedule_hash, _identity_isolation_classification


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


def test_identity_anchored_generator_has_zero_code_identity_and_zero_wy() -> None:
    generator = IdentityAnchoredConditionalAmplitudeGenerator(shape=(8, 8))
    labels = torch.tensor([2, 7])
    zero = torch.zeros(2, 8)
    command = generator(zero, labels)
    assert command.shape == (2, 1, 8, 8)
    assert torch.allclose(command, generator.class_carrier_command(labels), atol=0.0, rtol=0.0)
    assert torch.count_nonzero(generator.Wy) == 0
    assert generator.command_gain == pytest.approx(0.5)
    assert generator.Wz.bias is None


def test_identity_anchored_generator_preserves_eight_code_directions() -> None:
    generator = IdentityAnchoredConditionalAmplitudeGenerator(shape=(12, 12))
    labels = torch.zeros(16, dtype=torch.long)
    codes = torch.cat((torch.eye(8), -torch.eye(8)))
    commands = generator(codes, labels)
    # Wy=0 at initialization means the coefficient response is tanh(z), so
    # each signed basis direction remains observable before any training.
    plus_minus = commands[:8].flatten(1) - commands[8:].flatten(1)
    assert torch.linalg.matrix_rank(plus_minus) == 8
    assert torch.isfinite(commands).all()


def test_identity_anchored_generator_initialization_is_deterministic() -> None:
    torch.manual_seed(1234)
    first = IdentityAnchoredConditionalAmplitudeGenerator(shape=(8, 8))
    torch.manual_seed(1234)
    second = IdentityAnchoredConditionalAmplitudeGenerator(shape=(8, 8))
    assert all(torch.equal(first_state, second_state) for first_state, second_state in zip(first.state_dict().values(), second.state_dict().values()))


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


def test_conditional_sinkhorn_normalizes_features_and_pools_intensity_cost() -> None:
    labels = torch.arange(10).repeat_interleave(2)
    generated_features = torch.randn(20, 8, generator=torch.Generator().manual_seed(21))
    real_features = torch.randn(20, 8, generator=torch.Generator().manual_seed(22))
    generated_view = torch.rand(20, 1, 28, 28, generator=torch.Generator().manual_seed(23))
    real_view = torch.rand(20, 1, 28, 28, generator=torch.Generator().manual_seed(24))
    loss = conditional_sinkhorn_loss(generated_features, generated_view, real_features, real_view, labels, epsilon=0.05, iterations=5)
    scaled = conditional_sinkhorn_loss(17.0 * generated_features, 3.0 * generated_view, 0.25 * real_features, 3.0 * real_view, labels, epsilon=0.05, iterations=5)
    # Feature normalization makes global feature scale irrelevant; pooled
    # intensity costs remain finite and preserve the expected 14x14 contract.
    assert sinkhorn_cost_view(generated_view).shape == (20, 1, 14, 14)
    assert torch.isfinite(loss)
    assert torch.isfinite(scaled)
    assert not torch.allclose(loss, scaled)
    feature_only = conditional_sinkhorn_loss(generated_features, generated_view, real_features, real_view, labels, epsilon=0.05, iterations=5)
    feature_scaled = conditional_sinkhorn_loss(17.0 * generated_features, generated_view, 0.25 * real_features, real_view, labels, epsilon=0.05, iterations=5)
    assert torch.allclose(feature_only, feature_scaled, atol=1.0e-5, rtol=1.0e-5)


def test_balanced_code_fraction_is_zero_when_codes_are_identical() -> None:
    """Repeated code probes must not be credited for class/spatial variation."""
    generator = torch.Generator().manual_seed(11)
    class_patterns = torch.randn(10, 1, 3, 5, generator=generator)
    values = class_patterns.expand(10, 17, 3, 5).clone()
    assert balanced_code_fraction(values) == pytest.approx(0.0, abs=1.0e-8)


def test_balanced_code_fraction_uses_per_pixel_grand_mean() -> None:
    """A common spatial template must not alter the code fraction."""
    generator = torch.Generator().manual_seed(12)
    # Code variation is shared across labels, while the large template varies
    # strongly by pixel.  A scalar grand mean would spuriously change eta².
    code_patterns = torch.randn(1, 17, 3, 5, generator=generator)
    values = code_patterns.expand(10, 17, 3, 5).clone()
    template = torch.linspace(-100.0, 100.0, 15).reshape(1, 1, 3, 5)
    fraction = balanced_code_fraction(values)
    shifted_fraction = balanced_code_fraction(values + template)
    assert fraction == pytest.approx(1.0, abs=1.0e-6)
    assert shifted_fraction == pytest.approx(fraction, abs=1.0e-6)


def test_balanced_code_fraction_is_bounded() -> None:
    values = torch.randn(10, 17, 4, 4, generator=torch.Generator().manual_seed(13))
    fraction = balanced_code_fraction(values)
    assert 0.0 <= fraction <= 1.0


def test_identity_isolation_allows_one_semantic_alignment_smoke_only_after_transport_passes() -> None:
    common = dict(harness_valid=True, command_fraction=0.15, first_collapse=None, camera_fraction=0.32, command_accuracy=0.97, saturation=0.05)
    assert _identity_isolation_classification(**common, camera_accuracy=0.18) == "semantic_alignment_required"
    assert _identity_isolation_classification(**common, camera_accuracy=0.30) == "controllable_upper_bound"
    assert _identity_isolation_classification(**{**common, "command_accuracy": 0.94}, camera_accuracy=0.18) == "carrier_semantics_invalid"
    assert _identity_isolation_classification(**{**common, "camera_fraction": 0.09}, camera_accuracy=0.18) == "optical_transport_attenuated"
    assert _identity_isolation_classification(**{**common, "saturation": 0.081}, camera_accuracy=0.18) == "exposure_saturation_excessive"


def test_identity_smoke_and_formal_training_share_the_first_hundred_schedule() -> None:
    labels = torch.arange(10).repeat_interleave(600)
    assert _first_hundred_schedule_hash(labels) == _first_hundred_schedule_hash(labels.clone())
    changed = labels.clone()
    changed[0], changed[600] = changed[600], changed[0]
    assert _first_hundred_schedule_hash(changed) != _first_hundred_schedule_hash(labels)


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


def test_identity_logit_v2_checkpoint_requires_complete_context(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    model.architecture_kind = "conditional_identity_logit_amplitude_v2"
    state_sig = state_signature(model.state_dict())
    content_hash = state_content_hash(model.state_dict())
    common = dict(
        schema_version=2,
        architecture_kind="conditional_identity_logit_amplitude_v2",
        system_signature="system",
        config_hash="config",
        dataset_split_manifest_hash="split",
        evaluator_checkpoint_hash="evaluator",
        exposure_scale=1.0,
        state_signature=state_sig,
        state_content_hash=content_hash,
        generator_contract_hash="generator-contract",
        metric_contract_hash="metric-contract",
        training_schedule_hash="schedule-contract",
        initialization_seed=1234,
        initialization_state_hash="initial-state",
    )
    manifest = ConditionalCheckpointManifestV2(**common)
    checkpoint = tmp_path / "identity_v2.pt"
    save_conditional_checkpoint(checkpoint, model, manifest)
    load_conditional_checkpoint(checkpoint, nn.Linear(2, 2), manifest)
    incomplete = ConditionalCheckpointManifestV2(**{**common, "metric_contract_hash": None})
    with pytest.raises(ValueError, match="complete contract hashes"):
        save_conditional_checkpoint(tmp_path / "incomplete.pt", model, incomplete)


def test_identity_logit_config_is_separate_and_locks_gain_and_seed() -> None:
    config = load_config("configs/v2/formal_identity_logit_amplitude_finite_crop_identity.yaml")
    assert config.architecture == "conditional_identity_logit_amplitude_v2"
    assert config.conditional_generation is not None
    assert config.conditional_generation.command_gain == pytest.approx(0.5)
    assert config.conditional_generation.initialization_seed == 1234


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
