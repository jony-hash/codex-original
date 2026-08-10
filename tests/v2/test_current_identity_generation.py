from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ldm_project.v2.config import load_config
from ldm_project.v2.evidence import canonical_sha256
from ldm_project.v2.student.admission import balanced_code_fraction
from ldm_project.v2.student.checkpoints import (
    ConditionalCheckpointManifestV2,
    load_conditional_checkpoint,
    save_conditional_checkpoint,
    state_content_hash,
    state_signature,
)
from ldm_project.v2.student.conditional import (
    conditional_loss,
    conditional_sinkhorn_loss,
    fixed_exposure_view,
    sinkhorn_cost_view,
)
from ldm_project.v2.student.conditional_acceptance import (
    chunked_nearest_distances,
    memorization_statistics,
    paired_bootstrap_lower_bound,
    per_label_feature_diversity_ratio,
)
from ldm_project.v2.student.generator import (
    IdentityAnchoredConditionalAmplitudeGenerator,
)
from ldm_project.v2.evidence.reviews import ConditionalManualReviewV2
from ldm_project.v2.application.conditional import (
    _first_hundred_schedule_hash,
    _identity_isolation_classification,
    _schedule_hash,
    _semantic_localization_classification,
)
from ldm_project.v2.student.evaluator import MNISTSplitManifest


class _ToyPhaseSystem(nn.Module):
    """Small differentiable camera preserving the production phase parameter boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.phase = nn.Parameter(torch.zeros(1, 1, 8, 8))
        self.decoder = nn.Module()
        self.decoder.phase = nn.Parameter(torch.zeros(1, 1, 8, 8))
        self.frozen = nn.Parameter(torch.ones(1), requires_grad=False)

    def command(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def forward(self, command: torch.Tensor) -> SimpleNamespace:
        gain = torch.sigmoid(self.encoder.phase.mean() + self.decoder.phase.mean())
        return SimpleNamespace(raw_camera=SimpleNamespace(tensor=command * gain))


class _ToyEvaluator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(28 * 28, 16, bias=False)
        self.classifier = nn.Linear(16, 10, bias=False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.projection(images.flatten(1))
        return features, self.classifier(features)



def test_identity_anchored_generator_has_zero_code_identity_and_zero_wy() -> None:
    generator = IdentityAnchoredConditionalAmplitudeGenerator(shape=(8, 8))
    labels = torch.tensor([2, 7])
    zero = torch.zeros(2, 8)
    command = generator(zero, labels)
    assert command.shape == (2, 1, 8, 8)
    assert torch.allclose(
        command, generator.class_carrier_command(labels), atol=0.0, rtol=0.0
    )
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
    assert all(
        torch.equal(first_state, second_state)
        for first_state, second_state in zip(
            first.state_dict().values(), second.state_dict().values()
        )
    )


def test_current_bundle_seed_is_bound_before_construction() -> None:
    from ldm_project.v2.application.current.runtime import _seeded_conditional_bundle

    config = load_config(
        "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
    )
    # Perturb the caller's RNG; construction must still use the scoped config
    # seed before any module parameter is created.
    torch.manual_seed(9876)
    first = _seeded_conditional_bundle(config)
    torch.manual_seed(4321)
    second = _seeded_conditional_bundle(config)
    assert state_content_hash(first.state_dict()) == state_content_hash(second.state_dict())


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
    loss = conditional_sinkhorn_loss(
        generated_features,
        generated_view,
        real_features,
        real_view,
        labels,
        epsilon=0.05,
        iterations=5,
    )
    permutation = torch.cat(
        [torch.nonzero(labels == label).flatten().flip(0) for label in range(10)]
    )
    shuffled = conditional_sinkhorn_loss(
        generated_features,
        generated_view,
        real_features.index_select(0, permutation),
        real_view.index_select(0, permutation),
        labels,
        epsilon=0.05,
        iterations=5,
    )
    assert torch.allclose(loss, shuffled)


def test_conditional_sinkhorn_normalizes_features_and_pools_intensity_cost() -> None:
    labels = torch.arange(10).repeat_interleave(2)
    generated_features = torch.randn(20, 8, generator=torch.Generator().manual_seed(21))
    real_features = torch.randn(20, 8, generator=torch.Generator().manual_seed(22))
    generated_view = torch.rand(
        20, 1, 28, 28, generator=torch.Generator().manual_seed(23)
    )
    real_view = torch.rand(20, 1, 28, 28, generator=torch.Generator().manual_seed(24))
    loss = conditional_sinkhorn_loss(
        generated_features,
        generated_view,
        real_features,
        real_view,
        labels,
        epsilon=0.05,
        iterations=5,
    )
    scaled = conditional_sinkhorn_loss(
        17.0 * generated_features,
        3.0 * generated_view,
        0.25 * real_features,
        3.0 * real_view,
        labels,
        epsilon=0.05,
        iterations=5,
    )
    # Feature normalization makes global feature scale irrelevant; pooled
    # intensity costs remain finite and preserve the expected 14x14 contract.
    assert sinkhorn_cost_view(generated_view).shape == (20, 1, 14, 14)
    assert torch.isfinite(loss)
    assert torch.isfinite(scaled)
    assert not torch.allclose(loss, scaled)
    feature_only = conditional_sinkhorn_loss(
        generated_features,
        generated_view,
        real_features,
        real_view,
        labels,
        epsilon=0.05,
        iterations=5,
    )
    feature_scaled = conditional_sinkhorn_loss(
        17.0 * generated_features,
        generated_view,
        0.25 * real_features,
        real_view,
        labels,
        epsilon=0.05,
        iterations=5,
    )
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


def test_identity_isolation_allows_one_semantic_alignment_smoke_only_after_transport_passes() -> (
    None
):
    common = dict(
        harness_valid=True,
        command_fraction=0.15,
        first_collapse=None,
        camera_fraction=0.32,
        command_accuracy=0.97,
        saturation=0.05,
    )
    assert (
        _identity_isolation_classification(**common, camera_accuracy=0.18)
        == "semantic_alignment_required"
    )
    assert (
        _identity_isolation_classification(**common, camera_accuracy=0.30)
        == "controllable_upper_bound"
    )
    assert (
        _identity_isolation_classification(
            **{**common, "command_accuracy": 0.94}, camera_accuracy=0.18
        )
        == "carrier_semantics_invalid"
    )
    assert (
        _identity_isolation_classification(
            **{**common, "camera_fraction": 0.09}, camera_accuracy=0.18
        )
        == "optical_transport_attenuated"
    )
    assert (
        _identity_isolation_classification(
            **{**common, "saturation": 0.081}, camera_accuracy=0.18
        )
        == "exposure_saturation_excessive"
    )


def test_semantic_localization_classifies_gradient_contract_without_updates() -> None:
    groups = (
        "class_carrier",
        "global_modes",
        "code_projection",
        "class_heads",
        "encoder_phase",
        "decoder_phase",
    )
    zero = {
        name: {"gradient_norm": 1.0, "gradient_finite": True, "gradient_nonzero": True}
        for name in groups
    }
    formal = {
        name: {"gradient_norm": 1.0, "gradient_finite": True, "gradient_nonzero": True}
        for name in groups
    }
    aligned = {name: {"cosine": 0.5} for name in groups}
    assert (
        _semantic_localization_classification(
            harness_valid=True,
            zero_gradients=zero,
            formal_gradients=formal,
            gradient_cosines=aligned,
        )
        == "local_semantic_alignment_plausible"
    )
    assert (
        _semantic_localization_classification(
            harness_valid=True,
            zero_gradients=zero,
            formal_gradients=formal,
            gradient_cosines={**aligned, "encoder_phase": {"cosine": -0.1}},
        )
        == "zero_code_joint_conflict"
    )
    assert (
        _semantic_localization_classification(
            harness_valid=True,
            zero_gradients=zero,
            formal_gradients=formal,
            gradient_cosines={**aligned, "decoder_phase": {"cosine": 0.0}},
        )
        == "indeterminate_alignment"
    )
    zero_missing = {
        **zero,
        "class_carrier": {
            "gradient_norm": 0.0,
            "gradient_finite": True,
            "gradient_nonzero": False,
        },
    }
    assert (
        _semantic_localization_classification(
            harness_valid=True,
            zero_gradients=zero_missing,
            formal_gradients=formal,
            gradient_cosines=aligned,
        )
        == "zero_code_gradient_absent"
    )
    assert (
        _semantic_localization_classification(
            harness_valid=False,
            zero_gradients=zero,
            formal_gradients=formal,
            gradient_cosines=aligned,
        )
        == "harness_invalid"
    )


def test_semantic_alignment_isolation_routes_only_to_localization() -> None:
    common = dict(
        harness_valid=True,
        command_fraction=0.15,
        first_collapse=None,
        camera_fraction=0.32,
        command_accuracy=0.97,
        saturation=0.05,
    )
    assert (
        _identity_isolation_classification(**common, camera_accuracy=0.18)
        == "semantic_alignment_required"
    )


def test_identity_smoke_and_formal_training_share_the_first_hundred_schedule() -> None:
    labels = torch.arange(10).repeat_interleave(600)
    assert _first_hundred_schedule_hash(labels) == _first_hundred_schedule_hash(
        labels.clone()
    )
    changed = labels.clone()
    changed[0], changed[600] = changed[600], changed[0]
    assert _first_hundred_schedule_hash(changed) != _first_hundred_schedule_hash(labels)


def test_semantic_recovery_schedule_extends_the_same_first_hundred_batches() -> None:
    labels = torch.arange(10).repeat_interleave(2500)
    assert _schedule_hash(labels, 100) == _first_hundred_schedule_hash(labels)
    assert _schedule_hash(labels, 400) != _first_hundred_schedule_hash(labels)


def test_conditional_checkpoint_rejects_architecture_mismatch(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    manifest = ConditionalCheckpointManifestV2(
        2,
        "conditional_global_amplitude_v1",
        "system",
        "config",
        "split",
        "evaluator",
        1.0,
        state_signature(model.state_dict()),
        state_content_hash(model.state_dict()),
    )
    checkpoint = tmp_path / "conditional.pt"
    save_conditional_checkpoint(checkpoint, model, manifest)
    target = nn.Linear(2, 2)
    load_conditional_checkpoint(checkpoint, target, manifest)
    incompatible = ConditionalCheckpointManifestV2(
        2,
        "conditional_global_amplitude_v1",
        "other-system",
        "config",
        "split",
        "evaluator",
        1.0,
        manifest.state_signature,
        manifest.state_content_hash,
    )
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
    incomplete = ConditionalCheckpointManifestV2(
        **{**common, "metric_contract_hash": None}
    )
    with pytest.raises(ValueError, match="complete contract hashes"):
        save_conditional_checkpoint(tmp_path / "incomplete.pt", model, incomplete)


def test_initialization_payload_rejects_seed_or_state_hash_tampering() -> None:
    from ldm_project.v2.application.current.runtime import _validate_initialization_payload

    config = load_config(
        "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
    )
    manifest = SimpleNamespace(
        state_content_hash="state",
        initialization_state_hash="state",
        initialization_seed=1234,
    )
    _validate_initialization_payload(
        config, manifest, {"initialization": True, "seed": 1234}
    )
    with pytest.raises(ValueError, match="state hash"):
        _validate_initialization_payload(
            config,
            SimpleNamespace(
                state_content_hash="changed",
                initialization_state_hash="state",
                initialization_seed=1234,
            ),
            {"initialization": True, "seed": 1234},
        )
    with pytest.raises(ValueError, match="seed"):
        _validate_initialization_payload(
            config, manifest, {"initialization": True, "seed": 4321}
        )


def test_initialization_payload_rejects_config_seed_mismatch() -> None:
    from ldm_project.v2.application.current.runtime import _validate_initialization_payload

    config = load_config(
        "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
    )
    with pytest.raises(ValueError, match="initialization seed"):
        _validate_initialization_payload(
            config,
            SimpleNamespace(
                state_content_hash="state",
                initialization_state_hash="state",
                initialization_seed=4321,
            ),
            {"initialization": True, "seed": 4321},
        )


def test_current_bundle_loader_rejects_initialization_manifest_tampering(
    tmp_path: Path,
) -> None:
    from ldm_project.v2.application.current.runtime import (
        CONDITIONAL_CONTROL_METRIC_CONTRACT,
        IDENTITY_GENERATOR_CONTRACT,
        TRAINING_SCHEDULE_CONTRACT,
        _load_bundle,
        _seeded_conditional_bundle,
        _system_signature,
    )

    config = load_config(
        "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
    )
    bundle = _seeded_conditional_bundle(config)
    settings = config.conditional_generation
    assert settings is not None
    state_hash = state_content_hash(bundle.state_dict())
    manifest = ConditionalCheckpointManifestV2(
        2,
        config.architecture,
        _system_signature(config),
        canonical_sha256(config.model_dump(mode="json")),
        "split",
        "evaluator",
        settings.fixed_exposure_scale,
        state_signature(bundle.state_dict()),
        state_hash,
        IDENTITY_GENERATOR_CONTRACT,
        canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT),
        TRAINING_SCHEDULE_CONTRACT,
        settings.initialization_seed,
        state_hash,
    )
    checkpoint = tmp_path / "initialization.pt"
    save_conditional_checkpoint(
        checkpoint,
        bundle,
        manifest,
        extra={"initialization": True, "seed": settings.initialization_seed},
    )
    loaded, loaded_manifest = _load_bundle(
        config, str(checkpoint), device=torch.device("cpu")
    )
    assert state_content_hash(loaded.state_dict()) == loaded_manifest.initialization_state_hash

    payload = torch.load(checkpoint, weights_only=False)
    payload["manifest"]["initialization_seed"] = 4321
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="initialization seed"):
        _load_bundle(config, str(checkpoint), device=torch.device("cpu"))

    payload["manifest"]["initialization_seed"] = settings.initialization_seed
    payload["manifest"]["initialization_state_hash"] = "0" * 64
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="state hash"):
        _load_bundle(config, str(checkpoint), device=torch.device("cpu"))


def test_identity_logit_config_is_separate_and_locks_gain_and_seed() -> None:
    config = load_config(
        "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
    )
    assert config.architecture == "conditional_identity_logit_amplitude_v2"
    assert config.conditional_generation is not None
    assert config.conditional_generation.command_gain == pytest.approx(0.5)
    assert config.conditional_generation.initialization_seed == 1234


def test_current_identity_bundle_preserves_legacy_checkpoint_topology() -> None:
    """The physical split must not alter formal state keys or generator output."""
    from ldm_project.v2.application.current.runtime import ConditionalBundle as current_bundle
    from ldm_project.v2.application.legacy.runtime import ConditionalBundle as legacy_bundle

    config = load_config(
        "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
    )
    torch.manual_seed(1234)
    current = current_bundle(config).eval()
    torch.manual_seed(1234)
    legacy = legacy_bundle(config).eval()
    assert state_signature(current.state_dict()) == state_signature(legacy.state_dict())
    assert tuple(current.state_dict()) == tuple(legacy.state_dict())
    codes = torch.tensor([[0.0] * 8, [0.25] * 8])
    labels = torch.tensor([0, 9])
    assert torch.equal(current.generator(codes, labels), legacy.generator(codes, labels))


def test_current_identity_parameter_groups_have_no_candidate_architecture_branch() -> None:
    """Identity admission must not consult diagnostic-only architecture constants."""
    from ldm_project.v2.application.current.admission import _conditional_parameter_groups
    from ldm_project.v2.application.current.runtime import ConditionalBundle

    config = load_config(
        "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
    )
    groups = _conditional_parameter_groups(ConditionalBundle(config))
    assert tuple(groups) == (
        "class_carrier",
        "global_modes",
        "code_projection",
        "class_heads",
        "encoder_phase",
        "decoder_phase",
    )


def test_current_optimizer_updates_every_declared_trainable_group() -> None:
    from ldm_project.v2.application.current.recovery import _conditional_optimizer

    config = load_config(
        "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
    )
    generator = IdentityAnchoredConditionalAmplitudeGenerator(shape=(8, 8))
    system = _ToyPhaseSystem()
    evaluator = _ToyEvaluator().eval()
    bundle = SimpleNamespace(
        generator=generator,
        system=system,
    )
    bundle.parameters = lambda: list(generator.parameters()) + list(system.parameters())
    optimizer = _conditional_optimizer(bundle, config)
    declared = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len({id(parameter) for parameter in declared}) == 6
    labels = torch.arange(10).repeat_interleave(2)
    codes = torch.randn(20, 8)
    real_views = torch.rand(20, 1, 28, 28)
    before = [parameter.detach().clone() for parameter in declared]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = conditional_loss(
            system=system,
            generator=generator,
            evaluator=evaluator,
            codes=codes,
            labels=labels,
            real_views=real_views,
            exposure_scale=1.0,
            sinkhorn_epsilon=0.1,
            sinkhorn_iterations=1,
            activation_checkpoint=False,
            microbatch_size=2,
        )
        loss.backward()
        optimizer.step()
    assert all(not torch.equal(previous, parameter) for previous, parameter in zip(before, declared))
    assert torch.equal(system.frozen.detach(), torch.ones(1))

    optimizer.zero_grad(set_to_none=True)
    loss, _, _ = conditional_loss(
        system=system,
        generator=generator,
        evaluator=evaluator,
        codes=codes,
        labels=labels,
        real_views=real_views,
        exposure_scale=1.0,
        sinkhorn_epsilon=0.1,
        sinkhorn_iterations=1,
        activation_checkpoint=False,
        microbatch_size=2,
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in declared)
    no_step = [parameter.detach().clone() for parameter in declared]
    assert all(torch.equal(previous, parameter) for previous, parameter in zip(no_step, declared))
    from ldm_project.v2.application.current.admission import _conditional_parameter_groups
    from ldm_project.v2.application.current.training import (
        _validate_current_optimizer_coverage,
        _validate_two_step_parameter_updates,
    )

    groups = _conditional_parameter_groups(bundle)
    _validate_current_optimizer_coverage(optimizer, groups)
    with pytest.raises(ValueError, match="did not update"):
        _validate_two_step_parameter_updates(
            bundle,
            groups,
            {id(parameter): parameter.detach().clone() for parameter in declared},
            {
                id(parameter): parameter.detach().clone()
                for parameter in bundle.parameters()
                if id(parameter) not in {id(item) for item in declared}
            },
            {id(parameter) for parameter in declared if parameter.grad is not None},
        )


def test_current_optimizer_coverage_rejects_missing_declared_parameter() -> None:
    from ldm_project.v2.application.current.admission import _conditional_parameter_groups
    from ldm_project.v2.application.current.recovery import _conditional_optimizer
    from ldm_project.v2.application.current.runtime import ConditionalBundle
    from ldm_project.v2.application.current.training import _validate_current_optimizer_coverage

    config = load_config(
        "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
    )
    bundle = ConditionalBundle(config)
    groups = _conditional_parameter_groups(bundle)
    optimizer = _conditional_optimizer(bundle, config)
    optimizer.param_groups[0]["params"] = optimizer.param_groups[0]["params"][1:]
    with pytest.raises(ValueError, match="coverage changed"):
        _validate_current_optimizer_coverage(optimizer, groups)


def test_config_migration_resolves_exact_legacy_path() -> None:
    with pytest.warns(UserWarning, match="deprecated v2 config path"):
        legacy = load_config("configs/v2/formal_amplitude_finite_crop_identity.yaml")
    assert legacy.architecture == "conditional_global_amplitude_v1"
    current = load_config(
        "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
    )
    assert current.architecture == "conditional_identity_logit_amplitude_v2"
    assert legacy.run.evidence_tier == "diagnostic"


def test_config_migration_does_not_guess_unknown_paths() -> None:
    with pytest.raises(FileNotFoundError, match="v2 config does not exist"):
        load_config(
            "configs/v2/formal_identity_logit_amplitude_finite_crop_identity_typo.yaml"
        )


def test_content_bound_mnist_split_identity_ignores_machine_path() -> None:
    common = (2, 1234, "train", "validation", 55000, 5000, "mnist-content")
    left = MNISTSplitManifest(
        common[0],
        "/machine/a/data",
        common[1],
        common[2],
        common[3],
        common[4],
        common[5],
        common[6],
    )
    right = MNISTSplitManifest(
        common[0],
        "/machine/b/data",
        common[1],
        common[2],
        common[3],
        common[4],
        common[5],
        common[6],
    )
    assert left.manifest_hash == right.manifest_hash


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
    review = ConditionalManualReviewV2(
        review_kind="conditional_generation",
        reviewed_tensor_sha256="raw",
        reviewed_preview_sha256="preview",
        samples_per_label=16,
        readable_counts=(14,) * 10,
        reviewer="unit",
    )
    review.validate_contract(tensor_hash="raw", preview_hash="preview")
    with pytest.raises(ValueError, match="sixteen"):
        ConditionalManualReviewV2(
            review_kind="conditional_generation",
            reviewed_tensor_sha256="raw",
            reviewed_preview_sha256="preview",
            samples_per_label=15,
            readable_counts=(14,) * 10,
            reviewer="unit",
        ).validate_contract(tensor_hash="raw", preview_hash="preview")


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
    assert paired_bootstrap_lower_bound(
        values, seed=1234, resamples=20
    ) == paired_bootstrap_lower_bound(values, seed=1234, resamples=20)


def test_current_bundle_different_config_seed_changes_initial_state() -> None:
    config = load_config(
        "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
    )
    assert config.conditional_generation is not None
    altered = config.conditional_generation.model_copy(
        update={"initialization_seed": config.conditional_generation.initialization_seed + 1}
    )
    alternate = config.model_copy(update={"conditional_generation": altered})
    from ldm_project.v2.application.current.runtime import _seeded_conditional_bundle

    first = _seeded_conditional_bundle(config)
    second = _seeded_conditional_bundle(alternate)
    assert state_content_hash(first.generator.state_dict()) != state_content_hash(
        second.generator.state_dict()
    )
