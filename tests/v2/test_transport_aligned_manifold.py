from pathlib import Path

import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.conditional_transport_aligned_manifold import (
    _artifact_hashes,
    _calibrate_scale,
)
from ldm_project.v2.application.conditional_transport_curvature import (
    _quartiles,
    _ratio,
)
from ldm_project.v2.config import load_config
from ldm_project.v2.application.conditional_runtime import ConditionalBundle
from ldm_project.v2.student.generator import (
    ConditionalTransportAlignedManifoldGenerator,
    ConditionalTransportAlignedPhaseCapacityGenerator,
)
from ldm_project.v2.student.evaluator import SmoothMNISTEvaluator


def _generator() -> ConditionalTransportAlignedManifoldGenerator:
    return ConditionalTransportAlignedManifoldGenerator(
        shape=(8, 8),
        class_carrier_logits=torch.zeros(10, 8, 8),
        candidate_modes=torch.randn(10, 32, 8, 8),
        transport_mixing=torch.randn(10, 32, 8),
        manifold_scales=torch.full((10, 8), 0.05),
    )


def test_transport_aligned_zero_code_is_frozen_carrier_and_8d_only():
    generator = _generator()
    labels = torch.arange(10)
    assert torch.equal(
        generator(torch.zeros(10, 8), labels), generator.class_carrier_command(labels)
    )
    assert not generator.class_carrier_logits.requires_grad
    assert not generator.candidate_modes.requires_grad
    assert generator.transport_mixing.requires_grad


def test_transport_aligned_cli_and_config_are_explicit():
    config = load_config(
        Path("configs/v2/diagnostic/conditional_transport_aligned_manifold_v71.yaml")
    )
    parser = _parser()
    design = parser.parse_args(
        [
            "audit",
            "conditional-transport-aligned-manifold-design",
            "--config",
            "config.yaml",
            "--dataset-root",
            "data",
            "--evaluator-checkpoint",
            "e.pt",
            "--electronic-initialization",
            "init.pt",
            "--v7-capacity-evidence",
            "v7.json",
            "--v7-basis-artifact",
            "basis.pt",
        ]
    )
    smoke = parser.parse_args(
        [
            "student",
            "transport-aligned-manifold-smoke",
            "--config",
            "config.yaml",
            "--dataset-root",
            "data",
            "--evaluator-checkpoint",
            "e.pt",
            "--checkpoint",
            "init.pt",
            "--tangent-evidence",
            "t.json",
        ]
    )
    assert config.architecture == "conditional_transport_aligned_manifold_v7_1"
    assert design.v7_basis_artifact == "basis.pt"
    assert smoke.tangent_evidence == "t.json"


def test_transport_design_artifact_hashes_bind_every_frozen_tensor():
    artifact = {
        "carrier": torch.zeros(10, 8, 8),
        "candidate_modes": torch.ones(10, 32, 8, 8),
        "transport_mixing": torch.ones(10, 32, 8),
        "manifold_scales": torch.full((10, 8), 0.05),
    }
    from ldm_project.v2.evidence import tensor_sha256

    artifact |= {
        "carrier_hash": tensor_sha256(artifact["carrier"]),
        "candidate_basis_hash": tensor_sha256(artifact["candidate_modes"]),
        "mixing_hash": tensor_sha256(artifact["transport_mixing"]),
        "scale_hash": tensor_sha256(artifact["manifold_scales"]),
    }
    assert _artifact_hashes(artifact)
    artifact["mixing_hash"] = "tampered"
    assert not _artifact_hashes(artifact)


def test_transport_scale_bisection_has_exactly_twenty_fixed_trials():
    scale, trace = _calibrate_scale(lambda value: (value <= 0.2, {"value": value}))
    assert len(trace) == 20
    assert 0.199 < scale <= 0.2
    assert all("passed" in item and "value" in item for item in trace)


def test_v72_phase_capacity_config_is_isolated_4_plus_4_and_798_mm():
    config = load_config(
        Path(
            "configs/v2/diagnostic/conditional_transport_aligned_phase_capacity_v72.yaml"
        )
    )
    bundle = ConditionalBundle(
        config,
        carrier_logits=torch.zeros(10, 200, 200),
        class_modes=torch.zeros(10, 32, 200, 200),
        transport_mixing=torch.zeros(10, 32, 8),
        manifold_scales=torch.full((10, 8), 0.05),
    )
    assert config.architecture == "conditional_transport_aligned_phase_capacity_v7_2"
    assert (config.system.encoder_layers, config.system.decoder_layers) == (4, 4)
    assert (
        abs(bundle.system.chip_plane.z_m - bundle.system.encoder_plane.z_m - 0.798)
        < 1e-12
    )
    assert isinstance(
        bundle.generator, ConditionalTransportAlignedPhaseCapacityGenerator
    )
    assert (
        ConditionalTransportAlignedManifoldGenerator.architecture_kind
        != bundle.generator.architecture_kind
    )


def test_v72_cli_requires_the_non_authorizing_v71_failure_fact():
    parser = _parser()
    parsed = parser.parse_args(
        [
            "audit",
            "conditional-transport-aligned-phase-capacity-design",
            "--config",
            "config.yaml",
            "--dataset-root",
            "data",
            "--evaluator-checkpoint",
            "e.pt",
            "--electronic-initialization",
            "init.pt",
            "--v7-capacity-evidence",
            "v7.json",
            "--v7-basis-artifact",
            "basis.pt",
            "--v71-failure-evidence",
            "v71.json",
        ]
    )
    assert parsed.v71_failure_evidence == "v71.json"


def test_curvature_ratio_and_argmax_keep_numerator_and_denominator():
    zero = torch.zeros(1, 1, 2, 2)
    first = _ratio(torch.ones_like(zero), -torch.ones_like(zero), zero)
    curved = _ratio(torch.full_like(zero, 2.0), torch.zeros_like(zero), zero)
    summary = _quartiles([first, curved])
    assert first["ratio"] == 0.0
    assert summary["argmax_direction"] == 1
    assert summary["argmax"]["numerator"] > 0.0


def test_evaluator_v2_cli_and_smooth_evaluator_contract():
    parser = _parser()
    parsed = parser.parse_args(
        [
            "audit",
            "conditional-evaluator-qualification-v2",
            "--config",
            "config.yaml",
            "--dataset-root",
            "data",
            "--primary-evaluator-checkpoint",
            "primary.pt",
        ]
    )
    model = SmoothMNISTEvaluator()
    features, logits = model(torch.zeros(2, 1, 28, 28))
    assert parsed.primary_evaluator_checkpoint == "primary.pt"
    assert model.architecture_kind == "smooth_mnist_evaluator_v2"
    assert features.shape == (2, 128) and logits.shape == (2, 10)


def test_evaluator_v21_cli_has_separate_calibration_and_qualification():
    parser = _parser()
    calibration = parser.parse_args(
        [
            "audit",
            "conditional-evaluator-calibration-v21",
            "--config",
            "config.yaml",
            "--dataset-root",
            "data",
            "--primary-evaluator-checkpoint",
            "primary.pt",
            "--evaluator-v2-evidence",
            "failed.json",
        ]
    )
    qualification = parser.parse_args(
        [
            "audit",
            "conditional-evaluator-qualification-v21",
            "--config",
            "config.yaml",
            "--dataset-root",
            "data",
            "--primary-evaluator-checkpoint",
            "primary.pt",
            "--calibration-evidence",
            "calibration.json",
        ]
    )
    assert calibration.evaluator_v2_evidence == "failed.json"
    assert qualification.calibration_evidence == "calibration.json"
