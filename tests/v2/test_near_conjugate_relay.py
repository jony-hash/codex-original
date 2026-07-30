import torch
from types import SimpleNamespace

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.conditional_near_conjugate import _replay_checks
from ldm_project.v2.application.conditional_runtime import ConditionalBundle
from ldm_project.v2.config import load_config
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics import (
    NearConjugateRelay,
    reference_near_conjugate_lens_relay,
)


def test_near_conjugate_config_is_explicit_and_generator_is_frozen_by_stage():
    config = load_config(
        "configs/v2/diagnostic/conditional_pca_near_conjugate_relay.yaml"
    )
    assert config.architecture == "conditional_pca_near_conjugate_relay_v6_4"
    assert config.system.kind == "diagnostic_amplitude_near_conjugate_relay_identity_v1"
    bundle = ConditionalBundle(config)
    assert bundle.system.illumination_intensity_transmission == 0.80
    assert tuple(bundle.system.encoder.phase.shape) == (2, 1, 100, 100)


def test_near_conjugate_relay_records_physical_and_effective_geometry():
    source = PlaneSpec("source", (16, 16), (2e-6, 2e-6), 532e-9, 0.0, "unit")
    target = source.with_plane("target", z_m=0.09975)
    value = torch.zeros(1, 16, 16, dtype=torch.complex64)
    value[..., 5, 7] = 1
    field = ComplexField(value, source, OperationProvenance("input", None, None))
    relay = NearConjugateRelay(source, target, name="unit")
    output = relay(field)
    reference = reference_near_conjugate_lens_relay(field, target, name="unit")
    assert abs(relay.abcd.a + 1) < 1e-12
    assert abs(relay.abcd.b_m - 0.25e-3) < 1e-12
    assert output.plane == target
    assert output.provenance.parameters["mechanical_segments_m"] == [
        0.025,
        0.05,
        0.02475,
    ]
    assert output.provenance.parameters["effective_B_m"] == 0.25e-3
    assert torch.isfinite(reference.tensor).all()


def test_near_conjugate_replay_keeps_missing_quality_gates_independent():
    def panel(*, ratio, recall=1.0):
        return {
            "accuracy": 1.0,
            "recall": [recall] * 10,
            "camera_fraction": 0.5,
            "ratios": [ratio] * 10,
            "delta_correlation": 0.9,
            "direction_response": [0.02] * 8,
            "saturation": 0.01,
        }

    history = {
        "probes": {
            str(step): {
                "axis": panel(ratio=0.4),
                "uniform_1234": panel(ratio=0.5),
                "uniform_5678": panel(ratio=0.7),
            }
            for step in (0, 50, 100, 200, 400)
        },
        "history": [
            {
                "sinkhorn": 0.1,
                "relative_diversity": 0.1 if step < 350 else 0.2,
                "delta": 0.1,
            }
            for step in range(400)
        ],
        "updates": {"encoder": 1.0, "decoder": 1.0},
    }
    initial = SimpleNamespace(
        state_dict=lambda: {
            "generator.value": torch.ones(1),
            "system.encoder.phase": torch.zeros(1),
            "system.decoder.phase": torch.zeros(1),
        }
    )
    final = SimpleNamespace(
        state_dict=lambda: {
            "generator.value": torch.ones(1),
            "system.encoder.phase": torch.ones(1),
            "system.decoder.phase": torch.ones(1),
        }
    )
    crops = {
        "axis": {"encoder": [0.95], "decoder": [0.95]},
        "uniform_1234": {"encoder": [0.95], "decoder": [0.95]},
        "uniform_5678": {"encoder": [0.95], "decoder": [0.95]},
    }
    checks, _ = _replay_checks(
        history,
        initial,
        final,
        True,
        {"axis": True, "uniform_1234": True, "uniform_5678": True},
        crops,
        crops,
        {"accuracy": 1.0},
    )
    by_name = {item.name: item for item in checks}
    assert not by_name["feature_ratio"].passed
    assert not by_name["distribution_loss_trend"].passed
    assert by_name["integrity"].passed


def test_near_conjugate_replay_and_distribution_tangent_cli_are_explicit():
    parser = _parser()
    replay = parser.parse_args(
        [
            "audit",
            "conditional-near-conjugate-smoke-replay",
            "--config", "config.yaml",
            "--dataset-root", "data",
            "--evaluator-checkpoint", "evaluator.pt",
            "--smoke-evidence", "smoke.json",
            "--history", "history.json",
            "--initialization-checkpoint", "initial.pt",
            "--diagnostic-final", "final.pt",
            "--observability-evidence", "observability.json",
        ]
    )
    tangent = parser.parse_args(
        [
            "audit",
            "conditional-near-conjugate-phase-distribution-tangent",
            "--config", "config.yaml",
            "--dataset-root", "data",
            "--evaluator-checkpoint", "evaluator.pt",
            "--replay-evidence", "replay.json",
            "--initialization-checkpoint", "initial.pt",
            "--diagnostic-final", "final.pt",
        ]
    )
    assert replay.history == "history.json"
    assert tangent.replay_evidence == "replay.json"
