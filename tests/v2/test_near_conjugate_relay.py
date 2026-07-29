import torch

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
