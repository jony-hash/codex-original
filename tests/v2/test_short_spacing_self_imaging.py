import torch

from ldm_project.v2.config import load_config
from ldm_project.v2.application.conditional_runtime import ConditionalBundle
from ldm_project.v2.application.conditional_short_spacing import (
    SHORT_SPACING_GENERATOR_CONTRACT,
    SHORT_SPACING_METRIC_CONTRACT,
)
from ldm_project.v2.student.generator import ConditionalPCASelfImagingTransportGenerator


def test_short_spacing_config_is_explicit_and_separate_from_formal_contract():
    config = load_config("configs/v2/diagnostic/conditional_pca_self_imaging_transport_finite_crop_identity_short_spacing.yaml")
    assert config.architecture == "conditional_pca_self_imaging_transport_v6_2"
    assert config.system.kind == "diagnostic_amplitude_finite_crop_identity_short_spacing_v1"
    assert config.system.layer_spacing_m == 0.25e-3
    assert not config.system.is_formal


def test_short_spacing_generator_is_bounded_and_zero_code_is_carrier():
    generator = ConditionalPCASelfImagingTransportGenerator()
    labels = torch.arange(10)
    zero = torch.zeros(10, 8)
    assert torch.equal(generator(zero, labels), generator.class_carrier_command(labels))
    assert torch.all(generator.scales() < 0.20)
    assert torch.allclose(generator.scales(), torch.full_like(generator.scales(), 0.05), atol=1e-6)
    modes = generator.lowpass_normalized_modes()
    assert torch.allclose(modes.mean(dim=(-2, -1)), torch.zeros(10, 8), atol=1e-5)
    assert torch.allclose(modes.square().mean(dim=(-2, -1)), torch.ones(10, 8), atol=1e-4)


def test_short_spacing_bundle_keeps_two_plus_two_layers_and_contracts():
    config = load_config("configs/v2/diagnostic/conditional_pca_self_imaging_transport_finite_crop_identity_short_spacing.yaml")
    bundle = ConditionalBundle(config)
    assert bundle.system.encoder.layers == 2
    assert bundle.system.decoder.layers == 2
    assert bundle.system.config.layer_spacing_m == 0.25e-3
    assert SHORT_SPACING_GENERATOR_CONTRACT
    assert SHORT_SPACING_METRIC_CONTRACT
