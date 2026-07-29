import torch

from ldm_project.v2.application.conditional_formal_spacing import (
    FORMAL_SPACING_GENERATOR_CONTRACT,
    FORMAL_SPACING_METRIC_CONTRACT,
    _direction_response,
)
from ldm_project.v2.application.conditional_runtime import ConditionalBundle
from ldm_project.v2.config import load_config


def test_formal_spacing_config_locks_the_formal_bench_geometry():
    config = load_config(
        "configs/v2/diagnostic/conditional_pca_formal_spacing_self_imaging_finite_crop_identity.yaml"
    )
    assert config.architecture == "conditional_pca_formal_spacing_self_imaging_v6_3"
    assert config.system.kind == "formal_amplitude_finite_crop_identity"
    assert config.system.layer_spacing_m == 1.50e-3
    assert config.system.is_formal


def test_formal_spacing_bundle_has_the_explicit_pca_state_and_two_plus_two_system():
    config = load_config(
        "configs/v2/diagnostic/conditional_pca_formal_spacing_self_imaging_finite_crop_identity.yaml"
    )
    bundle = ConditionalBundle(config)
    assert tuple(bundle.generator.class_modes.shape) == (10, 8, 200, 200)
    assert bundle.system.encoder.layers == 2
    assert bundle.system.decoder.layers == 2
    assert FORMAL_SPACING_GENERATOR_CONTRACT
    assert FORMAL_SPACING_METRIC_CONTRACT


def test_direction_response_uses_plus_minus_axes_only():
    raw = torch.zeros(170, 1, 2, 2)
    raw.view(10, 17, 1, 2, 2)[:, 1, 0].fill_(2.0)
    raw.view(10, 17, 1, 2, 2)[:, 9, 0].fill_(1.0)
    response = _direction_response(raw)
    assert response[0] > 0.0
    assert response[1:] == [0.0] * 7
