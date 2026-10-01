from __future__ import annotations

import inspect

import pytest
import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research.optical_manifold import optical_manifold as manifold
from ldm_project.v2.config import load_config


def test_optical_manifold_config_and_cli_are_registered() -> None:
    spec = manifold._load_spec(
        "configs/research/optical_manifold/lightgen_optical_manifold_identity_v1.yaml"
    )
    manifold._assert_research_contract(spec)
    parsed = _parser().parse_args(
        [
            "research",
            "optical-manifold",
            "validate",
            "--config",
            "config.yaml",
            "--dataset-root",
            "data/MNIST",
            "--output",
            "validation.json",
        ]
    )
    assert parsed.research_program == "optical-manifold"
    assert parsed.research_action == "validate"
    assert parsed.dataset_root == "data/MNIST"


def test_rendered_intensity_is_the_only_resized_domain() -> None:
    intensity = torch.zeros(2, 1, 28, 28)
    intensity[0, 0, 10:18, 10:18] = 1.0
    rendered = manifold.render_intensity_batch(intensity)
    assert rendered.intensity.shape == (2, 1, 200, 200)
    assert rendered.amplitude.shape == (2, 1, 200, 200)
    assert rendered.target.shape == (2, 1, 200, 200)
    assert rendered.intensity.dtype == torch.float32
    assert torch.allclose(rendered.amplitude.square(), rendered.target, atol=1e-7, rtol=0)
    assert float(rendered.amplitude.min()) >= 0.0
    assert float(rendered.amplitude.max()) <= 1.0
    source = inspect.getsource(manifold.render_intensity_batch)
    assert "F.interpolate" in source
    assert "ComplexField" not in source


def test_identity_autoencoder_has_only_four_phase_masks_trainable() -> None:
    config = load_config(
        "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
    )
    model = manifold.IdentityOpticalManifoldAutoencoder(config.system)
    contract = manifold._parameter_contract(model)
    assert tuple(contract["trainable_parameter_names"]) == manifold.EXPECTED_TRAINABLE
    assert contract["phase_shapes"]["system.encoder.phase"] == [2, 1, 100, 100]
    assert contract["phase_shapes"]["system.decoder.phase"] == [2, 1, 100, 100]
    assert all(torch.equal(parameter, torch.zeros_like(parameter)) for parameter in model.parameters())
    assert model.system.config.kind == manifold.EXPECTED_FORMAL_KIND
    assert model.system.config.propagation_grid_size == 2048
    assert model.system.ols.__class__.__name__ == "IdentityOLS"


def test_research_module_has_no_evaluator_or_training_dispatch() -> None:
    source = inspect.getsource(manifold)
    assert "load_frozen_evaluator" not in source
    assert "SmoothMNISTEvaluator" not in source
    assert "def train" not in source
    assert "formal_promotion" in source


def test_invalid_input_encoding_is_rejected_before_model_contract() -> None:
    spec = manifold._load_spec(
        "configs/research/optical_manifold/lightgen_optical_manifold_identity_v1.yaml"
    )
    payload = dict(spec.payload)
    payload["input_encoding"] = "complex_resize"
    invalid = manifold.OpticalManifoldSpec(
        spec.path, payload, manifold.canonical_sha256(payload)
    )
    with pytest.raises(ValueError, match="sqrt intensity"):
        manifold._assert_research_contract(invalid)
