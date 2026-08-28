from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.research import optical_manifold_smf_vae_pilot as vae


def test_posterior_is_shared_per_port_and_identity_residual() -> None:
    posterior = vae.SharedPerPortPosterior()
    values = torch.randn(4, vae.PORT_COUNT, 2)
    mean, logvar = posterior(values)
    assert mean.shape == (4, vae.PORT_COUNT, 2)
    assert logvar.shape == mean.shape
    assert torch.equal(mean, values)
    assert torch.allclose(logvar, torch.full_like(logvar, vae.INITIAL_LOGVAR))


def test_complex_real_round_trip_is_exact() -> None:
    coefficients = torch.randn(3, vae.PORT_COUNT, dtype=torch.complex64)
    mean = torch.randn(vae.PORT_COUNT, 2)
    scale = torch.rand(vae.PORT_COUNT, 2) + 0.2
    normalized = vae._normalize(coefficients, mean, scale)
    restored = vae._denormalize(normalized, mean, scale)
    assert torch.allclose(restored, coefficients, atol=1.0e-6, rtol=0.0)


def test_standard_diagonal_gaussian_kl_matches_reference() -> None:
    mean = torch.zeros(1, vae.PORT_COUNT, 2)
    logvar = torch.zeros_like(mean)
    mean[0, 0] = torch.tensor([1.0, -2.0])
    logvar[0, 0] = torch.tensor([0.5, -0.25])
    expected = 0.5 * (mean.square() + logvar.exp() - 1.0 - logvar).mean()
    assert torch.equal(vae.posterior_kl(mean, logvar), expected)


def test_beta_calibration_uses_registered_gradient_ratio() -> None:
    assert vae._calibrate_beta(2.0, 5.0) == pytest.approx(0.04)


@pytest.mark.parametrize(
    "reconstruction, kl", [(0.0, 1.0), (1.0, 0.0), (float("nan"), 1.0)]
)
def test_beta_calibration_rejects_invalid_gradient_norms(
    reconstruction: float, kl: float
) -> None:
    with pytest.raises(ValueError, match="gradient norms"):
        vae._calibrate_beta(reconstruction, kl)


def test_posterior_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        vae.SharedPerPortPosterior()(torch.zeros(2, 100))


def test_config_review_and_lazy_cli_contract() -> None:
    root = Path(__file__).parents[2]
    config = root / "configs/research/lightgen_optical_manifold_smf_vae_pilot_v1.yaml"
    review = (
        root / "configs/research/lightgen_optical_manifold_smf_vae_pilot_review_v1.json"
    )
    spec = vae._load_spec(config)
    vae._assert_contract(spec)
    assert vae._load_review(review, spec).name.endswith("review_v1.json")
    source = (root / "src/ldm_project/v2/application/cli.py").read_text(
        encoding="utf-8"
    )
    assert "optical_manifold_smf_vae_pilot" in source
    assert "importlib.import_module" in source


def test_config_has_explicit_hybrid_boundary() -> None:
    root = Path(__file__).parents[2]
    spec = vae._load_spec(
        root / "configs/research/lightgen_optical_manifold_smf_vae_pilot_v1.yaml"
    )
    assert (
        spec.payload["optical_contract"]["posterior_status"] == "project_approximation"
    )
    assert (
        spec.payload["optical_contract"]["injector_status"]
        == "research_hybrid_active_latent_injector"
    )
    assert spec.payload["authorization"]["training_authorized"] is False
