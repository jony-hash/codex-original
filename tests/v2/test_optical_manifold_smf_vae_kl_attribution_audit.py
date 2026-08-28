from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.research import (
    optical_manifold_smf_vae_kl_attribution_audit as audit,
)


def test_config_review_and_lazy_cli_contract() -> None:
    root = Path(__file__).parents[2]
    config = root / "configs/research/lightgen_optical_manifold_smf_vae_kl_attribution_audit_v1.yaml"
    review = root / "configs/research/lightgen_optical_manifold_smf_vae_kl_attribution_audit_review_v1.json"
    spec = audit._load_spec(config)
    audit._assert_contract(spec)
    assert audit._load_review(review, spec).name.endswith("review_v1.json")
    source = (root / "src/ldm_project/v2/application/cli.py").read_text(encoding="utf-8")
    router = (root / "src/ldm_project/v2/application/_cli_research_optical_manifold.py").read_text(encoding="utf-8")
    assert "importlib.import_module" in source
    assert "smf-kl-chain-audit" in router
    assert "optical_manifold_smf_vae_kl_attribution_audit" in router


def test_kl_decomposition_matches_standard_formula() -> None:
    mean = torch.tensor([[[1.0, -2.0], [0.25, -0.5]]])
    logvar = torch.tensor([[[0.5, -0.25], [-1.0, -2.0]]])
    result = audit._kl_terms(mean, logvar)
    expected_mu = 0.5 * mean.square().mean()
    expected_var = 0.5 * (logvar.exp() - 1.0 - logvar).mean()
    assert result["kl_mu"] == pytest.approx(float(expected_mu))
    assert result["kl_variance"] == pytest.approx(float(expected_var))
    assert result["kl_total"] == pytest.approx(float(expected_mu + expected_var))


def test_dose_grid_is_fixed_and_no_update_contract_is_explicit() -> None:
    assert audit.DOSES == (1.0, 2.0, 4.0)
    assert audit.MIN_FREE_BYTES == 4 * 1024**3
    assert audit.MAX_PEAK_BYTES == 5 * 1024**3
    spec = audit._load_spec(
        Path(__file__).parents[2]
        / "configs/research/lightgen_optical_manifold_smf_vae_kl_attribution_audit_v1.yaml"
    )
    assert spec.payload["authorization"]["no_update_diagnostic_authorized"] is True
    assert spec.payload["authorization"]["training_authorized"] is False
    assert spec.payload["authorization"]["formal_promotion"] is False


def test_vector_compatibility_uses_descent_gradient_inner_product() -> None:
    left = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])}
    right = {"a": torch.tensor([2.0, 1.0]), "b": torch.tensor([-1.0])}
    assert audit._vector_dot(left, right) == pytest.approx(1.0)
