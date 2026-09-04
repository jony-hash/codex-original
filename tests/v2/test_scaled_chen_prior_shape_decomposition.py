from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.research.measure_scaled_chen_prior_shape_decomposition import (
    LATENT_DIM,
    classify_pcr,
    classify_structure,
    covariance_and_correlation,
    mean_variance_decomposition,
    next_write_once_output,
    prior_null_replicates,
    summarize_port_statistics,
    tail_masses,
    theoretical_truncated_normal,
)


def test_exact_registered_truncated_prior_moments() -> None:
    moments = theoretical_truncated_normal()
    assert moments["mean"] == pytest.approx(1.027623931339495, abs=1.0e-12)
    assert moments["variance"] == pytest.approx(0.2216129870778559, abs=1.0e-12)
    assert moments["skew"] == pytest.approx(0.22093570483049968, abs=1.0e-12)
    assert moments["quantiles"] == pytest.approx(
        [0.07752654296325345, 1.0142584632954588, 1.8279921783569142, 2.167484910379014],
        abs=1.0e-12,
    )


def test_per_port_statistics_and_variance_decomposition() -> None:
    z = torch.empty((3, LATENT_DIM), dtype=torch.float64)
    z[:, :50] = torch.tensor([[0.0], [1.0], [2.0]])
    z[:, 50:] = torch.tensor([[1.0], [2.0], [3.0]])
    stats = summarize_port_statistics(z, prior=theoretical_truncated_normal())
    assert torch.equal(stats["mean"][:50], torch.ones(50, dtype=torch.float64))
    assert torch.equal(stats["mean"][50:], torch.full((50,), 2.0, dtype=torch.float64))
    assert torch.equal(
        stats["variance"], torch.full((LATENT_DIM,), 2.0 / 3.0, dtype=torch.float64)
    )
    decomposition = mean_variance_decomposition(z, stats["mean"], stats["variance"])
    assert decomposition["within"] == pytest.approx(2.0 / 3.0)
    assert decomposition["between"] == pytest.approx(0.25)
    assert decomposition["total"] == pytest.approx(11.0 / 12.0)
    assert decomposition["identity_residual"] == pytest.approx(0.0, abs=1.0e-12)


def test_quantiles_and_prior_defined_tail_masses() -> None:
    prior = theoretical_truncated_normal()
    z = torch.ones((4, LATENT_DIM))
    z[:, 0] = torch.tensor([0.0, 1.0, 1.0, 2.0])
    z[:, 1] = torch.tensor([1.0, 2.2, 3.0, 4.0])
    stats = summarize_port_statistics(z, prior=prior)
    assert tuple(stats["quantiles"].shape) == (LATENT_DIM, 4)
    masses = tail_masses(z, prior["quantiles"])
    assert masses["lower"][:2].tolist() == pytest.approx([0.25, 0.0])
    assert masses["upper"][:2].tolist() == pytest.approx([0.0, 0.75])
    assert masses["central"][:2].tolist() == pytest.approx([0.75, 0.25])


def test_covariance_and_correlation_are_unwhitened_population_statistics() -> None:
    z = torch.empty((3, LATENT_DIM), dtype=torch.float64)
    z[:, 0] = torch.tensor([1.0, 2.0, 3.0])
    z[:, 1] = torch.tensor([2.0, 4.0, 6.0])
    z[:, 2:] = torch.tensor([[1.0], [2.0], [3.0]])
    covariance, correlation = covariance_and_correlation(z)
    expected_covariance = torch.tensor(
        [[2.0 / 3.0, 4.0 / 3.0], [4.0 / 3.0, 8.0 / 3.0]], dtype=torch.float64
    )
    assert torch.allclose(covariance[:2, :2], expected_covariance)
    assert torch.allclose(correlation, torch.ones((LATENT_DIM, LATENT_DIM), dtype=torch.float64))


def test_independent_prior_null_is_deterministic_and_scalar_only() -> None:
    first = prior_null_replicates(
        sample_count=12,
        replicates=5,
        seed=91,
        quantiles=theoretical_truncated_normal()["quantiles"],
    )
    second = prior_null_replicates(
        sample_count=12,
        replicates=5,
        seed=91,
        quantiles=theoretical_truncated_normal()["quantiles"],
    )
    assert first == second
    assert first["replicate_count"] == 5
    assert "samples" not in first
    assert len(first["statistics"]["mean_heterogeneity_std"]) == 5


def test_structure_and_pcr_classification_require_replicated_evidence() -> None:
    supported = {
        "selection": {"outside_upper": True},
        "confirmation": {"outside_upper": True},
    }
    panel_specific = {
        "selection": {"outside_upper": True},
        "confirmation": {"outside_upper": False},
    }
    assert classify_structure(supported) == (
        "P-S3_SUPPORTED",
        "REPLICATED_DIRECTION",
    )
    assert classify_structure(panel_specific) == ("PANEL_SPECIFIC", "PANEL_SPECIFIC")
    assert classify_pcr(
        contract_sanity=True,
        coordinate_consistent=True,
        observer_complete=True,
        panel_interpretable=True,
    ) == "PCR1_PRIOR_CONTRACT_INTERNALLY_COHERENT"
    assert classify_pcr(
        contract_sanity=False,
        coordinate_consistent=True,
        observer_complete=True,
        panel_interpretable=True,
    ) == "PCR2_PRIOR_PARAMETER_SEMANTICS_MISMATCH"
    assert classify_pcr(
        contract_sanity=True,
        coordinate_consistent=False,
        observer_complete=True,
        panel_interpretable=True,
    ) == "PCR3_PRIOR_COORDINATE_CONTRACT_INVALID"
    assert classify_pcr(
        contract_sanity=True,
        coordinate_consistent=True,
        observer_complete=False,
        panel_interpretable=True,
    ) == "PCR4_EXISTING_EVIDENCE_INSUFFICIENT"


def test_selection_confirmation_mapping_and_write_once_output(tmp_path: Path) -> None:
    assert LATENT_DIM == 100
    mapping = {"selection": 1000, "confirmation": 5000}
    assert mapping["selection"] != mapping["confirmation"]
    base = tmp_path / "lightgen_scaled_chen_prior_shape_decomposition_v1_attempt_01"
    first = next_write_once_output(base)
    first.mkdir()
    second = next_write_once_output(base)
    assert second.name.endswith("attempt_02")
