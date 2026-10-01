from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.research.scaled_chen.measure_scaled_chen_g1_prior_support_accessibility import DEFAULT_CONFIG_PATH, DEFAULT_REVIEW_PATH, LATENT_DIM, classify_p4_gate, exact_normalized_nearest_distances, bootstrap_ratio_ci, next_write_once_output, sample_registered_prior, validate_latent_bank, validate_registered_lineage


def test_latent_contract_is_exact_100d_real_nonnegative_finite() -> None:
    valid = torch.ones((3, LATENT_DIM), dtype=torch.float32)
    validate_latent_bank(valid)

    with pytest.raises(ValueError, match="100"):
        validate_latent_bank(torch.ones((3, LATENT_DIM - 1)))
    with pytest.raises(ValueError, match="real"):
        validate_latent_bank(torch.ones((3, LATENT_DIM), dtype=torch.complex64))
    with pytest.raises(ValueError, match="nonnegative"):
        validate_latent_bank(valid.clone().index_fill(1, torch.tensor([0]), -1.0))
    with pytest.raises(ValueError, match="finite"):
        validate_latent_bank(valid.clone().fill_(float("nan")))


def test_prior_sampler_matches_registered_truncated_normal() -> None:
    actual = sample_registered_prior(4, seed=1237)
    generator = torch.Generator(device="cpu").manual_seed(1237)
    from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt as bogt

    expected = bogt.truncated_normal_sample(
        (4, LATENT_DIM), generator=generator, device="cpu", dtype=torch.float32
    )
    assert torch.equal(actual, expected)
    validate_latent_bank(actual)


def test_exact_nearest_distance_uses_sqrt_100_normalization() -> None:
    support = torch.zeros((2, LATENT_DIM), dtype=torch.float32)
    support[1, 0] = 10.0
    query = torch.zeros((2, LATENT_DIM), dtype=torch.float32)
    query[0, 0] = 5.0
    query[1, 1] = 10.0

    distances = exact_normalized_nearest_distances(
        query, support, query_chunk_size=1, support_chunk_size=1
    )
    assert torch.allclose(
        distances,
        torch.tensor([0.5, 1.0], dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_exact_nearest_distance_is_chunk_invariant() -> None:
    generator = torch.Generator(device="cpu").manual_seed(17)
    support = torch.rand((19, LATENT_DIM), generator=generator)
    query = torch.rand((7, LATENT_DIM), generator=generator)
    reference = exact_normalized_nearest_distances(
        query, support, query_chunk_size=7, support_chunk_size=19
    )
    chunked = exact_normalized_nearest_distances(
        query, support, query_chunk_size=2, support_chunk_size=3
    )
    assert torch.equal(reference, chunked)


def test_bootstrap_ratio_ci_is_deterministic() -> None:
    holdout = torch.linspace(0.2, 0.8, 31, dtype=torch.float64)
    prior = holdout * 1.5
    first = bootstrap_ratio_ci(holdout, prior, replicates=2000, seed=1236)
    second = bootstrap_ratio_ci(holdout, prior, replicates=2000, seed=1236)
    assert first == second
    assert first["r50"] == pytest.approx(1.5)
    assert first["r95"] == pytest.approx(1.5)


def test_p4_gate_has_supported_not_supported_and_boundary_states() -> None:
    assert classify_p4_gate((1.1, 1.3), (1.2, 1.4)) == "P4_SUPPORTED"
    assert classify_p4_gate((0.8, 1.0), (0.7, 1.0)) == "P4_NOT_SUPPORTED"
    assert classify_p4_gate((0.8, 1.2), (0.7, 1.0)) == "P4_UNDERDETERMINED"
    assert classify_p4_gate((1.0, 1.4), (1.0, 1.4)) == "P4_UNDERDETERMINED"


def test_write_once_output_never_reuses_existing_attempt(tmp_path: Path) -> None:
    base = tmp_path / "lightgen_scaled_chen_g1_prior_support_accessibility_v1_attempt_01"
    first = next_write_once_output(base)
    first.mkdir()
    (first / "observer_manifest.json").write_text("{}", encoding="utf-8")
    second = next_write_once_output(base)
    assert second != first
    assert second.name.endswith("attempt_02")
    assert (first / "observer_manifest.json").read_text(encoding="utf-8") == "{}"


def test_registered_attempt_05_lineage_replays_without_loading_or_updating() -> None:
    lineage = validate_registered_lineage(DEFAULT_CONFIG_PATH, DEFAULT_REVIEW_PATH)
    assert lineage["task_id"] == "H-LOMA-GEN-G1-SA"
    assert lineage["mra_attempt"] == "lightgen_optical_manifold_smf_bogt_manifold_readiness_v1_attempt_05"
    assert lineage["checkpoint_step"] == 5000
    assert lineage["parent_prior"]["kind"] == "truncated_normal"
