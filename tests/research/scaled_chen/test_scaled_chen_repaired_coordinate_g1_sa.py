from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from ldm_project.v2.evidence import tensor_sha256
from scripts.research.scaled_chen.replay_scaled_chen_repaired_coordinate_g1_sa import RC1_CALIBRATION_WAS_DOMINANT_DRIVER, RC2_CALIBRATION_WAS_PARTIAL_DRIVER_P4_REMAINS, RC4_CALIBRATION_REPAIR_DOES_NOT_RESOLVE_MISMATCH, RC5_UNDERDETERMINED, P4_NOT_SUPPORTED_REPAIRED, P4_SUPPORTED_REPAIRED, P4_UNDERDETERMINED_REPAIRED, bootstrap_ratio_ci, audit_layout_manifest, classify_final_causal_result, classify_repaired_p4_gate, compare_old_and_repaired_support, exact_normalized_nearest_distances, learned_state_freeze, next_write_once_output, sample_registered_prior, state_digest, validate_latent_bank, validate_repaired_calibration_identity, validate_support_sample_identities


def _valid_repaired_identity() -> dict[str, object]:
    return {
        "schema": "scaled_chen_bogt_calibration.v2",
        "contract_version": "unit_aware_raw_mean_v2",
        "raw_port_mean": torch.ones(100, dtype=torch.float32),
        "port_scale": torch.ones(100, dtype=torch.float32),
        "sample_count": 1000,
        "port_count": 100,
        "calibration_indices_sha256": "indices",
        "dataset_manifest_hash": "dataset",
        "mode_bank_sha256": "mode-bank",
        "zero_phase_reference": "independent_train_only_zero_phase",
        "plane_identity": "pre_ols_plane_power_reference_v1",
        "raw_port_mean_equals_port_scale": True,
    }


def _support_metrics(
    h50: float, h95: float, p50: float, p95: float, r50: float, r95: float
) -> dict[str, object]:
    return {
        "holdout": {"median": h50, "p95": h95},
        "prior": {"median": p50, "p95": p95},
        "ratios": {"r50": r50, "r95": r95},
    }


def test_repaired_calibration_identity_rejects_old_v1() -> None:
    validate_repaired_calibration_identity(
        _valid_repaired_identity(),
        expected_schema="scaled_chen_bogt_calibration.v2",
        expected_contract="unit_aware_raw_mean_v2",
        expected_calibration_indices_sha256="indices",
        expected_dataset_manifest_hash="dataset",
        expected_mode_bank_sha256="mode-bank",
    )

    with pytest.raises(ValueError, match="calibration schema"):
        validate_repaired_calibration_identity(
            _valid_repaired_identity() | {"schema": "scaled_chen_bogt_calibration.v1"},
            expected_schema="scaled_chen_bogt_calibration.v2",
            expected_contract="unit_aware_raw_mean_v2",
            expected_calibration_indices_sha256="indices",
            expected_dataset_manifest_hash="dataset",
            expected_mode_bank_sha256="mode-bank",
        )


def test_support_sample_identity_replay_is_exact_and_fail_closed() -> None:
    train = torch.arange(5, dtype=torch.long)
    heldout = torch.tensor([5, 6, 7], dtype=torch.long)
    result = validate_support_sample_identities(
        train,
        heldout,
        expected_train_start=0,
        expected_train_count=5,
        expected_heldout_indices=heldout,
        expected_heldout_sha256=tensor_sha256(heldout),
    )
    assert result["train_support_count"] == 5
    assert result["heldout_query_count"] == 3
    with pytest.raises(ValueError, match="heldout indices"):
        validate_support_sample_identities(
            train,
            heldout.flip(0),
            expected_train_start=0,
            expected_train_count=5,
            expected_heldout_indices=heldout,
            expected_heldout_sha256=tensor_sha256(heldout),
        )


def test_prior_seed_and_exact_distance_are_deterministic_and_chunk_invariant() -> None:
    first = sample_registered_prior(7, seed=1237)
    second = sample_registered_prior(7, seed=1237)
    assert torch.equal(first, second)
    validate_latent_bank(first)

    generator = torch.Generator(device="cpu").manual_seed(17)
    support = torch.rand((19, 100), generator=generator)
    query = torch.rand((7, 100), generator=generator)
    reference = exact_normalized_nearest_distances(
        query, support, query_chunk_size=7, support_chunk_size=19
    )
    chunked = exact_normalized_nearest_distances(
        query, support, query_chunk_size=2, support_chunk_size=3
    )
    assert torch.equal(reference, chunked)


def test_bootstrap_and_repaired_p4_boundaries_reuse_registered_rule() -> None:
    holdout = torch.linspace(0.2, 0.8, 31, dtype=torch.float64)
    prior = holdout * 1.5
    assert bootstrap_ratio_ci(holdout, prior, seed=1236) == bootstrap_ratio_ci(
        holdout, prior, seed=1236
    )
    assert classify_repaired_p4_gate((1.1, 1.3), (1.2, 1.4)) == P4_SUPPORTED_REPAIRED
    assert classify_repaired_p4_gate((0.8, 1.0), (0.7, 1.0)) == P4_NOT_SUPPORTED_REPAIRED
    assert classify_repaired_p4_gate((0.8, 1.2), (0.7, 1.0)) == P4_UNDERDETERMINED_REPAIRED
    assert classify_repaired_p4_gate((1.0, 1.4), (1.0, 1.4)) == P4_UNDERDETERMINED_REPAIRED


def test_old_new_comparison_exposes_only_registered_support_metrics() -> None:
    old = _support_metrics(0.001, 0.002, 1.0, 2.0, 1000.0, 1000.0)
    repaired = _support_metrics(0.002, 0.004, 1.0, 2.0, 500.0, 500.0)
    comparison = compare_old_and_repaired_support(old, repaired)
    assert comparison["old_to_repaired"]["H50"] == {"old": 0.001, "repaired": 0.002}
    assert comparison["reduction_factor"]["R50"] == pytest.approx(2.0)
    assert set(comparison["old_to_repaired"]) == {"H50", "H95", "P50", "P95", "R50", "R95"}


def test_final_causal_classification_has_registered_rc_boundaries() -> None:
    rc1 = classify_final_causal_result(
        P4_NOT_SUPPORTED_REPAIRED,
        old_ratios={"r50": 1681.0, "r95": 1046.0},
        repaired_ratios={"r50": 1.1, "r95": 1.2},
        selection_mean=1.01,
        confirmation_mean=0.99,
        prior_contract_coherent=True,
    )
    assert rc1["classification"] == RC1_CALIBRATION_WAS_DOMINANT_DRIVER

    rc2 = classify_final_causal_result(
        P4_SUPPORTED_REPAIRED,
        old_ratios={"r50": 1681.0, "r95": 1046.0},
        repaired_ratios={"r50": 2.0, "r95": 1.5},
        selection_mean=1.01,
        confirmation_mean=0.99,
        prior_contract_coherent=True,
    )
    assert rc2["classification"] == RC2_CALIBRATION_WAS_PARTIAL_DRIVER_P4_REMAINS

    rc4 = classify_final_causal_result(
        P4_SUPPORTED_REPAIRED,
        old_ratios={"r50": 1681.0, "r95": 1046.0},
        repaired_ratios={"r50": 500.0, "r95": 400.0},
        selection_mean=1.01,
        confirmation_mean=0.99,
        prior_contract_coherent=True,
    )
    assert rc4["classification"] == RC4_CALIBRATION_REPAIR_DOES_NOT_RESOLVE_MISMATCH

    rc5 = classify_final_causal_result(
        P4_UNDERDETERMINED_REPAIRED,
        old_ratios={"r50": 1681.0, "r95": 1046.0},
        repaired_ratios={"r50": 2.0, "r95": 1.5},
        selection_mean=1.01,
        confirmation_mean=0.99,
        prior_contract_coherent=True,
    )
    assert rc5["classification"] == RC5_UNDERDETERMINED


def test_write_once_and_learned_state_freeze(tmp_path: Path) -> None:
    requested = tmp_path / "lightgen_scaled_chen_g1_repaired_attempt_01"
    first = next_write_once_output(requested)
    first.mkdir()
    second = next_write_once_output(requested)
    assert second.name.endswith("attempt_02")

    module = nn.Linear(3, 2)
    before = state_digest(module)
    _ = module(torch.ones((1, 3)))
    after = state_digest(module)
    freeze = learned_state_freeze(before, after, model_training=False)
    assert freeze["state_unchanged"] is True
    assert freeze["optimizer_created"] is False
    with pytest.raises(ValueError, match="learned state"):
        learned_state_freeze(before, "different", model_training=False)


def test_single_panel_alignment_manifest_is_fail_closed(tmp_path: Path) -> None:
    for stem in ("support_distance_distribution", "support_distance_summary"):
        (tmp_path / f"{stem}.alignment.json").write_text(
            "{\"schema\": \"lightgen_scaled_chen_repaired_coordinate_g1_sa_figure_alignment.v1\", \"figure\": \"%s\", \"backend\": \"matplotlib\", \"verdict\": \"NOT_APPLICABLE_SINGLE_PANEL\", \"panel_count\": 1, \"axes_count\": 1, \"text_editable\": true, \"reason\": \"single-panel latent support-accessibility diagnostic figure\"}" % stem,
            encoding="utf-8",
        )
    audit_layout_manifest(tmp_path)
    (tmp_path / "support_distance_summary.alignment.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="alignment manifest"):
        audit_layout_manifest(tmp_path)
