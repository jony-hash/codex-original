from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.research.scaled_chen.replay_scaled_chen_repaired_coordinate_mra import COORDINATE_METRIC_NAMES, MRA_PRIOR_CONTRACT_REVIEW_REQUIRED, MRA_REPAIRED_COORDINATE_VALID, MRA_REPLAY_INVALID, assess_prior_shape, classify_stage_a_gate, compare_metric, mra_participation_rank, next_write_once_output, repaired_latent_anchor, state_digest, validate_checkpoint_payload, validate_repaired_calibration_identity


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


def _valid_checks() -> dict[str, bool]:
    return {
        "calibration_identity": True,
        "checkpoint_identity": True,
        "latent_anchor": True,
        "finite": True,
        "invariant_controls": True,
        "learned_state_freeze": True,
        "mra_metrics_replayed": True,
    }


def test_repaired_calibration_identity_is_exact_and_v1_is_rejected() -> None:
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


def test_checkpoint_identity_requires_registered_schema_step_and_state() -> None:
    payload = {
        "schema": "scaled_chen_bogt_horizon_checkpoint.v1",
        "global_step": 5000,
        "model_state": {"weight": torch.ones(1)},
    }
    validate_checkpoint_payload(payload, expected_step=5000)

    with pytest.raises(ValueError, match="checkpoint"):
        validate_checkpoint_payload(payload | {"global_step": 1000}, expected_step=5000)


def test_repaired_latent_anchor_requires_finite_nonnegative_100d_values() -> None:
    z = torch.ones((4, 100), dtype=torch.float32)
    anchor = repaired_latent_anchor(z)
    assert anchor["shape"] == [4, 100]
    assert anchor["dtype"] == "torch.float32"
    assert anchor["finite"] is True
    assert anchor["nonnegative"] is True
    assert anchor["global_mean"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="100"):
        repaired_latent_anchor(torch.ones((4, 99)))
    with pytest.raises(ValueError, match="nonnegative"):
        repaired_latent_anchor(z.clone().index_fill(1, torch.tensor([0]), -1.0))


def test_mra_participation_rank_matches_the_centered_svd_definition() -> None:
    generator = torch.Generator(device="cpu").manual_seed(11)
    value = torch.randn((12, 100), generator=generator)
    expected = float(
        1.0
        / (
            (
                torch.linalg.svdvals(value - value.mean(dim=0, keepdim=True))
                .square()
                / torch.linalg.svdvals(value - value.mean(dim=0, keepdim=True))
                .square()
                .sum()
            )
            .square()
            .sum()
        )
    )
    assert mra_participation_rank(value) == pytest.approx(expected)


def test_old_new_metric_mapping_reports_direction_without_a_new_gate() -> None:
    assert set(COORDINATE_METRIC_NAMES) == {
        "z_mean",
        "z_variance",
        "gaussian_kl",
        "numerical_truncated_normal_kl",
        "power_effective_rank",
    }
    assert compare_metric("z_mean", 0.2, 0.9)["direction"] == "IMPROVED"
    assert compare_metric("gaussian_kl", 3.0, 2.0)["direction"] == "IMPROVED"
    assert compare_metric("power_effective_rank", 4.0, 3.0)["direction"] == "WORSENED"
    assert compare_metric("z_variance", 0.25, 0.25 + 1.0e-12)["direction"] == "UNCHANGED_WITHIN_TOLERANCE"


def test_stage_a_gate_has_valid_prior_review_and_invalid_states() -> None:
    checks = _valid_checks()
    assert classify_stage_a_gate(checks, {"dominant_inconsistency": False}) == MRA_REPAIRED_COORDINATE_VALID
    assert classify_stage_a_gate(checks, {"dominant_inconsistency": True}) == MRA_PRIOR_CONTRACT_REVIEW_REQUIRED
    assert classify_stage_a_gate(checks | {"invariant_controls": False}, {"dominant_inconsistency": True}) == MRA_REPLAY_INVALID


def test_prior_shape_review_requires_mean_alignment_and_both_panels() -> None:
    aligned_shape_mismatch = {
        "selection": {
            "z_mean": 1.02,
            "z_variance": 1.50,
            "gaussian_kl_decomposition": {"mean_offset_term": 0.01, "variance_shape_term": 1.1},
            "gaussian_kl": 1.11,
            "numerical_truncated_normal_kl": 2.0,
        },
        "confirmation": {
            "z_mean": 0.98,
            "z_variance": 1.40,
            "gaussian_kl_decomposition": {"mean_offset_term": 0.01, "variance_shape_term": 1.0},
            "gaussian_kl": 1.01,
            "numerical_truncated_normal_kl": 2.1,
        },
    }
    assessment = assess_prior_shape(aligned_shape_mismatch)
    assert assessment["dominant_inconsistency"] is True
    assert "variance" in assessment["dominant_assumption"]

    mean_misaligned = aligned_shape_mismatch | {
        "selection": aligned_shape_mismatch["selection"] | {"z_mean": 0.2}
    }
    assert assess_prior_shape(mean_misaligned)["dominant_inconsistency"] is False


def test_write_once_output_never_reuses_an_existing_attempt(tmp_path: Path) -> None:
    requested = tmp_path / "lightgen_scaled_chen_repaired_coordinate_mra_v1_attempt_01"
    first = next_write_once_output(requested)
    first.mkdir()
    second = next_write_once_output(requested)
    assert first != second
    assert second.name.endswith("attempt_02")


def test_learned_state_digest_is_stable_without_mutation() -> None:
    module = nn.Linear(3, 2)
    before = state_digest(module)
    _ = module(torch.ones((1, 3)))
    after = state_digest(module)
    assert before == after
