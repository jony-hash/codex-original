from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.research import (
    optical_manifold_smf_vae_kl_effect as effect,
)


def test_effect_contract_locks_corrected_candidate_beta() -> None:
    root = Path(__file__).parents[2]
    spec = effect._load_spec(
        root / "configs/research/lightgen_optical_manifold_smf_vae_kl_effect_v1.yaml"
    )
    effect._assert_contract(spec)
    review = effect._load_review(
        root / "configs/research/lightgen_optical_manifold_smf_vae_kl_effect_review_v1.json",
        spec,
    )
    assert review.name.endswith("review_v1.json")
    assert effect._beta_for_arm("smf_current_beta_vae", effect.REGISTERED_BETA) == pytest.approx(effect.REGISTERED_BETA)
    assert effect._beta_for_arm("smf_candidate_beta_vae", effect.REGISTERED_BETA) == pytest.approx(0.014605162095621654)
    assert effect._beta_for_arm("smf_posterior_no_kl", effect.REGISTERED_BETA) == 0.0


def test_candidate_beta_is_registered_beta_times_selected_multiplier() -> None:
    assert effect.CANDIDATE_BETA == pytest.approx(
        effect.REGISTERED_BETA * effect.CANDIDATE_MULTIPLIER
    )


def test_phase_update_gate_rejects_empty_or_frozen_layers() -> None:
    complete = {
        "update_norms": {
            name: 1.0 for name in effect.base.EXPECTED_TRAINABLE
        }
        | {"posterior.mean_delta.weight": 1.0}
    }
    assert effect._phase_update_gate([complete], posterior_required=True)
    empty = {"update_norms": {name: 0.0 for name in effect.base.EXPECTED_TRAINABLE}}
    assert not effect._phase_update_gate([empty], posterior_required=False)
    assert not effect._phase_update_gate([], posterior_required=False)


def test_tree_equal_is_exact_for_resume_state() -> None:
    left = {"weight": torch.ones(2, 2), "cursor": {"step": 2}}
    right = {"weight": torch.ones(2, 2), "cursor": {"step": 2}}
    changed = {"weight": torch.ones(2, 2), "cursor": {"step": 3}}
    assert effect._tree_equal(left, right)
    assert not effect._tree_equal(left, changed)


def test_safety_gate_compares_reconstruction_mse_not_beta_kl_objective() -> None:
    report = {"finite": True, "mse": 0.09, "loss": 0.14}
    assert effect._safety_ok(report, 0.05)
    assert not effect._safety_ok({"finite": True, "mse": 0.11, "loss": 0.15}, 0.05)
