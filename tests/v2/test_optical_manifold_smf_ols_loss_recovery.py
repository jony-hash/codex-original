from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.research import (
    optical_manifold_smf_ols_loss_recovery as recovery,
)


def test_fixed_ols_contract_and_review_are_locked() -> None:
    root = Path(__file__).parents[2]
    spec = recovery._load_spec(
        root / "configs/research/lightgen_optical_manifold_smf_ols_loss_recovery_v1.yaml"
    )
    recovery._assert_contract(spec)
    review = recovery._load_review(
        root
        / "configs/research/lightgen_optical_manifold_smf_ols_loss_recovery_review_v1.json",
        spec,
    )
    assert review.name.endswith("review_v1.json")
    assert recovery.CANDIDATE_BETA == pytest.approx(
        8.0 * 0.0018256452619527068
    )
    assert recovery.MIN_VARIANCE_RETENTION == 1.0


def test_update_layer_gate_requires_actual_per_layer_updates() -> None:
    complete = {
        "system.encoder.phase[0]": 1.0,
        "system.encoder.phase[1]": 1.0,
        "system.decoder.phase[0]": 1.0,
        "system.decoder.phase[1]": 1.0,
    }
    assert recovery.canary._four_phase_nonzero(complete)
    assert not recovery.canary._four_phase_nonzero(
        complete | {"system.encoder.phase[0]": 0.0}
    )
    assert not recovery.canary._four_phase_nonzero(
        {"system.encoder.phase": 1.0, "system.decoder.phase": 1.0}
    )


def test_projection_loss_is_not_a_free_gate() -> None:
    assert recovery.MIN_VARIANCE_RETENTION > 0.0
    assert recovery.SAFETY_MSE_RATIO == pytest.approx(2.0)


def test_tree_equal_is_exact_for_fixed_ols_resume() -> None:
    left = {"phase": torch.ones(2, 3), "step": 2}
    right = {"phase": torch.ones(2, 3), "step": 2}
    changed = {"phase": torch.ones(2, 3), "step": 3}
    assert recovery._tree_equal(left, right)
    assert not recovery._tree_equal(left, changed)


def test_projection_gradient_layer_norms_are_explicit() -> None:
    values = {
        "system.encoder.phase": torch.ones(2, 3),
        "system.decoder.phase": torch.zeros(2, 3),
    }
    norms = recovery._layer_norms(values)
    assert norms["system.encoder.phase[0]"] == pytest.approx(3**0.5)
    assert norms["system.encoder.phase[1]"] == pytest.approx(3**0.5)
    assert norms["system.decoder.phase[0]"] == 0.0


def test_cli_dispatch_keeps_fixed_ols_module_lazy() -> None:
    source = (
        Path(__file__).parents[2]
        / "src/ldm_project/v2/application/_cli_research_optical_manifold.py"
    ).read_text(encoding="utf-8")
    assert "optical_manifold_smf_ols_loss_recovery" in source
    assert "import optical_manifold_smf_ols_loss_recovery" not in source
