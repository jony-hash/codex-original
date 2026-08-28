from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.research import (
    optical_manifold_smf_bottleneck_attribution as audit,
)


def _root() -> Path:
    return Path(__file__).parents[2]


def test_bottleneck_config_and_review_are_locked() -> None:
    root = _root()
    spec = audit._load_spec(
        root / "configs/research/lightgen_optical_manifold_smf_bottleneck_attribution_v1.yaml"
    )
    audit._assert_contract(spec)
    review = audit._load_review(
        root / "configs/research/lightgen_optical_manifold_smf_bottleneck_attribution_review_v1.json",
        spec,
    )
    assert review.name.endswith("review_v1.json")
    assert audit.DOSE_MULTIPLIERS == (1.0, 2.0, 4.0, 8.0)
    assert audit.REGISTERED_LAMBDA == pytest.approx(0.0071572503388103665)


def test_data_panels_are_balanced_and_disjoint() -> None:
    data = audit._load_data(_root() / "data")
    all_panels = torch.cat((data.assessment_indices, *data.gradient_panels))
    assert all_panels.numel() == 1300
    assert torch.unique(all_panels).numel() == all_panels.numel()
    assert not torch.isin(all_panels, data.heldout_indices).any()
    assert [int(torch.bincount(data.labels.index_select(0, panel), minlength=10).min()) for panel in (data.assessment_indices, *data.gradient_panels)] == [100, 10, 10, 10]


def test_permuted_gradient_preserves_shapes_but_changes_direction() -> None:
    values = {
        "system.encoder.phase": torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
        "system.decoder.phase": torch.arange(12, 24, dtype=torch.float32).reshape(2, 2, 3),
    }
    generator = torch.Generator(device="cpu").manual_seed(audit.SEED)
    result = audit._permuted(values, generator)
    assert {name: tensor.shape for name, tensor in result.items()} == {name: tensor.shape for name, tensor in values.items()}
    assert not all(torch.equal(result[name], values[name]) for name in values)


def test_tree_equal_detects_mode_bank_or_model_mutation() -> None:
    left = {"input_modes": torch.ones(2, 3), "transport": torch.eye(2)}
    right = {"input_modes": torch.ones(2, 3), "transport": torch.eye(2)}
    changed = {"input_modes": torch.ones(2, 3), "transport": torch.zeros(2, 2)}
    assert audit._tree_equal(left, right)
    assert not audit._tree_equal(left, changed)


def test_source_training_is_the_authoritative_valid_failed_attempt() -> None:
    root = _root()
    payload = json.loads(
        (
            root
            / "outputs/20260828_v2/runs/lightgen_optical_manifold_smf_ols_loss_recovery_v1_attempt_02/training.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["evidence"]["status"] == "valid_failed"
    assert payload["contract"]["outcome"] == "valid_failed"
    assert payload["evidence"]["permitted_next_actions"] == [
        "manual_review_smf_bottleneck_attribution_contract"
    ]


def test_cli_router_is_lazy_for_bottleneck_audit() -> None:
    source = (
        _root() / "src/ldm_project/v2/application/_cli_research_optical_manifold.py"
    ).read_text(encoding="utf-8")
    assert "smf-bottleneck-attribution-validate" in source
    assert "optical_manifold_smf_bottleneck_attribution" in source
    assert "import optical_manifold_smf_bottleneck_attribution" not in source

