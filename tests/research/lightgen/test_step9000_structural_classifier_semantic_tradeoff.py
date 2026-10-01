from __future__ import annotations

import pytest
import torch

from scripts.research.lightgen import execute_lightgen_2d_attribute_step9000_structural_classifier_semantic_tradeoff_v1 as attribution


def test_fixed_dev_subset_is_class_balanced_and_deterministic() -> None:
    labels = torch.arange(10).repeat_interleave(12)
    development_indices = torch.arange(labels.numel())

    selected = attribution._select_dev_subset(labels, development_indices)

    assert selected.tolist() == torch.cat([torch.arange(d * 12, d * 12 + 10) for d in range(10)]).tolist()
    assert torch.bincount(labels.index_select(0, selected), minlength=10).tolist() == [10] * 10


def test_joint_directions_use_the_frozen_rms_contract() -> None:
    tensors = {
        "g_structural": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "g_E_S": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "g_E_I": torch.tensor([0.0, 0.0, 1.0, 0.0]),
        "g_E_C": torch.tensor([0.0, 0.0, 0.0, 1.0]),
    }

    directions, details = attribution._directions({}, tensors)

    assert set(directions) == {"d_struct", "d_SIC", "d_joint"}
    for value in directions.values():
        assert torch.isclose(value.square().mean().sqrt(), torch.tensor(1.0), atol=1.0e-6)
    assert details["v_joint_l2_norm"] > 0.0


def test_stable_descent_rejects_a_wrong_signed_probe() -> None:
    records = {
        "d_joint": [
            {
                "plus": {"deltas": {"L_struct": -1.0}},
                "minus": {"deltas": {"L_struct": 1.0}},
            }
        ]
    }

    assert attribution._stable_descent(records, "d_joint", "L_struct") is True
    records["d_joint"][0]["minus"]["deltas"]["L_struct"] = -1.0
    assert attribution._stable_descent(records, "d_joint", "L_struct") is False


def test_zero_joint_direction_is_rejected() -> None:
    tensors = {
        "g_structural": torch.tensor([1.0, 0.0]),
        "g_E_S": torch.tensor([-1.0, 0.0]),
        "g_E_I": torch.tensor([-1.0, 0.0]),
        "g_E_C": torch.tensor([-1.0, 0.0]),
    }

    with pytest.raises(attribution.TradeoffAttributionError):
        attribution._directions({}, tensors)
