from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from ldm_project.v2.application.research.capacity import electronic_o0_prior_gradient_conflict as pgc


def test_halfspace_projection_keeps_unconstrained_target() -> None:
    target = torch.tensor([-1.0, -2.0])
    first = torch.tensor([1.0, 0.0])
    second = torch.tensor([0.0, 1.0])
    projected, record = pgc._project_halfspaces(target, first, second)
    assert torch.equal(projected, target)
    assert record["active_constraints"] == []
    assert record["alignment_retention"] == pytest.approx(1.0)


def test_halfspace_projection_solves_single_active_constraint() -> None:
    target = torch.tensor([1.0, -1.0])
    first = torch.tensor([1.0, 0.0])
    second = torch.tensor([0.0, 1.0])
    projected, record = pgc._project_halfspaces(target, first, second)
    assert projected == pytest.approx(torch.tensor([0.0, -1.0]))
    assert float(torch.dot(first, projected)) <= 1.0e-6
    assert float(torch.dot(second, projected)) <= 1.0e-6
    assert record["active_constraints"] == ["reconstruction"]


def test_halfspace_projection_solves_double_active_constraint() -> None:
    target = torch.tensor([1.0, 1.0])
    first = torch.tensor([1.0, 0.0])
    second = torch.tensor([0.0, 1.0])
    projected, record = pgc._project_halfspaces(target, first, second)
    assert projected == pytest.approx(torch.zeros(2))
    assert record["active_constraints"] == ["reconstruction", "cycle"]
    assert record["alignment_retention"] == pytest.approx(0.0)


def test_gradient_tree_maps_none_to_zero_and_aggregates() -> None:
    encoder_parameter = nn.Parameter(torch.ones(2))
    decoder_parameter = nn.Parameter(torch.ones(3))
    params = (encoder_parameter, decoder_parameter)
    groups = {
        "encoder_features": (encoder_parameter,),
        "encoder_label_projection": tuple(),
        "decoder_input_label": (decoder_parameter,),
        "decoder_block": tuple(),
        "decoder_upsample_output": tuple(),
    }
    result = pgc._gradient_tree(
        (torch.tensor([1.0, 2.0]), None),
        params,
        groups,
        device=torch.device("cpu"),
    )
    assert torch.equal(result["encoder_features"], torch.tensor([1.0, 2.0]))
    assert torch.equal(result["decoder_input_label"], torch.zeros(3))
    assert torch.equal(result["full_joint"], torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0]))


def test_measurement_valid_rejects_missing_required_gradients() -> None:
    record = {
        "finite": True,
        "alignment_decoder_gradient_norm": 0.0,
        "groups": {
            "encoder_aggregate": {
                "gradient_norms": {"reconstruction": 0.0, "alignment": 1.0, "cycle": 1.0}
            },
            "decoder_aggregate": {
                "gradient_norms": {"reconstruction": 1.0, "alignment": 0.0, "cycle": 1.0}
            },
        },
        "finite_difference": {},
    }
    valid, failures = pgc._measurement_valid([record])
    assert not valid
    assert "measurement_0_encoder_aggregate_reconstruction_zero" in failures


def test_registered_config_is_exact() -> None:
    root = Path(__file__).parents[3]
    config = pgc._config(root / "configs/research/capacity/electronic_o0_prior_gradient_conflict_v1.yaml")
    assert config["audit_seed"] == 8731
    assert config["batch_seeds"] == [11, 23, 37, 51]
    assert config["projection_seeds"] == [271829, 314159]


def test_classification_keeps_paired_arm_and_requires_each_bank() -> None:
    def record(seed: int, cycle_rho: float) -> dict:
        return {
            "projection_seed": seed,
            "groups": {
                "full_joint": {
                    "fixed_rho": {"cycle": cycle_rho},
                    "projected": {"alignment_retention": 1.0},
                }
            },
            "finite_difference": {
                "projected": {
                    "alignment": {"plus": 0.0, "before": 1.0},
                    "reconstruction": {"plus": 1.0, "before": 1.0},
                    "cycle": {"plus": 1.0, "before": 1.0},
                }
            },
        }

    sced = [record(seed, -0.2) for seed in (271829, 314159) for _ in range(4)]
    swae = [record(seed, 0.2) for seed in (271829, 314159) for _ in range(4)]
    classification, _, decision = pgc._classify(
        sced, swae, [0.95] * 4, controls_passed=True
    )
    assert classification == "gradient_conflict_not_supported"
    assert decision["swae_cycle_rho_by_bank"][271829] == [0.2] * 4
    assert decision["alignment_estimator_independent"] is True
    assert decision["sced_specific"] is False
