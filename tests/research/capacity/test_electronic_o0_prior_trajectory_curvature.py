from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from ldm_project.v2.application import _cli_schema_research
from ldm_project.v2.application.research.capacity import electronic_o0_prior_trajectory_curvature as tca


def test_registered_config_is_exact() -> None:
    root = Path(__file__).parents[3]
    config = tca._config(root / "configs/research/capacity/electronic_o0_prior_trajectory_curvature_v1.yaml")
    assert config["trajectory_seed"] == 8741
    assert config["historical_seed"] == 8601
    assert config["historical_steps"] == 625
    assert config["landmark_steps"][-1] == 625


def test_cli_registers_manifest_only_tca_actions() -> None:
    parser = _cli_schema_research._parser() if hasattr(_cli_schema_research, "_parser") else None
    # Build through the public v2 parser; importing the schema module directly
    # must remain side-effect free.
    from ldm_project.v2.application.cli import _parser

    parser = _parser()
    for action in ("validate", "preflight", "audit", "report", "closeout"):
        args = parser.parse_args(
            [
                "research",
                "electronic-o0-prior-trajectory-curvature",
                action,
                "--run-manifest",
                "manifest.yaml",
            ]
        )
        assert args.research_action == action


def test_rho_and_direction_delta_are_finite_and_signed() -> None:
    assert tca._rho(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])) == pytest.approx(1.0)
    assert tca._rho(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])) == pytest.approx(-1.0)
    first = nn.Parameter(torch.ones(2))
    second = nn.Parameter(torch.ones(1))
    values = tca._direction_delta((first, second), torch.tensor([1.0, 2.0, 3.0]))
    assert len(values) == 2
    assert all(torch.isfinite(value).all() for value in values)


def test_state_hash_uses_tensor_content_not_serialization() -> None:
    state = {"weight": torch.tensor([1.0, 2.0])}
    assert tca._state_dict_hash(state) == tca._state_dict_hash({"weight": state["weight"].clone()})
    assert tca._state_dict_hash({"weight": torch.tensor([1.0, 3.0])}) != tca._state_dict_hash(state)


def _landmark(
    *,
    cycle_rho: float,
    adam_rho: float = -0.2,
    cycle_delta: float = -1.0e-8,
    training_cycle_delta: float | None = None,
) -> dict:
    records = []
    for batch_seed in tca.BATCH_SEEDS:
        for projection_seed in tca.PROJECTION_SEEDS:
            records.append(
                {
                    "batch_seed": batch_seed,
                    "projection_seed": projection_seed,
                    "fixed_rho": {"cycle": cycle_rho},
                    "adam_rho": adam_rho,
                    "actual_delta": {"cycle": cycle_delta},
                }
            )
    landmark = {"step": len(records), "records": records, "curvature": []}
    if training_cycle_delta is not None:
        landmark["training_cycle_delta"] = training_cycle_delta
    return landmark


def test_classification_detects_shared_gradient_rotation() -> None:
    arms = {
        "swae": {"landmarks": [_landmark(cycle_rho=0.2) for _ in range(4)]},
        "sced": {"landmarks": [_landmark(cycle_rho=0.2) for _ in range(4)]},
    }
    classification, next_action, decision = tca._classify(arms)
    assert classification == "trajectory_gradient_rotation"
    assert next_action == "register_trajectory_mechanism_hypothesis"
    assert decision["alignment_estimator_independent"] is True


def test_classification_keeps_underdetermined_path_fail_closed() -> None:
    arms = {
        "swae": {"landmarks": [_landmark(cycle_rho=-0.2) for _ in range(4)]},
        "sced": {"landmarks": [_landmark(cycle_rho=-0.2) for _ in range(4)]},
    }
    classification, next_action, decision = tca._classify(arms)
    assert classification == "trajectory_attribution_underdetermined"
    assert next_action == "register_denser_local_trajectory_attribution"
    assert decision["shared_supported_mechanisms"] == []


def test_observer_mismatch_requires_nonincreasing_training_batch() -> None:
    mismatch_arms = {
        route: {
            "landmarks": [
                _landmark(cycle_rho=-0.2, cycle_delta=2.0e-7, training_cycle_delta=-1.0e-8)
                for _ in range(4)
            ]
        }
        for route in ("swae", "sced")
    }
    classification, _, decision = tca._classify(mismatch_arms)
    assert classification == "observer_distribution_mismatch"
    assert decision["shared_supported_mechanisms"] == ["observer_distribution_mismatch"]

    observer_only_arms = {
        route: {
            "landmarks": [
                _landmark(cycle_rho=-0.2, cycle_delta=2.0e-7, training_cycle_delta=2.0e-7)
                for _ in range(4)
            ]
        }
        for route in ("swae", "sced")
    }
    classification, _, decision = tca._classify(observer_only_arms)
    assert classification == "trajectory_attribution_underdetermined"
    assert decision["shared_supported_mechanisms"] == []


def test_curvature_support_requires_first_order_and_remainder_fields() -> None:
    base = _landmark(cycle_rho=-0.2)
    valid_curvature = {
        "direction": "adam",
        "finite": True,
        "hvp_quadratic": 1.0,
        "finite_difference_forward": 2.0e-6,
        "first_order_cycle": 0.0,
        "actual_cycle_delta": 2.0e-6,
        "taylor_remainder_cycle": 1.0e-6,
    }
    invalid_curvature = dict(valid_curvature)
    invalid_curvature.pop("first_order_cycle")
    invalid_curvature.pop("taylor_remainder_cycle")
    valid_arms = {
        route: {"landmarks": [{**base, "curvature": [dict(valid_curvature)]} for _ in range(2)]}
        for route in ("swae", "sced")
    }
    assert tca._classify(valid_arms)[0] == "finite_step_curvature_cycle_drift"
    invalid_arms = {
        route: {"landmarks": [{**base, "curvature": [dict(invalid_curvature)]} for _ in range(2)]}
        for route in ("swae", "sced")
    }
    assert tca._classify(invalid_arms)[0] == "trajectory_attribution_underdetermined"


def test_progress_schema_contains_partial_history_window(tmp_path: Path) -> None:
    encoder = nn.Linear(2, 2)
    decoder = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(tuple(encoder.parameters()) + tuple(decoder.parameters()), lr=1e-3)
    target = tmp_path / "progress.pt"
    tca._save_progress(
        target,
        route="sced",
        step=1,
        epoch=0,
        cursor=1,
        encoder=encoder,
        decoder=decoder,
        optimizer=optimizer,
        history=[],
        history_window=[{"total": 1.0}],
        landmarks=[],
    )
    payload = tca.safe_torch_load(target, map_location="cpu")
    assert payload["schema"] == "electronic_o0_prior_trajectory_curvature_replay.v1"
    assert payload["history_window"] == [{"total": 1.0}]
