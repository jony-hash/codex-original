from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from ldm_project.v2.application import _cli_schema_research
from ldm_project.v2.application.research import electronic_o0_prior_trajectory_curvature as tca


def test_registered_config_is_exact() -> None:
    root = Path(__file__).parents[2]
    config = tca._config(root / "configs/research/electronic_o0_prior_trajectory_curvature_v1.yaml")
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


def _landmark(*, cycle_rho: float, adam_rho: float = -0.2, cycle_delta: float = -1.0e-8) -> dict:
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
    return {"step": len(records), "records": records, "curvature": []}


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
