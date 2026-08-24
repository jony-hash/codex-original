from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ldm_project.v2.application import _cli_schema_research
from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research import electronic_o0_prior_dense_local_attribution as dlta


def test_registered_config_and_fixed_landmarks_are_exact() -> None:
    root = Path(__file__).parents[2]
    config = dlta._config(root / "configs/research/electronic_o0_prior_dense_local_attribution_v1.yaml")
    assert config["dense_local_attribution_seed"] == 8751
    assert config["landmark_steps"] == list(dlta.LANDMARK_STEPS)
    assert len(dlta.LANDMARK_STEPS) == 39
    assert dlta.LANDMARK_STEPS[-1] == 625


def test_cli_registers_manifest_only_dlta_actions() -> None:
    assert _cli_schema_research is not None
    for action in ("validate", "preflight", "audit", "report", "closeout"):
        args = _parser().parse_args(
            [
                "research",
                "electronic-o0-prior-dense-local-attribution",
                action,
                "--run-manifest",
                "manifest.yaml",
            ]
        )
        assert args.research_action == action


def test_adamw_predicted_displacement_matches_one_real_step() -> None:
    encoder = nn.Linear(3, 2)
    decoder = nn.Linear(2, 3)
    optimizer = torch.optim.AdamW(
        tuple(encoder.parameters()) + tuple(decoder.parameters()),
        lr=dlta.LEARNING_RATE,
        betas=dlta.BETAS,
        eps=dlta.EPS,
        weight_decay=dlta.WEIGHT_DECAY,
    )
    x = torch.randn(4, 3)
    loss = decoder(encoder(x)).square().mean()
    params = tuple(encoder.parameters()) + tuple(decoder.parameters())
    gradients = torch.autograd.grad(loss, params, retain_graph=True)
    predicted = dlta._decompose_adamw(encoder, decoder, optimizer, gradients)["predicted"]
    before = torch.cat([parameter.detach().reshape(-1) for parameter in params])
    loss.backward()
    optimizer.step()
    after = torch.cat([parameter.detach().reshape(-1) for parameter in params])
    gate = dlta._component_gate(predicted, after - before)
    assert gate["finite"] is True
    assert gate["passed"] is True


def _landmark(*, rho: float = -0.2, training_delta: float = -1.0e-6) -> dict:
    records = []
    for batch_seed in dlta.BATCH_SEEDS:
        for projection_seed in dlta.PROJECTION_SEEDS:
            records.append(
                {
                    "batch_seed": batch_seed,
                    "projection_seed": projection_seed,
                    "losses": {"cycle": 0.01},
                    "component_rho": {
                        "gradient": rho,
                        "momentum": rho,
                        "adaptive": rho,
                        "predicted": rho,
                        "decay": -0.2,
                    },
                    "component_directional_derivatives": {
                        "predicted": -1.0e-4,
                        "decay": -1.0e-5,
                    },
                    "actual_delta": {"cycle": -1.0e-6},
                }
            )
    return {"step": 250, "training_cycle_delta": training_delta, "training_losses_before": {"cycle": 0.01}, "records": records, "curvature": []}


def test_classification_requires_three_consecutive_landmarks_and_supports_under() -> None:
    arms = {route: {"landmarks": [_landmark() for _ in range(3)]} for route in ("swae", "sced")}
    classification, next_action, decision = dlta._classify(arms)
    assert classification == "dense_local_attribution_underdetermined"
    assert next_action == "close_optimizer_trajectory_attribution_family"
    assert decision["shared_supported_mechanisms"] == []


def test_curvature_support_predicate_requires_positive_exact_step() -> None:
    item = {
        "finite": True,
        "baseline_cycle": 0.01,
        "first_order_cycle": -1.0e-5,
        "hvp_quadratic": 1.0e-3,
        "actual_cycle_delta": 2.0e-5,
        "taylor_remainder_cycle": 1.5e-5,
        "finite_difference": {"0.25": 0.010001, "0.5": 0.01001, "1.0": 0.01002},
    }
    assert dlta._curvature_supported(item) is True
    item["finite_difference"]["1.0"] = 0.009
    assert dlta._curvature_supported(item) is False
