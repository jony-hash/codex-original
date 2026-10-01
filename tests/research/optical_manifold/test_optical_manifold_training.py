from __future__ import annotations

import torch
from torch import nn

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research.optical_manifold import optical_manifold as base
from ldm_project.v2.application.research.optical_manifold import optical_manifold_training as pilot


def test_pilot_contract_and_review_are_exactly_registered() -> None:
    spec = pilot._load_pilot_spec(
        "configs/research/optical_manifold/lightgen_optical_manifold_identity_pilot_v1.yaml"
    )
    review, _ = pilot._load_review(
        "configs/research/optical_manifold/lightgen_optical_manifold_identity_pilot_review_v1.json",
        spec,
    )
    assert spec.payload["training"]["max_steps"] == 1000
    assert spec.payload["gates"]["heldout_relative_mse_reduction_min"] == 0.05
    assert review["reviewer_role"] == "project_owner"
    assert review["formal_training_authorized"] is False


def test_pilot_cli_actions_are_lazy_and_registered() -> None:
    parser = _parser()
    common = [
        "research",
        "optical-manifold",
        "train-validate",
        "--config",
        "pilot.yaml",
        "--review-contract",
        "review.json",
        "--source-validation-evidence",
        "validation.json",
        "--source-preflight-evidence",
        "preflight.json",
        "--dataset-root",
        "data",
        "--output",
        "validation.json",
    ]
    args = parser.parse_args(common)
    assert args.research_action == "train-validate"
    assert args.review_contract == "review.json"
    for action, extra in (
        (
            "train-preflight",
            ["--training-validation-evidence", "training_validation.json"],
        ),
        (
            "train",
            [
                "--training-validation-evidence",
                "training_validation.json",
                "--training-preflight-evidence",
                "training_preflight.json",
            ],
        ),
    ):
        args = parser.parse_args(
            [
                "research",
                "optical-manifold",
                action,
                "--config",
                "pilot.yaml",
                "--review-contract",
                "review.json",
                "--dataset-root",
                "data",
                "--output",
                "training.json",
                *extra,
            ]
        )
        assert args.research_action == action


def test_stratified_schedule_has_one_unused_example_per_label() -> None:
    labels = torch.arange(10).repeat_interleave(6000)
    train_indices, validation_indices = pilot._stratified_split(labels)
    schedule = pilot._schedule(train_indices, labels)
    assert train_indices.numel() == 55000
    assert validation_indices.numel() == 5000
    assert schedule.shape == (1000, 10)
    schedule_labels = labels.index_select(0, schedule.reshape(-1)).reshape(1000, 10)
    assert torch.equal(schedule_labels, torch.arange(10).expand(1000, -1))
    for label in range(10):
        selected = schedule[:, label]
        assert torch.unique(selected).numel() == 1000


def test_pilot_step_rejects_nan_measurements() -> None:
    assert pilot._finite_step(
        {
            "loss": float("nan"),
            "camera_min": 0.0,
            "camera_max": 1.0,
            "gradient_norms": {},
            "update_norms": {},
            "camera_finite": True,
        }
    ) is False


def test_heldout_permutation_is_across_panel_examples() -> None:
    class Echo(nn.Module):
        def forward(self, amplitude, *, activation_checkpoint=False):
            del activation_checkpoint
            return base.OpticalManifoldOutput(amplitude, amplitude)

    images = torch.stack(
        [torch.full((1, 28, 28), value) for value in (0.1, 0.2, 0.4, 0.8)]
    )
    data = pilot.PilotData(
        images=images,
        labels=torch.zeros(4, dtype=torch.long),
        train_indices=torch.empty(0, dtype=torch.long),
        validation_indices=torch.empty(0, dtype=torch.long),
        schedule=torch.empty(0, 10, dtype=torch.long),
        heldout_indices=torch.empty(0, dtype=torch.long),
        preflight_heldout_indices=torch.empty(0, dtype=torch.long),
        manifest={},
    )
    indices = torch.arange(4)
    paired = pilot._evaluate(Echo(), data, indices, torch.device("cpu"))
    permuted = pilot._evaluate(
        Echo(), data, indices, torch.device("cpu"), target_roll=1
    )
    assert paired != permuted
