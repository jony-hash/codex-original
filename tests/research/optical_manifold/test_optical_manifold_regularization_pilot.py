from __future__ import annotations

import inspect

import pytest
import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research.optical_manifold import optical_manifold as base
from ldm_project.v2.application.research.optical_manifold import optical_manifold_full_epoch as full_epoch
from ldm_project.v2.application.research.optical_manifold import optical_manifold_regularization_pilot as pilot
from ldm_project.v2.config import load_config


CONFIG = "configs/research/optical_manifold/lightgen_optical_manifold_identity_regularization_pilot_v1.yaml"
REVIEW = "configs/research/optical_manifold/lightgen_optical_manifold_identity_regularization_pilot_review_v1.json"


def test_pilot_contract_review_and_lazy_cli_are_registered() -> None:
    spec = pilot._load_spec(CONFIG)
    pilot._assert_contract(spec)
    review, _ = pilot._load_review(REVIEW, spec)
    assert review["bounded_research_training_authorized"] is True
    assert review["training_authorized"] is False
    assert spec.payload["regularization"]["lambda"] == pilot.LAMBDA

    parser = _parser()
    for action, extra in (
        ("manifold-regularization-pilot-validate", []),
        (
            "manifold-regularization-pilot-preflight",
            ["--training-validation-evidence", "validation.json"],
        ),
        (
            "manifold-regularization-pilot-train",
            [
                "--training-validation-evidence",
                "validation.json",
                "--training-preflight-evidence",
                "preflight.json",
            ],
        ),
    ):
        args = parser.parse_args(
            [
                "research",
                "optical-manifold",
                action,
                "--config",
                CONFIG,
                "--review-contract",
                REVIEW,
                "--dataset-root",
                "data",
                "--output",
                "result.json",
                *extra,
            ]
        )
        assert args.research_action == action


def test_pilot_schedule_is_balanced_deterministic_and_disjoint() -> None:
    labels = torch.arange(60000, dtype=torch.long).remainder(10)
    train_indices = torch.arange(55000, dtype=torch.long)
    schedule = pilot._pilot_schedule(train_indices, labels)
    repeat = pilot._pilot_schedule(train_indices, labels)
    assert torch.equal(schedule, repeat)
    assert schedule.shape == (pilot.STEPS_PER_ARM, pilot.LOGICAL_BATCH_SIZE)
    assert torch.unique(schedule).numel() == schedule.numel()
    schedule_labels = labels.index_select(0, schedule.reshape(-1)).reshape(schedule.shape)
    expected_labels = torch.arange(10).repeat_interleave(10).expand(pilot.STEPS_PER_ARM, -1)
    assert torch.equal(schedule_labels, expected_labels)
    assert not torch.isin(schedule.reshape(-1), torch.arange(55000, 56000)).any()


def test_fixed_rank_replay_covers_every_rank_for_every_direction() -> None:
    generator = torch.Generator(device="cpu").manual_seed(1234)
    scores = torch.randn(100, pilot.PCA_COMPONENTS, generator=generator)
    directions = torch.randn(pilot.FIXED_DIRECTIONS, pilot.PCA_COMPONENTS, generator=generator)
    directions = torch.nn.functional.normalize(directions, dim=1)
    coefficients, ranks = pilot._fixed_rank_coefficients(scores, directions)
    assert coefficients.shape == (100, 64)
    assert ranks.shape == (100, 64)
    assert pilot.audit._rank_assignment_is_permutation(ranks)
    assert torch.equal(
        torch.sort(ranks, dim=0).values,
        torch.arange(100).reshape(-1, 1).expand(100, 64),
    )


def test_pilot_parameter_and_reset_adam_contract() -> None:
    config = load_config(
        "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
    )
    model = base.IdentityOpticalManifoldAutoencoder(config.system)
    contract = pilot._parameter_contract(model)
    assert tuple(contract["trainable_parameter_names"]) == pilot.EXPECTED_TRAINABLE
    optimizer = full_epoch._make_full_optimizer(model)
    assert optimizer.state_dict()["state"] == {}
    assert [group["params"] for group in optimizer.state_dict()["param_groups"]] == [[0, 1]]


def test_resume_header_rejects_wrong_stage_checkpoint(tmp_path) -> None:
    path = tmp_path / "wrong.pt"
    torch.save({"schema": "other", "arm": "regularized", "step": 2}, path)
    with pytest.raises(ValueError, match="resume checkpoint"):
        pilot._resume_header(path)


def test_resume_payload_binds_both_arm_completion_and_no_formal_candidate() -> None:
    config = load_config(
        "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
    )
    model = base.IdentityOpticalManifoldAutoencoder(config.system)
    optimizer = full_epoch._make_full_optimizer(model)
    data = pilot.PilotData(
        regularization_data=None,  # type: ignore[arg-type]
        schedule=torch.empty(0, pilot.LOGICAL_BATCH_SIZE, dtype=torch.long),
        heldout_indices=torch.empty(0, dtype=torch.long),
        manifest={"manifest_hash": "manifest"},
    )
    payload = pilot._resume_payload(
        model,
        optimizer,
        arm="mse_only",
        step=2,
        history=[{"step": 1}, {"step": 2}],
        spec=pilot.PilotSpec(
            path=CONFIG,
            payload={},
            config_hash="config",
        ),
        data=data,
        evidence_hashes={"source": "hash"},
        completed_arms={
            "regularized": {
                "status": "completed",
                "path": "/tmp/regularized.pt",
                "sha256": "0" * 64,
            }
        },
    )
    assert payload["completed_arms"]["regularized"]["status"] == "completed"
    assert payload["formal_candidate_checkpoint"] is False
    assert payload["reset_adam_from_source"] is True


def test_pilot_does_not_self_bind_training_evidence_hash() -> None:
    source = inspect.getsource(pilot)
    assert "training_evidence_placeholder" not in source
    assert '"training_evidence": canonical_sha256(payload)' not in source
