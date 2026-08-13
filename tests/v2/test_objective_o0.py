"""Contracts for the research-only Objective E electronic O0 lifecycle."""

from __future__ import annotations

import torch
import pytest

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research.objective_o0 import (
    OBJECTIVE_E_CONTRACT,
    camera_safety_penalty,
    coverage_penalty,
    objective_e2_diversity_penalty,
    deterministic_joint_companion_indices,
    objective_e_dispatch,
    _training_batch,
)
from ldm_project.v2.application.research.objective_o0_spec import load_objective_o0_spec
from ldm_project.v2.application.research.o0_models import ElectronicImageOracleGenerator, build_o0_candidate


def test_objective_o0_cli_and_specs_are_separate_from_formal_runtime() -> None:
    parsed = _parser().parse_args([
        "research", "objective-o0", "preflight",
        "--config", "configs/research/electronic_objective_o0_v1_image.yaml",
        "--device", "cpu",
        "--validate-evidence", "validate.json",
        "--evaluator-checkpoint", "primary.pt",
        "--primary-qualification-evidence", "primary.json",
        "--secondary-evaluator-checkpoint", "secondary.pt",
        "--secondary-training-evidence", "secondary.json",
        "--measurement-bundle", "bundle.json",
        "--measurement-evidence", "measurement.json",
    ])
    assert parsed.research_program == "objective-o0"
    assert load_objective_o0_spec(parsed.config).candidate == "O0-I"
    assert OBJECTIVE_E_CONTRACT


def test_objective_e_dispatch_boundary_and_satisfied_penalties() -> None:
    weights = {"camera": 1.0, "diversity": 1.0, "coverage": 1.0}
    assert objective_e_dispatch(399, weights) == ("A", {name: 0.0 for name in weights})
    assert objective_e_dispatch(400, weights) == ("E", weights)
    labels = torch.arange(10).repeat_interleave(2)
    raw = torch.rand(20, 1, 2, 2, generator=torch.Generator().manual_seed(7))
    assert camera_safety_penalty(raw, labels).item() == 0.0
    assert coverage_penalty(torch.tensor(1.35)).item() == 0.0
    assert coverage_penalty(torch.tensor(1.50)).item() > 0.0


def test_o0_candidates_have_fixed_domain_and_finite_outputs() -> None:
    image = ElectronicImageOracleGenerator()
    command = build_o0_candidate("O0-G")
    labels = torch.arange(10)
    codes = torch.randn(10, 8)
    assert image(codes, labels).shape == (10, 1, 28, 28)
    assert command(codes, labels).shape == (10, 1, 200, 200)
    assert torch.isfinite(image(codes, labels)).all()
    assert torch.isfinite(command(codes, labels)).all()


def test_o0_spec_rejects_nonregistered_activation() -> None:
    spec = load_objective_o0_spec("configs/research/electronic_objective_o0_v1_image.yaml")
    payload = spec.__dict__.copy()
    payload["activation_step"] = 399
    with pytest.raises(ValueError):
        type(spec)(**payload).validate()


def test_e2_diversity_is_exact_zero_when_all_classes_are_in_range() -> None:
    labels = torch.arange(10).repeat_interleave(4)
    features = torch.nn.functional.normalize(torch.randn(40, 8, generator=torch.Generator().manual_seed(11)), dim=1).requires_grad_()
    value, telemetry = objective_e2_diversity_penalty(features, features.detach(), labels)
    assert value.item() == 0.0
    assert value.requires_grad
    value.backward()
    assert features.grad is not None
    assert torch.equal(features.grad, torch.zeros_like(features))
    assert len(telemetry["ratios"]) == 10
    assert all(0.70 <= ratio <= 1.30 for ratio in telemetry["ratios"])


def test_joint_companion_schedule_is_classwise_disjoint_and_deterministic() -> None:
    labels = torch.arange(10).repeat_interleave(10)
    first, first_labels = deterministic_joint_companion_indices(labels, seed=1234, global_step=400, primary_per_class=5)
    second, second_labels = deterministic_joint_companion_indices(labels, seed=1234, global_step=400, primary_per_class=5)
    assert torch.equal(first, second)
    assert torch.equal(first_labels, second_labels)
    assert torch.equal(first_labels[:50], first_labels[50:])
    for label in range(10):
        primary = first[:50][first_labels[:50] == label]
        companion = first[50:][first_labels[50:] == label]
        assert primary.numel() == 5
        assert companion.numel() == 5
        assert torch.unique(torch.cat([primary, companion])).numel() == 10


def test_training_batch_binds_same_schedule_across_candidates_and_steps() -> None:
    labels = torch.arange(10).repeat_interleave(10)
    first = _training_batch(labels, run_seed=1234, global_step=400, batch_size=50, code_dim=8, device=torch.device("cpu"))
    second = _training_batch(labels, run_seed=1234, global_step=400, batch_size=50, code_dim=8, device=torch.device("cpu"))
    next_step = _training_batch(labels, run_seed=1234, global_step=401, batch_size=50, code_dim=8, device=torch.device("cpu"))
    assert first[-1] == second[-1]
    assert first[-1]["schedule_hash"] != next_step[-1]["schedule_hash"]
    assert set(first[-1]) == {"global_step", "run_seed", "code_seed", "primary_indices_hash", "companion_indices_hash", "schedule_hash"}


def test_o0_harness_revision_is_bumped_for_lifecycle_contract() -> None:
    image = load_objective_o0_spec("configs/research/electronic_objective_o0_v1_image.yaml")
    generator = load_objective_o0_spec("configs/research/electronic_objective_o0_v1_generator.yaml")
    assert image.harness_revision == generator.harness_revision == 4
