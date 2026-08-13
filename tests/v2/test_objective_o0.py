"""Contracts for the research-only Objective E electronic O0 lifecycle."""

from __future__ import annotations

import torch
import pytest

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research.objective_o0 import (
    OBJECTIVE_E_CONTRACT,
    camera_safety_penalty,
    coverage_penalty,
    objective_e_dispatch,
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
