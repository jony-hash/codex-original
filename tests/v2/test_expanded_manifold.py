from pathlib import Path

import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.conditional_expanded_manifold import _rank_summary
from ldm_project.v2.config import load_config
from ldm_project.v2.student.generator import ConditionalNonlinearExpandedManifoldGenerator


def _generator() -> ConditionalNonlinearExpandedManifoldGenerator:
    return ConditionalNonlinearExpandedManifoldGenerator(
        class_carrier_logits=torch.zeros(10, 8, 8),
        class_modes=torch.randn(10, 32, 8, 8),
        manifold_scales=torch.full((10, 32), 0.05),
        shape=(8, 8),
    )


def test_expanded_manifold_zero_code_is_exact_carrier_anchor():
    generator = _generator()
    labels = torch.arange(10)
    command = generator(torch.zeros(10, 8), labels)
    assert torch.equal(command, generator.class_carrier_command(labels))
    assert generator.coefficients(torch.zeros(10, 8), labels).shape == (10, 32)


def test_expanded_manifold_is_class_conditioned_and_nonlinear():
    generator = _generator()
    code = torch.full((1, 8), 0.5)
    first = generator.coefficients(code, torch.tensor([0]))
    second = generator.coefficients(code, torch.tensor([1]))
    assert not torch.equal(first[:, 8:], second[:, 8:])
    assert not torch.equal(first[:, 8:], -generator.coefficients(-code, torch.tensor([0]))[:, 8:])


def test_expanded_manifold_cli_and_diagnostic_config_are_explicit():
    config = load_config(Path("configs/v2/diagnostic/conditional_nonlinear_expanded_manifold.yaml"))
    parser = _parser()
    capacity = parser.parse_args(["audit", "conditional-nonlinear-expanded-manifold-capacity", "--config", "config.yaml", "--dataset-root", "data", "--evaluator-checkpoint", "evaluator.pt", "--electronic-initialization", "source.pt"])
    admission = parser.parse_args(["audit", "conditional-nonlinear-expanded-manifold-admission", "--config", "config.yaml", "--dataset-root", "data", "--evaluator-checkpoint", "evaluator.pt", "--electronic-initialization", "source.pt", "--capacity-evidence", "capacity.json", "--basis-artifact", "basis.pt"])
    assert config.architecture == "conditional_nonlinear_expanded_manifold_v7"
    assert capacity.electronic_initialization == "source.pt"
    assert admission.basis_artifact == "basis.pt"


def test_rank_gate_uses_the_eighth_singular_value_not_centered_null_tail():
    # Centering 17 samples always introduces a final zero singular value.  A
    # true rank-8 panel must still pass the sigma8 gate.
    left = torch.randn(17, 8)
    right = torch.randn(8, 24)
    rank8 = left @ right
    rank8 -= rank8.mean(0, keepdim=True)
    assert _rank_summary(rank8)["rank"] >= 8
    assert _rank_summary(rank8)["sigma8_over_sigma1"] > 1.0e-3
    rank7 = rank8[:, :7] @ torch.randn(7, 24)
    assert _rank_summary(rank7)["rank"] == 7
