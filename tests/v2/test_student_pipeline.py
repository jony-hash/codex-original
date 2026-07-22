from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ldm_project.v2.evidence import EvidenceContextV2, EvidenceRecordV2, GateResult
from ldm_project.v2.student.capacity import CodeBasisArtifactV2
from ldm_project.v2.student.evaluation import CodeAcceptanceMetrics, code_acceptance_gate
from ldm_project.v2.student.generator import LabelPrototypeBasisGenerator
from ldm_project.v2.student.training import calibrate_prototypes, train_code_conditioned


class ToyRawCameraSystem(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.phase = nn.Parameter(torch.tensor(0.0))

    def command(self, command: torch.Tensor) -> torch.Tensor:
        return command

    def forward(self, command: torch.Tensor):
        return SimpleNamespace(raw_camera=SimpleNamespace(tensor=command * torch.sigmoid(self.phase)))

    def forward_from_generator(self, generator, code: torch.Tensor, labels: torch.Tensor, **kwargs):
        return self.forward(generator(code, labels))


def _artifact() -> CodeBasisArtifactV2:
    basis = torch.randn(10, 8, 4, 4)
    normalizer = basis.abs().sum(dim=1).amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
    return CodeBasisArtifactV2(basis, normalizer, 8, 1234, "unit")


def _evidence(kind: str) -> EvidenceRecordV2:
    return EvidenceRecordV2(2, kind, "acceptance", "passed", "commit", None, "config", EvidenceContextV2("system", "config"), {}, (GateResult(kind, True, {}),), ("train",))


def test_generator_uses_fixed_label_normalizer_and_reports_clipping() -> None:
    generator = LabelPrototypeBasisGenerator(code_dim=3, basis_artifact=_artifact())
    command, diagnostic = generator.forward_with_diagnostics(torch.randn(2, 3), torch.tensor([0, 1]))
    assert command.shape == (2, 1, 4, 4)
    assert diagnostic.coefficients.shape == (2, 8)
    assert generator.label_normalizer.shape == (10, 1, 1)


def test_training_requires_passed_and_matching_capacity_evidence() -> None:
    generator = LabelPrototypeBasisGenerator(code_dim=3, basis_artifact=_artifact())
    system = ToyRawCameraSystem()
    summary = calibrate_prototypes(system, generator, torch.full((10, 1, 4, 4), 0.5), exposure_scale=1.0, epochs=1, learning_rate=1e-3)
    assert summary.gradients_finite
    codes = torch.randn(20, 3)
    labels = torch.arange(10).repeat_interleave(2)
    targets = torch.full((20, 1, 4, 4), 0.5)
    weights = {"prototype": 1.0, "paired_residual": 1.0, "code_transport": 0.01, "clipping": 0.1, "saturation": 0.1, "phase_tv": 1e-5}
    trained = train_code_conditioned(system, generator, codes, labels, targets, exposure_scale=1.0, prototype_evidence=_evidence("prototype"), capacity_evidence=_evidence("capacity"), expected_basis_hash=generator.basis_artifact_hash, epochs=1, learning_rate=1e-3, batch_size=20, weights=weights)
    assert torch.isfinite(torch.tensor(trained.loss))
    with pytest.raises(ValueError, match="basis artifact"):
        train_code_conditioned(system, generator, codes, labels, targets, exposure_scale=1.0, prototype_evidence=_evidence("prototype"), capacity_evidence=_evidence("capacity"), expected_basis_hash="other", epochs=1, learning_rate=1e-3, batch_size=20, weights=weights)


def test_code_acceptance_requires_bootstrap_and_manual_review() -> None:
    metrics = CodeAcceptanceMetrics(0.25, 0.70, 0.68, 0.001, 1234, 10000)
    assert code_acceptance_gate(metrics, list(range(10))).passed
    assert not code_acceptance_gate(metrics, list(range(9))).passed
