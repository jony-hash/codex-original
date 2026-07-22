from __future__ import annotations

import pytest
import torch

from ldm_project.v2.evidence import EvidenceRecordV2, GateResult, require_passed_evidence
from ldm_project.v2.student.admission import CapacityMetrics, PrototypeMetrics, balanced_code_fraction, capacity_gate, prototype_gate
from ldm_project.v2.student.capacity import build_code_basis, capacity_metrics, supported_dct_candidates
from ldm_project.v2.student.capacity import finite_difference_response_matrices


def _prototype_pass() -> GateResult:
    return prototype_gate(PrototypeMetrics(0.8, 0.8, 1.0, 0.01, True, True, True, True), list(range(10)))


def test_prototype_gate_requires_every_manual_label() -> None:
    assert _prototype_pass().passed
    failed = prototype_gate(PrototypeMetrics(0.8, 0.8, 1.0, 0.01, True, True, True, True), list(range(9)))
    assert not failed.passed


def test_capacity_gate_blocks_when_code_is_not_transportable() -> None:
    gate = capacity_gate(CapacityMetrics(0.25, 0.24, 0.95, 0.08, True, False, True), _prototype_pass())
    assert not gate.passed
    passed = capacity_gate(CapacityMetrics(0.25, 0.25, 0.95, 0.08, True, False, True), _prototype_pass())
    assert passed.passed


def test_dct_svd_basis_and_capacity_metrics_are_label_fixed() -> None:
    support = torch.ones(10, 8, 8)
    candidates = supported_dct_candidates(support[0], candidate_count=8)
    assert candidates.shape == (8, 8, 8)
    response = torch.randn(10, 8, 16)
    artifact = build_code_basis(response, support, candidate_count=8, seed=1234, source_hash="source")
    assert artifact.basis.shape == (10, 8, 8, 8)
    assert artifact.normalizer.shape == (10, 1, 1)
    commands = torch.rand(10, 4, 8, 8)
    camera = commands.square()
    values = capacity_metrics(commands, commands.square(), camera, support)
    assert all(0.0 <= value <= 1.0 for value in values)
    assert balanced_code_fraction(commands) >= 0.0


def test_evidence_state_machine_blocks_diagnostic_and_dirty_acceptance() -> None:
    gate = GateResult("unit", True, {})
    diagnostic = EvidenceRecordV2(2, "capacity", "diagnostic", "diagnostic", "commit", "patch", "config", {}, (gate,), ())
    with pytest.raises(ValueError, match="passed"):
        require_passed_evidence(diagnostic, kind="capacity")
    dirty = EvidenceRecordV2(2, "capacity", "acceptance", "passed", "commit", "patch", "config", {}, (gate,), ("train",))
    with pytest.raises(ValueError, match="clean"):
        dirty.validate()


def test_finite_difference_responses_are_frozen_and_finite() -> None:
    class System(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def command(self, value: torch.Tensor) -> torch.Tensor:
            return value

        def forward(self, value: torch.Tensor):
            return type("Result", (), {"raw_camera": type("Raw", (), {"tensor": value * self.scale})()})()

    responses, unchanged = finite_difference_response_matrices(System(), torch.full((10, 1, 4, 4), 0.5), torch.ones(10, 4, 4), candidate_count=8, epsilon=1.0e-3)
    assert responses.shape == (10, 8, 16)
    assert torch.isfinite(responses).all() and unchanged
